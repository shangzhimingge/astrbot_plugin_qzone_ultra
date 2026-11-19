from __future__ import annotations

import asyncio
import multiprocessing
import threading
from pathlib import Path
from typing import Any

import pytest

from qzone_bridge.drafts import DraftPost, DraftStore, public_publish_result
from qzone_bridge.json_store import StoreReadError


def _claim_in_process(path: str, barrier: Any, queue: Any) -> None:
    async def run() -> None:
        barrier.wait(timeout=20)
        result = await DraftStore(Path(path)).claim_publish_async(
            1, account_uin=99, content="same", media=[], sync_weibo=False,
        )
        queue.put(result["state"])
    asyncio.run(run())


@pytest.mark.asyncio
async def test_claim_is_shared_by_store_instances_and_fenced(tmp_path: Path) -> None:
    path = tmp_path / "drafts.json"
    left, right = DraftStore(path), DraftStore(path)
    draft = await left.add_async(author_uin=11, content="original")
    args = dict(draft_id=draft.id, account_uin=99, content="edited", media=[], sync_weibo=False)
    first, second = await asyncio.gather(left.claim_publish_async(**args), right.claim_publish_async(**args))
    assert sorted([first["state"], second["state"]]) == ["acquired", "blocked"]
    claim = first if first["state"] == "acquired" else second
    await right.save_async(DraftPost.from_dict({**draft.to_dict(), "status": "pending", "content": "stale"}))
    current = await left.get_async(draft.id)
    assert current is not None and current.status == "publishing" and current.content == "edited"
    assert await left.complete_publish_async(
        draft.id, token=claim["token"], account_uin=99, digest=claim["digest"], result={"fid": "fid-1"}
    )
    repeat = await right.claim_publish_async(**args)
    assert repeat["state"] == "already_published"
    changed = await right.claim_publish_async(**{**args, "content": "different"})
    assert changed["state"] == "conflict"


@pytest.mark.asyncio
async def test_unknown_and_legacy_approved_are_never_reclaimed(tmp_path: Path) -> None:
    store = DraftStore(tmp_path / "drafts.json")
    draft = await store.add_async(author_uin=1, content="x")
    claim = await store.claim_publish_async(draft.id, account_uin=2, content="x", media=[])
    assert claim["state"] == "acquired"
    assert await store.mark_unknown_async(
        draft.id, token=claim["token"], account_uin=2, digest=claim["digest"],
        diagnostic={"code": "NETWORK", "phase": "transport"},
    )
    assert (await store.claim_publish_async(draft.id, account_uin=2, content="x", media=[]))["state"] == "blocked"
    legacy = DraftPost.from_dict({"id": 40, "author_uin": 1, "status": "approved"})
    await store.save_async(legacy)
    assert (await store.claim_publish_async(40, account_uin=2, content="x", media=[]))["state"] == "blocked"


def test_strict_store_corruption_does_not_quarantine_or_reset(tmp_path: Path) -> None:
    path = tmp_path / "drafts.json"
    original = b"{ malformed"
    path.write_bytes(original)
    with pytest.raises(StoreReadError):
        DraftStore(path).list()
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.corrupt.*"))


def test_separate_processes_share_one_publication_claim(tmp_path: Path) -> None:
    path = tmp_path / "drafts.json"
    asyncio.run(DraftStore(path).add_async(author_uin=11, content="same"))
    context = multiprocessing.get_context("spawn")
    barrier, queue = context.Barrier(2), context.Queue()
    workers = [context.Process(target=_claim_in_process, args=(str(path), barrier, queue)) for _ in range(2)]
    for worker in workers:
        worker.start()
    states = [queue.get(timeout=30), queue.get(timeout=30)]
    for worker in workers:
        worker.join(30)
        assert worker.exitcode == 0
    assert sorted(states) == ["acquired", "blocked"]


@pytest.mark.asyncio
async def test_cancelled_claim_settles_worker_and_releases_only_pre_dispatch_token(tmp_path: Path, monkeypatch: Any) -> None:
    store = DraftStore(tmp_path / "drafts.json")
    draft = await store.add_async(author_uin=11, content="x")
    entered, release = threading.Event(), threading.Event()
    original = store._store.transact

    def slow(updater: Any) -> Any:
        entered.set()
        assert release.wait(10)
        return original(updater)

    monkeypatch.setattr(store._store, "transact", slow)
    task = asyncio.create_task(store.claim_publish_async(draft.id, account_uin=99, content="x", media=[]))
    assert await asyncio.to_thread(entered.wait, 10)
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    monkeypatch.setattr(store._store, "transact", original)
    assert (await store.get_async(draft.id)).status == "pending"
    # A later explicit action can acquire a fresh token; that live claim remains
    # blocked from a concurrent duplicate attempt.
    retry = await store.claim_publish_async(draft.id, account_uin=99, content="x", media=[])
    assert retry["state"] == "acquired"
    duplicate = await store.claim_publish_async(draft.id, account_uin=99, content="x", media=[])
    assert duplicate["state"] == "blocked"
    assert await store.release_publish_async(
        draft.id, token="stale-cancelled-token", account_uin=99, digest=retry["digest"]
    ) is False
    assert (await store.get_async(draft.id)).status == "publishing"
    assert (await store.claim_publish_async(draft.id, account_uin=99, content="x", media=[]))["state"] == "blocked"


def test_public_publish_receipt_excludes_raw_and_media_paths() -> None:
    receipt = public_publish_result({
        "fid": "feed-123", "message": "published", "media_count": 1, "photo_count": 1,
        "raw": {"response": "RAW_SENTINEL"}, "post": {"images": ["C:/secret/photo.png"]},
        "path": "C:/secret/photo.png",
    })
    assert receipt == {"fid": "feed-123", "media_count": 1, "photo_count": 1}
    assert "RAW_SENTINEL" not in str(receipt) and "photo.png" not in str(receipt)


@pytest.mark.asyncio
async def test_claim_lock_or_write_failure_never_persists_admission(tmp_path: Path, monkeypatch: Any) -> None:
    store = DraftStore(tmp_path / "drafts.json")
    draft = await store.add_async(author_uin=11, content="original")

    def fail_write(_payload: Any) -> None:
        raise OSError("fixture write failure")

    monkeypatch.setattr(store._store, "_write_unlocked", fail_write)
    with pytest.raises(OSError):
        await store.claim_publish_async(draft.id, account_uin=99, content="edited", media=[])
    monkeypatch.undo()
    saved = await DraftStore(store.path).get_async(draft.id)
    assert saved is not None and saved.status == "pending" and saved.content == "original"

    from contextlib import contextmanager
    @contextmanager
    def fail_lock():
        raise OSError("fixture lock failure")
        yield
    monkeypatch.setattr(store._store, "_process_lock", fail_lock)
    with pytest.raises(OSError):
        await store.claim_publish_async(draft.id, account_uin=99, content="edited", media=[])
    assert (await DraftStore(store.path).get_async(draft.id)).status == "pending"


@pytest.mark.asyncio
async def test_inconsistent_pending_claim_fields_are_blocked_without_rewrite(tmp_path: Path) -> None:
    path = tmp_path / "inconsistent.json"
    store = DraftStore(path)
    draft = await store.add_async(author_uin=1, content="source")
    payload = store._store.read()
    raw = payload["items"][0]
    raw.update({"publish_claim_token": "stale-token", "publish_account_uin": 77,
                "publish_digest": "a" * 64, "publish_snapshot": {"content": "old"}})
    store._store.write(payload)
    before = path.read_bytes()
    outcome = await store.claim_publish_async(
        draft.id, account_uin=77, content="source", media=[]
    )
    assert outcome["state"] == "blocked"
    assert path.read_bytes() == before
