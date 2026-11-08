from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

import main
from qzone_bridge.persona_guard import PersonaConsistencyError
from qzone_bridge.social import QzonePost
from qzone_bridge.drafts import DraftStore
from qzone_bridge.media import PostPayload
from qzone_bridge.page_api import QzonePageApi
from qzone_bridge.page_life import PageLifeJobStore


def test_public_batch_comment_keeps_successes_around_review_failure_and_like_error() -> None:
    async def scenario() -> None:
        plugin = object.__new__(main.QzoneStablePlugin)
        sent: list[str] = []
        liked: list[str] = []

        async def generate(_event, post):
            if post.fid == "two":
                raise PersonaConsistencyError("BUDGET_EXHAUSTED")
            return f"comment-{post.fid}"

        class PostService:
            async def comment_post(self, post, content, *, private=False):
                sent.append(post.fid)
                return {"commentid": f"comment-{post.fid}"}

            async def like_post(self, post):
                liked.append(post.fid)
                if post.fid == "one":
                    raise RuntimeError("TOKEN must not appear")
                return {"liked": True}

        plugin._generate_comment_text = generate
        plugin._post_service = lambda: PostService()
        posts = [QzonePost(hostuin=100, fid=fid, summary=fid, local_id=index)
                 for index, fid in enumerate(("one", "two", "three"), 1)]
        results = await plugin._comment_posts_for_tool(
            SimpleNamespace(), posts, auto_generate=True, private=False, like_after_comment=True,
        )
        assert sent == ["one", "three"]
        assert liked == ["one", "three"]
        assert [item["status"] for item in results] == ["sent", "failed", "sent"]
        assert results[0]["like_error"]["code"] == "LIKE_FAILED"
        assert "TOKEN" not in str(results)
        summary = plugin._comment_tool_payload(results)
        assert summary["ok"] and summary["partial"]
        assert summary["result"]["count"] == 2 and summary["result"]["failed_count"] == 2

    asyncio.run(scenario())


def test_public_batch_comment_propagates_native_cancellation() -> None:
    async def scenario() -> None:
        plugin = object.__new__(main.QzoneStablePlugin)
        async def generate(_event, _post):
            raise asyncio.CancelledError()
        plugin._generate_comment_text = generate
        plugin._post_service = lambda: SimpleNamespace(comment_post=lambda *args, **kwargs: None)
        post = QzonePost(hostuin=100, fid="one", summary="one", local_id=1)
        with pytest.raises(asyncio.CancelledError):
            await plugin._comment_posts_for_tool(SimpleNamespace(), [post], auto_generate=True)

    asyncio.run(scenario())


def test_page_and_chat_publish_share_durable_life_claim_in_both_orders(tmp_path: Path) -> None:
    class Controller:
        def __init__(self) -> None:
            self.published: list[dict[str, object]] = []
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False,
                    "login_uin": 123, "login_nickname": "Tester"}
        async def publish_post(self, **kwargs):
            self.published.append(kwargs)
            return {"fid": f"fid-{len(self.published)}", "media_count": 0, "photo_count": 0}

    async def make_api(path: Path, controller: Controller):
        async def generate(_progress):
            return PostPayload("daily", [])
        return QzonePageApi(
            controller=controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=DraftStore(path),
        )

    async def scenario() -> None:
        first_controller = Controller()
        first_api = await make_api(tmp_path / "page-first.json", first_controller)
        first_job = await first_api.start_life_draft({"action": "generate_draft", "request_id": "page-first"})
        first_id = first_job["data"]["id"]
        await first_api._life_jobs.get(first_id, 123)
        await first_api._life_jobs._jobs[first_id].task
        page_result = await first_api.publish({"content": "edited by page", "life_job_id": first_id})
        first_store = DraftStore(tmp_path / "page-first.json")
        first_draft = await first_store.get_async(page_result["data"]["draft_id"] if "draft_id" in page_result["data"] else 1)
        assert len(first_controller.published) == 1 and first_draft is not None and first_draft.status == "published"
        page_repeat = await first_store.claim_publish_async(
            first_draft.id, account_uin=123, content="edited by page", media=[], life_job_id=first_id, life_owner_uin=123,
        )
        assert page_repeat["state"] == "already_published"

        second_controller = Controller()
        second_api = await make_api(tmp_path / "chat-first.json", second_controller)
        second_job = await second_api.start_life_draft({"action": "generate_draft", "request_id": "chat-first"})
        second_id = second_job["data"]["id"]
        second_life = await second_api._life_jobs.get(second_id, 123)
        await second_life.task
        second_store = DraftStore(tmp_path / "chat-first.json")
        chat_claim = await second_store.claim_publish_async(
            second_life.draft_id, account_uin=123, content="edited by chat", media=[],
            life_job_id=second_id, life_owner_uin=123,
        )
        assert chat_claim["state"] == "acquired"
        assert await second_store.complete_publish_async(
            second_life.draft_id, token=chat_claim["token"], account_uin=123,
            digest=chat_claim["digest"], result={"fid": "chat-fid"}, fid="chat-fid",
        )
        page_again = await second_api.publish({"content": "edited by chat", "life_job_id": second_id})
        assert page_again["data"]["fid"] == "chat-fid"
        assert second_controller.published == []

        await first_api.close()
        await second_api.close()

    asyncio.run(scenario())


def test_lost_page_cache_and_stale_save_cannot_reset_unknown_or_dispatch(tmp_path: Path) -> None:
    class Controller:
        def __init__(self): self.published = []
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False, "login_uin": 123}
        async def publish_post(self, **kwargs):
            self.published.append(kwargs)
            return {"fid": "unexpected", "media_count": 0}

    async def scenario() -> None:
        controller = Controller()
        path = tmp_path / "unknown.json"
        async def generate(_progress): return PostPayload("daily", [])
        store = DraftStore(path)
        api = QzonePageApi(
            controller=controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "unknown"})
        job_id = started["data"]["id"]
        job = await api._life_jobs.get(job_id, 123)
        await job.task
        draft = await store.get_async(job.draft_id)
        claim = await store.claim_publish_async(
            draft.id, account_uin=123, content="daily", media=[], life_job_id=job_id, life_owner_uin=123,
        )
        assert claim["state"] == "acquired"
        assert await store.mark_unknown_async(
            draft.id, token=claim["token"], account_uin=123, digest=claim["digest"],
            diagnostic={"code": "TIMEOUT", "phase": "controller_call"},
        )
        await store.save_async(type(draft).from_dict({**draft.to_dict(), "status": "pending", "content": "stale"}))
        api._life_jobs = PageLifeJobStore()  # Simulate a service restart/lost in-memory job cache.
        recovered = await api.get_life_draft({"id": job_id})
        assert recovered["data"]["publish_state"] == "unknown"
        assert "source" not in str(recovered["data"]) and "source" not in str(recovered["data"].get("media", []))
        with pytest.raises(Exception) as caught:
            await api.publish({"content": "daily", "life_job_id": job_id})
        assert getattr(caught.value, "code", "") == "LIFE_UNKNOWN"
        assert controller.published == []
        current = await store.get_async(draft.id)
        assert current.status == "unknown" and current.content == "daily"
        await api.close()

    asyncio.run(scenario())


def test_page_claim_storage_failure_stops_before_controller_dispatch(tmp_path: Path, monkeypatch) -> None:
    class Controller:
        def __init__(self): self.published = []
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False, "login_uin": 123}
        async def publish_post(self, **kwargs):
            self.published.append(kwargs)
            return {"fid": "unexpected", "media_count": 0}

    async def scenario() -> None:
        controller = Controller()
        store = DraftStore(tmp_path / "write-failure.json")
        async def generate(_progress): return PostPayload("daily", [])
        api = QzonePageApi(
            controller=controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "write-failure"})
        job = await api._life_jobs.get(started["data"]["id"], 123)
        await job.task
        def fail_write(_payload): raise OSError("fixture storage failure")
        monkeypatch.setattr(store._store, "_write_unlocked", fail_write)
        with pytest.raises(OSError):
            await api.publish({"content": "edited", "life_job_id": job.id})
        assert controller.published == []
        await api.close()

    asyncio.run(scenario())
