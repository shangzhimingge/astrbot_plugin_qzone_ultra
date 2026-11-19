from __future__ import annotations

import asyncio
from types import SimpleNamespace
from pathlib import Path

import pytest

import main
from qzone_bridge.persona_guard import PersonaConsistencyError
from qzone_bridge.social import QzonePost
from qzone_bridge.drafts import DraftStore, public_publish_result
from qzone_bridge.media import PostPayload
from qzone_bridge.page_api import QzonePageApi
from qzone_bridge.page_life import PageLifeJobStore
from qzone_bridge.errors import DaemonUnavailableError


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


def test_registered_public_comment_tool_returns_partial_batch_and_cancellation(monkeypatch) -> None:
    async def scenario():
        plugin = object.__new__(main.QzoneStablePlugin)
        sent = []
        logs = []
        payloads = []
        posts = [QzonePost(hostuin=100, fid=f"public-{i}", summary="post", local_id=i)
                 for i in range(1, 4)]
        async def generate(_event, post):
            if post.fid == "public-2":
                raise PersonaConsistencyError("BUDGET_EXHAUSTED")
            return f"public comment {post.fid}"
        class Service:
            async def comment_post(self, post, content, *, private=False):
                sent.append(post.fid)
                return {"commentid": post.fid}
            async def like_post(self, post): return {"liked": True}
        async def ready(*_args, **_kwargs): return None
        async def selected(*_args, **_kwargs): return SimpleNamespace()
        async def feed(*_args, **_kwargs): return posts
        async def model_reply(_event, payload, _fallback):
            payloads.append(payload)
            return "public partial result"
        plugin._is_admin = lambda _event: True
        plugin._ensure_cookie_ready = ready
        plugin._ensure_daemon = ready
        plugin._selection_from_tool_args = lambda *_args, **_kwargs: SimpleNamespace()
        plugin._posts_for_selection = feed
        plugin._generate_comment_text = generate
        plugin._post_service = lambda: Service()
        plugin._self_id = lambda _event: 100
        plugin._log_tool_call_result = logs.append
        plugin._ask_llm_tool_reply = model_reply
        result = await plugin.tool_comment_post(SimpleNamespace(), auto_generate=True, like_after_comment=True)
        assert result == "public partial result"
        assert sent == ["public-1", "public-3"]
        summary = payloads[0]
        assert summary["ok"] and summary["partial"]
        assert summary["result"]["count"] == 2 and summary["result"]["failed_count"] == 1
        assert logs[0]["result"][1]["status"] == "failed"

        async def cancelled(_event, _post): raise asyncio.CancelledError()
        plugin._generate_comment_text = cancelled
        with pytest.raises(asyncio.CancelledError):
            await plugin.tool_comment_post(SimpleNamespace(), auto_generate=True)

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


def test_page_life_draft_initial_latest_query_returns_idle_and_latest_unknown_after_cache_loss(tmp_path: Path) -> None:
    class Controller:
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False, "login_uin": 123}
        async def publish_post(self, **kwargs): raise AssertionError("query must not publish")

    async def scenario() -> None:
        path = tmp_path / "latest-life.json"
        store = DraftStore(path)
        async def generate(_progress): return PostPayload("daily", [])
        api = QzonePageApi(
            controller=Controller(), post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        assert (await api.get_life_draft({}))['data']['state'] == "idle"
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "latest-unknown"})
        job_id = started["data"]["id"]
        job = await api._life_jobs.get(job_id, 123)
        await job.task
        draft = await store.get_async(job.draft_id)
        claim = await store.claim_publish_async(
            draft.id, account_uin=123, content="daily", media=[], life_job_id=job_id, life_owner_uin=123,
        )
        assert claim["state"] == "acquired"
        await store.mark_unknown_async(
            draft.id, token=claim["token"], account_uin=123, digest=claim["digest"],
            diagnostic={"code": "TIMEOUT", "phase": "controller_call"},
        )
        api._life_jobs = PageLifeJobStore()
        api.draft_store = DraftStore(path)
        latest = await api.get_life_draft({})
        assert latest["data"]["id"] == job_id and latest["data"]["publish_state"] == "unknown"
        await api.close()

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


def test_real_chat_approval_then_page_status_uses_durable_published_result(tmp_path: Path) -> None:
    class Controller:
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False,
                    "login_uin": 123, "login_nickname": "Tester"}
        async def publish_post(self, **kwargs):
            raise AssertionError("the real chat flow calls the prepared publisher below")

    async def scenario() -> None:
        path = tmp_path / "chat-approve.json"
        store = DraftStore(path)
        async def generate(_progress): return PostPayload("daily", [])
        api = QzonePageApi(
            controller=Controller(), post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "real-chat"})
        job = await api._life_jobs.get(started["data"]["id"], 123)
        await job.task
        draft = await store.get_async(job.draft_id)
        plugin = object.__new__(main.QzoneStablePlugin)
        plugin.drafts = store
        plugin.settings = SimpleNamespace(show_name=False)
        plugin._self_id = lambda _event: 123
        plugin.controller = api.controller
        plugin._draft_id_from_event = lambda _event, _names: (draft.id, "")
        plugin._ensure_cookie_ready = lambda _event: asyncio.sleep(0)
        plugin._schedule_publisher_profile = lambda _event: None
        plugin._post_store = lambda: SimpleNamespace(upsert_async=lambda _post: asyncio.sleep(0))
        plugin._notify_draft_author = lambda *_args: asyncio.sleep(0, result=None)
        async def publish(post, *, on_prepared, on_dispatch, **_kwargs):
            await on_prepared(post)
            on_dispatch()
            return post, {"fid": "chat-approved-fid", "media_count": 0, "photo_count": 0}
        plugin._publish_post_payload = publish
        plugin._command_result = lambda _event, text: text
        async def fail_notification(*_args): raise RuntimeError("fixture notification failure")
        plugin._notify_draft_author = fail_notification
        event = SimpleNamespace(message_str=f"过稿 {draft.id}")
        outcomes = [item async for item in plugin.approve_post(event)]
        assert outcomes and "失败" in outcomes[0]
        saved = await store.get_async(draft.id)
        assert saved.status == "published" and saved.published_fid == "chat-approved-fid"
        page_status = await api.get_life_draft({"id": job.id})
        assert page_status["data"]["publish_state"] == "published"
        assert page_status["data"]["content"] == "daily"
        assert page_status["data"]["publish_result"]["fid"] == "chat-approved-fid"
        restarted = QzonePageApi(
            controller=api.controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), draft_store=DraftStore(path),
        )
        restarted_status = await restarted.get_life_draft({"id": job.id})
        assert restarted_status["data"]["publish_state"] == "published"
        assert restarted_status["data"]["publish_result"]["fid"] == "chat-approved-fid"
        await restarted.close()
        await api.close()

    asyncio.run(scenario())


def test_page_publish_then_real_chat_approval_is_idempotent(tmp_path: Path) -> None:
    class Controller:
        def __init__(self): self.posts = []
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False,
                    "login_uin": 123, "login_nickname": "Tester"}
        async def publish_post(self, **kwargs):
            self.posts.append(kwargs)
            return {"fid": "page-first-fid", "media_count": 0, "photo_count": 0}

    async def scenario() -> None:
        path = tmp_path / "page-first-chat.json"
        store = DraftStore(path)
        controller = Controller()
        async def generate(_progress): return PostPayload("daily", [])
        api = QzonePageApi(
            controller=controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "page-first-chat"})
        job_id = started["data"]["id"]
        job = await api._life_jobs.get(job_id, 123)
        await job.task
        page_result = await api.publish({"content": "daily", "life_job_id": job_id})
        plugin = object.__new__(main.QzoneStablePlugin)
        plugin.drafts = store
        plugin.controller = controller
        plugin._ensure_cookie_ready = lambda _event: asyncio.sleep(0)
        draft = await store.get_async(job.draft_id)
        plugin._draft_id_from_event = lambda _event, _names: (draft.id, "")
        plugin._command_result = lambda _event, text: text
        outcomes = [item async for item in plugin.approve_post(SimpleNamespace(message_str="过稿"))]
        assert len(controller.posts) == 1
        assert "page-first-fid" in str(outcomes)
        published_draft = await store.get_async(job.draft_id)
        assert page_result["data"]["post"]["content"] == "daily"
        assert published_draft is not None
        assert published_draft.status == "published"
        assert published_draft.published_fid == "page-first-fid"
        await api.close()

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
        api.draft_store = DraftStore(path)  # Reconstruct the durable store from disk as a new process would.
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


def test_expired_pending_life_draft_does_not_recover_page_publish_capability(tmp_path: Path) -> None:
    class Controller:
        def __init__(self): self.published = []
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False, "login_uin": 123}
        async def publish_post(self, **kwargs):
            self.published.append(kwargs)
            return {"fid": "unexpected"}

    async def scenario() -> None:
        controller = Controller()
        path = tmp_path / "expired-pending.json"
        async def generate(_progress): return PostPayload("daily", [])
        store = DraftStore(path)
        api = QzonePageApi(
            controller=controller, post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": "expired-pending"})
        job_id = started["data"]["id"]
        job = await api._life_jobs.get(job_id, 123)
        await job.task
        job.completed_at = 1.0
        with pytest.raises(Exception) as query_error:
            await api.get_life_draft({"id": job_id})
        assert getattr(query_error.value, "code", "") == "LIFE_EXPIRED"
        with pytest.raises(Exception) as publish_error:
            await api.publish({"content": "daily", "life_job_id": job_id})
        assert getattr(publish_error.value, "code", "") == "LIFE_EXPIRED"
        assert controller.published == []
        assert (await store.get_async(job.draft_id)).status == "pending"
        await api.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("request_dispatched", [False, True])
def test_page_publish_uses_only_trusted_pre_dispatch_proof_to_release_claim(tmp_path: Path, request_dispatched: bool) -> None:
    async def scenario() -> None:
        calls = []
        class Controller:
            async def get_status(self, *, probe_daemon=False):
                return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False,
                        "login_uin": 123, "login_nickname": "Tester"}
            async def publish_post(self, **kwargs):
                calls.append(kwargs)
                if len(calls) == 1:
                    raise DaemonUnavailableError("fixture", detail={"request_dispatched": request_dispatched})
                return {"fid": "published-once", "media_count": 0, "photo_count": 0}

        store = DraftStore(tmp_path / f"page-pre-dispatch-{request_dispatched}.json")
        async def generate(_progress):
            return PostPayload("daily", [])
        api = QzonePageApi(
            controller=Controller(), post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20), life_draft_generator=generate,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        started = await api.start_life_draft({"action": "generate_draft", "request_id": f"proof-{request_dispatched}"})
        job_id = started["data"]["id"]
        job = await api._life_jobs.get(job_id, 123)
        await job.task
        saved = await store.get_async(job.draft_id)
        assert saved is not None
        if request_dispatched is False:
            with pytest.raises(Exception) as caught:
                await api.publish({"content": "daily", "life_job_id": job_id})
            assert getattr(caught.value, "code", "") == "LIFE_PRE_DISPATCH"
            assert (await store.get_async(job.draft_id)).status == "pending"
            assert (await api._life_jobs.get(job_id, 123)).publish_state == "ready"
            result = await api.publish({"content": "daily", "life_job_id": job_id})
            assert result["data"]["fid"] == "published-once"
            assert len(calls) == 2
        else:
            with pytest.raises(Exception) as caught:
                await api.publish({"content": "daily", "life_job_id": job_id})
            assert getattr(caught.value, "code", "") == "LIFE_UNKNOWN"
            assert (await store.get_async(job.draft_id)).status == "unknown"
            assert (await api._life_jobs.get(job_id, 123)).publish_state == "unknown"
            with pytest.raises(Exception):
                await api.publish({"content": "daily", "life_job_id": job_id})
            assert len(calls) == 1
        await api.close()
    asyncio.run(scenario())


def test_chat_approval_replays_only_same_owner_safe_published_receipt(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = DraftStore(tmp_path / "chat-receipt.json")
        media = [{"kind": "image", "source": "https://example.test/a.png", "name": "a.png",
                  "mime_type": "image/png", "size": 123, "upload_id": "stable-upload-id"}]
        draft = await store.add_async(author_uin=123, content="daily", media=media)
        claim = await store.claim_publish_async(draft.id, account_uin=123, content="daily", media=media)
        await store.complete_publish_async(
            draft.id, token=claim["token"], account_uin=123, digest=claim["digest"],
            result=public_publish_result({"fid": "safe-fid", "media_count": 0, "photo_count": 0,
                    "raw": {"secret": "RAW_SENTINEL"}, "post": {"images": ["C:/private.png"]}}),
            fid="safe-fid",
        )

        class Controller:
            login = 123
            async def get_status(self, *, probe_daemon=False): return {"login_uin": self.login}

        controller = Controller()
        plugin = object.__new__(main.QzoneStablePlugin)
        plugin.drafts = store
        plugin.controller = controller
        plugin.settings = SimpleNamespace(show_name=True)
        plugin._draft_id_from_event = lambda _event, _names: (draft.id, "")
        plugin._command_result = lambda _event, text: text
        plugin._ensure_cookie_ready = lambda _event: asyncio.sleep(0)
        plugin._draft_publish_content = lambda _draft: "formatted-mutated-content"
        plugin._publish_post_payload = lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not send"))
        own = "\n".join([item async for item in plugin.approve_post(SimpleNamespace(message_str="過稿"))])
        assert "safe-fid" in own and "RAW_SENTINEL" not in own and "private.png" not in own
        controller.login = 999
        other = "\n".join([item async for item in plugin.approve_post(SimpleNamespace(message_str="過稿"))])
        assert "safe-fid" not in other and "RAW_SENTINEL" not in other
    asyncio.run(scenario())


def test_chat_approval_empty_response_remains_unknown(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = DraftStore(tmp_path / "chat-empty-receipt.json")
        draft = await store.add_async(author_uin=123, content="daily")
        class Controller:
            async def get_status(self, *, probe_daemon=False): return {"login_uin": 123}
        plugin = object.__new__(main.QzoneStablePlugin)
        plugin.drafts = store
        plugin.controller = Controller()
        plugin._draft_id_from_event = lambda _event, _names: (draft.id, "")
        plugin._command_result = lambda _event, text: text
        plugin._ensure_cookie_ready = lambda _event: asyncio.sleep(0)
        plugin._draft_publish_content = lambda _draft: "daily"
        plugin._schedule_publisher_profile = lambda _event: None
        async def malformed(post, *, on_prepared, on_dispatch, **_kwargs):
            await on_prepared(post)
            on_dispatch()
            return post, {"message": "created maybe", "raw": "RAW_SENTINEL"}
        plugin._publish_post_payload = malformed
        outcome = "\n".join([item async for item in plugin.approve_post(SimpleNamespace(message_str="過稿"))])
        saved = await store.get_async(draft.id)
        assert saved is not None and saved.status == "unknown"
        assert "RAW_SENTINEL" not in outcome and "created maybe" not in outcome
    asyncio.run(scenario())
