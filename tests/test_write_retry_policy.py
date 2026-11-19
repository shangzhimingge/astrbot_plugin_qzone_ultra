from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
from PIL import Image
import base64
import io
import pytest
from aiohttp import web

from qzone_bridge.client import QzoneClient
from qzone_bridge.controller import QzoneDaemonController
from qzone_bridge.daemon import QzoneDaemonService, create_app
from qzone_bridge.errors import DaemonUnavailableError, QzoneRequestError
from qzone_bridge.models import SessionState
from qzone_bridge.storage import StateStore


def test_protected_write_attempts_once_while_reads_keep_retries() -> None:
    async def scenario() -> None:
        writes: list[int] = []
        reads: list[int] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.method == "POST":
                writes.append(1)  # Model QQ accepting the mutation before the response fails.
                return httpx.Response(500, request=request, text="accepted then response error")
            reads.append(1)
            if len(reads) == 1:
                return httpx.Response(503, request=request, text="temporary read failure")
            return httpx.Response(200, request=request, text="ok")

        client = QzoneClient(SessionState(uin=12345, cookies={"uin": "o12345", "p_skey": "ps-key"}), max_retries=3)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        try:
            async with client.write_retry_policy(False):
                response = await client._request_text("GET", "https://unit.test/read", login_required=False)
                assert response.text == "ok"
                with pytest.raises(QzoneRequestError):
                    await client.publish_mood("hello")
            assert len(reads) == 2
            assert len(writes) == 1
            # The protected policy is task-local and does not leak into later operations.
            with pytest.raises(QzoneRequestError):
                await client.publish_mood("generic retry")
            assert len(writes) == 4
        finally:
            await client.close()

    asyncio.run(scenario())


def test_protected_reply_write_is_single_attempt() -> None:
    async def scenario() -> None:
        attempts = 0

        async def handler(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            attempts += 1
            return httpx.Response(500, request=request, text="accepted then response error")

        client = QzoneClient(SessionState(uin=12345, cookies={"uin": "o12345", "p_skey": "ps-key"}), max_retries=4)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        try:
            async with client.write_retry_policy(False):
                with pytest.raises(QzoneRequestError):
                    await client.reply_comment(12345, "fid", "cid", 999, "reply")
            assert attempts == 1
        finally:
            await client.close()

    asyncio.run(scenario())


@pytest.mark.parametrize("external_failure", ["timeout", "read_error", "http_5xx"])
def test_controller_daemon_client_chain_dispatches_protected_mutation_once(tmp_path, external_failure: str) -> None:
    async def scenario() -> None:
        data_dir = tmp_path / "daemon-data"
        secret = "FIXTURE_SECRET"
        store = StateStore(data_dir)
        service = QzoneDaemonService(store, secret=secret, port=18999, request_timeout=2.0)
        service.state.session.uin = 12345
        service.state.session.cookies = {"uin": "o12345", "p_skey": "fixture"}
        service.client.update_session(service.state.session)
        fixture_calls = 0

        async def fixture(request: httpx.Request) -> httpx.Response:
            nonlocal fixture_calls
            fixture_calls += 1
            if external_failure == "timeout":
                raise httpx.ReadTimeout("response lost after accepted mutation", request=request)
            if external_failure == "read_error":
                raise httpx.ReadError("response stream lost after accepted mutation", request=request)
            return httpx.Response(500, request=request, text="accepted then server error")

        await service.client._client.aclose()
        service.client._client = httpx.AsyncClient(transport=httpx.MockTransport(fixture), follow_redirects=False)
        app = create_app(service)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        service.state.runtime.daemon_port = port
        service.save()

        controller = QzoneDaemonController(
            plugin_root=Path(__file__).resolve().parents[1], data_dir=data_dir,
            default_port=port, request_timeout=2.0, auto_start_daemon=False,
        )
        try:
            with pytest.raises(Exception):
                await controller.publish_post(
                    content="fixture", content_sanitized=True, retry_writes=False,
                )
            assert fixture_calls == 1
        finally:
            await controller.close()
            await runner.cleanup()
            await service.close()

    asyncio.run(scenario())


def test_controller_response_loss_after_daemon_write_does_not_repeat_daemon_dispatch(tmp_path) -> None:
    async def scenario() -> None:
        data_dir = tmp_path / "response-loss-data"
        secret = "FIXTURE_SECRET"
        store = StateStore(data_dir)
        service = QzoneDaemonService(store, secret=secret, port=18999, request_timeout=2.0)
        service.state.session.uin = 12345
        service.state.session.cookies = {"uin": "o12345", "p_skey": "fixture"}
        service.client.update_session(service.state.session)
        external_mutations = 0
        async def qzone_fixture(request: httpx.Request) -> httpx.Response:
            nonlocal external_mutations
            external_mutations += 1
            return httpx.Response(200, request=request, json={"ret": 0, "data": {"fid": "fid-1"}})
        await service.client._client.aclose()
        service.client._client = httpx.AsyncClient(transport=httpx.MockTransport(qzone_fixture), follow_redirects=False)
        runner = web.AppRunner(create_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        service.state.runtime.daemon_port = port
        service.save()
        controller = QzoneDaemonController(
            plugin_root=Path(__file__).resolve().parents[1], data_dir=data_dir,
            default_port=port, request_timeout=2.0, auto_start_daemon=False,
        )
        downstream = httpx.AsyncClient(trust_env=False)
        daemon_dispatches = 0
        async def lose_response_after_dispatch(request: httpx.Request) -> httpx.Response:
            nonlocal daemon_dispatches
            response = await downstream.send(request)
            if request.url.path == "/post":
                daemon_dispatches += 1
                raise httpx.ReadTimeout("fixture response lost after dispatch", request=request)
            return response
        await controller._client.aclose()
        controller._client = httpx.AsyncClient(transport=httpx.MockTransport(lose_response_after_dispatch), timeout=2.0)
        try:
            with pytest.raises(DaemonUnavailableError):
                await controller.publish_post(content="fixture", content_sanitized=True, retry_writes=False)
            assert daemon_dispatches == 1 and external_mutations == 1
        finally:
            await downstream.aclose()
            await controller.close()
            await runner.cleanup()
            await service.close()
    asyncio.run(scenario())


def test_scheduled_life_publish_uses_protected_controller_daemon_transport(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        from test_security_hardening import _import_main_with_stubs

        main = _import_main_with_stubs(monkeypatch)
        data_dir = tmp_path / "scheduled-life-data"
        service = QzoneDaemonService(StateStore(data_dir), secret="FIXTURE_SECRET", port=18999, request_timeout=2.0)
        service.state.session.uin = 12345
        service.state.session.cookies = {"uin": "o12345", "p_skey": "fixture"}
        service.client.update_session(service.state.session)
        external_mutations = 0
        async def qzone_fixture(request: httpx.Request) -> httpx.Response:
            nonlocal external_mutations
            external_mutations += 1
            raise httpx.ReadTimeout("accepted scheduled Life write, response lost", request=request)
        await service.client._client.aclose()
        service.client._client = httpx.AsyncClient(transport=httpx.MockTransport(qzone_fixture), follow_redirects=False)
        runner = web.AppRunner(create_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        service.state.runtime.daemon_port = port
        service.save()
        controller = QzoneDaemonController(
            plugin_root=Path(__file__).resolve().parents[1], data_dir=data_dir,
            default_port=port, request_timeout=2.0, auto_start_daemon=False,
        )
        plugin = object.__new__(main.QzoneStablePlugin)
        plugin.settings = type("Settings", (), {"life_publish_mode": "publish"})()
        plugin.controller = controller
        async def ready(*_args, **_kwargs): return None
        plugin._ensure_cookie_ready = ready
        plugin._ensure_daemon = ready
        try:
            with pytest.raises(QzoneRequestError):
                await plugin._publish_or_draft_scheduled_post(main.PostPayload(content="fixture", media=[]))
            assert external_mutations == 1
        finally:
            await controller.close()
            await runner.cleanup()
            await service.close()
    asyncio.run(scenario())


@pytest.mark.parametrize("operation", ["native_final_create", "native_visibility_update"])
def test_protected_native_video_final_write_does_not_replay_after_response_loss(operation: str) -> None:
    async def scenario() -> None:
        calls = []
        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadTimeout("accepted then response lost", request=request)
        client = QzoneClient(SessionState(uin=12345, cookies={"uin": "o12345", "p_skey": "fixture"}), max_retries=4)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        try:
            async with client.write_retry_policy(False):
                with pytest.raises(httpx.TimeoutException):
                    if operation == "native_final_create":
                        await client.publish_mood("fixture", photos=[{"richval": "native-video-fixture"}])
                    else:
                        await client.update_mood_visibility_public(fid="native-fid", content="")
            assert len(calls) == 1
        finally:
            await client.close()
    asyncio.run(scenario())


def test_protected_photo_upload_ambiguous_failure_is_not_replayed_sequentially() -> None:
    async def scenario() -> None:
        buffer = io.BytesIO()
        Image.new("RGB", (128, 128), "blue").save(buffer, "PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        source = "data:image/png;base64," + encoded
        second_image = Image.new("RGB", (128, 128), "red")
        second_buffer = io.BytesIO()
        second_image.save(second_buffer, "PNG")
        second_source = "data:image/png;base64," + base64.b64encode(second_buffer.getvalue()).decode("ascii")
        calls = []
        async def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            raise httpx.ReadTimeout("photo upload response lost", request=request)
        client = QzoneClient(SessionState(uin=12345, cookies={"uin": "o12345", "p_skey": "fixture"}), max_retries=4)
        await client._client.aclose()
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=False)
        try:
            async with client.write_retry_policy(False):
                with pytest.raises(httpx.TimeoutException):
                    await client.publish_mood("fixture", photos=[{"source": source}, {"source": second_source}])
            assert len(calls) == 2  # one ambiguous request per distinct photo; no sequential replay/final create
        finally:
            await client.close()
    asyncio.run(scenario())


def test_generic_photo_upload_keeps_sequential_failover() -> None:
    async def scenario() -> None:
        client = QzoneClient(SessionState(uin=12345, cookies={"uin": "o12345"}))
        attempts: dict[str, int] = {}
        async def upload(media):
            source = media["source"]
            attempts[source] = attempts.get(source, 0) + 1
            if source == "retry-me" and attempts[source] == 1:
                raise QzoneRequestError("transient upload response")
            return {"source": source, "richval": source}
        client.upload_photo = upload
        try:
            result = await client._prepare_publish_photos([{"source": "retry-me"}, {"source": "ok"}])
            assert [item["source"] for item in result] == ["retry-me", "ok"]
            assert attempts == {"retry-me": 2, "ok": 1}
        finally:
            await client.close()
    asyncio.run(scenario())
