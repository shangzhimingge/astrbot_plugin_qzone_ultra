import asyncio
import copy
import types
from types import SimpleNamespace

import pytest
from test_security_hardening import _import_main_with_stubs


def test_real_qzone_llm_persona_and_provider_snapshot_survives_selector_mutation():
    from qzone_bridge.llm import QzoneLLM
    from qzone_bridge.persona_guard import generation_scope

    calls = []
    class Provider:
        async def text_chat(self, **kwargs):
            calls.append(kwargs)
            return {"completion_text": "ok"}
    old_provider = Provider()
    new_provider = Provider()
    class PersonaManager:
        def __init__(self):
            self.prompts = {"voice-1": "ORIGINAL_PERSONA_VOICE", "voice-2": "MUTATED_PERSONA_VOICE"}
        async def get_persona_v3_by_id(self, persona_id):
            return {"prompt": self.prompts[persona_id]}
    class Context:
        def __init__(self):
            self.persona_manager = PersonaManager()
            self.providers = {"provider-1": old_provider, "provider-2": new_provider}
        def get_provider_by_id(self, provider_id):
            return self.providers.get(provider_id)
    context = Context()
    settings = types.SimpleNamespace(persona_id="voice-1", persona_consistency_check=False)
    llm = QzoneLLM(context, settings)

    async def scenario():
        async with generation_scope():
            snapshot = await llm.prepare_persona_snapshot(None, provider_id="provider-1", prefer_current_provider=True, require_pinned_provider=False)
            # Mirrors scheduler/OmniDraw selection mutating config after the Life
            # operation has already resolved its actual Persona/provider snapshot.
            settings.persona_id = "voice-2"
            context.providers["provider-1"] = new_provider
            again = await llm.prepare_persona_snapshot(None, provider_id="provider-2", prefer_current_provider=True, require_pinned_provider=False)
            assert again is snapshot
            assert snapshot["persona_prompt"] == "ORIGINAL_PERSONA_VOICE"
            assert snapshot["provider"] is old_provider and snapshot["provider_id"] == "provider-1"
            assert await llm.generate_text(None, "caption prompt", provider_id="provider-2") == "ok"
        assert len(calls) == 1
        assert calls[0]["system_prompt"].startswith("ORIGINAL_PERSONA_VOICE")
    import asyncio
    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("use_image_llm", "auto_caption", "inherit_visual", "visual_prompt", "expected_phase"),
    [
        (True, True, False, "", "image_prompt"),
        (False, True, False, "", "caption"),
        (False, False, True, "fixed visual", "image_prompt"),
    ],
)
def test_real_persona_resolution_timeout_uses_owning_text_phase_before_scheduler(
    monkeypatch, use_image_llm, auto_caption, inherit_visual, visual_prompt, expected_phase
):
    from qzone_bridge.errors import LifeGenerationError

    main = _import_main_with_stubs(monkeypatch)
    scheduler_calls, image_calls, publish_calls = [], [], []
    class Manager:
        async def get_persona_v3_by_id(self, _persona_id):
            await asyncio.sleep(0.2)
            return {"prompt": "slow resolved voice"}
    class Provider:
        async def text_chat(self, **kwargs): return {"completion_text": "ok"}
    class Context:
        persona_manager = Manager()
        def get_provider_by_id(self, provider_id):
            assert provider_id == "voice-provider"
            return Provider()
        def get_registered_star(self, _name): return None
    context = Context()
    settings = SimpleNamespace(
        life_publish_timezone="UTC", life_publish_image_style_prompt="",
        life_publish_use_life_context=True, life_publish_use_llm_image_prompt=use_image_llm,
        life_publish_use_omnidraw_selfie=True, life_publish_auto_caption=auto_caption,
        life_publish_inherit_persona_visual=inherit_visual, life_publish_persona_visual_prompt=visual_prompt,
        life_publish_failure_policy="text_only", life_publish_mode="draft", life_publish_image_retry_count=0,
        life_publish_generation_timeout_seconds=2, life_publish_image_timeout_seconds=1,
        life_publish_text_timeout_seconds=0.03, life_publish_image_prompt_template="",
        life_publish_caption_prompt="caption", life_publish_static_caption="static",
        post_provider_id="voice-provider", persona_id="voice-1", admin_uins=[123], manage_group=0,
    )
    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = settings
    plugin._context = context
    plugin.llm = main.QzoneLLM(context, settings)
    plugin._life_budget_snapshot = lambda _settings: (2.0, 1.0, 0.03)
    async def scheduler():
        scheduler_calls.append(True)
        return {"date": "2026-11-19"}, "schedule"
    plugin._get_life_context = scheduler
    async def prompt(*_args, **_kwargs): return "prompt"
    async def image(*_args, **_kwargs):
        image_calls.append(True)
        return [main.PostMedia(kind="image", source="fixture.png")], {"success": True}
    async def caption(*_args, **_kwargs): return "caption"
    async def publish(*_args, **_kwargs): publish_calls.append(True)
    plugin._generate_life_image_prompt = prompt
    plugin._generate_life_selfie_media = image
    plugin._generate_life_caption = caption
    plugin._publish_or_draft_scheduled_post = publish
    with pytest.raises(LifeGenerationError) as caught:
        asyncio.run(plugin._auto_life_publish_once())
    assert (caught.value.code, caught.value.phase) == ("LIFE_TEXT_TIMEOUT", expected_phase)
    assert scheduler_calls == [] and image_calls == [] and publish_calls == []


def test_real_persona_resolution_native_cancellation_propagates_before_scheduler(monkeypatch):
    from qzone_bridge.persona_guard import current_scope

    main = _import_main_with_stubs(monkeypatch)
    entered = asyncio.Event()
    class Manager:
        async def get_persona_v3_by_id(self, _persona_id):
            entered.set()
            await asyncio.Event().wait()
    class Provider:
        async def text_chat(self, **kwargs): return {"completion_text": "ok"}
    class Context:
        persona_manager = Manager()
        def get_provider_by_id(self, _provider_id): return Provider()
        def get_registered_star(self, _name): return None
    context = Context()
    settings = SimpleNamespace(
        life_publish_timezone="UTC", life_publish_image_style_prompt="",
        life_publish_use_life_context=True, life_publish_use_llm_image_prompt=True,
        life_publish_use_omnidraw_selfie=True, life_publish_auto_caption=True,
        life_publish_inherit_persona_visual=False, life_publish_persona_visual_prompt="",
        life_publish_failure_policy="skip", life_publish_mode="draft", life_publish_image_retry_count=0,
        life_publish_image_prompt_template="", life_publish_caption_prompt="caption",
        life_publish_static_caption="", post_provider_id="voice-provider", persona_id="voice-1",
        admin_uins=[123], manage_group=0,
    )
    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = settings
    plugin._context = context
    plugin.llm = main.QzoneLLM(context, settings)
    plugin._life_budget_snapshot = lambda _settings: (2.0, 1.0, 1.0)
    scheduler_calls = []
    async def scheduler():
        scheduler_calls.append(True)
        return {}, ""
    plugin._get_life_context = scheduler

    async def scenario():
        task = asyncio.create_task(plugin._auto_life_publish_once())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert scheduler_calls == [] and current_scope() is None
    asyncio.run(scenario())


def test_early_snapshot_consumes_only_its_own_text_budget(monkeypatch):
    from qzone_bridge.errors import LifeGenerationError

    main = _import_main_with_stubs(monkeypatch)
    calls = {"snapshot": 0, "scheduler": 0, "selector": 0, "image_prompt": 0, "image": 0}
    class LLM:
        async def prepare_persona_snapshot(self, *_args, **_kwargs):
            calls["snapshot"] += 1
            await asyncio.sleep(0.015)
    class Omni:
        async def resolve_selfie_context(self, _request, *, life_context):
            calls["selector"] += 1
            await asyncio.sleep(0.06)
            return {}
    settings = SimpleNamespace(
        life_publish_timezone="UTC", life_publish_image_style_prompt="",
        life_publish_use_life_context=True, life_publish_use_llm_image_prompt=True,
        life_publish_use_omnidraw_selfie=True, life_publish_auto_caption=False,
        life_publish_inherit_persona_visual=False, life_publish_persona_visual_prompt="",
        life_publish_failure_policy="skip", life_publish_mode="draft", life_publish_image_retry_count=0,
        life_publish_image_prompt_template="", life_publish_caption_prompt="caption",
        life_publish_static_caption="", post_provider_id="voice-provider", admin_uins=[123], manage_group=0,
    )
    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = settings
    plugin.llm = LLM()
    plugin._llm_adapter = lambda: plugin.llm
    plugin._life_budget_snapshot = lambda _settings: (2.0, 1.0, 0.08)
    plugin._omnidraw_plugin = lambda: Omni()
    async def scheduler():
        calls["scheduler"] += 1
        await asyncio.sleep(0.06)
        return {"date": "2026-11-19"}, "context"
    async def image_prompt(*_args, **_kwargs):
        calls["image_prompt"] += 1
        await asyncio.sleep(0.075)
        return "too slow"
    async def image(*_args, **_kwargs):
        calls["image"] += 1
        return [], {"success": False}
    plugin._get_life_context = scheduler
    plugin._generate_life_image_prompt = image_prompt
    plugin._generate_life_selfie_media = image
    with pytest.raises(LifeGenerationError) as caught:
        asyncio.run(plugin._auto_life_publish_once())
    assert (caught.value.code, caught.value.phase) == ("LIFE_TEXT_TIMEOUT", "image_prompt")
    assert calls == {"snapshot": 1, "scheduler": 1, "selector": 1, "image_prompt": 1, "image": 0}


@pytest.mark.parametrize("mode", ["draft", "publish"])
@pytest.mark.parametrize("use_llm_prompt", [True, False])
def test_schedule_and_page_paths_use_real_frozen_life_prompt_helpers(
    monkeypatch, tmp_path, mode, use_llm_prompt
):
    from datetime import datetime, timezone
    from qzone_bridge.life_context import capture_life_inputs
    from qzone_bridge.drafts import DraftStore
    from qzone_bridge.page_api import QzonePageApi

    main = _import_main_with_stubs(monkeypatch)
    real_capture = capture_life_inputs
    frozen_utc = datetime(2026, 11, 8, 1, 15, tzinfo=timezone.utc)
    monkeypatch.setattr(main, "capture_life_inputs", lambda zone, style: real_capture(zone, style, utc_now=frozen_utc))
    captured = {"image_prompts": [], "caption_facts": [], "personas": [], "order": []}
    original = {
        "date": "2026-11-08",
        "outfit": "旧来源穿搭",
        "timeline": [{"start": "09:00", "end": "10:00", "activity": "写代码"}],
        "schedule": "09:00-10:00 写代码",
        "current_activity": "scheduler-current-activity",
        "current_time": "09:15",
    }

    class Omni:
        async def resolve_selfie_context(self, action, *, life_context):
            captured["order"].append("selector")
            captured["resolver_called"] = True
            assert life_context == original
            selected = copy.deepcopy(life_context)
            selected.update({
                "outfit": "深蓝工装卫衣", "selfie_persona": "成熟版工装",
                "current_activity": "SELECTOR_INVENTED_ACTIVITY", "current_time": "23:59",
                "date": "2099-01-01",
            })
            # Mutate provider and caption controls while selector awaits; this
            # generation must retain the operation-start snapshot.
            plugin.settings.post_provider_id = "mutated-provider"
            plugin.settings.life_publish_image_prompt_template = "MUTATED_IMAGE_TEMPLATE"
            plugin.settings.life_publish_auto_caption = False
            plugin.settings.life_publish_caption_prompt = "MUTATED_CAPTION_TEMPLATE"
            plugin.settings.life_publish_static_caption = "MUTATED_STATIC_CAPTION"
            return {"persona_id": "workwear", "life_context": selected}

    class LLM:
        async def prepare_persona_snapshot(self, *args, **kwargs):
            captured["order"].append("snapshot")
            captured["voice_snapshot"] = True
        async def generate_persona_visual_projection(self, *args, **kwargs):
            if use_llm_prompt:
                assert captured.get("voice_snapshot")
            return "voice-persona visual reference"
        async def generate_text(self, event, prompt, **kwargs):
            captured["image_prompts"].append(prompt)
            captured.setdefault("image_provider_ids", []).append(kwargs.get("provider_id"))
            return "在窗边写代码的生活自拍"
        async def generate_guarded_text(self, event, prompt, **kwargs):
            captured["caption_facts"].append(kwargs["authoritative_facts"])
            captured["caption_prompts"] = captured.get("caption_prompts", []) + [prompt]
            captured.setdefault("caption_provider_ids", []).append(kwargs.get("provider_id"))
            return "今天继续写代码"

    class Controller:
        async def get_status(self, *, probe_daemon=False):
            return {"daemon_state": "ready", "cookie_count": 1, "needs_rebind": False,
                    "login_uin": 123, "login_nickname": "Tester"}

    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = types.SimpleNamespace(
        life_publish_enabled=True, life_publish_mode=mode,
        life_publish_failure_policy="skip", life_publish_timezone="Asia/Shanghai",
        life_publish_image_style_prompt="", life_publish_use_llm_image_prompt=use_llm_prompt,
        life_publish_image_prompt_template="FROZEN_IMAGE_TEMPLATE {life_context}", life_publish_auto_caption=True,
        life_publish_caption_prompt="FROZEN_CAPTION_TEMPLATE 自然地概括本次日常。", life_publish_static_caption="",
        post_provider_id="voice-provider", max_feed_limit=20,
    )
    plugin._omnidraw_plugin = lambda: Omni()
    plugin._scheduled_plugin_event = lambda **kwargs: SimpleNamespace(message_str=kwargs.get("prompt", ""))
    plugin._llm_adapter = lambda: LLM()
    async def get_context():
        captured["order"].append("scheduler")
        captured["context_called"] = True
        return original, plugin._life_context_text(original)
    plugin._get_life_context = get_context
    async def generate_media(event, prompt, *, persona_id, settings_snapshot=None):
        captured["personas"].append(persona_id)
        captured.setdefault("media_prompts", []).append(prompt)
        captured.setdefault("draw_settings", []).append(dict(settings_snapshot or {}))
        return [main.PostMedia(kind="image", source=str(tmp_path / "fixture.png"))], {"success": True}
    plugin._generate_life_selfie_media = generate_media
    async def publish_or_draft(post, **kwargs):
        captured["scheduled_post"] = post
        return {"draft_id": 1}
    plugin._publish_or_draft_scheduled_post = publish_or_draft

    async def scenario():
        result = await plugin._auto_life_publish_once()
        assert result == {"draft_id": 1}, captured
        assert captured["scheduled_post"].content == "今天继续写代码"
        store = DraftStore(tmp_path / "page-drafts.json")
        api = QzonePageApi(
            controller=Controller(), post_service_factory=lambda: None,
            settings=SimpleNamespace(max_feed_limit=20),
            life_draft_generator=plugin._generate_life_web_draft,
            life_capability_provider=lambda: {"available": True}, draft_store=store,
        )
        plugin.settings.post_provider_id = "voice-provider"
        plugin.settings.life_publish_image_prompt_template = "FROZEN_IMAGE_TEMPLATE {life_context}"
        plugin.settings.life_publish_auto_caption = True
        plugin.settings.life_publish_caption_prompt = "FROZEN_CAPTION_TEMPLATE 自然地概括本次日常。"
        plugin.settings.life_publish_static_caption = ""
        started = await api.start_life_draft({"action": "generate_draft", "request_id": f"page-{mode}"})
        job = await api._life_jobs.get(started["data"]["id"], 123)
        await job.task
        saved = await store.get_async(job.draft_id)
        assert saved is not None and saved.content == "今天继续写代码"
        await api.close()

    asyncio.run(scenario())
    assert captured["personas"] == ["workwear", "workwear"]
    assert captured["order"].index("snapshot") < captured["order"].index("scheduler") < captured["order"].index("selector")
    assert all("深蓝工装卫衣" in prompt for prompt in captured["image_prompts"])
    assert all("成熟版工装" in prompt for prompt in captured["image_prompts"])
    assert all("选定穿搭意图" in prompt and "不代表已确认的当前穿着" in prompt for prompt in captured["image_prompts"])
    assert all("SELECTOR_INVENTED_ACTIVITY" not in prompt and "23:59" not in prompt for prompt in captured["image_prompts"])
    assert all("2026-11-08 09:15" in prompt for prompt in captured["image_prompts"])
    assert all("2099-01-01" not in prompt for prompt in captured["image_prompts"])
    if use_llm_prompt:
        assert captured["image_provider_ids"] and set(captured["image_provider_ids"]) == {"voice-provider"}
        assert all("FROZEN_IMAGE_TEMPLATE" in prompt for prompt in captured["image_prompts"])
    else:
        assert not captured.get("image_provider_ids")
        assert all("FROZEN_IMAGE_TEMPLATE" in prompt for prompt in captured["media_prompts"])
    assert captured["caption_provider_ids"] and set(captured["caption_provider_ids"]) == {"voice-provider"}
    assert all("FROZEN_CAPTION_TEMPLATE" in prompt for prompt in captured["caption_prompts"])
    assert all(item["provider_id"] == "voice-provider" and item["auto_caption"] is True for item in captured["draw_settings"])
    assert all("深蓝工装卫衣" in facts and "选定穿搭意图" in facts for facts in captured["caption_facts"])
    assert all("SELECTOR_INVENTED_ACTIVITY" not in facts and "23:59" not in facts for facts in captured["caption_facts"])
    assert original["outfit"] == "旧来源穿搭"


def test_selfie_persona_id_reaches_return_result_tool(monkeypatch, tmp_path):
    main = _import_main_with_stubs(monkeypatch)
    captured = {}

    class Omni:
        async def tool_generate_selfie(
            self, event, action, *, persona_id, return_result, **kwargs
        ):
            captured.update(persona_id=persona_id, return_result=return_result)
            return {
                "success": True,
                "images": [{"file_path": str(tmp_path / "selfie.png")}],
            }

    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = types.SimpleNamespace(
        life_publish_use_omnidraw_selfie=True, life_publish_image_retry_count=0
    )
    plugin._omnidraw_plugin = lambda: Omni()
    plugin._scheduled_plugin_event = lambda **kwargs: object()
    media, result = asyncio.run(
        plugin._generate_life_selfie_media(None, "Q版写代码", persona_id="chibi")
    )
    assert result["success"] and len(media) == 1
    assert captured == {"persona_id": "chibi", "return_result": True}


@pytest.mark.parametrize("resolver_raises", [False, True])
def test_legacy_or_failed_selector_keeps_existing_generation_path(
    monkeypatch, tmp_path, resolver_raises
):
    main = _import_main_with_stubs(monkeypatch)
    calls = []

    class Omni:
        pass

    if resolver_raises:

        async def resolve(action, *, life_context):
            raise RuntimeError("selector unavailable")

        Omni.resolve_selfie_context = staticmethod(resolve)

    plugin = object.__new__(main.QzoneStablePlugin)
    plugin.settings = types.SimpleNamespace(
        life_publish_failure_policy="skip", life_publish_use_llm_image_prompt=False,
        life_publish_auto_caption=False,
    )
    plugin._omnidraw_plugin = lambda: Omni()
    plugin._llm_adapter = lambda: SimpleNamespace(prepare_persona_snapshot=lambda *args, **kwargs: asyncio.sleep(0))

    async def life():
        return {"outfit": "女仆服"}, "今日穿搭：女仆服"

    async def prompt(event, context, **kwargs):
        assert "女仆服" in context
        return "在窗边喝咖啡"

    async def generate(event, prompt, **kwargs):
        calls.append(prompt)
        return [main.PostMedia(kind="image", source=str(tmp_path / "selfie.png"))], {
            "success": True
        }

    async def caption(event, **kwargs):
        return "咖啡时间"

    async def publish(post, **kwargs):
        return {"fid": "fixture"}

    plugin._get_life_context = life
    plugin._generate_life_image_prompt = prompt
    plugin._generate_life_selfie_media = generate
    plugin._generate_life_caption = caption
    plugin._publish_or_draft_scheduled_post = publish
    assert asyncio.run(plugin._auto_life_publish_once()) == {"fid": "fixture"}
    assert calls == ["在窗边喝咖啡"]
