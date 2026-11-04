import asyncio

import pytest

from qzone_bridge.llm import QzoneLLM
from qzone_bridge.persona import PersonaResolutionError, resolve_persona_prompt
from qzone_bridge.settings import PluginSettings


class Manager:
    def __init__(self):
        self.lookups = []

    async def resolve_selected_persona(self, **kwargs):
        self.lookups.append(kwargs)
        pid = kwargs.get("conversation_persona_id") or "default"
        if pid == "[%None]":
            return pid, None, None, False
        return pid, {"prompt": f"persona:{pid}"}, None, False

    async def get_persona_v3_by_id(self, persona_id):
        return {"prompt": "persona:explicit"} if persona_id == "explicit" else None

    async def get_default_persona_v3(self, umo=None):
        return {"prompt": "persona:default"}


class Conversations:
    async def get_curr_conversation_id(self, umo):
        return "cid"

    async def get_conversation(self, umo, cid):
        return {"persona_id": "chosen"}


class Context:
    def __init__(self):
        self.persona_manager = Manager()
        self.conversation_manager = Conversations()
        self.calls = []
        self.platform_manager = type("PlatformManager", (), {
            "get_insts": lambda _self: [type("Platform", (), {
                "meta": lambda _self: type("Meta", (), {"name": "aiocqhttp", "id": "custom-instance-7"})()
            })()]
        })()

    async def get_config(self, umo):
        return {"provider_settings": {"provider": "configured"}}

    async def llm_generate(self, **kwargs):
        self.calls.append(kwargs)
        return "generated"


def test_incoming_event_uses_selected_persona_and_conversation():
    async def scenario():
        context = Context()
        prompt = await resolve_persona_prompt(
            context, PluginSettings(), type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        )
        assert prompt == "persona:chosen"
        assert context.persona_manager.lookups[0]["umo"] == "custom-instance-7:GroupMessage:1"
        assert context.persona_manager.lookups[0]["provider_settings"] == {"provider": "configured"}
    asyncio.run(scenario())


def test_context_free_override_disable_and_invalid_id():
    async def scenario():
        context = Context()
        settings = PluginSettings(persona_id="explicit")
        assert await resolve_persona_prompt(context, settings, None) == "persona:explicit"
        settings.persona_id = "[%None]"
        assert await resolve_persona_prompt(context, settings, None) == ""
        settings.persona_id = "missing"
        with pytest.raises(PersonaResolutionError):
            await resolve_persona_prompt(context, settings, None)
    asyncio.run(scenario())


def test_context_free_uses_management_group_and_resolver():
    async def scenario():
        context = Context()
        assert await resolve_persona_prompt(context, PluginSettings(manage_group=123), None) == "persona:chosen"
        assert context.persona_manager.lookups[0]["umo"] == "custom-instance-7:GroupMessage:123"
    asyncio.run(scenario())


def test_context_umo_uses_astrbot_session_contract_and_ambiguous_falls_back():
    from astrbot.core.platform.message_session import MessageSession

    async def scenario():
        context = Context()
        await resolve_persona_prompt(context, PluginSettings(manage_group=123), None)
        group_umo = context.persona_manager.lookups[-1]["umo"]
        assert str(MessageSession.from_str(group_umo)) == "custom-instance-7:GroupMessage:123"

        context.platform_manager.get_insts = lambda: [
            type("Platform", (), {"meta": lambda _self: type("Meta", (), {"name": "aiocqhttp", "id": "custom-instance-7"})()})(),
            type("Platform", (), {"meta": lambda _self: type("Meta", (), {"name": "aiocqhttp", "id": "other-instance"})()})(),
        ]
        context.persona_manager.lookups.clear()
        assert await resolve_persona_prompt(context, PluginSettings(manage_group=123), None) == "persona:default"
        assert not context.persona_manager.lookups
        context.platform_manager.get_insts = lambda: []
        assert await resolve_persona_prompt(context, PluginSettings(manage_group=123), None) == "persona:default"
        assert not context.persona_manager.lookups

        context.platform_manager.get_insts = lambda: [
            type("Platform", (), {"meta": lambda _self: type("Meta", (), {"name": "aiocqhttp", "id": "custom-instance-7"})()})()
        ]
        await resolve_persona_prompt(context, PluginSettings(admin_uins=[456]), None)
        assert str(MessageSession.from_str(context.persona_manager.lookups[-1]["umo"])) == "custom-instance-7:FriendMessage:456"

        explicit = PluginSettings(context_umo="custom-instance-7:GroupMessage:789")
        await resolve_persona_prompt(context, explicit, None)
        assert context.persona_manager.lookups[-1]["umo"] == explicit.context_umo
        with pytest.raises(PersonaResolutionError):
            await resolve_persona_prompt(context, PluginSettings(context_umo="aiocqhttp:group:123"), None)
    asyncio.run(scenario())


def test_context_umo_setting_roundtrips_and_real_event_ignores_it():
    async def scenario():
        settings = PluginSettings.from_mapping({"llm": {"context_umo": "custom-instance-7:GroupMessage:789"}})
        assert settings.context_umo == "custom-instance-7:GroupMessage:789"
        context = Context()
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:FriendMessage:1"})()
        await resolve_persona_prompt(context, settings, event)
        assert context.persona_manager.lookups[-1]["umo"] == event.unified_msg_origin
    asyncio.run(scenario())


def test_scheduled_event_uses_shared_resolved_umo():
    from main import QzoneStablePlugin

    context = Context()
    plugin = object.__new__(QzoneStablePlugin)
    plugin._context = context
    plugin.settings = PluginSettings(manage_group=123)
    event = plugin._scheduled_plugin_event(prompt="draft")
    assert event.unified_msg_origin == "custom-instance-7:GroupMessage:123"
    assert event.get_session_id() == event.unified_msg_origin

    plugin.settings.context_umo = "custom-instance-7:FriendMessage:456"
    event = plugin._scheduled_plugin_event(prompt="draft")
    assert event.unified_msg_origin == plugin.settings.context_umo

    context.platform_manager.get_insts = lambda: []
    plugin.settings.context_umo = ""
    event = plugin._scheduled_plugin_event(prompt="draft")
    assert event.unified_msg_origin is None


def test_text_chat_keeps_persona_and_task_prompt():
    async def scenario():
        class Provider:
            async def text_chat(self, *, prompt, contexts, system_prompt):
                assert contexts == []
                assert "persona:chosen" in system_prompt
                assert "task instructions" in system_prompt
                return "ok"

        class ProviderContext(Context):
            async def get_using_provider(self, **kwargs):
                return Provider()

        context = ProviderContext()
        llm = QzoneLLM(context, PluginSettings())
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await llm.generate_text(event, "write", system_prompt="task instructions") == "ok"
    asyncio.run(scenario())


def test_context_llm_fallback_preserves_system_prompt():
    async def scenario():
        context = Context()
        llm = QzoneLLM(context, PluginSettings())
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await llm.generate_text(event, "write", system_prompt="task instructions") == "generated"
        assert "persona:chosen" in context.calls[0]["system_prompt"]
    asyncio.run(scenario())


def test_selected_disabled_persona_does_not_fallback_to_default():
    async def scenario():
        context = Context()
        async def disabled(**kwargs):
            return "[%None]", None, None, False
        context.persona_manager.resolve_selected_persona = disabled
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await resolve_persona_prompt(context, PluginSettings(), event) == ""
    asyncio.run(scenario())


def test_persona_runtime_errors_stop_instead_of_falling_back():
    async def scenario():
        context = Context()
        async def broken_resolver(**kwargs):
            raise RuntimeError("sensitive provider/account marker")
        context.persona_manager.resolve_selected_persona = broken_resolver
        llm = QzoneLLM(context, PluginSettings())
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        with pytest.raises(PersonaResolutionError):
            await llm.generate_text(event, "prompt")
        assert context.calls == []

        class BrokenConversation:
            async def get_curr_conversation_id(self, umo):
                raise RuntimeError("conversation storage failure")
        context.conversation_manager = BrokenConversation()
        with pytest.raises(PersonaResolutionError):
            await resolve_persona_prompt(context, PluginSettings(), event)
    asyncio.run(scenario())


def test_explicit_override_uses_legacy_database_persona_and_missing_manager_errors():
    async def scenario():
        class LegacyManager:
            async def get_persona(self, persona_id):
                assert persona_id == "db-id"
                return type("DBPersona", (), {"system_prompt": "legacy DB prompt"})()

        context = type("LegacyContext", (), {"persona_manager": LegacyManager()})()
        assert await resolve_persona_prompt(context, PluginSettings(persona_id="db-id"), None) == "legacy DB prompt"
        with pytest.raises(PersonaResolutionError):
            await resolve_persona_prompt(None, PluginSettings(persona_id="db-id"), None)
    asyncio.run(scenario())


def test_forced_persona_selection_overrides_conversation_and_default():
    async def scenario():
        context = Context()
        async def forced(**kwargs):
            assert kwargs["conversation_persona_id"] == "chosen"
            return "forced-id", {"prompt": "forced persona"}, "forced-id", False
        context.persona_manager.resolve_selected_persona = forced
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await resolve_persona_prompt(context, PluginSettings(), event) == "forced persona"
    asyncio.run(scenario())


def test_context_free_without_umo_uses_astrbot_default_persona():
    async def scenario():
        context = Context()
        context.conversation_manager = None
        assert await resolve_persona_prompt(context, PluginSettings(), None) == "persona:default"
    asyncio.run(scenario())


def test_real_incoming_event_ignores_context_free_override():
    async def scenario():
        context = Context()
        settings = PluginSettings(persona_id="explicit")
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await resolve_persona_prompt(context, settings, event) == "persona:chosen"
        assert context.persona_manager.lookups
    asyncio.run(scenario())


def test_configured_and_current_provider_receive_same_persona_and_task_prompt():
    async def scenario():
        prompts = []
        class Provider:
            async def text_chat(self, *, prompt, contexts, system_prompt):
                assert contexts == []
                prompts.append(system_prompt)
                return "draft"
        provider = Provider()
        class ProviderContext(Context):
            async def get_provider_by_id(self, provider_id):
                assert provider_id == "chosen-provider"
                return provider
            async def get_using_provider(self, **kwargs):
                return provider
        context = ProviderContext()
        llm = QzoneLLM(context, PluginSettings())
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        task = "same task instructions"
        await llm.generate_text(event, "prompt", provider_id="chosen-provider", system_prompt=task)
        await llm.generate_text(event, "prompt", system_prompt=task, prefer_current_provider=True)
        assert len(prompts) == 2 and prompts[0] == prompts[1]
        assert "persona:chosen" in prompts[0] and task in prompts[0]
    asyncio.run(scenario())


def test_inherit_persona_false_skips_resolver_and_keeps_task_prompt():
    async def scenario():
        class Provider:
            async def text_chat(self, *, prompt, contexts, system_prompt):
                assert "task only" == system_prompt
                return "draft"
        class ProviderContext(Context):
            async def get_using_provider(self, **kwargs):
                return Provider()
        context = ProviderContext()
        settings = PluginSettings(persona_id="invalid")
        llm = QzoneLLM(context, settings)
        event = type("Event", (), {"unified_msg_origin": "custom-instance-7:GroupMessage:1"})()
        assert await llm.generate_text(event, "prompt", system_prompt="task only", inherit_persona=False) == "draft"
        assert not context.persona_manager.lookups
    asyncio.run(scenario())
