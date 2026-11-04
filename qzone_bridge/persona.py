"""Resolve the AstrBot persona applicable to a Qzone generation request."""

from __future__ import annotations

import inspect
from typing import Any


class PersonaResolutionError(ValueError):
    """A configured Persona override cannot be resolved."""


async def _await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


async def _persona_by_id(manager: Any, persona_id: str) -> Any:
    getter = getattr(manager, "get_persona_v3_by_id", None)
    if not callable(getter):
        getter = getattr(manager, "get_persona", None)
    if not callable(getter):
        raise PersonaResolutionError("当前 AstrBot 未提供按 ID 查询人格的兼容接口。")
    try:
        persona = await _await(getter(persona_id))
    except Exception as exc:
        raise PersonaResolutionError("按 ID 读取 AstrBot 人格失败。") from exc
    if persona is None:
        raise PersonaResolutionError(f"配置的人格 ID 不存在：{persona_id}")
    return persona


def _value(obj: Any, key: str, default: Any = None) -> Any:
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _synthetic_umo(settings: Any) -> str | None:
    group = int(getattr(settings, "manage_group", 0) or 0)
    if group > 0:
        return f"aiocqhttp:group:{group}"
    admins = getattr(settings, "admin_uins", ()) or ()
    for admin in admins:
        if str(admin).isdigit() and int(admin) > 0:
            return f"aiocqhttp:private:{int(admin)}"
    return None


async def resolve_persona_prompt(context: Any, settings: Any, event: Any) -> str:
    """Return the selected Persona prompt without caching mutable Persona state."""
    synthetic = event is None or bool(getattr(event, "_qzone_context_free", False))
    manager = getattr(context, "persona_manager", None)
    if manager is None:
        if synthetic and str(getattr(settings, "persona_id", "") or "").strip():
            raise PersonaResolutionError("已配置人格 ID，但当前 AstrBot 未提供人格管理器。")
        return ""
    umo = getattr(event, "unified_msg_origin", None) if not synthetic else _synthetic_umo(settings)
    override = str(getattr(settings, "persona_id", "") or "").strip()

    if synthetic and override:
        if override == "[%None]":
            return ""
        persona = await _persona_by_id(manager, override)
        return str(_value(persona, "prompt", "") or _value(persona, "system_prompt", "") or "")

    config = None
    getter = getattr(context, "get_config", None)
    if callable(getter) and umo:
        try:
            config = await _await(getter(umo))
        except Exception as exc:
            raise PersonaResolutionError("读取当前人格配置失败。") from exc
    provider_settings = _value(config, "provider_settings", None)
    conversation_persona_id = None
    conversations = getattr(context, "conversation_manager", None)
    if umo and conversations is not None:
        try:
            cid = await _await(conversations.get_curr_conversation_id(umo))
            if cid:
                conversation = await _await(conversations.get_conversation(umo, cid))
                conversation_persona_id = _value(conversation, "persona_id", None)
        except Exception as exc:
            raise PersonaResolutionError("读取当前会话人格失败。") from exc

    resolver = getattr(manager, "resolve_selected_persona", None)
    persona = None
    resolved_selection = False
    if callable(resolver) and umo:
        platform_name = str(umo).split(":", 1)[0]
        try:
            resolved = await _await(resolver(
                umo=umo,
                conversation_persona_id=conversation_persona_id,
                platform_name=platform_name,
                provider_settings=provider_settings,
            ))
            if isinstance(resolved, (tuple, list)) and len(resolved) > 1:
                resolved_selection = True
                if str(resolved[0] or "") == "[%None]":
                    return ""
                persona = resolved[1]
                if persona is None and resolved[0]:
                    raise PersonaResolutionError("AstrBot 当前选中人格未能解析。")
        except Exception as exc:
            raise PersonaResolutionError("解析 AstrBot 当前人格失败。") from exc
    elif conversation_persona_id and conversation_persona_id != "[%None]":
        persona = await _persona_by_id(manager, str(conversation_persona_id))
    elif conversation_persona_id == "[%None]":
        return ""

    if persona is None and not resolved_selection:
        default_getter = getattr(manager, "get_default_persona_v3", None)
        if callable(default_getter):
            try:
                persona = await _await(default_getter(umo))
            except TypeError:
                try:
                    persona = await _await(default_getter())
                except Exception as exc:
                    raise PersonaResolutionError("读取 AstrBot 默认人格失败。") from exc
            except Exception as exc:
                raise PersonaResolutionError("读取 AstrBot 默认人格失败。") from exc
    return str(_value(persona, "prompt", "") or _value(persona, "system_prompt", "") or "")
