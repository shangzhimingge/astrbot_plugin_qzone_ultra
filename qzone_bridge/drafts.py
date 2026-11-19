"""Persistent draft store for target-style campus-wall workflows."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import secrets
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from time import time

from .json_store import AtomicItemStoreFile

DraftStatus = Literal["pending", "approved", "rejected", "recalled", "published", "publishing", "unknown"]
_PROTECTED_STATES = {"approved", "published", "publishing", "unknown"}


def public_publish_result(value: Any) -> dict[str, Any]:
    """Keep durable/idempotent publish receipts to non-sensitive scalar fields."""
    if not isinstance(value, dict):
        return {}
    result: dict[str, Any] = {}
    for key in ("fid", "media_count", "photo_count"):
        item = value.get(key)
        if key in {"media_count", "photo_count"}:
            if isinstance(item, int) and not isinstance(item, bool) and item >= 0:
                result[key] = item
        elif isinstance(item, str) and item:
            result[key] = item[:160]
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class DraftPost:
    id: int
    author_uin: int
    author_name: str = ""
    group_id: int = 0
    content: str = ""
    media: list[dict[str, Any]] = field(default_factory=list)
    anonymous: bool = False
    status: DraftStatus = "pending"
    reject_reason: str = ""
    published_fid: str = ""
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    life_job_id: str = ""
    life_owner_uin: int = 0
    publish_claim_token: str = ""
    publish_account_uin: int = 0
    publish_digest: str = ""
    publish_snapshot: dict[str, Any] = field(default_factory=dict)
    publish_claimed_at: str = ""
    publish_result: dict[str, Any] = field(default_factory=dict)
    publish_diagnostic: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DraftPost":
        status = str(data.get("status") or "pending")
        allowed = {"pending", "approved", "rejected", "recalled", "published", "publishing", "unknown"}
        if status not in allowed:
            # Preserve an unrecognized state as unknown rather than authorizing a send.
            status = "unknown"
        protected_fields = (
            "publish_claim_token", "publish_account_uin", "publish_digest",
            "publish_snapshot", "publish_claimed_at", "publish_result", "publish_diagnostic",
        )
        if status == "pending" and any(data.get(field) not in (None, "", 0, {}) for field in protected_fields):
            # A contradictory pending+claim record is never re-authorized as a fresh send.
            status = "unknown"
        return cls(
            id=int(data.get("id") or 0),
            author_uin=int(data.get("author_uin") or 0),
            author_name=str(data.get("author_name") or ""),
            group_id=int(data.get("group_id") or 0),
            content=str(data.get("content") or ""),
            media=list(data.get("media") or []),
            anonymous=bool(data.get("anonymous") or False),
            status=status,  # type: ignore[arg-type]
            reject_reason=str(data.get("reject_reason") or ""),
            published_fid=str(data.get("published_fid") or ""),
            created_at=str(data.get("created_at") or now_iso()),
            updated_at=str(data.get("updated_at") or now_iso()),
            life_job_id=str(data.get("life_job_id") or ""),
            life_owner_uin=int(data.get("life_owner_uin") or 0),
            publish_claim_token=str(data.get("publish_claim_token") or ""),
            publish_account_uin=int(data.get("publish_account_uin") or 0),
            publish_digest=str(data.get("publish_digest") or ""),
            publish_snapshot=dict(data.get("publish_snapshot") or {}),
            publish_claimed_at=str(data.get("publish_claimed_at") or ""),
            publish_result=dict(data.get("publish_result") or {}),
            publish_diagnostic=dict(data.get("publish_diagnostic") or {}),
        )

    def title(self) -> str:
        author = "匿名投稿" if self.anonymous else (self.author_name or str(self.author_uin or "未知用户"))
        return f"稿件 #{self.id} · {author} · {self.status}"

    def preview(self, *, include_private: bool = True) -> str:
        lines = [self.title()]
        if include_private and not self.anonymous:
            lines.append(f"投稿人: {self.author_name or self.author_uin}")
        if self.group_id:
            lines.append(f"来源群: {self.group_id}")
        if self.content:
            lines.append(self.content)
        if self.media:
            lines.append(f"媒体: {len(self.media)} 个")
        if self.reject_reason:
            lines.append(f"拒绝原因: {self.reject_reason}")
        if self.published_fid:
            lines.append(f"已发布 fid: {self.published_fid}")
        return "\n".join(lines)


class DraftStore:
    def __init__(self, path: Path):
        self.path = path
        self._store = AtomicItemStoreFile(path, strict=True)

    def _read_payload(self) -> dict[str, Any]:
        return self._store.read()

    def _write_payload(self, payload: dict[str, Any]) -> None:
        self._store.write(payload)

    async def _read_payload_async(self) -> dict[str, Any]:
        return await self._store.read_async()

    async def _write_payload_async(self, payload: dict[str, Any]) -> None:
        await self._store.write_async(payload)

    def list(self, *, status: str | None = None) -> list[DraftPost]:
        payload = self._read_payload()
        return self._items_from_payload(payload, status=status)

    async def list_async(self, *, status: str | None = None) -> list[DraftPost]:
        payload = await self._read_payload_async()
        return self._items_from_payload(payload, status=status)

    @staticmethod
    def _items_from_payload(payload: dict[str, Any], *, status: str | None = None) -> list[DraftPost]:
        items = [DraftPost.from_dict(item) for item in payload.get("items") or [] if isinstance(item, dict)]
        if status:
            items = [item for item in items if item.status == status]
        return sorted(items, key=lambda item: item.id)

    def get(self, draft_id: int | None = None) -> DraftPost | None:
        target_id = self._normalize_id(draft_id)
        if target_id <= 0:
            return None
        items = self.list()
        for item in items:
            if item.id == target_id:
                return item
        return None

    async def get_async(self, draft_id: int | None = None) -> DraftPost | None:
        target_id = self._normalize_id(draft_id)
        if target_id <= 0:
            return None
        items = await self.list_async()
        for item in items:
            if item.id == target_id:
                return item
        return None

    @staticmethod
    def _normalize_id(draft_id: int | None = None) -> int:
        try:
            return int(draft_id or 0)
        except (TypeError, ValueError):
            return 0

    def latest_pending(self) -> DraftPost | None:
        items = self.list(status="pending")
        return items[-1] if items else None

    async def latest_pending_async(self) -> DraftPost | None:
        items = await self.list_async(status="pending")
        return items[-1] if items else None

    def add(
        self,
        *,
        author_uin: int,
        author_name: str = "",
        group_id: int = 0,
        content: str = "",
        media: list[dict[str, Any]] | None = None,
        anonymous: bool = False,
        life_job_id: str = "",
        life_owner_uin: int = 0,
    ) -> DraftPost:
        return self._store.transact(
            lambda payload: self._add_to_payload(
                payload,
                author_uin=author_uin,
                author_name=author_name,
                group_id=group_id,
                content=content,
                media=media,
                anonymous=anonymous,
                life_job_id=life_job_id,
                life_owner_uin=life_owner_uin,
            )
        )

    async def add_async(
        self,
        *,
        author_uin: int,
        author_name: str = "",
        group_id: int = 0,
        content: str = "",
        media: list[dict[str, Any]] | None = None,
        anonymous: bool = False,
        life_job_id: str = "",
        life_owner_uin: int = 0,
    ) -> DraftPost:
        return await self._store.transact_async(
            lambda payload: self._add_to_payload(
                payload,
                author_uin=author_uin,
                author_name=author_name,
                group_id=group_id,
                content=content,
                media=media,
                anonymous=anonymous,
                life_job_id=life_job_id,
                life_owner_uin=life_owner_uin,
            )
        )

    def _add_to_payload(
        self,
        payload: dict[str, Any],
        *,
        author_uin: int,
        author_name: str,
        group_id: int,
        content: str,
        media: list[dict[str, Any]] | None,
        anonymous: bool,
        life_job_id: str,
        life_owner_uin: int,
    ) -> DraftPost:
        draft_id = int(payload.get("next_id") or 1)
        draft = DraftPost(
            id=draft_id,
            author_uin=author_uin,
            author_name=author_name,
            group_id=group_id,
            content=content,
            media=list(media or []),
            anonymous=anonymous,
            life_job_id=life_job_id,
            life_owner_uin=life_owner_uin,
        )
        items = [item for item in payload.get("items") or [] if isinstance(item, dict)]
        items.append(draft.to_dict())
        payload["items"] = items
        payload["next_id"] = draft_id + 1
        return draft

    def save(self, draft: DraftPost) -> DraftPost:
        draft.updated_at = now_iso()
        return self._store.transact(lambda payload: self._save_to_payload(payload, draft))

    async def save_async(self, draft: DraftPost) -> DraftPost:
        draft.updated_at = now_iso()
        return await self._store.transact_async(lambda payload: self._save_to_payload(payload, draft))

    def _save_to_payload(self, payload: dict[str, Any], draft: DraftPost) -> DraftPost:
        items: list[dict[str, Any]] = []
        found = False
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            if int(item.get("id") or 0) == draft.id:
                existing = DraftPost.from_dict(item)
                if existing.status in _PROTECTED_STATES:
                    items.append(existing.to_dict())
                    found = True
                    continue
                items.append(draft.to_dict())
                found = True
            else:
                items.append(item)
        if not found:
            items.append(draft.to_dict())
        payload["items"] = items
        payload["next_id"] = max(int(payload.get("next_id") or 1), draft.id + 1)
        return draft

    def update(self, draft_id: int, mutator: Callable[[DraftPost], None]) -> DraftPost | None:
        target_id = self._normalize_id(draft_id)
        if target_id <= 0:
            return None
        return self._store.transact(lambda payload: self._update_payload(payload, target_id, mutator))

    async def update_async(self, draft_id: int, mutator: Callable[[DraftPost], None]) -> DraftPost | None:
        target_id = self._normalize_id(draft_id)
        if target_id <= 0:
            return None
        return await self._store.transact_async(lambda payload: self._update_payload(payload, target_id, mutator))

    @staticmethod
    def _update_payload(
        payload: dict[str, Any],
        target_id: int,
        mutator: Callable[[DraftPost], None],
    ) -> DraftPost | None:
        items: list[dict[str, Any]] = []
        updated: DraftPost | None = None
        for item in payload.get("items") or []:
            if not isinstance(item, dict):
                continue
            if int(item.get("id") or 0) == target_id:
                draft = DraftPost.from_dict(item)
                before = draft.to_dict()
                mutator(draft)
                if before.get("status") in _PROTECTED_STATES:
                    # A generic update/save path is never a publication-state transition.
                    draft = DraftPost.from_dict(before)
                draft.updated_at = now_iso()
                updated = draft
                items.append(draft.to_dict())
            else:
                items.append(item)
        payload["items"] = items
        return updated

    @staticmethod
    def _canonical_snapshot(content: str, media: list[dict[str, Any]], sync_weibo: bool) -> tuple[dict[str, Any], str]:
        snapshot = {"content": str(content), "media": media, "sync_weibo": bool(sync_weibo)}
        digest = hashlib.sha256(json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
        return snapshot, digest

    async def _transaction_uncancellable(self, updater: Callable[[dict[str, Any]], Any]) -> Any:
        task = asyncio.create_task(self._store.transact_async(updater))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # The worker thread continues; settle it before returning cancellation to the caller.
            try:
                await asyncio.shield(task)
            finally:
                raise

    async def claim_publish_async(
        self, draft_id: int, *, account_uin: int, content: str,
        media: list[dict[str, Any]], sync_weibo: bool = False,
        life_job_id: str = "", life_owner_uin: int = 0,
    ) -> dict[str, Any]:
        snapshot, digest = self._canonical_snapshot(content, media, sync_weibo)
        token = secrets.token_urlsafe(24)
        def claim(payload: dict[str, Any]) -> dict[str, Any]:
            for raw in payload.get("items") or []:
                if not isinstance(raw, dict) or int(raw.get("id") or 0) != int(draft_id):
                    continue
                draft = DraftPost.from_dict(raw)
                if draft.status == "published":
                    if draft.publish_account_uin == account_uin and draft.publish_digest == digest:
                        return {"state": "already_published", "draft": draft, "result": draft.publish_result}
                    return {"state": "conflict", "draft": draft}
                if draft.status in {"publishing", "unknown", "approved"}:
                    return {"state": "blocked", "draft": draft}
                if draft.status != "pending":
                    return {"state": "conflict", "draft": draft}
                if draft.life_job_id and (draft.life_job_id != life_job_id or draft.life_owner_uin != account_uin):
                    return {"state": "conflict", "draft": draft}
                if life_job_id and (draft.life_job_id != life_job_id or draft.life_owner_uin != account_uin):
                    return {"state": "conflict", "draft": draft}
                draft.status = "publishing"
                draft.content = str(content)
                draft.media = list(media)
                draft.publish_claim_token = token
                draft.publish_account_uin = int(account_uin)
                draft.publish_digest = digest
                draft.publish_snapshot = snapshot
                draft.publish_claimed_at = now_iso()
                draft.updated_at = draft.publish_claimed_at
                raw.clear()
                raw.update(draft.to_dict())
                return {"state": "acquired", "draft": draft, "token": token, "digest": digest, "snapshot": snapshot}
            return {"state": "missing"}
        task = asyncio.create_task(self._store.transact_async(claim))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                result = await asyncio.shield(task)
                if isinstance(result, dict) and result.get("state") == "acquired":
                    await asyncio.shield(self.release_publish_async(
                        draft_id, token=token, account_uin=account_uin, digest=digest,
                    ))
            finally:
                raise

    async def complete_publish_async(self, draft_id: int, *, token: str, account_uin: int, digest: str,
                                     result: dict[str, Any], fid: str = "") -> bool:
        def complete(payload: dict[str, Any]) -> bool:
            for raw in payload.get("items") or []:
                if isinstance(raw, dict) and int(raw.get("id") or 0) == int(draft_id):
                    draft = DraftPost.from_dict(raw)
                    if (draft.status != "publishing" or draft.publish_claim_token != token
                            or draft.publish_account_uin != account_uin or draft.publish_digest != digest):
                        return False
                    draft.status = "published"
                    draft.published_fid = str(fid or result.get("fid") or "")
                    draft.publish_result = dict(result)
                    draft.updated_at = now_iso()
                    raw.clear()
                    raw.update(draft.to_dict())
                    return True
            return False
        return await self._transaction_uncancellable(complete)

    async def snapshot_publish_async(self, draft_id: int, *, token: str, account_uin: int,
                                    content: str, media: list[dict[str, Any]],
                                    sync_weibo: bool = False) -> str | None:
        snapshot, digest = self._canonical_snapshot(content, media, sync_weibo)
        def update(payload: dict[str, Any]) -> str | None:
            for raw in payload.get("items") or []:
                if isinstance(raw, dict) and int(raw.get("id") or 0) == int(draft_id):
                    draft = DraftPost.from_dict(raw)
                    if (draft.status != "publishing" or draft.publish_claim_token != token
                            or draft.publish_account_uin != account_uin):
                        return None
                    draft.content = str(content)
                    draft.media = list(media)
                    draft.publish_digest = digest
                    draft.publish_snapshot = snapshot
                    draft.updated_at = now_iso()
                    raw.clear()
                    raw.update(draft.to_dict())
                    return digest
            return None
        return await self._transaction_uncancellable(update)

    async def release_publish_async(self, draft_id: int, *, token: str, account_uin: int, digest: str) -> bool:
        def release(payload: dict[str, Any]) -> bool:
            for raw in payload.get("items") or []:
                if isinstance(raw, dict) and int(raw.get("id") or 0) == int(draft_id):
                    draft = DraftPost.from_dict(raw)
                    if (draft.status != "publishing" or draft.publish_claim_token != token
                            or draft.publish_account_uin != account_uin or draft.publish_digest != digest):
                        return False
                    draft.status = "pending"
                    draft.publish_claim_token = ""
                    draft.publish_digest = ""
                    draft.publish_snapshot = {}
                    draft.publish_claimed_at = ""
                    draft.publish_account_uin = 0
                    draft.updated_at = now_iso()
                    raw.clear()
                    raw.update(draft.to_dict())
                    return True
            return False
        return await self._transaction_uncancellable(release)

    async def mark_unknown_async(self, draft_id: int, *, token: str, account_uin: int, digest: str, diagnostic: dict[str, str]) -> bool:
        allowed_codes = {
            "INTERFACE_ERROR", "REQUEST_ERROR", "SOURCE_OR_FORMAT_ERROR", "AUTH_REQUIRED",
            "DAEMON_UNAVAILABLE", "TIMEOUT", "CANCELLED", "INTERNAL_ERROR",
        }
        allowed_phases = {"controller_call", "result_processing", "bookkeeping"}
        diagnostic_messages = {
            "INTERFACE_ERROR": "调用接口时发生错误", "REQUEST_ERROR": "请求处理失败",
            "SOURCE_OR_FORMAT_ERROR": "媒体来源或格式未通过处理", "AUTH_REQUIRED": "登录状态需要重新确认",
            "DAEMON_UNAVAILABLE": "后台服务暂不可用", "TIMEOUT": "操作等待超时",
            "CANCELLED": "操作被取消，结果按未确认保留", "INTERNAL_ERROR": "内部处理发生错误",
        }
        code = str(diagnostic.get("code") or "INTERNAL_ERROR")
        phase = str(diagnostic.get("phase") or "controller_call")
        code = code if code in allowed_codes else "INTERNAL_ERROR"
        safe_diagnostic = {
            "id": (str(diagnostic.get("id")) if re.fullmatch(r"[0-9a-f]{12}", str(diagnostic.get("id") or ""))
                   else secrets.token_hex(6)),
            "code": code,
            "phase": phase if phase in allowed_phases else "controller_call",
            "message": diagnostic_messages[code],
            "at": diagnostic.get("at") if isinstance(diagnostic.get("at"), (int, float)) else time(),
        }
        def mark(payload: dict[str, Any]) -> bool:
            for raw in payload.get("items") or []:
                if isinstance(raw, dict) and int(raw.get("id") or 0) == int(draft_id):
                    draft = DraftPost.from_dict(raw)
                    if (draft.status != "publishing" or draft.publish_claim_token != token
                            or draft.publish_account_uin != account_uin or draft.publish_digest != digest):
                        return False
                    draft.status = "unknown"
                    draft.publish_diagnostic = safe_diagnostic
                    draft.updated_at = now_iso()
                    raw.clear()
                    raw.update(draft.to_dict())
                    return True
            return False
        return await self._transaction_uncancellable(mark)
