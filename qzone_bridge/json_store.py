"""Small atomic JSON persistence helper for plugin-local caches."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar


T = TypeVar("T")


class StoreReadError(RuntimeError):
    """A strict store could not be read without risking an authorization reset."""

_LOCKS: dict[Path, threading.RLock] = {}
_LOCKS_LOCK = threading.RLock()


def _default_payload() -> dict[str, Any]:
    return {"next_id": 1, "items": []}


def _chmod_private(path: Path) -> None:
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def _lock_for(path: Path) -> threading.RLock:
    key = path.resolve(strict=False)
    with _LOCKS_LOCK:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _LOCKS[key] = lock
        return lock


class AtomicItemStoreFile:
    def __init__(self, path: Path, *, strict: bool = False):
        self.path = Path(path)
        self._lock = _lock_for(self.path)
        self.strict = strict

    @contextlib.contextmanager
    def _process_lock(self):
        """Hold a kernel-owned lock across the full read/modify/replace sequence."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_name(self.path.name + ".lock")
        with lock_path.open("a+b") as handle:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @contextlib.contextmanager
    def _guard(self):
        with self._lock:
            if self.strict:
                with self._process_lock():
                    yield
            else:
                yield

    def read(self) -> dict[str, Any]:
        with self._guard():
            return self._read_unlocked()

    def write(self, payload: dict[str, Any]) -> None:
        with self._guard():
            self._write_unlocked(payload)

    def transact(self, updater: Callable[[dict[str, Any]], T]) -> T:
        with self._guard():
            payload = self._read_unlocked()
            result = updater(payload)
            self._write_unlocked(payload)
            return result

    async def read_async(self) -> dict[str, Any]:
        return await asyncio.to_thread(self.read)

    async def write_async(self, payload: dict[str, Any]) -> None:
        await asyncio.to_thread(self.write, payload)

    async def transact_async(self, updater: Callable[[dict[str, Any]], T]) -> T:
        return await asyncio.to_thread(self.transact, updater)

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_payload()
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            if self.strict:
                raise StoreReadError("strict JSON store is unreadable") from exc
            self._quarantine_corrupt_unlocked()
            return _default_payload()
        if not isinstance(payload, dict):
            if self.strict:
                raise StoreReadError("strict JSON store has an invalid root")
            self._quarantine_corrupt_unlocked()
            return _default_payload()
        payload.setdefault("next_id", 1)
        if not isinstance(payload.get("items"), list):
            if self.strict:
                raise StoreReadError("strict JSON store has an invalid items collection")
            payload["items"] = []
        if self.strict and any(not isinstance(item, dict) for item in payload["items"]):
            raise StoreReadError("strict JSON store contains an invalid item")
        return payload

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_name(f"{self.path.name}.tmp.{uuid.uuid4().hex}")
        data = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        try:
            with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(data)
                handle.flush()
                if self.strict:
                    os.fsync(handle.fileno())
            _chmod_private(tmp_path)
            os.replace(tmp_path, self.path)
            _chmod_private(self.path)
            if self.strict and os.name != "nt":
                directory_fd = os.open(str(self.path.parent), os.O_RDONLY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass

    def _quarantine_corrupt_unlocked(self) -> None:
        if not self.path.exists():
            return
        backup = self.path.with_name(f"{self.path.name}.corrupt.{uuid.uuid4().hex}")
        try:
            os.replace(self.path, backup)
            _chmod_private(backup)
        except OSError:
            pass
