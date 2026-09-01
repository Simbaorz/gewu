"""Conversation-scoped file knowledge used by read-before-write tools."""

from __future__ import annotations

from collections.abc import Set
from typing import Any

from pydantic import BaseModel


class FileState(BaseModel):
    """Version and exact range returned by the latest retained Read result."""

    version: str = ""
    offset: int | None = None
    limit: int | None = None


class FileStateCache:
    """Mutable working copy persisted as one Runtime conversation state."""

    def __init__(self) -> None:
        self._cache: dict[str, FileState] = {}
        self._dirty = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> FileStateCache:
        cache = cls()
        values = payload.get("files")
        if not isinstance(values, dict):
            return cache
        for path, value in values.items():
            if not isinstance(path, str) or not isinstance(value, dict):
                continue
            try:
                cache._cache[cache._normalize_path(path)] = FileState.model_validate(value)
            except ValueError:
                continue
        return cache

    @staticmethod
    def _normalize_path(path: str) -> str:
        return path.replace("\\", "/").strip("/")

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False

    def replace_with(self, other: FileStateCache) -> None:
        self._cache = {path: state.model_copy(deep=True) for path, state in other._cache.items()}
        self._dirty = False

    def to_payload(self) -> dict[str, Any]:
        return {
            "files": {path: state.model_dump(mode="json") for path, state in self._cache.items()}
        }

    def get(self, path: str) -> FileState | None:
        return self._cache.get(self._normalize_path(path))

    def set(self, path: str, state: FileState) -> None:
        self._cache[self._normalize_path(path)] = state
        self._dirty = True

    def has(self, path: str) -> bool:
        return self._normalize_path(path) in self._cache

    def delete(self, path: str) -> None:
        if self._cache.pop(self._normalize_path(path), None) is not None:
            self._dirty = True

    def delete_prefix(self, path: str) -> None:
        normalized = self._normalize_path(path)
        prefix = f"{normalized}/"
        deleted = False
        for cached_path in tuple(self._cache):
            if cached_path == normalized or cached_path.startswith(prefix):
                self._cache.pop(cached_path)
                deleted = True
        self._dirty = self._dirty or deleted

    def clear(self) -> None:
        if self._cache:
            self._cache.clear()
            self._dirty = True

    def items(self) -> list[tuple[str, FileState]]:
        return list(self._cache.items())

    def retain_read_states(self, retained_reads: Set[tuple[str, int, int]]) -> None:
        normalized = {
            (self._normalize_path(path), offset, limit) for path, offset, limit in retained_reads
        }
        for path, state in tuple(self._cache.items()):
            if (
                state.offset is None
                or state.limit is None
                or (path, state.offset, state.limit) not in normalized
            ):
                self._cache.pop(path)
                self._dirty = True
