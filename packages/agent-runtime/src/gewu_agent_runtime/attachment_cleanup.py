"""Lifecycle cleanup for Runtime-owned attachment references."""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.persistence import RuntimeStore
from gewu_core.time import utc_now

logger = logging.getLogger(__name__)


class AttachmentObjectStore(Protocol):
    """Delete opaque attachment objects from one configured storage backend."""

    storage_backend: str

    async def delete(self, resource_key: str) -> None: ...


class AttachmentCleanupResult(BaseModel):
    """Summary of one bounded attachment cleanup pass."""

    model_config = ConfigDict(frozen=True)

    scanned: int = Field(ge=0)
    deleted: int = Field(ge=0)
    failed: int = Field(ge=0)


class AttachmentCleanupService:
    """Remove abandoned uploads and media owned by archived conversations."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        object_store: AttachmentObjectStore,
        pending_ttl_hours: int,
        batch_size: int = 200,
    ) -> None:
        if pending_ttl_hours < 1:
            raise ValueError("Attachment pending TTL must be positive.")
        if batch_size < 1:
            raise ValueError("Attachment cleanup batch size must be positive.")
        self._store = store
        self._object_store = object_store
        self._pending_ttl_hours = pending_ttl_hours
        self._batch_size = batch_size

    async def cleanup(self) -> AttachmentCleanupResult:
        """Delete one bounded batch while retaining failed references for retry."""

        candidates = await self._store.list_attachment_cleanup_candidates(
            pending_before=utc_now() - timedelta(hours=self._pending_ttl_hours),
            storage_backend=self._object_store.storage_backend,
            limit=self._batch_size,
        )
        deleted_ids: list[str] = []
        for attachment in candidates:
            if attachment.storage_backend != self._object_store.storage_backend:
                logger.error(
                    "Unable to clean attachment from unavailable storage backend "
                    "attachment_id=%s",
                    attachment.attachment_id,
                )
                continue
            try:
                await self._object_store.delete(attachment.resource_key)
            except Exception as exc:
                logger.error(
                    "Unable to delete expired attachment object attachment_id=%s "
                    "exception_type=%s",
                    attachment.attachment_id,
                    type(exc).__name__,
                )
                continue
            deleted_ids.append(attachment.attachment_id)
        await self._store.mark_attachments_deleted(deleted_ids)
        return AttachmentCleanupResult(
            scanned=len(candidates),
            deleted=len(deleted_ids),
            failed=len(candidates) - len(deleted_ids),
        )
