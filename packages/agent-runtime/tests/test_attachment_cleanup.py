"""Runtime-owned attachment lifecycle cleanup."""

from __future__ import annotations

import logging
from datetime import timedelta

import pytest

from gewu_agent_runtime import AttachmentCleanupService
from gewu_agent_runtime.domain import Conversation, StoredAttachment
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_core.time import utc_now


class _ObjectStore:
    storage_backend = "local"

    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, resource_key: str) -> None:
        if resource_key.endswith("failed.png"):
            raise OSError("attachment-delete-private-secret")
        self.deleted.append(resource_key)


async def test_cleanup_marks_only_successfully_removed_objects_deleted(
    principal,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    for attachment_id in ("deleted", "failed"):
        await store.create_attachment(
            StoredAttachment(
                attachment_id=attachment_id,
                owner=principal,
                conversation_id=conversation.conversation_id,
                request_id=f"request-{attachment_id}",
                storage_backend="local",
                resource_key=f"chat/{attachment_id}.png",
                mime_type="image/png",
                size_bytes=10,
                created_at=utc_now() - timedelta(hours=25),
            )
        )
    object_store = _ObjectStore()

    with caplog.at_level(logging.ERROR, logger="gewu_agent_runtime.attachment_cleanup"):
        result = await AttachmentCleanupService(
            store=store,
            object_store=object_store,
            pending_ttl_hours=24,
        ).cleanup()

    assert result.model_dump() == {"scanned": 2, "deleted": 1, "failed": 1}
    assert object_store.deleted == ["chat/deleted.png"]
    assert await store.get_active_attachment("deleted", principal) is None
    assert await store.get_active_attachment("failed", principal) is not None
    assert "attachment_id=failed" in caplog.text
    assert "exception_type=OSError" in caplog.text
    assert "chat/failed.png" not in caplog.text
    assert "attachment-delete-private-secret" not in caplog.text
