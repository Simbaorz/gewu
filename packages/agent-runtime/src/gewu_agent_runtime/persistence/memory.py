"""Concurrency-safe in-memory fact store."""

from __future__ import annotations

import asyncio
import copy
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    AttachmentStatus,
    Conversation,
    ConversationCompaction,
    ConversationCompactionCommit,
    ConversationMessage,
    ConversationPage,
    ConversationState,
    ConversationStatus,
    MessageKind,
    NewConversationMessage,
    StoredAttachment,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.persistence.errors import (
    ConcurrentWriteError,
    EntityNotFoundError,
    IdempotencyConflictError,
    MessageWriteConflictError,
)
from gewu_agent_runtime.persistence.protected import (
    ProtectedPayloadCipher,
    hydrated_compaction,
    hydrated_message,
    stored_compaction,
    stored_message,
)
from gewu_core.time import utc_now


class InMemoryRuntimeStore:
    """Process-local Runtime store for tests and ephemeral deployments."""

    def __init__(
        self,
        *,
        protected_payload_cipher: ProtectedPayloadCipher | None = None,
        encrypt_protected_payloads: bool = False,
        encrypt_compactions: bool = False,
    ) -> None:
        if (encrypt_protected_payloads or encrypt_compactions) and protected_payload_cipher is None:
            raise ValueError("Enabled Runtime encryption requires a protected payload cipher.")
        self._conversations: dict[str, Conversation] = {}
        self._messages: dict[str, list[ConversationMessage]] = {}
        self._attachments: dict[str, StoredAttachment] = {}
        self._runs: dict[str, AgentRun] = {}
        self._idempotency: dict[tuple[str, str, str], str] = {}
        self._states: dict[tuple[str, str], ConversationState] = {}
        self._compactions: dict[str, ConversationCompaction] = {}
        self._protected_payload_cipher = protected_payload_cipher
        self._encrypt_protected_payloads = encrypt_protected_payloads
        self._encrypt_compactions = encrypt_compactions
        self._lock = asyncio.Lock()

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with self._lock:
            if conversation.conversation_id in self._conversations:
                raise ConcurrentWriteError("Conversation already exists.")
            stored = conversation.model_copy(deep=True)
            self._conversations[stored.conversation_id] = stored
            self._messages[stored.conversation_id] = []
            return stored.model_copy(deep=True)

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        async with self._lock:
            value = self._conversations.get(conversation_id)
            return value.model_copy(deep=True) if value is not None else None

    async def list_conversations(
        self,
        owner: PrincipalRef,
        *,
        include_archived: bool = False,
        updated_after: datetime | None = None,
        before_updated_at: datetime | None = None,
        before_conversation_id: str = "",
        limit: int = 30,
    ) -> ConversationPage:
        if limit < 1:
            raise ValueError("Conversation page limit must be positive.")
        async with self._lock:
            values = [
                value
                for value in self._conversations.values()
                if value.owner == owner
                and (include_archived or value.status is ConversationStatus.ACTIVE)
                and (updated_after is None or value.updated_at >= updated_after)
            ]
            total = len(values)
            if before_updated_at is not None:
                values = [
                    value
                    for value in values
                    if value.updated_at < before_updated_at
                    or (
                        value.updated_at == before_updated_at
                        and value.conversation_id < before_conversation_id
                    )
                ]
            values.sort(
                key=lambda value: (value.updated_at, value.conversation_id),
                reverse=True,
            )
            return ConversationPage(
                items=tuple(value.model_copy(deep=True) for value in values[:limit]),
                total=total,
            )

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: ConversationStatus | None = None,
    ) -> Conversation | None:
        async with self._lock:
            current = self._conversations.get(conversation_id)
            if current is None:
                return None
            changes: dict[str, object] = {"updated_at": utc_now()}
            if title is not None:
                changes["title"] = title
            if status is not None:
                changes["status"] = status
            updated = current.model_copy(update=changes, deep=True)
            self._conversations[conversation_id] = updated
            return updated.model_copy(deep=True)

    async def compare_and_patch_conversation_metadata(
        self,
        conversation_id: str,
        *,
        expected_values: Mapping[str, Any],
        set_values: Mapping[str, Any],
        remove_keys: Sequence[str] = (),
    ) -> Conversation | None:
        async with self._lock:
            current = self._conversations.get(conversation_id)
            if current is None or any(
                current.metadata.get(key) != expected for key, expected in expected_values.items()
            ):
                return None
            metadata = copy.deepcopy(current.metadata)
            metadata.update(copy.deepcopy(dict(set_values)))
            for key in remove_keys:
                metadata.pop(key, None)
            updated = current.model_copy(
                update={"metadata": metadata, "updated_at": utc_now()},
                deep=True,
            )
            self._conversations[conversation_id] = updated
            return updated.model_copy(deep=True)

    async def create_attachment(self, attachment: StoredAttachment) -> StoredAttachment:
        async with self._lock:
            conversation = self._require_conversation(attachment.conversation_id)
            if conversation.owner != attachment.owner:
                raise ValueError("Attachment owner must match its conversation owner.")
            if attachment.attachment_id in self._attachments:
                raise ConcurrentWriteError("Attachment already exists.")
            stored = attachment.model_copy(deep=True)
            self._attachments[stored.attachment_id] = stored
            return stored.model_copy(deep=True)

    async def get_active_attachment(
        self,
        attachment_id: str,
        owner: PrincipalRef,
    ) -> StoredAttachment | None:
        async with self._lock:
            value = self._attachments.get(attachment_id)
            if value is None or value.owner != owner or value.status is not AttachmentStatus.ACTIVE:
                return None
            return value.model_copy(deep=True)

    async def list_active_attachments(
        self,
        owner: PrincipalRef,
        *,
        conversation_id: str,
        request_id: str,
        attachment_ids: Sequence[str],
    ) -> tuple[StoredAttachment, ...]:
        async with self._lock:
            result: list[StoredAttachment] = []
            for attachment_id in dict.fromkeys(attachment_ids):
                value = self._attachments.get(attachment_id)
                if (
                    value is not None
                    and value.owner == owner
                    and value.conversation_id == conversation_id
                    and value.request_id == request_id
                    and value.status is AttachmentStatus.ACTIVE
                ):
                    result.append(value.model_copy(deep=True))
            return tuple(result)

    async def list_attachment_cleanup_candidates(
        self,
        *,
        pending_before: datetime,
        storage_backend: str,
        limit: int,
    ) -> tuple[StoredAttachment, ...]:
        if limit < 1:
            raise ValueError("Attachment cleanup limit must be positive.")
        async with self._lock:
            candidates = []
            for attachment in self._attachments.values():
                if (
                    attachment.status is not AttachmentStatus.ACTIVE
                    or attachment.storage_backend != storage_backend
                ):
                    continue
                conversation = self._conversations.get(attachment.conversation_id)
                archived = (
                    conversation is not None and conversation.status is ConversationStatus.ARCHIVED
                )
                submitted = any(
                    message.kind is MessageKind.INPUT
                    and message.request_id == attachment.request_id
                    for message in self._messages.get(attachment.conversation_id, ())
                )
                if archived or (attachment.created_at < pending_before and not submitted):
                    candidates.append(attachment)
            candidates.sort(key=lambda value: (value.created_at, value.attachment_id))
            return tuple(value.model_copy(deep=True) for value in candidates[:limit])

    async def mark_attachments_deleted(self, attachment_ids: Sequence[str]) -> None:
        async with self._lock:
            for attachment_id in dict.fromkeys(attachment_ids):
                value = self._attachments.get(attachment_id)
                if value is not None and value.status is AttachmentStatus.ACTIVE:
                    self._attachments[attachment_id] = value.model_copy(
                        update={"status": AttachmentStatus.DELETED},
                        deep=True,
                    )

    async def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[NewConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        if not messages:
            return ()
        async with self._lock:
            conversation = self._require_conversation(conversation_id)
            existing_ids = self._existing_message_ids()
            message_ids = [item.message_id for item in messages]
            if len(message_ids) != len(set(message_ids)) or any(
                message_id in existing_ids for message_id in message_ids
            ):
                raise MessageWriteConflictError("Message ID already exists.")
            sequence = conversation.next_sequence
            persisted = tuple(
                stored_message(
                    message,
                    conversation_id=conversation_id,
                    sequence=sequence + index,
                    cipher=self._protected_payload_cipher,
                    encrypt=self._encrypt_protected_payloads,
                )
                for index, message in enumerate(messages)
            )
            self._messages[conversation_id].extend(persisted)
            self._conversations[conversation_id] = conversation.model_copy(
                update={"next_sequence": sequence + len(persisted), "updated_at": utc_now()}
            )
            return tuple(self._hydrate_message(item) for item in persisted)

    async def append_messages_for_run(
        self,
        run_id: str,
        messages: Sequence[NewConversationMessage],
        *,
        finish_status: AgentRunStatus | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> tuple[ConversationMessage, ...] | None:
        if finish_status is not None and finish_status not in {
            AgentRunStatus.COMPLETED,
            AgentRunStatus.FAILED,
            AgentRunStatus.CANCELLED,
        }:
            raise ValueError("A run-fenced message commit can only finish in a terminal state.")
        if any(message.run_id != run_id for message in messages):
            raise ValueError("Every run-fenced message must reference its owning run.")
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return None
            conversation = self._require_conversation(run.conversation_id)
            if run.status is not AgentRunStatus.RUNNING or conversation.active_run_id != run_id:
                return None
            existing_ids = self._existing_message_ids()
            message_ids = [item.message_id for item in messages]
            if len(message_ids) != len(set(message_ids)) or any(
                message_id in existing_ids for message_id in message_ids
            ):
                raise MessageWriteConflictError("Message ID already exists.")
            now = utc_now()
            persisted = tuple(
                stored_message(
                    message,
                    conversation_id=run.conversation_id,
                    sequence=conversation.next_sequence + index,
                    cipher=self._protected_payload_cipher,
                    encrypt=self._encrypt_protected_payloads,
                )
                for index, message in enumerate(messages)
            )
            self._messages[run.conversation_id].extend(persisted)
            conversation_changes: dict[str, object] = {
                "next_sequence": conversation.next_sequence + len(persisted),
                "updated_at": now,
            }
            if finish_status is not None:
                conversation_changes["active_run_id"] = None
                self._runs[run_id] = run.model_copy(
                    update={
                        "status": finish_status,
                        "error_code": error_code,
                        "error_message": error_message,
                        "finished_at": now,
                        "updated_at": now,
                    }
                )
            self._conversations[run.conversation_id] = conversation.model_copy(
                update=conversation_changes
            )
            return tuple(self._hydrate_message(item) for item in persisted)

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        async with self._lock:
            self._require_conversation(conversation_id)
            return tuple(
                self._hydrate_message(item)
                for item in self._messages[conversation_id]
                if item.sequence > after_sequence
                and (through_sequence is None or item.sequence <= through_sequence)
            )

    async def list_messages_for_run(
        self,
        run_id: str,
    ) -> tuple[ConversationMessage, ...]:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return ()
            return tuple(
                self._hydrate_message(item)
                for item in self._messages[run.conversation_id]
                if item.run_id == run_id
            )

    async def list_message_page(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> tuple[ConversationMessage, ...]:
        if limit < 1:
            raise ValueError("Message page limit must be positive.")
        async with self._lock:
            self._require_conversation(conversation_id)
            values = (
                item
                for item in self._messages[conversation_id]
                if item.sequence > after_sequence
                and (before_sequence is None or item.sequence < before_sequence)
            )
            return tuple(self._hydrate_message(item) for item in list(values)[:limit])

    async def list_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        if limit < 1:
            raise ValueError("Recent message limit must be positive.")
        async with self._lock:
            self._require_conversation(conversation_id)
            values = [
                item
                for item in self._messages[conversation_id]
                if before_sequence is None or item.sequence < before_sequence
            ][-limit:]
            return tuple(self._hydrate_message(item) for item in values)

    async def create_run(self, run: AgentRun) -> AgentRun:
        async with self._lock:
            self._require_conversation(run.conversation_id)
            if run.run_id in self._runs:
                raise ConcurrentWriteError("Run ID already exists.")
            if run.idempotency_key is not None:
                key = (run.invoker.subscriber_id, run.conversation_id, run.idempotency_key)
                existing_id = self._idempotency.get(key)
                if existing_id is not None:
                    existing = self._runs[existing_id]
                    if (
                        existing.request_id != run.request_id
                        or existing.input_snapshot != run.input_snapshot
                    ):
                        raise IdempotencyConflictError(
                            "Idempotency key belongs to a different request."
                        )
                    return existing.model_copy(deep=True)
                self._idempotency[key] = run.run_id
            self._runs[run.run_id] = run.model_copy(deep=True)
            return run.model_copy(deep=True)

    async def get_run(self, run_id: str) -> AgentRun | None:
        async with self._lock:
            value = self._runs.get(run_id)
            return value.model_copy(deep=True) if value is not None else None

    async def get_latest_run(self, conversation_id: str) -> AgentRun | None:
        async with self._lock:
            values = [
                value for value in self._runs.values() if value.conversation_id == conversation_id
            ]
            if not values:
                return None
            latest = max(values, key=lambda value: (value.created_at, value.run_id))
            return latest.model_copy(deep=True)

    async def transition_run(
        self,
        run_id: str,
        *,
        expected: Sequence[AgentRunStatus],
        status: AgentRunStatus,
        error_code: str = "",
        error_message: str = "",
        expected_active_run_id: str | None = None,
    ) -> AgentRun:
        async with self._lock:
            current = self._runs.get(run_id)
            if current is None:
                raise EntityNotFoundError("Run does not exist.")
            if current.status not in expected:
                raise ConcurrentWriteError(
                    f"Run is {current.status.value}; expected one of "
                    f"{', '.join(item.value for item in expected)}."
                )
            conversation = self._require_conversation(current.conversation_id)
            conversation_changes: dict[str, object] = {}
            if status is AgentRunStatus.RUNNING:
                if conversation.active_run_id != expected_active_run_id:
                    raise ConcurrentWriteError("Run ownership changed before it was claimed.")
                conversation_changes["active_run_id"] = run_id
            elif current.status is AgentRunStatus.RUNNING:
                if conversation.active_run_id != run_id:
                    raise ConcurrentWriteError("Run ownership was lost before transition.")
                conversation_changes["active_run_id"] = None
            now = utc_now()
            updated = current.model_copy(
                update={
                    "status": status,
                    "error_code": error_code,
                    "error_message": error_message,
                    "started_at": current.started_at
                    or (now if status is AgentRunStatus.RUNNING else None),
                    "finished_at": (
                        now
                        if status
                        in {
                            AgentRunStatus.COMPLETED,
                            AgentRunStatus.FAILED,
                            AgentRunStatus.CANCELLED,
                        }
                        else None
                    ),
                    "updated_at": now,
                }
            )
            self._runs[run_id] = updated
            if conversation_changes:
                conversation_changes["updated_at"] = now
                self._conversations[current.conversation_id] = conversation.model_copy(
                    update=conversation_changes
                )
            return updated.model_copy(deep=True)

    async def get_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        include_expired: bool = False,
    ) -> ConversationState | None:
        async with self._lock:
            value = self._states.get((conversation_id, kind))
            if (
                not include_expired
                and value is not None
                and value.expires_at is not None
                and value.expires_at <= utc_now()
            ):
                return None
            return value.model_copy(deep=True) if value is not None else None

    async def list_expired_states(
        self,
        kind: str,
        expires_before: datetime,
        *,
        after: ConversationState | None = None,
        limit: int = 100,
    ) -> tuple[ConversationState, ...]:
        if limit < 1:
            raise ValueError("Expired state page limit must be positive.")
        if after is not None and after.expires_at is None:
            raise ValueError("Expired state cursor must have expires_at.")
        async with self._lock:
            values = [
                value
                for value in self._states.values()
                if value.kind == kind
                and value.expires_at is not None
                and value.expires_at <= expires_before
                and (
                    after is None
                    or (value.expires_at, value.conversation_id)
                    > (after.expires_at, after.conversation_id)
                )
            ]
            values.sort(key=lambda value: (value.expires_at, value.conversation_id))
            return tuple(value.model_copy(deep=True) for value in values[:limit])

    async def save_state(
        self,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState:
        async with self._lock:
            self._require_conversation(state.conversation_id)
            current = self._states.get((state.conversation_id, state.kind))
            actual_revision = current.revision if current is not None else 0
            if actual_revision != expected_revision or state.revision != expected_revision + 1:
                raise ConcurrentWriteError("Conversation state revision changed.")
            stored = state.model_copy(update={"updated_at": utc_now()}, deep=True)
            self._states[(state.conversation_id, state.kind)] = stored
            return stored.model_copy(deep=True)

    async def save_state_for_run(
        self,
        run_id: str,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState | None:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.conversation_id != state.conversation_id:
                return None
            conversation = self._require_conversation(state.conversation_id)
            if run.status is not AgentRunStatus.RUNNING or conversation.active_run_id != run_id:
                return None
            current = self._states.get((state.conversation_id, state.kind))
            actual_revision = current.revision if current is not None else 0
            if actual_revision != expected_revision or state.revision != expected_revision + 1:
                raise ConcurrentWriteError("Conversation state revision changed.")
            stored = state.model_copy(update={"updated_at": utc_now()}, deep=True)
            self._states[(state.conversation_id, state.kind)] = stored
            return stored.model_copy(deep=True)

    async def delete_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        async with self._lock:
            current = self._states.get((conversation_id, kind))
            if current is None:
                return
            if expected_revision is not None and current.revision != expected_revision:
                raise ConcurrentWriteError("Conversation state revision changed.")
            self._states.pop((conversation_id, kind), None)

    async def delete_state_for_run(
        self,
        run_id: str,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.conversation_id != conversation_id:
                return False
            conversation = self._require_conversation(conversation_id)
            if run.status is not AgentRunStatus.RUNNING or conversation.active_run_id != run_id:
                return False
            current = self._states.get((conversation_id, kind))
            if current is None:
                return True
            if expected_revision is not None and current.revision != expected_revision:
                raise ConcurrentWriteError("Conversation state revision changed.")
            self._states.pop((conversation_id, kind), None)
            return True

    async def commit_pending_ask(
        self,
        run_id: str,
        *,
        state: ConversationState,
        message: NewConversationMessage,
    ) -> AgentRun | None:
        if (
            state.kind != "pending_ask"
            or state.revision != 1
            or state.payload.get("run_id") != run_id
            or message.run_id != run_id
        ):
            raise ValueError("Pending Ask commit must reference one new state for its active run.")
        async with self._lock:
            run = self._runs.get(run_id)
            if (
                run is None
                or run.status is not AgentRunStatus.RUNNING
                or run.conversation_id != state.conversation_id
                or (state.conversation_id, state.kind) in self._states
            ):
                return None
            conversation = self._require_conversation(run.conversation_id)
            if conversation.active_run_id != run_id:
                return None
            if message.message_id in self._existing_message_ids():
                raise MessageWriteConflictError("Message ID already exists.")
            now = utc_now()
            persisted = stored_message(
                message,
                conversation_id=run.conversation_id,
                sequence=conversation.next_sequence,
                cipher=self._protected_payload_cipher,
                encrypt=self._encrypt_protected_payloads,
            )
            self._messages[run.conversation_id].append(persisted)
            self._states[(state.conversation_id, state.kind)] = state.model_copy(
                update={"updated_at": now},
                deep=True,
            )
            self._conversations[run.conversation_id] = conversation.model_copy(
                update={
                    "next_sequence": conversation.next_sequence + 1,
                    "active_run_id": None,
                    "updated_at": now,
                }
            )
            waiting = run.model_copy(
                update={"status": AgentRunStatus.WAITING_INPUT, "updated_at": now}
            )
            self._runs[run_id] = waiting
            return waiting.model_copy(deep=True)

    async def resolve_pending_ask(
        self,
        run_id: str,
        *,
        state_revision: int,
        message: NewConversationMessage,
        error_code: str,
        error_message: str,
    ) -> AgentRun | None:
        if message.run_id != run_id:
            raise ValueError("Pending Ask resolution message must reference its waiting run.")
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status is not AgentRunStatus.WAITING_INPUT:
                return None
            state_key = (run.conversation_id, "pending_ask")
            state = self._states.get(state_key)
            if (
                state is None
                or state.revision != state_revision
                or state.payload.get("run_id") != run_id
            ):
                return None
            conversation = self._require_conversation(run.conversation_id)
            if message.message_id in self._existing_message_ids():
                raise MessageWriteConflictError("Message ID already exists.")
            now = utc_now()
            persisted = stored_message(
                message,
                conversation_id=run.conversation_id,
                sequence=conversation.next_sequence,
                cipher=self._protected_payload_cipher,
                encrypt=self._encrypt_protected_payloads,
            )
            self._messages[run.conversation_id].append(persisted)
            self._states.pop(state_key, None)
            self._conversations[run.conversation_id] = conversation.model_copy(
                update={"next_sequence": conversation.next_sequence + 1, "updated_at": now}
            )
            resolved = run.model_copy(
                update={
                    "status": AgentRunStatus.CANCELLED,
                    "error_code": error_code,
                    "error_message": error_message,
                    "finished_at": now,
                    "updated_at": now,
                }
            )
            self._runs[run_id] = resolved
            return resolved.model_copy(deep=True)

    async def commit_ask_answer(
        self,
        run_id: str,
        *,
        state_revision: int,
        message: NewConversationMessage,
    ) -> bool:
        if message.run_id != run_id:
            raise ValueError("Ask answer message must reference its active run.")
        async with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.status is not AgentRunStatus.RUNNING:
                return False
            state_key = (run.conversation_id, "pending_ask")
            state = self._states.get(state_key)
            if (
                state is None
                or state.revision != state_revision
                or state.payload.get("run_id") != run_id
            ):
                return False
            conversation = self._require_conversation(run.conversation_id)
            if conversation.active_run_id != run_id:
                return False
            if message.message_id in self._existing_message_ids():
                raise MessageWriteConflictError("Message ID already exists.")
            now = utc_now()
            persisted = stored_message(
                message,
                conversation_id=run.conversation_id,
                sequence=conversation.next_sequence,
                cipher=self._protected_payload_cipher,
                encrypt=self._encrypt_protected_payloads,
            )
            self._messages[run.conversation_id].append(persisted)
            self._states.pop(state_key, None)
            self._conversations[run.conversation_id] = conversation.model_copy(
                update={"next_sequence": conversation.next_sequence + 1, "updated_at": now}
            )
            return True

    async def get_latest_compaction(
        self,
        conversation_id: str,
    ) -> ConversationCompaction | None:
        async with self._lock:
            conversation = self._require_conversation(conversation_id)
            value = self._compactions.get(conversation.latest_compaction_id)
            return (
                hydrated_compaction(value, self._protected_payload_cipher)
                if value is not None
                else None
            )

    async def get_compaction(
        self,
        conversation_id: str,
        compaction_id: str,
    ) -> ConversationCompaction | None:
        async with self._lock:
            self._require_conversation(conversation_id)
            value = self._compactions.get(compaction_id)
            if value is None or value.conversation_id != conversation_id:
                return None
            return hydrated_compaction(value, self._protected_payload_cipher)

    async def save_compaction(
        self,
        compaction: ConversationCompaction,
        *,
        expected_previous_id: str,
    ) -> ConversationCompaction:
        async with self._lock:
            conversation = self._require_conversation(compaction.conversation_id)
            if conversation.latest_compaction_id != expected_previous_id:
                raise ConcurrentWriteError("Conversation compaction boundary changed.")
            if compaction.previous_compaction_id != expected_previous_id:
                raise ConcurrentWriteError("Compaction does not extend the current boundary.")
            self._compactions[compaction.compaction_id] = stored_compaction(
                compaction,
                self._protected_payload_cipher,
                encrypt=self._encrypt_compactions,
            )
            self._conversations[conversation.conversation_id] = conversation.model_copy(
                update={
                    "latest_compaction_id": compaction.compaction_id,
                    "updated_at": utc_now(),
                }
            )
            return compaction.model_copy(deep=True)

    async def commit_compaction_for_run(
        self,
        commit: ConversationCompactionCommit,
        run_id: str,
    ) -> bool:
        async with self._lock:
            compaction = commit.compaction
            conversation = self._require_conversation(compaction.conversation_id)
            run = self._runs.get(run_id)
            if (
                run is None
                or run.conversation_id != conversation.conversation_id
                or run.status is not AgentRunStatus.RUNNING
                or conversation.active_run_id != run_id
                or conversation.latest_compaction_id != commit.expected_previous_id
            ):
                return False
            changed_kinds = set(commit.state_payloads) | set(commit.delete_state_kinds)
            if len(changed_kinds) != len(commit.state_payloads) + len(commit.delete_state_kinds):
                raise ValueError("Compaction cannot save and delete the same state kind.")
            existing_ids = self._existing_message_ids()
            message_ids = [item.message_id for item in commit.messages]
            if len(message_ids) != len(set(message_ids)) or any(
                value in existing_ids for value in message_ids
            ):
                raise MessageWriteConflictError("Compaction lifecycle message ID already exists.")

            start = conversation.next_sequence
            persisted = tuple(
                stored_message(
                    message,
                    conversation_id=conversation.conversation_id,
                    sequence=start + index,
                    cipher=self._protected_payload_cipher,
                    encrypt=self._encrypt_protected_payloads,
                )
                for index, message in enumerate(commit.messages)
            )
            new_states: dict[str, ConversationState] = {}
            for kind, payload in commit.state_payloads.items():
                current = self._states.get((conversation.conversation_id, kind))
                new_states[kind] = ConversationState(
                    conversation_id=conversation.conversation_id,
                    kind=kind,
                    revision=(current.revision if current is not None else 0) + 1,
                    payload=payload,
                )

            self._messages[conversation.conversation_id].extend(persisted)
            for kind, state in new_states.items():
                self._states[(conversation.conversation_id, kind)] = state
            for kind in commit.delete_state_kinds:
                self._states.pop((conversation.conversation_id, kind), None)
            self._compactions[compaction.compaction_id] = stored_compaction(
                compaction,
                self._protected_payload_cipher,
                encrypt=self._encrypt_compactions,
            )
            self._conversations[conversation.conversation_id] = conversation.model_copy(
                update={
                    "next_sequence": start + len(persisted),
                    "latest_compaction_id": compaction.compaction_id,
                    "updated_at": utc_now(),
                }
            )
            return True

    def _hydrate_message(self, message: ConversationMessage) -> ConversationMessage:
        return hydrated_message(message, self._protected_payload_cipher)

    def _existing_message_ids(self) -> set[str]:
        return {
            message.message_id
            for conversation_messages in self._messages.values()
            for message in conversation_messages
        }

    def _require_conversation(self, conversation_id: str) -> Conversation:
        try:
            return self._conversations[conversation_id]
        except KeyError as exc:
            raise EntityNotFoundError("Conversation does not exist.") from exc


class InMemoryStateCache:
    """Bounded disposable LRU state cache with expiry handling."""

    def __init__(self, *, max_entries: int = 10_000, ttl_seconds: int = 86_400) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive.")
        if ttl_seconds < 1:
            raise ValueError("ttl_seconds must be positive.")
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._values: OrderedDict[
            tuple[str, str],
            tuple[ConversationState, float],
        ] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, conversation_id: str, kind: str) -> ConversationState | None:
        async with self._lock:
            key = (conversation_id, kind)
            cached = self._values.get(key)
            if cached is None:
                return None
            value, deadline = cached
            if deadline <= time.monotonic() or (
                value.expires_at is not None and value.expires_at <= utc_now()
            ):
                self._values.pop(key, None)
                return None
            self._values.move_to_end(key)
            return value.model_copy(deep=True)

    async def set(self, state: ConversationState) -> None:
        async with self._lock:
            key = (state.conversation_id, state.kind)
            self._values[key] = (
                state.model_copy(deep=True),
                time.monotonic() + self._ttl_seconds,
            )
            self._values.move_to_end(key)
            while len(self._values) > self._max_entries:
                self._values.popitem(last=False)

    async def delete(self, conversation_id: str, kind: str) -> None:
        async with self._lock:
            self._values.pop((conversation_id, kind), None)
