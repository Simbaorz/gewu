"""Persistent entities owned by the Agent Runtime."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import MessageRole
from gewu_core.ids import new_id
from gewu_core.time import utc_now


class ConversationStatus(StrEnum):
    """Lifecycle of one runtime conversation."""

    ACTIVE = "active"
    ARCHIVED = "archived"


class MessageKind(StrEnum):
    """Semantic kinds in the append-only conversation log."""

    INPUT = "input"
    META = "meta"
    SYSTEM = "system"
    ASSISTANT = "assistant"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    ASK = "ask"
    ERROR = "error"
    MEMORY_COMPACTION = "memory_compaction"


class AgentRunStatus(StrEnum):
    """Durable lifecycle of one turn invocation."""

    PENDING = "pending"
    RUNNING = "running"
    WAITING_INPUT = "waiting_input"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttachmentStatus(StrEnum):
    """Lifecycle of one Runtime-owned attachment reference."""

    ACTIVE = "active"
    DELETED = "deleted"


class StoredAttachment(BaseModel):
    """Persistent attachment metadata without storage-provider semantics."""

    model_config = ConfigDict(frozen=True)

    attachment_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    owner: PrincipalRef
    conversation_id: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=128)
    storage_backend: str = Field(min_length=1, max_length=32)
    resource_key: str = Field(min_length=1, max_length=512)
    original_name: str = Field(default="", max_length=255)
    mime_type: str = Field(min_length=1, max_length=64)
    size_bytes: int = Field(ge=0)
    status: AttachmentStatus = AttachmentStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)


class Conversation(BaseModel):
    """Conversation metadata scoped to one subscriber and owning principal."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    owner: PrincipalRef
    title: str = Field(default="", max_length=256)
    status: ConversationStatus = ConversationStatus.ACTIVE
    metadata: dict[str, Any] = Field(default_factory=dict)
    next_sequence: int = Field(default=1, ge=1)
    latest_compaction_id: str = Field(default="", max_length=64)
    active_run_id: str | None = Field(default=None, min_length=1, max_length=64)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationPage(BaseModel):
    """One owner-scoped page of Runtime conversations."""

    model_config = ConfigDict(frozen=True)

    items: tuple[Conversation, ...] = ()
    total: int = Field(default=0, ge=0)


class ProtectedMessageBody(BaseModel):
    """Transient cleartext body encrypted by a capable Runtime store."""

    model_config = ConfigDict(frozen=True)

    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)


class NewConversationMessage(BaseModel):
    """Message content before its sequence is allocated by the store."""

    model_config = ConfigDict(frozen=True)

    message_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    role: MessageRole
    kind: MessageKind
    content: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    protected_body: ProtectedMessageBody | None = Field(
        default=None,
        exclude=True,
        repr=False,
    )
    run_id: str = Field(default="", max_length=64)
    request_id: str = Field(default="", max_length=128)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationMessage(NewConversationMessage):
    """One append-only persisted message with a monotonic sequence."""

    conversation_id: str = Field(min_length=1, max_length=64)
    sequence: int = Field(ge=1)
    body_encryption_version: int = Field(default=0, ge=0, exclude=True, repr=False)


class ConversationState(BaseModel):
    """Versioned mutable runtime state for one conversation."""

    model_config = ConfigDict(frozen=True)

    conversation_id: str = Field(min_length=1, max_length=64)
    kind: str = Field(min_length=1, max_length=64)
    revision: int = Field(ge=1)
    payload: dict[str, Any] = Field(default_factory=dict)
    expires_at: datetime | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class ConversationCompaction(BaseModel):
    """Immutable cumulative summary of one conversation prefix."""

    model_config = ConfigDict(frozen=True)

    compaction_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    generation: int = Field(ge=1)
    previous_compaction_id: str = Field(default="", max_length=64)
    source_from_sequence: int | None = Field(default=None, ge=1)
    through_sequence: int = Field(ge=1)
    summary: str = Field(min_length=1)
    model_ref: str = Field(default="", max_length=256)
    model_name: str = Field(default="", max_length=128)
    context_window: int = Field(default=1, ge=1)
    pre_compaction_tokens: int = Field(default=0, ge=0)
    post_compaction_tokens: int = Field(default=0, ge=0)
    summary_encryption_version: int = Field(default=0, ge=0, exclude=True, repr=False)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationCompactionCommit(BaseModel):
    """Atomic summary, lifecycle-message and Runtime-state transition."""

    model_config = ConfigDict(frozen=True)

    compaction: ConversationCompaction
    expected_previous_id: str = Field(default="", max_length=64)
    messages: tuple[NewConversationMessage, ...] = ()
    state_payloads: dict[str, dict[str, Any]] = Field(default_factory=dict)
    delete_state_kinds: tuple[str, ...] = ()


class AgentRun(BaseModel):
    """Durable record for one caller invocation or suspended continuation."""

    model_config = ConfigDict(frozen=True)

    run_id: str = Field(default_factory=new_id, min_length=1, max_length=64)
    conversation_id: str = Field(min_length=1, max_length=64)
    invoker: PrincipalRef
    status: AgentRunStatus = AgentRunStatus.PENDING
    request_id: str = Field(default_factory=new_id, min_length=1, max_length=128)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    input_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_snapshot: dict[str, Any] = Field(default_factory=dict)
    error_code: str = Field(default="", max_length=64)
    error_message: str = ""
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
