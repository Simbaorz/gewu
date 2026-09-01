"""SQLAlchemy tables exclusively owned by the Agent Runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class AgentRuntimeBase(DeclarativeBase):
    """Declarative base that hosts can create or add to migrations."""


LongText = Text().with_variant(LONGTEXT(), "mysql")


class ConversationRow(AgentRuntimeBase):
    __tablename__ = "agent_conversation"
    __table_args__ = (
        Index(
            "idx_agent_conversation_owner",
            "subscriber_id",
            "owner_principal_type",
            "owner_principal_id",
            "status",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subscriber_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_principal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    title: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    extra: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    next_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1)
    latest_compaction_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    active_run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ConversationMessageRow(AgentRuntimeBase):
    __tablename__ = "agent_conversation_message"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "sequence_number",
            name="uk_agent_message_sequence",
        ),
        Index("idx_agent_message_run", "run_id"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sequence_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    message_kind: Mapped[str] = mapped_column(String(32), nullable=False)
    content: Mapped[str] = mapped_column(LongText, nullable=False, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    body_encryption_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    run_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    request_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AttachmentRow(AgentRuntimeBase):
    __tablename__ = "agent_attachment"
    __table_args__ = (
        Index("idx_agent_attachment_cleanup", "status", "created_at"),
        Index(
            "idx_agent_attachment_request",
            "subscriber_id",
            "owner_principal_type",
            "owner_principal_id",
            "conversation_id",
            "request_id",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    subscriber_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_principal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation.id", ondelete="CASCADE"),
        nullable=False,
    )
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    storage_backend: Mapped[str] = mapped_column(String(32), nullable=False)
    resource_key: Mapped[str] = mapped_column(String(512), nullable=False)
    original_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    mime_type: Mapped[str] = mapped_column(String(64), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ConversationStateRow(AgentRuntimeBase):
    __tablename__ = "agent_conversation_state"
    __table_args__ = (Index("idx_agent_state_expiry", "state_kind", "expires_at"),)

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation.id", ondelete="CASCADE"),
        primary_key=True,
    )
    state_kind: Mapped[str] = mapped_column(String(64), primary_key=True)
    revision: Mapped[int] = mapped_column(BigInteger, nullable=False)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ConversationCompactionRow(AgentRuntimeBase):
    __tablename__ = "agent_conversation_compaction"
    __table_args__ = (
        UniqueConstraint(
            "conversation_id",
            "generation",
            name="uk_agent_compaction_generation",
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    previous_compaction_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    source_from_sequence: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    through_sequence: Mapped[int] = mapped_column(BigInteger, nullable=False)
    summary: Mapped[str] = mapped_column(LongText, nullable=False)
    summary_encryption_version: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    model_ref: Mapped[str] = mapped_column(String(256), nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    context_window: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    pre_compaction_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    post_compaction_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AgentRunRow(AgentRuntimeBase):
    __tablename__ = "agent_run"
    __table_args__ = (
        UniqueConstraint(
            "subscriber_id",
            "conversation_id",
            "idempotency_key",
            name="uk_agent_run_idempotency",
        ),
        Index("idx_agent_run_conversation", "conversation_id", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversation.id", ondelete="CASCADE"),
        nullable=False,
    )
    subscriber_id: Mapped[str] = mapped_column(String(128), nullable=False)
    invoker_principal_id: Mapped[str] = mapped_column(String(128), nullable=False)
    invoker_principal_type: Mapped[str] = mapped_column(String(16), nullable=False)
    status: Mapped[str] = mapped_column(String(24), nullable=False)
    request_id: Mapped[str] = mapped_column(String(128), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    input_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    model_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error_code: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    error_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
