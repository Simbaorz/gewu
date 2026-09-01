"""Fact-store contract used by the in-process Runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Protocol

from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationCompaction,
    ConversationCompactionCommit,
    ConversationMessage,
    ConversationPage,
    ConversationState,
    ConversationStatus,
    NewConversationMessage,
    StoredAttachment,
)
from gewu_agent_runtime.identity import PrincipalRef


class RuntimeStore(Protocol):
    """Durable store for all state owned by the Agent Runtime."""

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        """Persist a new conversation."""

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Load one conversation."""

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
        """Load one owner-scoped page ordered by recency and stable ID."""

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: ConversationStatus | None = None,
    ) -> Conversation | None:
        """Update mutable conversation fields when the row exists."""

    async def compare_and_patch_conversation_metadata(
        self,
        conversation_id: str,
        *,
        expected_values: Mapping[str, Any],
        set_values: Mapping[str, Any],
        remove_keys: Sequence[str] = (),
    ) -> Conversation | None:
        """Patch metadata only while all expected top-level values still match."""

    async def create_attachment(self, attachment: StoredAttachment) -> StoredAttachment:
        """Persist one attachment reference owned by a conversation principal."""

    async def get_active_attachment(
        self,
        attachment_id: str,
        owner: PrincipalRef,
    ) -> StoredAttachment | None:
        """Load one active attachment reference."""

    async def list_active_attachments(
        self,
        owner: PrincipalRef,
        *,
        conversation_id: str,
        request_id: str,
        attachment_ids: Sequence[str],
    ) -> tuple[StoredAttachment, ...]:
        """Load exact active request attachments in caller order."""

    async def list_attachment_cleanup_candidates(
        self,
        *,
        pending_before: datetime,
        storage_backend: str,
        limit: int,
    ) -> tuple[StoredAttachment, ...]:
        """Load active orphaned or archived-conversation attachment references."""

    async def mark_attachments_deleted(self, attachment_ids: Sequence[str]) -> None:
        """Mark active attachment references deleted."""

    async def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[NewConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        """Atomically allocate sequences and append messages."""

    async def append_messages_for_run(
        self,
        run_id: str,
        messages: Sequence[NewConversationMessage],
        *,
        finish_status: AgentRunStatus | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> tuple[ConversationMessage, ...] | None:
        """Append and optionally finish only while ``run_id`` owns its conversation."""

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        """Load an ordered inclusive range after a sequence."""

    async def list_messages_for_run(
        self,
        run_id: str,
    ) -> tuple[ConversationMessage, ...]:
        """Load all messages persisted for one run in conversation order."""

    async def list_message_page(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> tuple[ConversationMessage, ...]:
        """Load one ascending bounded page between exclusive sequence bounds."""

    async def list_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        """Load the newest bounded suffix in ascending order."""

    async def create_run(self, run: AgentRun) -> AgentRun:
        """Persist a run or return its exact idempotent predecessor."""

    async def get_run(self, run_id: str) -> AgentRun | None:
        """Load one run."""

    async def get_latest_run(self, conversation_id: str) -> AgentRun | None:
        """Load the newest run for one conversation."""

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
        """Compare-and-set a run lifecycle transition and its ownership marker."""

    async def get_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        include_expired: bool = False,
    ) -> ConversationState | None:
        """Load one state value, optionally retaining expired fact-store evidence."""

    async def list_expired_states(
        self,
        kind: str,
        expires_before: datetime,
        *,
        after: ConversationState | None = None,
        limit: int = 100,
    ) -> tuple[ConversationState, ...]:
        """Return one keyset page of expired state facts."""

    async def save_state(
        self,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState:
        """Create or compare-and-set one state value."""

    async def save_state_for_run(
        self,
        run_id: str,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState | None:
        """Save one state value only while ``run_id`` owns its conversation."""

    async def delete_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        """Delete one state value, optionally with a revision precondition."""

    async def delete_state_for_run(
        self,
        run_id: str,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        """Delete one state value only while ``run_id`` owns its conversation."""

    async def commit_pending_ask(
        self,
        run_id: str,
        *,
        state: ConversationState,
        message: NewConversationMessage,
    ) -> AgentRun | None:
        """Atomically append one Ask, save its state, and suspend the active Run."""

    async def commit_ask_answer(
        self,
        run_id: str,
        *,
        state_revision: int,
        message: NewConversationMessage,
    ) -> bool:
        """Atomically append one valid Ask answer and consume its exact pending state."""

    async def resolve_pending_ask(
        self,
        run_id: str,
        *,
        state_revision: int,
        message: NewConversationMessage,
        error_code: str,
        error_message: str,
    ) -> AgentRun | None:
        """Atomically record a synthetic result and cancel one exact waiting Ask."""

    async def get_latest_compaction(
        self,
        conversation_id: str,
    ) -> ConversationCompaction | None:
        """Load the conversation's committed cumulative summary."""

    async def get_compaction(
        self,
        conversation_id: str,
        compaction_id: str,
    ) -> ConversationCompaction | None:
        """Load one exact compaction owned by a conversation."""

    async def save_compaction(
        self,
        compaction: ConversationCompaction,
        *,
        expected_previous_id: str,
    ) -> ConversationCompaction:
        """Atomically commit a summary and advance its conversation pointer."""

    async def commit_compaction_for_run(
        self,
        commit: ConversationCompactionCommit,
        run_id: str,
    ) -> bool:
        """Commit all compaction effects only while the Run and boundary still match."""


class StateCache(Protocol):
    """Disposable cache for versioned conversation state."""

    async def get(self, conversation_id: str, kind: str) -> ConversationState | None:
        """Load a cached state value."""

    async def set(self, state: ConversationState) -> None:
        """Cache one state value."""

    async def delete(self, conversation_id: str, kind: str) -> None:
        """Invalidate one state value."""
