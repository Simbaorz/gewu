"""SQLAlchemy implementation of the Runtime fact store."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, overload

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from gewu_agent_runtime.adapters.mysql.models import (
    AgentRunRow,
    AttachmentRow,
    ConversationCompactionRow,
    ConversationMessageRow,
    ConversationRow,
    ConversationStateRow,
)
from gewu_agent_runtime.adapters.mysql.transactions import committed_session
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
from gewu_agent_runtime.identity import PrincipalRef, PrincipalType
from gewu_agent_runtime.llm import MessageRole
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


def _is_message_id_conflict(error: IntegrityError) -> bool:
    """Return whether an integrity failure is the message ID primary-key conflict."""

    original = error.orig
    if str(original).strip() == "UNIQUE constraint failed: agent_conversation_message.id":
        return True
    arguments = getattr(original, "args", None)
    if not isinstance(arguments, tuple) or len(arguments) < 2 or arguments[0] != 1062:
        return False
    detail = str(arguments[1])
    return any(
        marker in detail
        for marker in (
            "for key 'PRIMARY'",
            "for key `PRIMARY`",
            "for key 'agent_conversation_message.PRIMARY'",
            "for key `agent_conversation_message.PRIMARY`",
        )
    )


class SqlAlchemyRuntimeStore:
    """Transactional Runtime store suitable for MySQL async engines."""

    def __init__(
        self,
        sessions: async_sessionmaker[AsyncSession],
        *,
        protected_payload_cipher: ProtectedPayloadCipher | None = None,
        encrypt_protected_payloads: bool = False,
        encrypt_compactions: bool = False,
    ) -> None:
        if (encrypt_protected_payloads or encrypt_compactions) and protected_payload_cipher is None:
            raise ValueError("Enabled Runtime encryption requires a protected payload cipher.")
        self._sessions = sessions
        self._protected_payload_cipher = protected_payload_cipher
        self._encrypt_protected_payloads = encrypt_protected_payloads
        self._encrypt_compactions = encrypt_compactions

    async def create_conversation(self, conversation: Conversation) -> Conversation:
        async with self._sessions() as session:
            try:
                async with session.begin():
                    session.add(_conversation_row(conversation))
            except IntegrityError as exc:
                raise ConcurrentWriteError("Conversation already exists.") from exc
        return conversation

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        async with self._sessions() as session:
            row = await session.get(ConversationRow, conversation_id)
            return _conversation(row) if row is not None else None

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
        filters = [
            ConversationRow.subscriber_id == owner.subscriber_id,
            ConversationRow.owner_principal_id == owner.principal_id,
            ConversationRow.owner_principal_type == owner.principal_type.value,
        ]
        if not include_archived:
            filters.append(ConversationRow.status == ConversationStatus.ACTIVE.value)
        if updated_after is not None:
            filters.append(ConversationRow.updated_at >= updated_after)
        async with self._sessions() as session:
            total = int(
                await session.scalar(select(func.count(ConversationRow.id)).where(*filters)) or 0
            )
            statement = select(ConversationRow).where(*filters)
            if before_updated_at is not None:
                statement = statement.where(
                    or_(
                        ConversationRow.updated_at < before_updated_at,
                        and_(
                            ConversationRow.updated_at == before_updated_at,
                            ConversationRow.id < before_conversation_id,
                        ),
                    )
                )
            rows = (
                await session.scalars(
                    statement.order_by(
                        ConversationRow.updated_at.desc(),
                        ConversationRow.id.desc(),
                    ).limit(limit)
                )
            ).all()
            return ConversationPage(
                items=tuple(_conversation(row) for row in rows),
                total=total,
            )

    async def update_conversation(
        self,
        conversation_id: str,
        *,
        title: str | None = None,
        status: ConversationStatus | None = None,
    ) -> Conversation | None:
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .with_for_update()
            )
            if row is None:
                return None
            if title is not None:
                row.title = title
            if status is not None:
                row.status = status.value
            row.updated_at = utc_now()
            await session.flush()
            return _conversation(row)

    async def compare_and_patch_conversation_metadata(
        self,
        conversation_id: str,
        *,
        expected_values: Mapping[str, Any],
        set_values: Mapping[str, Any],
        remove_keys: Sequence[str] = (),
    ) -> Conversation | None:
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .with_for_update()
            )
            if row is None:
                return None
            metadata = dict(row.extra)  # noqa
            if any(metadata.get(key) != expected for key, expected in expected_values.items()):
                return None
            metadata.update(dict(set_values))
            for key in remove_keys:
                metadata.pop(key, None)
            row.extra = metadata
            row.updated_at = utc_now()
            await session.flush()
            return _conversation(row)

    async def create_attachment(self, attachment: StoredAttachment) -> StoredAttachment:
        try:
            async with committed_session(
                self._sessions,
                operation="create Agent attachment",
            ) as session:
                conversation = await session.get(ConversationRow, attachment.conversation_id)
                if conversation is None:
                    raise EntityNotFoundError("Conversation does not exist.")
                if _conversation(conversation).owner != attachment.owner:
                    raise ValueError("Attachment owner must match its conversation owner.")
                session.add(_attachment_row(attachment))
        except IntegrityError as exc:
            raise ConcurrentWriteError("Attachment already exists.") from exc
        return attachment

    async def get_active_attachment(
        self,
        attachment_id: str,
        owner: PrincipalRef,
    ) -> StoredAttachment | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(AttachmentRow).where(
                    AttachmentRow.id == attachment_id,
                    AttachmentRow.subscriber_id == owner.subscriber_id,
                    AttachmentRow.owner_principal_id == owner.principal_id,
                    AttachmentRow.owner_principal_type == owner.principal_type.value,
                    AttachmentRow.status == AttachmentStatus.ACTIVE.value,
                )
            )
            return _attachment(row) if row is not None else None

    async def list_active_attachments(
        self,
        owner: PrincipalRef,
        *,
        conversation_id: str,
        request_id: str,
        attachment_ids: Sequence[str],
    ) -> tuple[StoredAttachment, ...]:
        unique_ids = tuple(dict.fromkeys(attachment_ids))
        if not unique_ids:
            return ()
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(AttachmentRow).where(
                        AttachmentRow.id.in_(unique_ids),
                        AttachmentRow.subscriber_id == owner.subscriber_id,
                        AttachmentRow.owner_principal_id == owner.principal_id,
                        AttachmentRow.owner_principal_type == owner.principal_type.value,
                        AttachmentRow.conversation_id == conversation_id,
                        AttachmentRow.request_id == request_id,
                        AttachmentRow.status == AttachmentStatus.ACTIVE.value,
                    )
                )
            ).all()
            by_id = {row.id: row for row in rows}
            return tuple(_attachment(by_id[value]) for value in unique_ids if value in by_id)

    async def list_attachment_cleanup_candidates(
        self,
        *,
        pending_before: datetime,
        storage_backend: str,
        limit: int,
    ) -> tuple[StoredAttachment, ...]:
        if limit < 1:
            raise ValueError("Attachment cleanup limit must be positive.")
        submitted_message_exists = (
            select(ConversationMessageRow.id)
            .where(
                ConversationMessageRow.conversation_id == AttachmentRow.conversation_id,
                ConversationMessageRow.request_id == AttachmentRow.request_id,
                ConversationMessageRow.message_kind == MessageKind.INPUT.value,
            )
            .correlate(AttachmentRow)
            .exists()
        )
        statement = (
            select(AttachmentRow)
            .join(ConversationRow, ConversationRow.id == AttachmentRow.conversation_id)
            .where(
                AttachmentRow.status == AttachmentStatus.ACTIVE.value,
                AttachmentRow.storage_backend == storage_backend,
                or_(
                    ConversationRow.status == ConversationStatus.ARCHIVED.value,
                    and_(
                        AttachmentRow.created_at < pending_before,
                        ~submitted_message_exists,
                    ),
                ),
            )
            .order_by(AttachmentRow.created_at, AttachmentRow.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
            return tuple(_attachment(row) for row in rows)

    async def mark_attachments_deleted(self, attachment_ids: Sequence[str]) -> None:
        unique_ids = tuple(dict.fromkeys(attachment_ids))
        if not unique_ids:
            return
        async with self._sessions() as session, session.begin():
            rows = (
                await session.scalars(
                    select(AttachmentRow)
                    .where(
                        AttachmentRow.id.in_(unique_ids),
                        AttachmentRow.status == AttachmentStatus.ACTIVE.value,
                    )
                    .with_for_update()
                )
            ).all()
            for row in rows:
                row.status = AttachmentStatus.DELETED.value

    async def append_messages(
        self,
        conversation_id: str,
        messages: Sequence[NewConversationMessage],
    ) -> tuple[ConversationMessage, ...]:
        if not messages:
            return ()
        async with self._sessions() as session:
            try:
                async with session.begin():
                    conversation = await session.scalar(
                        self._conversation_append_lock_stmt(conversation_id)
                    )
                    if conversation is None:
                        raise EntityNotFoundError("Conversation does not exist.")
                    start = conversation.next_sequence
                    persisted = tuple(
                        stored_message(
                            message,
                            conversation_id=conversation_id,
                            sequence=start + index,
                            cipher=self._protected_payload_cipher,
                            encrypt=self._encrypt_protected_payloads,
                        )
                        for index, message in enumerate(messages)
                    )
                    session.add_all(_message_row(message) for message in persisted)
                    conversation.next_sequence = start + len(persisted)
                    conversation.updated_at = utc_now()
            except IntegrityError as exc:
                if _is_message_id_conflict(exc):
                    raise MessageWriteConflictError("Message append conflicted.") from exc
                raise
        return tuple(self._hydrate_message(message) for message in persisted)

    @staticmethod
    def _conversation_append_lock_stmt(
        conversation_id: str,
    ) -> Select[tuple[ConversationRow]]:
        """Lock sequence allocation on the owning conversation row."""

        return (
            select(ConversationRow).where(ConversationRow.id == conversation_id).with_for_update()
        )

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
        async with self._sessions() as session:
            try:
                async with session.begin():
                    conversation_id = await session.scalar(
                        select(AgentRunRow.conversation_id).where(AgentRunRow.id == run_id)
                    )
                    if conversation_id is None:
                        return None
                    conversation = await session.scalar(
                        select(ConversationRow)
                        .where(ConversationRow.id == conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.status != AgentRunStatus.RUNNING.value
                        or conversation.active_run_id != run_id
                    ):
                        return None
                    start = conversation.next_sequence
                    persisted = tuple(
                        stored_message(
                            message,
                            conversation_id=conversation.id,
                            sequence=start + index,
                            cipher=self._protected_payload_cipher,
                            encrypt=self._encrypt_protected_payloads,
                        )
                        for index, message in enumerate(messages)
                    )
                    now = utc_now()
                    session.add_all(_message_row(message) for message in persisted)
                    conversation.next_sequence = start + len(persisted)
                    conversation.updated_at = now
                    if finish_status is not None:
                        conversation.active_run_id = None
                        run.status = finish_status.value
                        run.error_code = error_code
                        run.error_message = error_message
                        run.finished_at = now
                        run.updated_at = now
            except IntegrityError as exc:
                if _is_message_id_conflict(exc):
                    raise MessageWriteConflictError(
                        "Run-fenced message append conflicted."
                    ) from exc
                raise
        return tuple(self._hydrate_message(message) for message in persisted)

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        statement = select(ConversationMessageRow).where(
            ConversationMessageRow.conversation_id == conversation_id,
            ConversationMessageRow.sequence_number > after_sequence,
        )
        if through_sequence is not None:
            statement = statement.where(ConversationMessageRow.sequence_number <= through_sequence)
        statement = statement.order_by(ConversationMessageRow.sequence_number)
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
            return tuple(_message(row, self._protected_payload_cipher) for row in rows)

    async def list_messages_for_run(
        self,
        run_id: str,
    ) -> tuple[ConversationMessage, ...]:
        statement = (
            select(ConversationMessageRow)
            .where(ConversationMessageRow.run_id == run_id)
            .order_by(ConversationMessageRow.sequence_number)
        )
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
            return tuple(_message(row, self._protected_payload_cipher) for row in rows)

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
        statement = select(ConversationMessageRow).where(
            ConversationMessageRow.conversation_id == conversation_id,
            ConversationMessageRow.sequence_number > after_sequence,
        )
        if before_sequence is not None:
            statement = statement.where(ConversationMessageRow.sequence_number < before_sequence)
        statement = statement.order_by(ConversationMessageRow.sequence_number).limit(limit)
        async with self._sessions() as session:
            rows = (await session.scalars(statement)).all()
            return tuple(_message(row, self._protected_payload_cipher) for row in rows)

    async def list_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        if limit < 1:
            raise ValueError("Recent message limit must be positive.")
        statement = select(ConversationMessageRow).where(
            ConversationMessageRow.conversation_id == conversation_id
        )
        if before_sequence is not None:
            statement = statement.where(ConversationMessageRow.sequence_number < before_sequence)
        statement = statement.order_by(ConversationMessageRow.sequence_number.desc()).limit(limit)
        async with self._sessions() as session:
            rows = list((await session.scalars(statement)).all())
            rows.reverse()
            return tuple(_message(row, self._protected_payload_cipher) for row in rows)

    async def create_run(self, run: AgentRun) -> AgentRun:
        if run.idempotency_key is not None:
            existing = await self._get_idempotent_run(run)
            if existing is not None:
                return _validate_idempotent_run(existing, run)
        async with self._sessions() as session:
            try:
                async with session.begin():
                    session.add(_run_row(run))
            except IntegrityError as exc:
                if run.idempotency_key is not None:
                    existing = await self._get_idempotent_run(run)
                    if existing is not None:
                        return _validate_idempotent_run(existing, run)
                raise ConcurrentWriteError("Run already exists.") from exc
        return run

    async def get_run(self, run_id: str) -> AgentRun | None:
        async with self._sessions() as session:
            row = await session.get(AgentRunRow, run_id)
            return _run(row) if row is not None else None

    async def get_latest_run(self, conversation_id: str) -> AgentRun | None:
        async with self._sessions() as session:
            row = await session.scalar(
                select(AgentRunRow)
                .where(AgentRunRow.conversation_id == conversation_id)
                .order_by(AgentRunRow.created_at.desc(), AgentRunRow.id.desc())
                .limit(1)
            )
            return _run(row) if row is not None else None

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
        async with self._sessions() as session, session.begin():
            conversation_id = await session.scalar(
                select(AgentRunRow.conversation_id).where(AgentRunRow.id == run_id)
            )
            if conversation_id is None:
                raise EntityNotFoundError("Run does not exist.")
            conversation = await session.scalar(
                select(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .with_for_update()
            )
            row = await session.scalar(
                select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
            )
            if row is None or conversation is None:
                raise EntityNotFoundError("Run does not exist.")
            current_status = AgentRunStatus(row.status)
            if current_status not in expected:
                raise ConcurrentWriteError("Run status changed.")
            conversation_changed = False
            if status is AgentRunStatus.RUNNING:
                if conversation.active_run_id != expected_active_run_id:
                    raise ConcurrentWriteError("Run ownership changed before it was claimed.")
                conversation.active_run_id = run_id
                conversation_changed = True
            elif current_status is AgentRunStatus.RUNNING:
                if conversation.active_run_id != run_id:
                    raise ConcurrentWriteError("Run ownership was lost before transition.")
                conversation.active_run_id = None
                conversation_changed = True
            now = utc_now()
            row.status = status.value
            row.error_code = error_code
            row.error_message = error_message
            if row.started_at is None and status is AgentRunStatus.RUNNING:
                row.started_at = now
            row.finished_at = (
                now
                if status
                in {AgentRunStatus.COMPLETED, AgentRunStatus.FAILED, AgentRunStatus.CANCELLED}
                else None
            )
            row.updated_at = now
            if conversation_changed:
                conversation.updated_at = now
            await session.flush()
            return _run(row)

    async def get_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        include_expired: bool = False,
    ) -> ConversationState | None:
        async with self._sessions() as session:
            row = await session.get(ConversationStateRow, (conversation_id, kind))
            if row is None:
                return None
            state = _state(row)
            if (
                not include_expired
                and state.expires_at is not None
                and state.expires_at <= utc_now()
            ):
                return None
            return state

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
        filters = [
            ConversationStateRow.state_kind == kind,
            ConversationStateRow.expires_at.is_not(None),
            ConversationStateRow.expires_at <= expires_before,
        ]
        if after is not None:
            if after.expires_at is None:
                raise ValueError("Expired state cursor must have expires_at.")
            filters.append(
                or_(
                    ConversationStateRow.expires_at > after.expires_at,
                    and_(
                        ConversationStateRow.expires_at == after.expires_at,
                        ConversationStateRow.conversation_id > after.conversation_id,
                    ),
                )
            )
        async with self._sessions() as session:
            rows = (
                await session.scalars(
                    select(ConversationStateRow)
                    .where(*filters)
                    .order_by(
                        ConversationStateRow.expires_at.asc(),
                        ConversationStateRow.conversation_id.asc(),
                    )
                    .limit(limit)
                )
            ).all()
            return tuple(_state(row) for row in rows)

    async def save_state(
        self,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState:
        async with self._sessions() as session:
            try:
                async with session.begin():
                    row = await session.scalar(  # noqa
                        select(ConversationStateRow)
                        .where(
                            ConversationStateRow.conversation_id == state.conversation_id,
                            ConversationStateRow.state_kind == state.kind,
                        )
                        .with_for_update()
                    )
                    actual = row.revision if row is not None else 0
                    if actual != expected_revision or state.revision != expected_revision + 1:
                        raise ConcurrentWriteError("Conversation state revision changed.")
                    if row is None:
                        row = _state_row(state)
                        session.add(row)
                    else:
                        row.revision = state.revision
                        row.payload = state.payload
                        row.expires_at = state.expires_at
                        row.updated_at = state.updated_at
            except IntegrityError as exc:
                raise ConcurrentWriteError("Conversation state revision changed.") from exc
        return state

    async def save_state_for_run(
        self,
        run_id: str,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState | None:
        async with self._sessions() as session:
            try:
                async with session.begin():
                    conversation = await session.scalar(
                        select(ConversationRow)
                        .where(ConversationRow.id == state.conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.conversation_id != state.conversation_id
                        or run.status != AgentRunStatus.RUNNING.value
                        or conversation.active_run_id != run_id
                    ):
                        return None
                    row = await session.scalar(  # noqa
                        select(ConversationStateRow)
                        .where(
                            ConversationStateRow.conversation_id == state.conversation_id,
                            ConversationStateRow.state_kind == state.kind,
                        )
                        .with_for_update()
                    )
                    actual = row.revision if row is not None else 0
                    if actual != expected_revision or state.revision != expected_revision + 1:
                        raise ConcurrentWriteError("Conversation state revision changed.")
                    stored = state.model_copy(update={"updated_at": utc_now()})
                    if row is None:
                        session.add(_state_row(stored))
                    else:
                        row.revision = stored.revision
                        row.payload = stored.payload
                        row.expires_at = stored.expires_at
                        row.updated_at = stored.updated_at
                    return stored
            except IntegrityError as exc:
                raise ConcurrentWriteError("Conversation state revision changed.") from exc

    async def delete_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        async with self._sessions() as session, session.begin():
            row = await session.scalar(
                select(ConversationStateRow)
                .where(
                    ConversationStateRow.conversation_id == conversation_id,
                    ConversationStateRow.state_kind == kind,
                )
                .with_for_update()
            )
            if row is None:
                return
            if expected_revision is not None and row.revision != expected_revision:
                raise ConcurrentWriteError("Conversation state revision changed.")
            await session.delete(row)

    async def delete_state_for_run(
        self,
        run_id: str,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> bool:
        async with self._sessions() as session, session.begin():
            conversation = await session.scalar(
                select(ConversationRow)
                .where(ConversationRow.id == conversation_id)
                .with_for_update()
            )
            run = await session.scalar(
                select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
            )
            if (
                conversation is None
                or run is None
                or run.conversation_id != conversation_id
                or run.status != AgentRunStatus.RUNNING.value
                or conversation.active_run_id != run_id
            ):
                return False
            row = await session.scalar(
                select(ConversationStateRow)
                .where(
                    ConversationStateRow.conversation_id == conversation_id,
                    ConversationStateRow.state_kind == kind,
                )
                .with_for_update()
            )
            if row is None:
                return True
            if expected_revision is not None and row.revision != expected_revision:
                raise ConcurrentWriteError("Conversation state revision changed.")
            await session.delete(row)
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
        async with self._sessions() as session:
            try:
                async with session.begin():
                    run_snapshot = await session.get(AgentRunRow, run_id)
                    if (
                        run_snapshot is None
                        or run_snapshot.conversation_id != state.conversation_id
                    ):
                        return None
                    conversation = await session.scalar(
                        select(ConversationRow)
                        .where(ConversationRow.id == state.conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    current_state = await session.scalar(
                        select(ConversationStateRow)
                        .where(
                            ConversationStateRow.conversation_id == state.conversation_id,
                            ConversationStateRow.state_kind == state.kind,
                        )
                        .with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.status != AgentRunStatus.RUNNING.value
                        or conversation.active_run_id != run_id
                        or current_state is not None
                    ):
                        return None
                    now = utc_now()
                    persisted = stored_message(
                        message,
                        conversation_id=conversation.id,
                        sequence=conversation.next_sequence,
                        cipher=self._protected_payload_cipher,
                        encrypt=self._encrypt_protected_payloads,
                    )
                    session.add(_message_row(persisted))
                    session.add(_state_row(state.model_copy(update={"updated_at": now})))
                    conversation.next_sequence += 1
                    conversation.active_run_id = None
                    conversation.updated_at = now
                    run.status = AgentRunStatus.WAITING_INPUT.value
                    run.updated_at = now
                    await session.flush()
                    return _run(run)
            except IntegrityError:
                return None

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
        async with self._sessions() as session:
            try:
                async with session.begin():
                    run_snapshot = await session.get(AgentRunRow, run_id)
                    if run_snapshot is None:
                        return None
                    conversation = await session.scalar(  # noqa
                        select(ConversationRow)
                        .where(ConversationRow.id == run_snapshot.conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    state = await session.scalar(
                        select(ConversationStateRow)
                        .where(
                            ConversationStateRow.conversation_id == run_snapshot.conversation_id,
                            ConversationStateRow.state_kind == "pending_ask",
                        )
                        .with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.status != AgentRunStatus.WAITING_INPUT.value
                        or state is None
                        or state.revision != state_revision
                        or state.payload.get("run_id") != run_id
                    ):
                        return None
                    persisted = stored_message(
                        message,
                        conversation_id=conversation.id,
                        sequence=conversation.next_sequence,
                        cipher=self._protected_payload_cipher,
                        encrypt=self._encrypt_protected_payloads,
                    )
                    session.add(_message_row(persisted))
                    await session.delete(state)
                    now = utc_now()
                    conversation.next_sequence += 1
                    conversation.updated_at = now
                    run.status = AgentRunStatus.CANCELLED.value
                    run.error_code = error_code
                    run.error_message = error_message
                    run.finished_at = now
                    run.updated_at = now
                    await session.flush()
                    return _run(run)
            except IntegrityError as exc:
                raise ConcurrentWriteError("Pending Ask resolution conflicted.") from exc

    async def commit_ask_answer(
        self,
        run_id: str,
        *,
        state_revision: int,
        message: NewConversationMessage,
    ) -> bool:
        if message.run_id != run_id:
            raise ValueError("Ask answer message must reference its active run.")
        async with self._sessions() as session:
            try:
                async with session.begin():
                    run_snapshot = await session.get(AgentRunRow, run_id)
                    if run_snapshot is None:
                        return False
                    conversation = await session.scalar(  # noqa
                        select(ConversationRow)
                        .where(ConversationRow.id == run_snapshot.conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    state = await session.scalar(
                        select(ConversationStateRow)
                        .where(
                            ConversationStateRow.conversation_id == run_snapshot.conversation_id,
                            ConversationStateRow.state_kind == "pending_ask",
                        )
                        .with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.status != AgentRunStatus.RUNNING.value
                        or conversation.active_run_id != run_id
                        or state is None
                        or state.revision != state_revision
                        or state.payload.get("run_id") != run_id
                    ):
                        return False
                    persisted = stored_message(
                        message,
                        conversation_id=conversation.id,
                        sequence=conversation.next_sequence,
                        cipher=self._protected_payload_cipher,
                        encrypt=self._encrypt_protected_payloads,
                    )
                    session.add(_message_row(persisted))
                    await session.delete(state)
                    conversation.next_sequence += 1
                    conversation.updated_at = utc_now()
                    await session.flush()
                    return True
            except IntegrityError as exc:
                raise ConcurrentWriteError("Ask answer commit conflicted.") from exc

    async def get_latest_compaction(
        self,
        conversation_id: str,
    ) -> ConversationCompaction | None:
        async with self._sessions() as session:
            conversation = await session.get(ConversationRow, conversation_id)
            if conversation is None or not conversation.latest_compaction_id:
                return None
            row = await session.get(
                ConversationCompactionRow,
                conversation.latest_compaction_id,
            )
            return (
                hydrated_compaction(_compaction(row), self._protected_payload_cipher)
                if row is not None
                else None
            )

    async def get_compaction(
        self,
        conversation_id: str,
        compaction_id: str,
    ) -> ConversationCompaction | None:
        async with self._sessions() as session:
            row = await session.get(ConversationCompactionRow, compaction_id)
            if row is None or row.conversation_id != conversation_id:
                return None
            return hydrated_compaction(_compaction(row), self._protected_payload_cipher)

    async def save_compaction(
        self,
        compaction: ConversationCompaction,
        *,
        expected_previous_id: str,
    ) -> ConversationCompaction:
        async with self._sessions() as session:
            try:
                async with session.begin():
                    conversation = await session.scalar(
                        select(ConversationRow)
                        .where(ConversationRow.id == compaction.conversation_id)
                        .with_for_update()
                    )
                    if conversation is None:
                        raise EntityNotFoundError("Conversation does not exist.")
                    if (
                        conversation.latest_compaction_id != expected_previous_id
                        or compaction.previous_compaction_id != expected_previous_id
                    ):
                        raise ConcurrentWriteError("Conversation compaction boundary changed.")
                    session.add(
                        _compaction_row(
                            stored_compaction(
                                compaction,
                                self._protected_payload_cipher,
                                encrypt=self._encrypt_compactions,
                            )
                        )
                    )
                    conversation.latest_compaction_id = compaction.compaction_id
                    conversation.updated_at = utc_now()
            except IntegrityError as exc:
                raise ConcurrentWriteError("Conversation compaction conflicted.") from exc
        return compaction

    async def commit_compaction_for_run(
        self,
        commit: ConversationCompactionCommit,
        run_id: str,
    ) -> bool:
        compaction = commit.compaction
        changed_kinds = set(commit.state_payloads) | set(commit.delete_state_kinds)
        if len(changed_kinds) != len(commit.state_payloads) + len(commit.delete_state_kinds):
            raise ValueError("Compaction cannot save and delete the same state kind.")
        async with self._sessions() as session:
            try:
                async with session.begin():
                    conversation = await session.scalar(
                        select(ConversationRow)
                        .where(ConversationRow.id == compaction.conversation_id)
                        .with_for_update()
                    )
                    run = await session.scalar(
                        select(AgentRunRow).where(AgentRunRow.id == run_id).with_for_update()
                    )
                    if (
                        conversation is None
                        or run is None
                        or run.conversation_id != compaction.conversation_id
                        or run.status != AgentRunStatus.RUNNING.value
                        or conversation.active_run_id != run_id
                        or conversation.latest_compaction_id != commit.expected_previous_id
                    ):
                        return False

                    start = conversation.next_sequence
                    persisted = tuple(
                        stored_message(
                            message,
                            conversation_id=compaction.conversation_id,
                            sequence=start + index,
                            cipher=self._protected_payload_cipher,
                            encrypt=self._encrypt_protected_payloads,
                        )
                        for index, message in enumerate(commit.messages)
                    )
                    state_rows: dict[str, ConversationStateRow | None] = {}
                    for kind in sorted(changed_kinds):
                        state_rows[kind] = await session.scalar(
                            select(ConversationStateRow)
                            .where(
                                ConversationStateRow.conversation_id == compaction.conversation_id,
                                ConversationStateRow.state_kind == kind,
                            )
                            .with_for_update()
                        )

                    session.add(
                        _compaction_row(
                            stored_compaction(
                                compaction,
                                self._protected_payload_cipher,
                                encrypt=self._encrypt_compactions,
                            )
                        )
                    )
                    session.add_all(_message_row(message) for message in persisted)
                    for kind, payload in commit.state_payloads.items():
                        row = state_rows[kind]
                        if row is None:
                            session.add(
                                _state_row(
                                    ConversationState(
                                        conversation_id=compaction.conversation_id,
                                        kind=kind,
                                        revision=1,
                                        payload=payload,
                                    )
                                )
                            )
                        else:
                            row.revision += 1
                            row.payload = payload
                            row.expires_at = None
                            row.updated_at = utc_now()
                    for kind in commit.delete_state_kinds:
                        row = state_rows[kind]
                        if row is not None:
                            await session.delete(row)
                    conversation.next_sequence = start + len(persisted)
                    conversation.latest_compaction_id = compaction.compaction_id
                    conversation.updated_at = utc_now()
            except IntegrityError:
                return False
        return True

    def _hydrate_message(self, message: ConversationMessage) -> ConversationMessage:
        return hydrated_message(message, self._protected_payload_cipher)

    async def _get_idempotent_run(self, run: AgentRun) -> AgentRun | None:
        assert run.idempotency_key is not None
        async with self._sessions() as session:
            row = await session.scalar(
                select(AgentRunRow).where(
                    AgentRunRow.subscriber_id == run.invoker.subscriber_id,
                    AgentRunRow.conversation_id == run.conversation_id,
                    AgentRunRow.idempotency_key == run.idempotency_key,
                )
            )
            return _run(row) if row is not None else None


def _validate_idempotent_run(existing: AgentRun, proposed: AgentRun) -> AgentRun:
    if (
        existing.request_id != proposed.request_id
        or existing.input_snapshot != proposed.input_snapshot
    ):
        raise IdempotencyConflictError("Idempotency key belongs to a different request.")
    return existing


def _conversation_row(value: Conversation) -> ConversationRow:
    return ConversationRow(
        id=value.conversation_id,
        subscriber_id=value.owner.subscriber_id,
        owner_principal_id=value.owner.principal_id,
        owner_principal_type=value.owner.principal_type.value,
        title=value.title,
        status=value.status.value,
        extra=value.metadata,
        next_sequence=value.next_sequence,
        latest_compaction_id=value.latest_compaction_id,
        active_run_id=value.active_run_id,
        created_at=_utc(value.created_at),
        updated_at=_utc(value.updated_at),
    )


def _conversation(value: ConversationRow) -> Conversation:
    return Conversation(
        conversation_id=value.id,
        owner=PrincipalRef(
            subscriber_id=value.subscriber_id,
            principal_id=value.owner_principal_id,
            principal_type=PrincipalType(value.owner_principal_type),
        ),
        title=value.title,
        status=ConversationStatus(value.status),
        metadata=value.extra,
        next_sequence=value.next_sequence,
        latest_compaction_id=value.latest_compaction_id,
        active_run_id=value.active_run_id,
        created_at=_utc(value.created_at),  # noqa
        updated_at=_utc(value.updated_at),  # noqa
    )


def _message_row(value: ConversationMessage) -> ConversationMessageRow:
    return ConversationMessageRow(
        id=value.message_id,
        conversation_id=value.conversation_id,
        sequence_number=value.sequence,
        role=value.role.value,
        message_kind=value.kind.value,
        content=value.content,
        payload=value.payload,
        body_encryption_version=value.body_encryption_version,
        run_id=value.run_id,
        request_id=value.request_id,
        created_at=_utc(value.created_at),
    )


def _attachment_row(value: StoredAttachment) -> AttachmentRow:
    return AttachmentRow(
        id=value.attachment_id,
        subscriber_id=value.owner.subscriber_id,
        owner_principal_id=value.owner.principal_id,
        owner_principal_type=value.owner.principal_type.value,
        conversation_id=value.conversation_id,
        request_id=value.request_id,
        storage_backend=value.storage_backend,
        resource_key=value.resource_key,
        original_name=value.original_name,
        mime_type=value.mime_type,
        size_bytes=value.size_bytes,
        status=value.status.value,
        created_at=_utc(value.created_at),
    )


def _attachment(value: AttachmentRow) -> StoredAttachment:
    return StoredAttachment(
        attachment_id=value.id,
        owner=PrincipalRef(
            subscriber_id=value.subscriber_id,
            principal_id=value.owner_principal_id,
            principal_type=PrincipalType(value.owner_principal_type),
        ),
        conversation_id=value.conversation_id,
        request_id=value.request_id,
        storage_backend=value.storage_backend,
        resource_key=value.resource_key,
        original_name=value.original_name,
        mime_type=value.mime_type,
        size_bytes=value.size_bytes,
        status=AttachmentStatus(value.status),
        created_at=_utc(value.created_at),  # noqa
    )


def _message(
    value: ConversationMessageRow,
    cipher: ProtectedPayloadCipher | None,
) -> ConversationMessage:
    stored = ConversationMessage(
        message_id=value.id,
        conversation_id=value.conversation_id,
        sequence=value.sequence_number,
        role=MessageRole(value.role),
        kind=MessageKind(value.message_kind),
        content=value.content,
        payload=value.payload,
        body_encryption_version=value.body_encryption_version,
        run_id=value.run_id,
        request_id=value.request_id,
        created_at=value.created_at,
    )
    return hydrated_message(stored, cipher)


def _run_row(value: AgentRun) -> AgentRunRow:
    return AgentRunRow(
        id=value.run_id,
        conversation_id=value.conversation_id,
        subscriber_id=value.invoker.subscriber_id,
        invoker_principal_id=value.invoker.principal_id,
        invoker_principal_type=value.invoker.principal_type.value,
        status=value.status.value,
        request_id=value.request_id,
        idempotency_key=value.idempotency_key,
        input_snapshot=value.input_snapshot,
        model_snapshot=value.model_snapshot,
        error_code=value.error_code,
        error_message=value.error_message,
        started_at=_utc(value.started_at),  # noqa
        finished_at=_utc(value.finished_at),  # noqa
        created_at=_utc(value.created_at),
        updated_at=_utc(value.updated_at),
    )


def _run(value: AgentRunRow) -> AgentRun:
    return AgentRun(
        run_id=value.id,
        conversation_id=value.conversation_id,
        invoker=PrincipalRef(
            subscriber_id=value.subscriber_id,
            principal_id=value.invoker_principal_id,
            principal_type=PrincipalType(value.invoker_principal_type),
        ),
        status=AgentRunStatus(value.status),
        request_id=value.request_id,
        idempotency_key=value.idempotency_key,
        input_snapshot=value.input_snapshot,
        model_snapshot=value.model_snapshot,
        error_code=value.error_code,
        error_message=value.error_message,
        started_at=_utc(value.started_at),
        finished_at=_utc(value.finished_at),
        created_at=_utc(value.created_at),  # noqa
        updated_at=_utc(value.updated_at),  # noqa
    )


def _state_row(value: ConversationState) -> ConversationStateRow:
    return ConversationStateRow(
        conversation_id=value.conversation_id,
        state_kind=value.kind,
        revision=value.revision,
        payload=value.payload,
        expires_at=_utc(value.expires_at),  # noqa
        updated_at=_utc(value.updated_at),  # noqa
    )


def _state(value: ConversationStateRow) -> ConversationState:
    return ConversationState(
        conversation_id=value.conversation_id,
        kind=value.state_kind,
        revision=value.revision,
        payload=value.payload,
        expires_at=_utc(value.expires_at),
        updated_at=_utc(value.updated_at),  # noqa
    )


def _compaction_row(value: ConversationCompaction) -> ConversationCompactionRow:
    return ConversationCompactionRow(
        id=value.compaction_id,
        conversation_id=value.conversation_id,
        generation=value.generation,
        previous_compaction_id=value.previous_compaction_id,
        source_from_sequence=value.source_from_sequence,
        through_sequence=value.through_sequence,
        summary=value.summary,
        summary_encryption_version=value.summary_encryption_version,
        model_ref=value.model_ref,
        model_name=value.model_name,
        context_window=value.context_window,
        pre_compaction_tokens=value.pre_compaction_tokens,
        post_compaction_tokens=value.post_compaction_tokens,
        created_at=_utc(value.created_at),
    )


@overload
def _utc(value: datetime) -> datetime: ...


@overload
def _utc(value: None) -> None: ...


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _compaction(value: ConversationCompactionRow) -> ConversationCompaction:
    return ConversationCompaction(
        compaction_id=value.id,
        conversation_id=value.conversation_id,
        generation=value.generation,
        previous_compaction_id=value.previous_compaction_id,
        source_from_sequence=value.source_from_sequence,
        through_sequence=value.through_sequence,
        summary=value.summary,
        summary_encryption_version=value.summary_encryption_version,
        model_ref=value.model_ref,
        model_name=value.model_name,
        context_window=value.context_window,
        pre_compaction_tokens=value.pre_compaction_tokens,
        post_compaction_tokens=value.post_compaction_tokens,
        created_at=_utc(value.created_at),  # noqa
    )
