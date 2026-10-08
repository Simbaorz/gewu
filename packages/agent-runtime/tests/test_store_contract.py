"""Runtime store contract tests for memory and SQLAlchemy implementations."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
from pymysql.err import IntegrityError as PyMySQLIntegrityError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError as SqlAlchemyIntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from gewu_agent_runtime.adapters.mysql import SqlAlchemyRuntimeStore, create_schema
from gewu_agent_runtime.adapters.mysql.models import (
    AttachmentRow,
    ConversationCompactionRow,
    ConversationMessageRow,
)
from gewu_agent_runtime.adapters.mysql.store import _is_message_id_conflict
from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationCompaction,
    ConversationCompactionCommit,
    ConversationState,
    ConversationStatus,
    MessageKind,
    NewConversationMessage,
    ProtectedMessageBody,
    StoredAttachment,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import MessageRole
from gewu_agent_runtime.persistence import (
    ConcurrentWriteError,
    InMemoryRuntimeStore,
    MessageWriteConflictError,
    RuntimeStore,
)
from gewu_core.errors import CommitOutcomeUnknownError
from gewu_core.secrets import JsonSecretCipher
from gewu_core.time import utc_now


class _CommitThenRaiseSession(AsyncSession):
    """Simulate a lost database acknowledgement after a durable COMMIT."""

    async def commit(self) -> None:
        await super().commit()
        raise RuntimeError("injected post-commit disconnect")


class _FailingFlushSession(AsyncSession):
    """Fail before COMMIT while pending attachment metadata is still reversible."""

    async def flush(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("injected flush failure")


def test_store_encryption_requires_an_explicit_cipher() -> None:
    with pytest.raises(ValueError, match="requires a protected payload cipher"):
        InMemoryRuntimeStore(encrypt_protected_payloads=True)


@pytest.fixture(params=["memory", "sqlalchemy"])
async def runtime_store(request: pytest.FixtureRequest) -> AsyncIterator[RuntimeStore]:
    if request.param == "memory":
        yield InMemoryRuntimeStore()
        return
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    yield SqlAlchemyRuntimeStore(
        async_sessionmaker(engine, expire_on_commit=False),
        protected_payload_cipher=JsonSecretCipher("runtime-store-contract-key"),
    )
    await engine.dispose()


async def test_stored_tool_response_groups_survive_paged_history(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    from gewu_agent_runtime.context import ConversationContextBuilder

    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    messages = []
    for call_id in ("first", "second"):
        messages.extend(
            (
                NewConversationMessage(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.TOOL_USE,
                    payload={
                        "assistant_message_id": "response",
                        "tool_call_id": call_id,
                        "tool_name": "lookup",
                        "arguments": {},
                    },
                ),
                NewConversationMessage(
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_RESULT,
                    payload={"tool_call_id": call_id, "result": {"value": call_id}},
                ),
            )
        )
    await runtime_store.append_messages(conversation.conversation_id, tuple(messages))
    restored = await ConversationContextBuilder(runtime_store, history_page_size=1).build(
        conversation.conversation_id
    )
    assert [message.role.value for message in restored] == ["assistant", "tool", "tool"]
    assert tuple(call.tool_call_id for call in restored[0].tool_calls) == ("first", "second")


async def test_store_allocates_sequences_and_transitions_run(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    messages = await runtime_store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(role=MessageRole.USER, kind=MessageKind.INPUT, content="one"),
            NewConversationMessage(
                role=MessageRole.ASSISTANT, kind=MessageKind.ASSISTANT, content="two"
            ),
        ),
    )
    run = await runtime_store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    running = await runtime_store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )

    assert [message.sequence for message in messages] == [1, 2]
    assert running.status is AgentRunStatus.RUNNING
    with pytest.raises(ConcurrentWriteError):
        await runtime_store.transition_run(
            run.run_id,
            expected=(AgentRunStatus.PENDING,),
            status=AgentRunStatus.COMPLETED,
        )


async def test_store_lists_only_messages_for_requested_run(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    first = await runtime_store.create_run(
        AgentRun(run_id="run-1", conversation_id=conversation.conversation_id, invoker=principal)
    )
    second = await runtime_store.create_run(
        AgentRun(run_id="run-2", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                message_id="run-1-message-1",
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                run_id=first.run_id,
            ),
            NewConversationMessage(
                message_id="run-2-message-1",
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                run_id=second.run_id,
            ),
            NewConversationMessage(
                message_id="run-1-message-2",
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                run_id=first.run_id,
            ),
        ),
    )

    messages = await runtime_store.list_messages_for_run(first.run_id)

    assert [message.message_id for message in messages] == [
        "run-1-message-1",
        "run-1-message-2",
    ]
    assert await runtime_store.list_messages_for_run("missing-run") == ()


async def test_store_pre_execution_run_failure_does_not_touch_conversation(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    original_updated_at = datetime(2026, 7, 14, 10, tzinfo=UTC)
    conversation = await runtime_store.create_conversation(
        Conversation(owner=principal, updated_at=original_updated_at)
    )
    run = await runtime_store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )

    await runtime_store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.FAILED,
        error_code="runtime_capacity",
    )

    unchanged = await runtime_store.get_conversation(conversation.conversation_id)
    assert unchanged is not None
    assert unchanged.active_run_id is None
    assert unchanged.updated_at == original_updated_at


async def test_store_conditionally_patches_conversation_metadata(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(
            conversation_id="c1",
            owner=principal,
            metadata={
                "run_state": "running",
                "active_run_id": "run-1",
                "run_started_at": "client-value",
                "keep": "value",
            },
        )
    )

    updated = await runtime_store.compare_and_patch_conversation_metadata(
        conversation.conversation_id,
        expected_values={"run_state": "running", "active_run_id": "run-1"},
        set_values={"run_state": "error"},
        remove_keys=("active_run_id", "run_started_at"),
    )
    stale = await runtime_store.compare_and_patch_conversation_metadata(
        conversation.conversation_id,
        expected_values={"run_state": "running", "active_run_id": "run-1"},
        set_values={"run_state": "idle"},
    )

    assert updated is not None
    assert updated.metadata == {"run_state": "error", "keep": "value"}
    assert stale is None
    assert await runtime_store.get_conversation(conversation.conversation_id) == updated


async def test_store_round_trips_protected_message_body(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    message = NewConversationMessage(
        role=MessageRole.TOOL,
        kind=MessageKind.TOOL_RESULT,
        protected_body=ProtectedMessageBody(
            content="protected error",
            payload={"result": {"customer": "sensitive-customer"}},
        ),
    )

    appended = await runtime_store.append_messages(conversation.conversation_id, (message,))
    listed = await runtime_store.list_messages(conversation.conversation_id)

    assert appended[0].content == "protected error"
    assert appended[0].payload == {"result": {"customer": "sensitive-customer"}}
    assert listed[0].message_id == appended[0].message_id
    assert listed[0].content == appended[0].content
    assert listed[0].payload == appended[0].payload
    assert "sensitive-customer" not in message.model_dump_json()


async def test_store_plaintext_protected_body_accepts_encryption_envelope_key(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    payload = {
        "_gewu_protected_message": {"kind": "ordinary-business-data"},
        "result": {"customer": "visible-customer"},
    }

    appended = await runtime_store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                protected_body=ProtectedMessageBody(payload=payload),
            ),
        ),
    )

    assert appended[0].payload == payload
    assert (await runtime_store.list_messages(conversation.conversation_id))[0].payload == payload


async def test_store_plaintext_compaction_accepts_encryption_prefix(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    compaction = ConversationCompaction(
        conversation_id=conversation.conversation_id,
        generation=1,
        through_sequence=1,
        summary="gewu-protected-compaction:v1:ordinary summary",
    )

    await runtime_store.save_compaction(compaction, expected_previous_id="")

    assert await runtime_store.get_latest_compaction(conversation.conversation_id) == compaction


async def test_store_rejects_duplicate_message_id_across_conversations(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    first = await runtime_store.create_conversation(
        Conversation(conversation_id="conversation-1", owner=principal)
    )
    second = await runtime_store.create_conversation(
        Conversation(conversation_id="conversation-2", owner=principal)
    )
    message = NewConversationMessage(
        message_id="shared-message-id",
        role=MessageRole.USER,
        kind=MessageKind.INPUT,
    )
    await runtime_store.append_messages(first.conversation_id, (message,))

    with pytest.raises(MessageWriteConflictError):
        await runtime_store.append_messages(second.conversation_id, (message,))

    assert await runtime_store.list_messages(second.conversation_id) == ()
    unchanged = await runtime_store.get_conversation(second.conversation_id)
    assert unchanged is not None
    assert unchanged.next_sequence == 1


def test_sql_store_identifies_only_message_primary_key_conflicts() -> None:
    duplicate_message_id = SqlAlchemyIntegrityError(
        "insert",
        {},
        PyMySQLIntegrityError(
            1062,
            "Duplicate entry 'message-1' for key 'agent_conversation_message.PRIMARY'",
        ),
    )
    duplicate_sequence = SqlAlchemyIntegrityError(
        "insert",
        {},
        PyMySQLIntegrityError(
            1062,
            "Duplicate entry 'conversation-1-1' for key 'uk_agent_message_sequence'",
        ),
    )

    assert _is_message_id_conflict(duplicate_message_id) is True
    assert _is_message_id_conflict(duplicate_sequence) is False


async def test_sql_store_does_not_reclassify_sequence_conflict(
    principal: PrincipalRef,
) -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyRuntimeStore(sessions)
    try:
        conversation = await store.create_conversation(Conversation(owner=principal))
        async with sessions.begin() as session:
            session.add(
                ConversationMessageRow(
                    id="reserved-sequence",
                    conversation_id=conversation.conversation_id,
                    sequence_number=1,
                    role=MessageRole.USER.value,
                    message_kind=MessageKind.INPUT.value,
                    content="reserved",
                    payload={},
                    run_id="",
                    request_id="",
                    created_at=utc_now(),
                )
            )

        with pytest.raises(SqlAlchemyIntegrityError):
            await store.append_messages(
                conversation.conversation_id,
                (
                    NewConversationMessage(
                        message_id="different-message-id",
                        role=MessageRole.USER,
                        kind=MessageKind.INPUT,
                    ),
                ),
            )
    finally:
        await engine.dispose()


async def test_sql_store_never_writes_protected_message_cleartext() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyRuntimeStore(
        sessions,
        protected_payload_cipher=JsonSecretCipher("protected-sql-test-key"),
        encrypt_protected_payloads=True,
    )
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    try:
        conversation = await store.create_conversation(Conversation(owner=owner))
        await store.append_messages(
            conversation.conversation_id,
            (
                NewConversationMessage(
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_RESULT,
                    protected_body=ProtectedMessageBody(
                        content="sensitive-error",
                        payload={"result": {"customer": "sensitive-customer"}},
                    ),
                ),
            ),
        )

        async with sessions() as session:
            row = await session.scalar(select(ConversationMessageRow))
            assert row is not None
            assert row.content == ""
            assert "sensitive" not in str(row.payload)
            assert row.body_encryption_version == 1

        restored = await store.list_messages(conversation.conversation_id)
        assert restored[0].content == "sensitive-error"
        assert restored[0].payload == {"result": {"customer": "sensitive-customer"}}
    finally:
        await engine.dispose()


async def test_sql_store_writes_protected_message_as_cleartext_by_default() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyRuntimeStore(
        sessions,
        protected_payload_cipher=JsonSecretCipher("protected-sql-test-key"),
    )
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    try:
        conversation = await store.create_conversation(Conversation(owner=owner))
        await store.append_messages(
            conversation.conversation_id,
            (
                NewConversationMessage(
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_RESULT,
                    protected_body=ProtectedMessageBody(
                        content="sensitive-error",
                        payload={"result": {"customer": "sensitive-customer"}},
                    ),
                ),
            ),
        )

        async with sessions() as session:
            row = await session.scalar(select(ConversationMessageRow))
            assert row is not None
            assert row.content == "sensitive-error"
            assert row.payload == {"result": {"customer": "sensitive-customer"}}
            assert row.body_encryption_version == 0
    finally:
        await engine.dispose()


async def test_sql_store_writes_compaction_summary_as_cleartext() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyRuntimeStore(
        sessions,
        protected_payload_cipher=JsonSecretCipher("protected-sql-test-key"),
    )
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    try:
        conversation = await store.create_conversation(Conversation(owner=owner))
        compaction = ConversationCompaction(
            conversation_id=conversation.conversation_id,
            generation=1,
            through_sequence=1,
            summary="Summary containing sensitive-customer-data.",
        )
        await store.save_compaction(compaction, expected_previous_id="")

        async with sessions() as session:
            row = await session.scalar(select(ConversationCompactionRow))
            assert row is not None
            assert row.summary == compaction.summary
            assert row.summary_encryption_version == 0

        restored = await store.get_latest_compaction(conversation.conversation_id)
        assert restored == compaction
    finally:
        await engine.dispose()


async def test_sql_store_encrypts_compaction_when_enabled_and_reads_it_after_disable() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    sessions = async_sessionmaker(engine, expire_on_commit=False)
    cipher = JsonSecretCipher("protected-sql-test-key")
    encrypted_store = SqlAlchemyRuntimeStore(
        sessions,
        protected_payload_cipher=cipher,
        encrypt_compactions=True,
    )
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    try:
        conversation = await encrypted_store.create_conversation(Conversation(owner=owner))
        compaction = ConversationCompaction(
            conversation_id=conversation.conversation_id,
            generation=1,
            through_sequence=1,
            summary="Summary containing sensitive-customer-data.",
        )
        await encrypted_store.save_compaction(compaction, expected_previous_id="")

        async with sessions() as session:
            row = await session.scalar(select(ConversationCompactionRow))
            assert row is not None
            assert compaction.summary not in row.summary
            assert row.summary_encryption_version == 1

        plain_writing_store = SqlAlchemyRuntimeStore(
            sessions,
            protected_payload_cipher=cipher,
        )
        assert (
            await plain_writing_store.get_latest_compaction(conversation.conversation_id)
            == compaction
        )
    finally:
        await engine.dispose()


async def test_store_fences_stale_run_messages_state_and_terminal_transition(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    """A replacement owner must fence every durable write from the stale run."""

    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    old_run = await runtime_store.create_run(
        AgentRun(run_id="run-old", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        old_run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    observed = await runtime_store.get_conversation(conversation.conversation_id)
    assert observed is not None and observed.active_run_id == old_run.run_id

    new_run = await runtime_store.create_run(
        AgentRun(run_id="run-new", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        new_run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
        expected_active_run_id=observed.active_run_id,
    )
    stale_message = NewConversationMessage(
        message_id="stale-message",
        role=MessageRole.ASSISTANT,
        kind=MessageKind.ASSISTANT,
        content="must not persist",
        run_id=old_run.run_id,
    )
    stale_state = ConversationState(
        conversation_id=conversation.conversation_id,
        kind="file",
        revision=1,
        payload={"files": {"stale.txt": {}}},
    )

    assert await runtime_store.append_messages_for_run(old_run.run_id, (stale_message,)) is None
    assert (
        await runtime_store.save_state_for_run(
            old_run.run_id,
            stale_state,
            expected_revision=0,
        )
        is None
    )
    assert (
        await runtime_store.delete_state_for_run(
            old_run.run_id,
            conversation.conversation_id,
            "file",
        )
        is False
    )
    with pytest.raises(ConcurrentWriteError, match="ownership"):
        await runtime_store.transition_run(
            old_run.run_id,
            expected=(AgentRunStatus.RUNNING,),
            status=AgentRunStatus.FAILED,
        )

    current_message = NewConversationMessage(
        message_id="current-message",
        role=MessageRole.ASSISTANT,
        kind=MessageKind.ASSISTANT,
        content="persisted",
        run_id=new_run.run_id,
    )
    persisted = await runtime_store.append_messages_for_run(
        new_run.run_id,
        (current_message,),
        finish_status=AgentRunStatus.COMPLETED,
    )

    assert persisted is not None
    assert [message.message_id for message in persisted] == ["current-message"]
    assert [message.message_id for message in await runtime_store.list_messages("c1")] == [
        "current-message"
    ]
    completed = await runtime_store.get_run(new_run.run_id)
    refreshed = await runtime_store.get_conversation(conversation.conversation_id)
    assert completed is not None and completed.status is AgentRunStatus.COMPLETED
    assert refreshed is not None and refreshed.active_run_id is None


async def test_chat_attachment_repository_finds_only_cleanup_candidates(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    now = utc_now()
    active = await runtime_store.create_conversation(
        Conversation(conversation_id="active", owner=principal)
    )
    archived = await runtime_store.create_conversation(
        Conversation(
            conversation_id="archived",
            owner=principal,
            status=ConversationStatus.ARCHIVED,
        )
    )
    attachments = (
        StoredAttachment(
            attachment_id="old-orphan",
            owner=principal,
            conversation_id=active.conversation_id,
            request_id="request-old",
            storage_backend="local",
            resource_key="chat/old.png",
            mime_type="image/png",
            size_bytes=1,
            created_at=now - timedelta(hours=25),
        ),
        StoredAttachment(
            attachment_id="persisted",
            owner=principal,
            conversation_id=active.conversation_id,
            request_id="request-persisted",
            storage_backend="local",
            resource_key="chat/persisted.png",
            mime_type="image/png",
            size_bytes=1,
            created_at=now - timedelta(hours=25),
        ),
        StoredAttachment(
            attachment_id="recent-orphan",
            owner=principal,
            conversation_id=active.conversation_id,
            request_id="request-recent",
            storage_backend="local",
            resource_key="chat/recent.png",
            mime_type="image/png",
            size_bytes=1,
            created_at=now - timedelta(hours=1),
        ),
        StoredAttachment(
            attachment_id="archived-media",
            owner=principal,
            conversation_id=archived.conversation_id,
            request_id="request-archived",
            storage_backend="local",
            resource_key="chat/archived.png",
            mime_type="image/png",
            size_bytes=1,
            created_at=now - timedelta(hours=1),
        ),
        StoredAttachment(
            attachment_id="wrong-backend",
            owner=principal,
            conversation_id=active.conversation_id,
            request_id="request-wrong",
            storage_backend="oss",
            resource_key="chat/wrong.png",
            mime_type="image/png",
            size_bytes=1,
            created_at=now - timedelta(hours=25),
        ),
    )
    for attachment in attachments:
        await runtime_store.create_attachment(attachment)
    await runtime_store.append_messages(
        active.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                request_id="request-persisted",
            ),
        ),
    )

    candidates = await runtime_store.list_attachment_cleanup_candidates(
        pending_before=now - timedelta(hours=24),
        storage_backend="local",
        limit=200,
    )

    assert {value.attachment_id for value in candidates} == {
        "old-orphan",
        "archived-media",
    }
    await runtime_store.mark_attachments_deleted(tuple(value.attachment_id for value in candidates))
    assert await runtime_store.get_active_attachment("old-orphan", principal) is None
    assert await runtime_store.get_active_attachment("archived-media", principal) is None
    assert await runtime_store.get_active_attachment("persisted", principal) is not None
    assert await runtime_store.get_active_attachment("recent-orphan", principal) is not None
    assert await runtime_store.get_active_attachment("wrong-backend", principal) is not None


async def test_store_state_cas_and_compaction_boundary(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    await runtime_store.append_messages(
        conversation.conversation_id,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=f"message {index}",
            )
            for index in range(4)
        ),
    )
    state = ConversationState(
        conversation_id=conversation.conversation_id,
        kind="pending_ask",
        revision=1,
        payload={"ask_id": "ask"},
        expires_at=utc_now() + timedelta(minutes=5),
    )
    await runtime_store.save_state(state, expected_revision=0)
    compaction = ConversationCompaction(
        conversation_id=conversation.conversation_id,
        generation=1,
        through_sequence=2,
        summary="Earlier context.",
    )
    await runtime_store.save_compaction(compaction, expected_previous_id="")

    loaded_state = await runtime_store.get_state(conversation.conversation_id, state.kind)
    loaded_compaction = await runtime_store.get_latest_compaction(conversation.conversation_id)
    assert loaded_state is not None
    assert loaded_state.revision == state.revision
    assert loaded_state.payload == state.payload
    assert loaded_compaction is not None
    assert loaded_compaction.compaction_id == compaction.compaction_id
    with pytest.raises(ConcurrentWriteError):
        await runtime_store.save_state(state, expected_revision=0)


async def test_store_can_recover_expired_state_as_fact_evidence(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    expired = ConversationState(
        conversation_id=conversation.conversation_id,
        kind="pending_ask",
        revision=1,
        payload={"ask_id": "ask-expired"},
        expires_at=utc_now() - timedelta(seconds=1),
    )
    await runtime_store.save_state(expired, expected_revision=0)

    assert await runtime_store.get_state(conversation.conversation_id, expired.kind) is None
    recovered = await runtime_store.get_state(
        conversation.conversation_id,
        expired.kind,
        include_expired=True,
    )

    assert recovered is not None
    assert recovered.payload == {"ask_id": "ask-expired"}


async def test_store_lists_expired_states_with_stable_cursor(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    deadline = utc_now()
    rows = (
        ("c1", deadline - timedelta(minutes=2), "pending_ask"),
        ("c2", deadline - timedelta(minutes=1), "pending_ask"),
        ("c3", deadline - timedelta(minutes=1), "pending_ask"),
        ("future", deadline + timedelta(minutes=1), "pending_ask"),
        ("other-kind", deadline - timedelta(minutes=3), "file"),
    )
    for conversation_id, expires_at, kind in rows:
        await runtime_store.create_conversation(
            Conversation(conversation_id=conversation_id, owner=principal)
        )
        await runtime_store.save_state(
            ConversationState(
                conversation_id=conversation_id,
                kind=kind,
                revision=1,
                expires_at=expires_at,
            ),
            expected_revision=0,
        )

    first = await runtime_store.list_expired_states(
        "pending_ask",
        deadline,
        limit=2,
    )
    second = await runtime_store.list_expired_states(
        "pending_ask",
        deadline,
        after=first[-1],
        limit=2,
    )

    assert [state.conversation_id for state in (*first, *second)] == ["c1", "c2", "c3"]


async def test_store_resolves_pending_ask_atomically(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    await runtime_store.create_run(
        AgentRun(run_id="run-1", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    await runtime_store.save_state(
        ConversationState(
            conversation_id=conversation.conversation_id,
            kind="pending_ask",
            revision=1,
            payload={"run_id": "run-1", "ask_id": "ask-1"},
            expires_at=utc_now() - timedelta(seconds=1),
        ),
        expected_revision=0,
    )
    resolution = NewConversationMessage(
        message_id="resolution-1",
        role=MessageRole.TOOL,
        kind=MessageKind.TOOL_RESULT,
        payload={"result": {"metadata": {"reason": "expired"}}},
        run_id="run-1",
        request_id="request-1",
    )

    resolved = await runtime_store.resolve_pending_ask(
        "run-1",
        state_revision=1,
        message=resolution,
        error_code="ask_expired",
        error_message="Pending ask_user request expired before it was answered.",
    )
    duplicate = await runtime_store.resolve_pending_ask(
        "run-1",
        state_revision=1,
        message=resolution.model_copy(update={"message_id": "resolution-2"}),
        error_code="ask_expired",
        error_message="Pending ask_user request expired before it was answered.",
    )

    assert resolved is not None
    assert resolved.status is AgentRunStatus.CANCELLED
    assert resolved.error_code == "ask_expired"
    assert duplicate is None
    assert (
        await runtime_store.get_state(
            conversation.conversation_id,
            "pending_ask",
            include_expired=True,
        )
        is None
    )
    messages = await runtime_store.list_messages(conversation.conversation_id)
    assert [message.message_id for message in messages] == ["resolution-1"]


async def test_store_commits_pending_ask_atomically(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    await runtime_store.create_run(
        AgentRun(run_id="run-1", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    state = ConversationState(
        conversation_id=conversation.conversation_id,
        kind="pending_ask",
        revision=1,
        payload={"run_id": "run-1", "ask_id": "ask-1"},
        expires_at=utc_now() + timedelta(minutes=5),
    )
    message = NewConversationMessage(
        message_id="ask-message-1",
        role=MessageRole.ASSISTANT,
        kind=MessageKind.ASK,
        payload={"ask_id": "ask-1"},
        run_id="run-1",
        request_id="request-1",
    )

    committed = await runtime_store.commit_pending_ask(
        "run-1",
        state=state,
        message=message,
    )
    duplicate = await runtime_store.commit_pending_ask(
        "run-1",
        state=state,
        message=message.model_copy(update={"message_id": "ask-message-2"}),
    )

    assert committed is not None
    assert committed.status is AgentRunStatus.WAITING_INPUT
    assert duplicate is None
    persisted_state = await runtime_store.get_state(conversation.conversation_id, "pending_ask")
    assert persisted_state is not None
    assert persisted_state.payload["ask_id"] == "ask-1"
    messages = await runtime_store.list_messages(conversation.conversation_id)
    assert [item.message_id for item in messages] == ["ask-message-1"]


async def test_store_commits_ask_answer_atomically(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    await runtime_store.create_run(
        AgentRun(run_id="run-1", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    await runtime_store.save_state(
        ConversationState(
            conversation_id=conversation.conversation_id,
            kind="pending_ask",
            revision=1,
            payload={"run_id": "run-1", "ask_id": "ask-1"},
            expires_at=utc_now() + timedelta(minutes=5),
        ),
        expected_revision=0,
    )
    answer = NewConversationMessage(
        message_id="answer-message-1",
        role=MessageRole.TOOL,
        kind=MessageKind.TOOL_RESULT,
        payload={"result": {"answers": {"Continue?": "Yes"}}},
        run_id="run-1",
        request_id="request-1",
    )

    committed = await runtime_store.commit_ask_answer(
        "run-1",
        state_revision=1,
        message=answer,
    )
    duplicate = await runtime_store.commit_ask_answer(
        "run-1",
        state_revision=1,
        message=answer.model_copy(update={"message_id": "answer-message-2"}),
    )

    assert committed is True
    assert duplicate is False
    assert (
        await runtime_store.get_state(
            conversation.conversation_id,
            "pending_ask",
            include_expired=True,
        )
        is None
    )
    messages = await runtime_store.list_messages(conversation.conversation_id)
    assert [item.message_id for item in messages] == ["answer-message-1"]


async def test_store_rolls_back_pending_ask_when_answer_message_conflicts(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    await runtime_store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                message_id="answer-message-1",
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="existing",
            ),
        ),
    )
    await runtime_store.create_run(
        AgentRun(run_id="run-1", conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    await runtime_store.save_state(
        ConversationState(
            conversation_id=conversation.conversation_id,
            kind="pending_ask",
            revision=1,
            payload={"run_id": "run-1", "ask_id": "ask-1"},
        ),
        expected_revision=0,
    )

    with pytest.raises(ConcurrentWriteError):
        await runtime_store.commit_ask_answer(
            "run-1",
            state_revision=1,
            message=NewConversationMessage(
                message_id="answer-message-1",
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                run_id="run-1",
            ),
        )

    state = await runtime_store.get_state(
        conversation.conversation_id,
        "pending_ask",
        include_expired=True,
    )
    assert state is not None and state.revision == 1
    messages = await runtime_store.list_messages(conversation.conversation_id)
    assert [(item.message_id, item.content) for item in messages] == [
        ("answer-message-1", "existing")
    ]


async def test_store_reads_bounded_message_pages(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    await runtime_store.append_messages(
        conversation.conversation_id,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=f"message {index}",
            )
            for index in range(1, 7)
        ),
    )

    page = await runtime_store.list_message_page(
        conversation.conversation_id,
        after_sequence=1,
        before_sequence=6,
        limit=3,
    )
    recent = await runtime_store.list_recent_messages(
        conversation.conversation_id,
        before_sequence=6,
        limit=2,
    )

    assert [value.sequence for value in page] == [2, 3, 4]
    assert [value.sequence for value in recent] == [4, 5]


async def test_store_lists_owner_conversations_with_stable_cursor(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    same_time = datetime(2026, 7, 14, 11, tzinfo=UTC)
    older_time = datetime(2026, 7, 13, 11, tzinfo=UTC)
    other = principal.model_copy(update={"principal_id": "other-user"})
    foreign_subscriber = principal.model_copy(update={"subscriber_id": "other-subscriber"})
    rows = (
        Conversation(conversation_id="c3", owner=principal, updated_at=same_time),
        Conversation(conversation_id="c2", owner=principal, updated_at=same_time),
        Conversation(conversation_id="c1", owner=principal, updated_at=older_time),
        Conversation(
            conversation_id="archived",
            owner=principal,
            status=ConversationStatus.ARCHIVED,
            updated_at=same_time,
        ),
        Conversation(conversation_id="other-user", owner=other, updated_at=same_time),
        Conversation(
            conversation_id="other-subscriber",
            owner=foreign_subscriber,
            updated_at=same_time,
        ),
    )
    for row in rows:
        await runtime_store.create_conversation(row)

    first = await runtime_store.list_conversations(principal, limit=2)
    second = await runtime_store.list_conversations(
        principal,
        before_updated_at=first.items[-1].updated_at,
        before_conversation_id=first.items[-1].conversation_id,
        limit=2,
    )
    with_archived = await runtime_store.list_conversations(
        principal,
        include_archived=True,
        updated_after=older_time,
        limit=10,
    )

    assert [value.conversation_id for value in first.items] == ["c3", "c2"]
    assert first.total == 3
    assert [value.conversation_id for value in second.items] == ["c1"]
    assert second.total == 3
    assert {value.conversation_id for value in with_archived.items} == {
        "archived",
        "c1",
        "c2",
        "c3",
    }
    assert with_archived.total == 4


async def test_store_updates_conversation_and_loads_latest_run(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal, title="Before")
    )
    older = await runtime_store.create_run(
        AgentRun(
            run_id="r1",
            conversation_id=conversation.conversation_id,
            invoker=principal,
            created_at=datetime(2026, 7, 14, 10, tzinfo=UTC),
        )
    )
    newer = await runtime_store.create_run(
        AgentRun(
            run_id="r2",
            conversation_id=conversation.conversation_id,
            invoker=principal,
            created_at=datetime(2026, 7, 14, 11, tzinfo=UTC),
        )
    )

    updated = await runtime_store.update_conversation(
        conversation.conversation_id,
        title="After",
        status=ConversationStatus.ARCHIVED,
    )

    assert older.run_id == "r1"
    assert updated is not None
    assert updated.title == "After"
    assert updated.status is ConversationStatus.ARCHIVED
    assert updated.updated_at > conversation.updated_at
    assert await runtime_store.get_latest_run(conversation.conversation_id) == newer
    assert await runtime_store.update_conversation("missing", title="Nope") is None
    assert await runtime_store.get_latest_run("missing") is None


async def test_chat_attachment_repository_creates_and_lists_for_message(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    other = principal.model_copy(update={"principal_id": "other-user"})
    await runtime_store.create_conversation(Conversation(conversation_id="c2", owner=other))
    attachments = (
        StoredAttachment(
            attachment_id="a1",
            owner=principal,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            storage_backend="local",
            resource_key="chat/2026/07/a1.png",
            mime_type="image/png",
            size_bytes=3,
        ),
        StoredAttachment(
            attachment_id="a2",
            owner=principal,
            conversation_id=conversation.conversation_id,
            request_id="request-1",
            storage_backend="local",
            resource_key="chat/2026/07/a2.jpg",
            mime_type="image/jpeg",
            size_bytes=4,
        ),
        StoredAttachment(
            attachment_id="other",
            owner=other,
            conversation_id="c2",
            request_id="request-1",
            storage_backend="local",
            resource_key="chat/2026/07/other.png",
            mime_type="image/png",
            size_bytes=3,
        ),
    )
    for attachment in attachments:
        await runtime_store.create_attachment(attachment)

    resolved = await runtime_store.list_active_attachments(
        principal,
        conversation_id=conversation.conversation_id,
        request_id="request-1",
        attachment_ids=("a2", "missing", "a1"),
    )
    await runtime_store.mark_attachments_deleted(("a2", "a2", "missing"))

    assert [value.attachment_id for value in resolved] == ["a2", "a1"]
    assert await runtime_store.get_active_attachment("a1", principal) == attachments[0]
    assert await runtime_store.get_active_attachment("a1", other) is None
    assert await runtime_store.get_active_attachment("a2", principal) is None
    assert (
        await runtime_store.list_active_attachments(
            other,
            conversation_id="c2",
            request_id="request-1",
            attachment_ids=("a1", "other"),
        )
    ) == (attachments[2],)


async def test_store_rejects_duplicate_attachment_id(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    await runtime_store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    attachment = StoredAttachment(
        attachment_id="a1",
        owner=principal,
        conversation_id="c1",
        request_id="request-1",
        storage_backend="local",
        resource_key="chat/a1.png",
        mime_type="image/png",
        size_bytes=3,
    )
    await runtime_store.create_attachment(attachment)

    with pytest.raises(ConcurrentWriteError):
        await runtime_store.create_attachment(attachment)


async def test_chat_attachment_refresh_failure_does_not_commit_metadata() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    normal_sessions = async_sessionmaker(engine, expire_on_commit=False)
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    attachment = StoredAttachment(
        attachment_id="attachment-1",
        owner=owner,
        conversation_id="conversation-1",
        request_id="request-1",
        storage_backend="local",
        resource_key="chat/attachment-1.png",
        mime_type="image/png",
        size_bytes=3,
    )
    try:
        normal_store = SqlAlchemyRuntimeStore(normal_sessions)
        await normal_store.create_conversation(
            Conversation(conversation_id=attachment.conversation_id, owner=owner)
        )
        failing_store = SqlAlchemyRuntimeStore(
            async_sessionmaker(
                engine,
                class_=_FailingFlushSession,
                expire_on_commit=False,
            )
        )

        with pytest.raises(RuntimeError, match="injected flush failure"):
            await failing_store.create_attachment(attachment)

        async with normal_sessions() as session:
            assert await session.get(AttachmentRow, attachment.attachment_id) is None
    finally:
        await engine.dispose()


async def test_chat_attachment_post_commit_disconnect_reports_unknown() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    normal_sessions = async_sessionmaker(engine, expire_on_commit=False)
    owner = PrincipalRef(
        subscriber_id="subscriber-1",
        principal_id="principal-1",
        principal_type="user",
    )
    attachment = StoredAttachment(
        attachment_id="attachment-1",
        owner=owner,
        conversation_id="conversation-1",
        request_id="request-1",
        storage_backend="local",
        resource_key="chat/attachment-1.png",
        mime_type="image/png",
        size_bytes=3,
    )
    try:
        normal_store = SqlAlchemyRuntimeStore(normal_sessions)
        await normal_store.create_conversation(
            Conversation(conversation_id=attachment.conversation_id, owner=owner)
        )
        failing_store = SqlAlchemyRuntimeStore(
            async_sessionmaker(
                engine,
                class_=_CommitThenRaiseSession,
                expire_on_commit=False,
            )
        )

        with pytest.raises(CommitOutcomeUnknownError):
            await failing_store.create_attachment(attachment)

        async with normal_sessions() as session:
            assert await session.get(AttachmentRow, attachment.attachment_id) is not None
    finally:
        await engine.dispose()


async def test_store_commits_all_compaction_effects_for_active_run(
    runtime_store: RuntimeStore,
    principal: PrincipalRef,
) -> None:
    conversation = await runtime_store.create_conversation(Conversation(owner=principal))
    run = await runtime_store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    await runtime_store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    await runtime_store.save_state(
        ConversationState(
            conversation_id=conversation.conversation_id,
            kind="file",
            revision=1,
            payload={"files": {"old.txt": {}}},
        ),
        expected_revision=0,
    )
    compaction = ConversationCompaction(
        conversation_id=conversation.conversation_id,
        generation=1,
        through_sequence=1,
        summary="Earlier context.",
    )
    commit = ConversationCompactionCommit(
        compaction=compaction,
        messages=(
            NewConversationMessage(
                role=MessageRole.SYSTEM,
                kind=MessageKind.MEMORY_COMPACTION,
                content="会话记忆压缩完毕",
                payload={"phase": "completed", "llm_ignore": True},
                run_id=run.run_id,
            ),
        ),
        state_payloads={"skill": {"sent_skill_names": ["review"]}},
        delete_state_kinds=("file",),
    )

    assert await runtime_store.commit_compaction_for_run(commit, run.run_id) is True
    assert await runtime_store.get_latest_compaction(conversation.conversation_id) == compaction
    exact = await runtime_store.get_compaction(
        conversation.conversation_id,
        compaction.compaction_id,
    )
    assert exact == compaction
    assert await runtime_store.get_state(conversation.conversation_id, "file") is None
    skill = await runtime_store.get_state(conversation.conversation_id, "skill")
    assert skill is not None and skill.payload == {"sent_skill_names": ["review"]}
    messages = await runtime_store.list_messages(conversation.conversation_id)
    assert messages[-1].kind is MessageKind.MEMORY_COMPACTION

    rejected = compaction.model_copy(
        update={"compaction_id": "compact-2", "generation": 2, "previous_compaction_id": ""}
    )
    assert (
        await runtime_store.commit_compaction_for_run(
            ConversationCompactionCommit(compaction=rejected),
            run.run_id,
        )
        is False
    )
    assert len(await runtime_store.list_messages(conversation.conversation_id)) == 1
