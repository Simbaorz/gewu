"""Direct parity evidence for durable conversation persistence behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import Select
from sqlalchemy.dialects import mysql, sqlite
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from gewu_agent_runtime.adapters.mysql import SqlAlchemyRuntimeStore, create_schema
from gewu_agent_runtime.adapters.mysql.models import (
    ConversationCompactionRow,
    ConversationMessageRow,
    ConversationRow,
)
from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationCompaction,
    ConversationCompactionCommit,
    ConversationState,
    MessageKind,
    NewConversationMessage,
)
from gewu_agent_runtime.identity import PrincipalRef, PrincipalType
from gewu_agent_runtime.llm import MessageRole
from gewu_agent_runtime.persistence import ConcurrentWriteError
from gewu_core.secrets import JsonSecretCipher
from gewu_core.time import utc_now


def _owner() -> PrincipalRef:
    return PrincipalRef(
        subscriber_id="subscriber-a",
        principal_id="user-a",
        principal_type=PrincipalType.USER,
    )


@pytest.fixture
async def sql_store() -> AsyncIterator[SqlAlchemyRuntimeStore]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    await create_schema(engine)
    yield SqlAlchemyRuntimeStore(
        async_sessionmaker(engine, expire_on_commit=False),
        protected_payload_cipher=JsonSecretCipher("mysql-conversation-test-key"),
    )
    await engine.dispose()


async def _claim_run(
    store: SqlAlchemyRuntimeStore,
    conversation_id: str,
    run_id: str,
    *,
    expected_active_run_id: str | None = None,
) -> AgentRun:
    run = await store.create_run(
        AgentRun(run_id=run_id, conversation_id=conversation_id, invoker=_owner())
    )
    return await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
        expected_active_run_id=expected_active_run_id,
    )


def test_conversation_message_content_uses_mysql_longtext() -> None:
    """Keep long model output portable while avoiding MySQL TEXT overflow."""
    content_type = ConversationMessageRow.__table__.c.content.type

    assert content_type.compile(dialect=mysql.dialect()) == "LONGTEXT"
    assert content_type.compile(dialect=sqlite.dialect()) == "TEXT"


def test_conversation_compaction_summary_uses_mysql_longtext() -> None:
    """Keep cumulative summaries portable beyond MySQL TEXT limits."""
    summary_type = ConversationCompactionRow.__table__.c.summary.type

    assert summary_type.compile(dialect=mysql.dialect()) == "LONGTEXT"
    assert summary_type.compile(dialect=sqlite.dialect()) == "TEXT"


async def test_compaction_commit_atomically_advances_boundary_and_agent_states(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Fence one summary, boundary pointer, states and messages by the active Run."""
    await sql_store.create_conversation(Conversation(conversation_id="c1", owner=_owner()))
    await _claim_run(sql_store, "c1", "run-1")
    compaction = ConversationCompaction(
        compaction_id="compact-1",
        conversation_id="c1",
        generation=1,
        source_from_sequence=1,
        through_sequence=10,
        summary="summary",
        model_ref="llm-1",
        model_name="model-1",
        context_window=32_768,
        pre_compaction_tokens=24_000,
        post_compaction_tokens=12_000,
    )

    committed = await sql_store.commit_compaction_for_run(
        ConversationCompactionCommit(
            compaction=compaction,
            messages=(
                NewConversationMessage(
                    message_id="compact-started",
                    role=MessageRole.SYSTEM,
                    kind=MessageKind.MEMORY_COMPACTION,
                    content="compacting",
                    payload={"phase": "started", "llm_ignore": True},
                    run_id="run-1",
                    request_id="input-1",
                ),
                NewConversationMessage(
                    message_id="compact-completed",
                    role=MessageRole.SYSTEM,
                    kind=MessageKind.MEMORY_COMPACTION,
                    content="compacted",
                    payload={"phase": "completed", "llm_ignore": True},
                    run_id="run-1",
                    request_id="input-1",
                ),
            ),
            state_payloads={
                "file": {"files": {"a.md": {"version": "1"}}},
                "skill": {"sent_skill_names": ["review"]},
            },
        ),
        "run-1",
    )
    conversation = await sql_store.get_conversation("c1")

    assert committed is True
    assert conversation is not None
    assert conversation.latest_compaction_id == "compact-1"
    assert await sql_store.get_compaction("c1", "compact-1") == compaction
    file_state = await sql_store.get_state("c1", "file")
    skill_state = await sql_store.get_state("c1", "skill")
    assert file_state is not None and file_state.revision == 1
    assert skill_state is not None and skill_state.revision == 1
    persisted_messages = await sql_store.list_messages("c1")
    assert [message.message_id for message in persisted_messages] == [
        "compact-started",
        "compact-completed",
    ]
    assert [message.sequence for message in persisted_messages] == [1, 2]

    stale_compaction = compaction.model_copy(
        update={
            "compaction_id": "compact-2",
            "generation": 2,
            "previous_compaction_id": "compact-1",
        }
    )
    assert not await sql_store.commit_compaction_for_run(
        ConversationCompactionCommit(
            compaction=stale_compaction,
            expected_previous_id="wrong-observation",
        ),
        "run-1",
    )
    assert not await sql_store.commit_compaction_for_run(
        ConversationCompactionCommit(
            compaction=stale_compaction,
            expected_previous_id="compact-1",
        ),
        "run-2",
    )
    assert await sql_store.get_compaction("c1", "compact-2") is None


def test_message_append_lock_uses_mysql_row_lock() -> None:
    """Serialize sequence allocation on the owning conversation row in MySQL."""
    statement: Select[tuple[ConversationRow]] = (
        SqlAlchemyRuntimeStore._conversation_append_lock_stmt("c1")
    )

    sql = str(statement.compile(dialect=mysql.dialect(), compile_kwargs={"literal_binds": True}))

    assert "FOR UPDATE" in sql


async def test_list_recent_filters_by_sequence_cursor(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Return the newest bounded page before the requested message sequence."""
    await sql_store.create_conversation(Conversation(conversation_id="c1", owner=_owner()))
    await sql_store.append_messages(
        "c1",
        tuple(
            NewConversationMessage(
                message_id=f"m{index}",
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content=str(index),
            )
            for index in range(1, 6)
        ),
    )

    messages = await sql_store.list_recent_messages("c1", limit=2, before_sequence=5)
    range_page = await sql_store.list_message_page(
        "c1",
        after_sequence=1,
        before_sequence=5,
        limit=2,
    )

    assert [message.sequence for message in messages] == [3, 4]
    assert [message.sequence for message in range_page] == [2, 3]


async def test_conversation_list_filters_date_and_uses_stable_cursor(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Page recent owner conversations without equal timestamps skipping rows."""
    now = utc_now()
    for conversation_id, updated_at in (
        ("c3", now - timedelta(hours=1)),
        ("c2", now - timedelta(hours=1)),
        ("c1", now - timedelta(days=2)),
        ("old", now - timedelta(days=31)),
    ):
        await sql_store.create_conversation(
            Conversation(
                conversation_id=conversation_id,
                owner=_owner(),
                title=conversation_id,
                created_at=updated_at,
                updated_at=updated_at,
            )
        )

    first = await sql_store.list_conversations(
        _owner(),
        updated_after=now - timedelta(days=7),
        limit=2,
    )
    second = await sql_store.list_conversations(
        _owner(),
        updated_after=now - timedelta(days=7),
        before_updated_at=first.items[-1].updated_at,
        before_conversation_id=first.items[-1].conversation_id,
        limit=2,
    )

    assert [item.conversation_id for item in first.items] == ["c3", "c2"]
    assert [item.conversation_id for item in second.items] == ["c1"]
    assert first.total == second.total == 3


async def test_skill_state_is_stored_separately_from_conversation_metadata(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Keep Runtime Skill state in its run-fenced state row."""
    await sql_store.create_conversation(
        Conversation(conversation_id="c1", owner=_owner(), metadata={"other": "keep"})
    )
    await _claim_run(sql_store, "c1", "run-1")
    saved = await sql_store.save_state_for_run(
        "run-1",
        ConversationState(
            conversation_id="c1",
            kind="skill",
            revision=1,
            payload={"sent_skill_names": ["review"]},
        ),
        expected_revision=0,
    )
    await sql_store.transition_run(
        "run-1",
        expected=(AgentRunStatus.RUNNING,),
        status=AgentRunStatus.COMPLETED,
    )
    conversation = await sql_store.get_conversation("c1")

    assert saved is not None
    assert conversation is not None
    assert conversation.metadata == {"other": "keep"}
    assert await sql_store.get_state("c1", "skill") == saved


async def test_patch_metadata_sets_and_removes_keys_without_replacing_metadata(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Patch generic host metadata without losing unrelated keys."""
    await sql_store.create_conversation(
        Conversation(
            conversation_id="c1",
            owner=_owner(),
            metadata={"other": "keep", "pending_ask": {"ask_id": "ask-1"}},
        )
    )

    saved = await sql_store.compare_and_patch_conversation_metadata(
        "c1",
        expected_values={},
        set_values={"run_state": "running"},
        remove_keys=("pending_ask",),
    )
    conversation = await sql_store.get_conversation("c1")

    assert saved is not None
    assert conversation is not None
    assert conversation.metadata == {"other": "keep", "run_state": "running"}


async def test_finish_run_rejects_stale_owner(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """A stale Run must not overwrite the state of a replacement owner."""
    await sql_store.create_conversation(Conversation(conversation_id="c1", owner=_owner()))
    await _claim_run(sql_store, "c1", "run-old")
    await _claim_run(
        sql_store,
        "c1",
        "run-new",
        expected_active_run_id="run-old",
    )

    with pytest.raises(ConcurrentWriteError, match="ownership"):
        await sql_store.transition_run(
            "run-old",
            expected=(AgentRunStatus.RUNNING,),
            status=AgentRunStatus.FAILED,
        )
    await sql_store.transition_run(
        "run-new",
        expected=(AgentRunStatus.RUNNING,),
        status=AgentRunStatus.COMPLETED,
    )
    conversation = await sql_store.get_conversation("c1")
    old_run = await sql_store.get_run("run-old")

    assert conversation is not None and conversation.active_run_id is None
    assert old_run is not None and old_run.status is AgentRunStatus.RUNNING


async def test_append_for_run_rejects_stale_owner_and_finishes_current_owner(
    sql_store: SqlAlchemyRuntimeStore,
) -> None:
    """Fence stale message writes and commit the current terminal state atomically."""
    await sql_store.create_conversation(Conversation(conversation_id="c1", owner=_owner()))
    await _claim_run(sql_store, "c1", "run-old")
    await _claim_run(
        sql_store,
        "c1",
        "run-new",
        expected_active_run_id="run-old",
    )

    stale = NewConversationMessage(
        message_id="stale",
        role=MessageRole.ASSISTANT,
        kind=MessageKind.ASSISTANT,
        content="must not persist",
        run_id="run-old",
        request_id="request-1",
    )
    current = stale.model_copy(update={"message_id": "m1", "content": "done", "run_id": "run-new"})

    assert await sql_store.append_messages_for_run("run-old", (stale,)) is None
    saved = await sql_store.append_messages_for_run(
        "run-new",
        (current,),
        finish_status=AgentRunStatus.COMPLETED,
    )
    conversation = await sql_store.get_conversation("c1")
    persisted = await sql_store.list_recent_messages("c1", limit=10)
    run = await sql_store.get_run("run-new")

    assert saved is not None
    assert [item.message_id for item in persisted] == ["m1"]
    assert conversation is not None and conversation.active_run_id is None
    assert run is not None and run.status is AgentRunStatus.COMPLETED
