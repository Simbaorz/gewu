from sqlalchemy.ext.asyncio import create_async_engine

from gewu_agent_runtime.adapters.mysql.schema import (
    AgentRuntimeBase,
    ConversationRow,
    create_schema,
)


async def test_create_schema_creates_the_complete_runtime_table_set() -> None:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    try:
        await create_schema(engine)
        assert set(AgentRuntimeBase.metadata.tables) == {
            "agent_attachment",
            "agent_conversation",
            "agent_conversation_compaction",
            "agent_conversation_message",
            "agent_conversation_state",
            "agent_run",
        }
    finally:
        await engine.dispose()


def test_conversation_schema_persists_the_active_run_owner() -> None:
    """Keep database ownership fencing independent from the Redis run lease."""

    assert ConversationRow.__table__.c.active_run_id.nullable is True
