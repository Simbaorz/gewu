"""Schema lifecycle helpers for tests and simple deployments."""

from sqlalchemy.ext.asyncio import AsyncEngine

from gewu_agent_runtime.adapters.mysql.schema.models import AgentRuntimeBase


async def create_schema(engine: AsyncEngine) -> None:
    """Create all Agent Runtime persistence tables."""

    async with engine.begin() as connection:
        await connection.run_sync(AgentRuntimeBase.metadata.create_all)
