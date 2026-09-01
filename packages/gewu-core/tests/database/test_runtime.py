"""Asynchronous database resource lifecycle behavior."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import text

from gewu_core.database import DatabaseRuntime, DatabaseSettings


async def test_database_runtime_owns_one_engine_and_session_factory(tmp_path: Path) -> None:
    runtime = DatabaseRuntime(DatabaseSettings(), tmp_path)

    await runtime.startup()
    first_engine = runtime.engine
    first_sessions = runtime.sessions
    await runtime.startup()

    assert runtime.started is True
    assert runtime.engine is first_engine
    assert runtime.sessions is first_sessions
    assert first_sessions is not None
    async with first_sessions() as session:
        assert await session.scalar(text("SELECT 1")) == 1

    await runtime.shutdown()

    assert runtime.started is False
    assert runtime.engine is None
    assert runtime.sessions is None


async def test_database_runtime_rejects_disabled_database(tmp_path: Path) -> None:
    runtime = DatabaseRuntime(DatabaseSettings(enabled=False), tmp_path)

    with pytest.raises(RuntimeError, match="db.enabled must be true"):
        await runtime.startup()

    assert runtime.started is False
    assert runtime.engine is None
    assert runtime.sessions is None
