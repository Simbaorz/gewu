"""Native asynchronous database pool instrumentation."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError
from sqlalchemy.ext.asyncio import create_async_engine

from gewu_core.database.pool import (
    InstrumentedAsyncAdaptedQueuePool,
    configure_db_checkout_recorder,
)


async def test_pool_records_success_and_timeout(tmp_path: Path) -> None:
    observations: list[tuple[float, str]] = []
    previous = configure_db_checkout_recorder(
        lambda seconds, outcome: observations.append((seconds, outcome))
    )
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{tmp_path / 'pool.db'}",
        poolclass=InstrumentedAsyncAdaptedQueuePool,
        pool_size=1,
        max_overflow=0,
        pool_timeout=0.01,
    )
    try:
        async with engine.connect():
            with pytest.raises(SqlAlchemyTimeoutError):
                async with engine.connect():
                    pass
    finally:
        await engine.dispose()
        configure_db_checkout_recorder(previous)

    assert [outcome for _, outcome in observations] == ["success", "timeout"]
    assert all(seconds >= 0 for seconds, _ in observations)
