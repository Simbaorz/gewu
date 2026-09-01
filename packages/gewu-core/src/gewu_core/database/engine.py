"""Shared SQLAlchemy asynchronous Engine options."""

from __future__ import annotations

from sqlalchemy.pool import NullPool

from gewu_core.database.pool import InstrumentedAsyncAdaptedQueuePool
from gewu_core.database.settings import DatabaseSettings


def build_async_engine_kwargs(
    settings: DatabaseSettings,
    *,
    use_null_pool: bool = False,
) -> dict[str, object]:
    """Build Engine options for the configured database and pooling mode."""
    engine_kwargs: dict[str, object] = {"echo": settings.echo}
    if settings.use_sqlite:
        return engine_kwargs

    engine_kwargs["connect_args"] = {
        "connect_timeout": settings.connect_timeout_seconds,
    }
    if use_null_pool:
        engine_kwargs["poolclass"] = NullPool
        return engine_kwargs

    engine_kwargs.update(
        poolclass=InstrumentedAsyncAdaptedQueuePool,
        pool_size=settings.pool_size,
        max_overflow=settings.max_overflow,
        pool_pre_ping=settings.pool_pre_ping,
        pool_recycle=settings.pool_recycle_seconds,
        pool_timeout=settings.pool_timeout_seconds,
        pool_use_lifo=True,
    )
    return engine_kwargs
