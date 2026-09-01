"""Shared asynchronous database Engine option behavior."""

from sqlalchemy.pool import NullPool

from gewu_core.database import (
    DatabaseSettings,
    InstrumentedAsyncAdaptedQueuePool,
    build_async_engine_kwargs,
)


def test_mysql_pooled_engine_uses_configured_connection_policy() -> None:
    settings = DatabaseSettings(
        use_sqlite=False,
        echo=True,
        pool_size=7,
        max_overflow=11,
        pool_pre_ping=False,
        pool_recycle_seconds=600,
        pool_timeout_seconds=12,
        connect_timeout_seconds=4,
    )

    assert build_async_engine_kwargs(settings) == {
        "echo": True,
        "connect_args": {"connect_timeout": 4},
        "poolclass": InstrumentedAsyncAdaptedQueuePool,
        "pool_size": 7,
        "max_overflow": 11,
        "pool_pre_ping": False,
        "pool_recycle": 600,
        "pool_timeout": 12,
        "pool_use_lifo": True,
    }


def test_mysql_unpooled_engine_only_uses_connection_timeout() -> None:
    settings = DatabaseSettings(use_sqlite=False, connect_timeout_seconds=4)

    assert build_async_engine_kwargs(settings, use_null_pool=True) == {
        "echo": False,
        "connect_args": {"connect_timeout": 4},
        "poolclass": NullPool,
    }


def test_sqlite_engine_ignores_mysql_connection_policy() -> None:
    assert build_async_engine_kwargs(DatabaseSettings(use_sqlite=True, echo=True)) == {"echo": True}
