"""Lifecycle ownership for one asynchronous SQLAlchemy database pool."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from gewu_core.database.engine import build_async_engine_kwargs
from gewu_core.database.settings import DatabaseSettings
from gewu_core.database.url import resolve_async_db_url


class DatabaseRuntime:
    """Own one process-local asynchronous Engine and session factory."""

    def __init__(self, settings: DatabaseSettings, project_home: str | Path) -> None:
        self.settings = settings
        self.project_home = Path(project_home)
        self.engine: AsyncEngine | None = None
        self.sessions: async_sessionmaker[AsyncSession] | None = None
        self._started = False

    async def startup(self) -> None:
        """Create the configured Engine and session factory once."""

        if self._started:
            return
        if not self.settings.enabled:
            raise RuntimeError("db.enabled must be true.")
        url = resolve_async_db_url(self.settings, self.project_home)
        kwargs = cast(dict[str, Any], build_async_engine_kwargs(self.settings))
        engine = create_async_engine(url, **kwargs)
        self.engine = engine
        self.sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._started = True

    async def shutdown(self) -> None:
        """Dispose the owned Engine and clear runtime resources."""

        self._started = False
        engine = self.engine
        self.engine = None
        self.sessions = None
        if engine is not None:
            await engine.dispose()

    @property
    def started(self) -> bool:
        return self._started
