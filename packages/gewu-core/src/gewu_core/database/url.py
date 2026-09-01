"""Database URL resolution policy."""

from __future__ import annotations

from pathlib import Path

from sqlalchemy.engine import make_url

from gewu_core.database.settings import DatabaseSettings


def resolve_async_db_url(settings: DatabaseSettings, project_home: str | Path) -> str:
    """Resolve the async SQLAlchemy URL from database configuration."""
    if settings.use_sqlite:
        project_dir = Path(project_home).expanduser().resolve()
        project_dir.mkdir(parents=True, exist_ok=True)
        return f"sqlite+aiosqlite:///{project_dir / settings.sqlite_file_name}"
    return settings.url


def to_sync_db_url(async_url: str) -> str:
    """Convert supported asynchronous drivers into synchronous equivalents."""
    parsed = make_url(async_url)
    if parsed.drivername == "mysql+aiomysql":
        parsed = parsed.set(drivername="mysql+pymysql")
    elif parsed.drivername == "sqlite+aiosqlite":
        parsed = parsed.set(drivername="sqlite")
    return parsed.render_as_string(hide_password=False)
