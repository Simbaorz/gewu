"""Asynchronous and synchronous database URL behavior."""

from pathlib import Path

import pytest

from gewu_core.database import DatabaseSettings, resolve_async_db_url, to_sync_db_url


def test_configured_async_url_is_preserved_for_mysql() -> None:
    settings = DatabaseSettings(
        enabled=True,
        use_sqlite=False,
        url="mysql+aiomysql://u:p@127.0.0.1:3306/subscriber",
    )

    assert (
        resolve_async_db_url(settings, "/tmp/project")
        == "mysql+aiomysql://u:p@127.0.0.1:3306/subscriber"
    )


def test_sqlite_uses_project_local_configured_file(tmp_path: Path) -> None:
    settings = DatabaseSettings(
        enabled=True,
        use_sqlite=True,
        sqlite_file_name="subscriber.db",
        url="mysql+aiomysql://ignored",
    )

    assert resolve_async_db_url(settings, tmp_path) == (
        f"sqlite+aiosqlite:///{tmp_path / 'subscriber.db'}"
    )


@pytest.mark.parametrize("file_name", [" ", ".", "..", "nested/db", "nested\\db"])
def test_sqlite_file_name_cannot_escape_project_directory(file_name: str) -> None:
    with pytest.raises(ValueError, match="sqlite_file_name"):
        DatabaseSettings(sqlite_file_name=file_name)


def test_sync_url_converts_supported_async_drivers() -> None:
    assert (
        to_sync_db_url("mysql+aiomysql://u:p@127.0.0.1:3306/subscriber")
        == "mysql+pymysql://u:p@127.0.0.1:3306/subscriber"
    )
    assert (
        to_sync_db_url("sqlite+aiosqlite:////tmp/subscriber.db") == "sqlite:////tmp/subscriber.db"
    )
