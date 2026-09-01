"""Typed relational database settings."""

from pydantic import Field, field_validator

from gewu_core.config import SettingsModel


class DatabaseSettings(SettingsModel):
    """Database and native asynchronous pool configuration."""

    enabled: bool = True
    use_sqlite: bool = True
    sqlite_file_name: str = Field(default="database.db", min_length=1)
    url: str = Field(default="", exclude=True, repr=False)
    echo: bool = False
    pool_size: int = Field(default=10, ge=1)
    max_overflow: int = Field(default=10, ge=0)
    pool_pre_ping: bool = True
    pool_recycle_seconds: int = Field(default=300, ge=1)
    pool_timeout_seconds: int = Field(default=10, ge=1)
    connect_timeout_seconds: int = Field(default=5, ge=1)

    @field_validator("sqlite_file_name")
    @classmethod
    def validate_sqlite_file_name(cls, value: str) -> str:
        """Keep the configured SQLite database inside the project directory."""

        normalized = value.strip()
        if not normalized or normalized in {".", ".."} or "/" in normalized or "\\" in normalized:
            raise ValueError("sqlite_file_name must be a file name, not a path.")
        return normalized
