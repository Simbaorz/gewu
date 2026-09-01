"""Business-neutral SQLAlchemy value and timestamp primitives."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.types import TypeDecorator


class TimezoneAwareDateTime(TypeDecorator[datetime]):
    """Persist UTC values and restore timezone awareness on read."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        del dialect
        if value is None:
            return None
        aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
        return aware.astimezone(UTC).replace(tzinfo=None)

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        del dialect
        if value is None or value.tzinfo is not None:
            return value.astimezone(UTC) if value is not None else None
        return value.replace(tzinfo=UTC)


def db_now() -> datetime:
    """Return current UTC time for database timestamps."""
    return datetime.now(UTC)
