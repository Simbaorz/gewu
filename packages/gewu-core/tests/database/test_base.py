"""Business-neutral SQLAlchemy value behavior."""

from datetime import UTC, datetime

from sqlalchemy.dialects import sqlite

from gewu_core.database import TimezoneAwareDateTime


def test_timezone_column_normalizes_bind_and_result_values_to_utc() -> None:
    column_type = TimezoneAwareDateTime()
    dialect = sqlite.dialect()
    aware = datetime(2026, 8, 1, tzinfo=UTC)

    bound = column_type.process_bind_param(aware, dialect)
    restored = column_type.process_result_value(bound, dialect)

    assert bound == aware.replace(tzinfo=None)
    assert restored == aware
