"""Timezone-aware clock helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now() -> datetime:
    """Return current UTC time as an aware datetime."""

    return datetime.now(UTC)
