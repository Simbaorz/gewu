"""System-generated opaque identifiers."""

from __future__ import annotations

from uuid import uuid4

ENTITY_ID_COLUMN_LENGTH = 64
ENTITY_ID_LENGTH = 32


def new_entity_id() -> str:
    """Return a 32-character lowercase UUID4 hex entity ID."""

    return uuid4().hex


def new_uuid4_id() -> str:
    """Return a 32-character lowercase UUID4 hex runtime ID."""

    return uuid4().hex


def new_id() -> str:
    """Return a UUID4 hex ID for callers without an entity distinction."""

    return new_uuid4_id()
