"""Tests for shared pure utility primitives."""

from __future__ import annotations

from datetime import UTC

import pytest

from gewu_core.ids import ENTITY_ID_LENGTH, new_entity_id, new_id, new_uuid4_id
from gewu_core.size import parse_size_bytes
from gewu_core.time import utc_now


def test_ids_are_lowercase_uuid4_hex() -> None:
    values = {new_entity_id(), new_uuid4_id(), new_id()}

    assert len(values) == 3
    assert all(len(value) == ENTITY_ID_LENGTH for value in values)
    assert all(value == value.lower() for value in values)
    assert all(int(value, 16) >= 0 for value in values)


def test_utc_now_is_timezone_aware() -> None:
    current = utc_now()

    assert current.tzinfo is UTC


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1024, 1024),
        (1024.0, 1024),
        ("512KB", 512 * 1024),
        ("20 MB", 20 * 1024 * 1024),
        ("1GB", 1024 * 1024 * 1024),
    ],
)
def test_parse_size_bytes(value: object, expected: int) -> None:
    assert parse_size_bytes(value) == expected


@pytest.mark.parametrize("value", [True, 1.5, "1TB", object()])
def test_parse_size_bytes_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        parse_size_bytes(value)
