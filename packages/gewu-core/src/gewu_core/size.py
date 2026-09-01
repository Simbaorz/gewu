"""Helpers for parsing byte-size configuration values."""

from __future__ import annotations

import re
from typing import Any

SIZE_PATTERN = re.compile(r"^\s*(\d+)\s*(B|KB|MB|GB)?\s*$", re.IGNORECASE)
SIZE_MULTIPLIERS = {
    "B": 1,
    "KB": 1024,
    "MB": 1024 * 1024,
    "GB": 1024 * 1024 * 1024,
}


def parse_size_bytes(value: Any) -> int:
    """Parse a numeric or B/KB/MB/GB byte-size value into bytes."""

    if isinstance(value, bool):
        raise ValueError("byte size cannot be a boolean")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError("byte size must resolve to a whole number of bytes")
        return int(value)
    if not isinstance(value, str):
        raise ValueError("byte size must be a number or a string with B/KB/MB/GB suffix")

    match = SIZE_PATTERN.match(value)
    if not match:
        raise ValueError("byte size must use a B, KB, MB, or GB suffix")
    number = int(match.group(1))
    unit = (match.group(2) or "B").upper()
    return number * SIZE_MULTIPLIERS[unit]
