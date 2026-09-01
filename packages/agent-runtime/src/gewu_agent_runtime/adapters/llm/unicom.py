"""China Unicom Open Service helpers."""

from __future__ import annotations

import hashlib
from typing import Any


def generate_unicom_token(params: dict[str, Any], app_secret: str) -> str:
    """Return the China Unicom Open Service request token."""

    sorted_items = sorted((key, value) for key, value in params.items() if key.upper() != "TOKEN")
    base_str = "".join(f"{key}{value}" for key, value in sorted_items) + app_secret
    return hashlib.md5(base_str.encode("utf-8")).hexdigest()  # noqa: S324
