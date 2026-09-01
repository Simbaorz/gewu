"""Generation configuration normalization shared by provider adapters."""

from __future__ import annotations

from typing import Any


def generation_kwargs(config: dict[str, Any], allowed_keys: set[str]) -> dict[str, Any]:
    """Return non-empty supported arguments with integer bounds applied."""

    cleaned: dict[str, Any] = {}
    for key, value in config.items():
        if key not in allowed_keys or value is None or value == "":
            continue
        if key in {"max_tokens", "seed"}:
            integer_value = _positive_int_value(value)
            if integer_value is None:
                continue
            cleaned[key] = integer_value
            continue
        cleaned[key] = value
    return cleaned


def stream_enabled(config: dict[str, Any], support_stream: bool) -> bool:
    """Return whether streaming is enabled by capability and runtime configuration."""

    if not support_stream:
        return False
    value = config.get("stream", True)
    return value if isinstance(value, bool) else True


def _positive_int_value(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 1 else None
