"""JSON helpers for public API datetime values."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import Any, ClassVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi.responses import JSONResponse

DATETIME_FIELD_SUFFIXES = ("_at", "_time")
API_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_TIMEZONE = "Asia/Shanghai"


class LocalTimeJSONResponse(JSONResponse):
    """JSON response that emits API datetime fields as local wall-clock strings."""

    response_timezone: ClassVar[tzinfo] = timezone(timedelta(hours=8), DEFAULT_TIMEZONE)

    def render(self, content: Any) -> bytes:
        """Render content after normalizing known datetime fields."""
        return super().render(normalize_api_datetimes(content, self.response_timezone))


def response_class_for_timezone(timezone_name: str) -> type[LocalTimeJSONResponse]:
    """Create an isolated response class bound to one bootstrap timezone."""
    resolved_timezone = resolve_timezone(timezone_name)

    class ConfiguredLocalTimeJSONResponse(LocalTimeJSONResponse):
        response_timezone = resolved_timezone

    return ConfiguredLocalTimeJSONResponse


def normalize_api_datetimes(content: Any, target_timezone: tzinfo) -> Any:
    """Normalize public datetime fields recursively into local wall-clock strings."""
    if isinstance(content, dict):
        return {
            key: _normalize_value(key, value, target_timezone) for key, value in content.items()
        }
    if isinstance(content, list):
        return [normalize_api_datetimes(item, target_timezone) for item in content]
    if isinstance(content, tuple):
        return [normalize_api_datetimes(item, target_timezone) for item in content]
    return content


def resolve_timezone(timezone_name: str) -> tzinfo:
    """Resolve the bootstrap timezone with safe fallbacks."""
    normalized = timezone_name.strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(normalized)
    except ZoneInfoNotFoundError:
        if normalized in {"Asia/Shanghai", "Asia/Chongqing", "Asia/Harbin", "PRC"}:
            return timezone(timedelta(hours=8), DEFAULT_TIMEZONE)
        if normalized == "UTC":
            return UTC
        return timezone(timedelta(hours=8), DEFAULT_TIMEZONE)


def _normalize_value(key: str, value: Any, target_timezone: tzinfo) -> Any:
    if key.endswith(DATETIME_FIELD_SUFFIXES):
        if isinstance(value, datetime):
            return _datetime_to_local_string(value, target_timezone)
        if isinstance(value, str):
            return _datetime_string_to_local_string(value, target_timezone)
    return normalize_api_datetimes(value, target_timezone)


def _datetime_to_local_string(value: datetime, target_timezone: tzinfo) -> str:
    aware = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return aware.astimezone(target_timezone).strftime(API_DATETIME_FORMAT)


def _datetime_string_to_local_string(value: str, target_timezone: tzinfo) -> str:
    normalized = value.strip()
    if not normalized:
        return value
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        return value
    return _datetime_to_local_string(parsed, target_timezone)
