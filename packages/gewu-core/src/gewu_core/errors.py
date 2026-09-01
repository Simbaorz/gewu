"""Transport-neutral application failure categories shared by process apps."""

from __future__ import annotations

from enum import StrEnum


class ApplicationErrorKind(StrEnum):
    """Stable categories for expected application failures."""

    BAD_REQUEST = "bad_request"
    UNAUTHENTICATED = "unauthenticated"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    INVALID_INPUT = "invalid_input"
    INTERNAL = "internal"
    UNAVAILABLE = "unavailable"
    TIMEOUT = "timeout"


class ApplicationError(Exception):
    """Expected use-case failure translated by an inbound adapter."""

    def __init__(self, kind: ApplicationErrorKind, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


class CommitOutcomeUnknownError(RuntimeError):
    """Raised when a failed commit may already have reached durable storage."""
