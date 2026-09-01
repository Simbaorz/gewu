"""HTTP mapping for transport-neutral application failures."""

from __future__ import annotations

from gewu_core.errors import ApplicationError, ApplicationErrorKind

_HTTP_STATUS_BY_KIND: dict[ApplicationErrorKind, int] = {
    ApplicationErrorKind.BAD_REQUEST: 400,
    ApplicationErrorKind.UNAUTHENTICATED: 401,
    ApplicationErrorKind.FORBIDDEN: 403,
    ApplicationErrorKind.NOT_FOUND: 404,
    ApplicationErrorKind.CONFLICT: 409,
    ApplicationErrorKind.PAYLOAD_TOO_LARGE: 413,
    ApplicationErrorKind.INVALID_INPUT: 422,
    ApplicationErrorKind.INTERNAL: 500,
    ApplicationErrorKind.UNAVAILABLE: 503,
    ApplicationErrorKind.TIMEOUT: 504,
}


def application_error_status_code(error: ApplicationError) -> int:
    """Return the public HTTP status mapped from an application failure."""
    return _HTTP_STATUS_BY_KIND[error.kind]
