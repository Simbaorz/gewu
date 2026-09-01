"""Static service API-key authentication for dedicated subscriber deployments."""

from __future__ import annotations

import hmac
from collections.abc import Sequence
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from gewu_agent_runtime import PrincipalType
from gewu_agent_runtime.host.authentication import AuthenticationRequest, VerifiedIdentity
from gewu_core import ApplicationError, ApplicationErrorKind, utc_now


class ServiceApiKeyRecord(BaseModel):
    """One configured service credential and its stable external identity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
    secret: SecretStr = Field(repr=False)
    issuer: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=512)
    expires_at: datetime | None = Field(default=None, repr=False)
    enabled: bool = True
    claims: dict[str, object] = Field(default_factory=dict, repr=False)


class ServiceApiKeyAuthenticator:
    """Verify ``key_id.secret`` credentials as service identities only."""

    def __init__(
        self,
        records: Sequence[ServiceApiKeyRecord],
        *,
        scheme: str = "api-key",
    ) -> None:
        normalized_scheme = scheme.strip().lower()
        if not normalized_scheme:
            raise ValueError("service API-key scheme is required")
        by_id = {record.key_id: record for record in records}
        if len(by_id) != len(records):
            raise ValueError("service API-key IDs must be unique")
        self._records = by_id
        self._scheme = normalized_scheme

    async def authenticate(self, request: AuthenticationRequest) -> VerifiedIdentity:
        if request.scheme.strip().lower() != self._scheme:
            raise _invalid_service_key()
        key_id, separator, supplied_secret = request.credential.get_secret_value().partition(".")
        record = self._records.get(key_id)
        if not separator or not supplied_secret or record is None:
            raise _invalid_service_key()
        if not hmac.compare_digest(record.secret.get_secret_value(), supplied_secret):
            raise _invalid_service_key()
        if not record.enabled or _is_expired(record.expires_at):
            raise _invalid_service_key()
        return VerifiedIdentity(
            issuer=record.issuer,
            subject=record.subject,
            principal_type=PrincipalType.SERVICE,
            claims=dict(record.claims),
            credential_id=record.key_id,
            expires_at=record.expires_at,
        )


def _is_expired(expires_at: datetime | None) -> bool:
    if expires_at is None:
        return False
    normalized = expires_at if expires_at.tzinfo is not None else expires_at.replace(tzinfo=UTC)
    return normalized.astimezone(UTC) <= utc_now().astimezone(UTC)


def _invalid_service_key() -> ApplicationError:
    return ApplicationError(
        ApplicationErrorKind.UNAUTHENTICATED,
        "Service API key is invalid or expired.",
    )
