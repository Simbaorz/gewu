"""Business-neutral authentication and principal-mapping contracts."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from gewu_agent_runtime import PrincipalRef, PrincipalType
from gewu_core import ApplicationError, ApplicationErrorKind


class AuthenticationRequest(BaseModel):
    """One untrusted credential presented to a subscriber host."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheme: str = Field(default="bearer", min_length=1, max_length=32)
    credential: SecretStr
    request_id: str = Field(default="", max_length=128)
    client_ip: str = Field(default="", max_length=128)


class VerifiedIdentity(BaseModel):
    """Identity claims accepted by one configured subscriber authenticator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    issuer: str = Field(min_length=1, max_length=512)
    subject: str = Field(min_length=1, max_length=512)
    principal_type: PrincipalType
    claims: dict[str, object] = Field(default_factory=dict, repr=False)
    credential_id: str = Field(default="", max_length=512, repr=False)
    expires_at: datetime | None = Field(default=None, repr=False)


class SubscriberAuthenticator(Protocol):
    """Verify one credential without assigning subscriber business permissions."""

    async def authenticate(self, request: AuthenticationRequest) -> VerifiedIdentity:
        """Return a verified identity or raise a categorized application error."""


class PrincipalMapper(Protocol):
    """Map a verified external identity to one stable Runtime principal."""

    async def map_identity(self, identity: VerifiedIdentity) -> PrincipalRef | None:
        """Return a principal, or none when the verified identity is not onboarded."""


class DirectSubjectPrincipalMapper:
    """Map trusted subjects directly for one dedicated subscriber deployment."""

    def __init__(self, subscriber_id: str, *, allowed_issuers: tuple[str, ...]) -> None:
        normalized_subscriber = subscriber_id.strip()
        normalized_issuers = tuple(value.strip() for value in allowed_issuers if value.strip())
        if not normalized_subscriber:
            raise ValueError("subscriber_id is required")
        if not normalized_issuers:
            raise ValueError("allowed_issuers is required")
        self._subscriber_id = normalized_subscriber
        self._allowed_issuers = frozenset(normalized_issuers)

    async def map_identity(self, identity: VerifiedIdentity) -> PrincipalRef | None:
        if identity.issuer not in self._allowed_issuers:
            return None
        return PrincipalRef(
            subscriber_id=self._subscriber_id,
            principal_id=identity.subject,
            principal_type=identity.principal_type,
        )


async def authenticate_principal(
    request: AuthenticationRequest,
    *,
    authenticator: SubscriberAuthenticator,
    mapper: PrincipalMapper,
) -> PrincipalRef:
    """Authenticate and map one caller without trusting caller-supplied identity fields."""

    identity = await authenticator.authenticate(request)
    principal = await mapper.map_identity(identity)
    if principal is None:
        raise ApplicationError(
            ApplicationErrorKind.FORBIDDEN,
            "Authenticated identity is not authorized for this subscriber.",
        )
    if principal.principal_type is not identity.principal_type:
        raise ApplicationError(
            ApplicationErrorKind.FORBIDDEN,
            "Authenticated identity type does not match the mapped principal.",
        )
    return principal


def string_claims(value: object) -> Mapping[str, object]:
    """Validate the optional structured claims returned by an authenticator."""

    if value is None:
        return {}
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ApplicationError(
            ApplicationErrorKind.UNAVAILABLE,
            "Authentication service returned an invalid response.",
        )
    return value
