"""Fail-closed HTTPS opaque-token authentication adapter."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx
from pydantic import Field, SecretStr, model_validator

from gewu_agent_runtime import PrincipalType
from gewu_agent_runtime.host.authentication import (
    AuthenticationRequest,
    VerifiedIdentity,
    string_claims,
)
from gewu_core import ApplicationError, ApplicationErrorKind, SettingsModel, utc_now

IdentityClock = Callable[[], datetime]


class OpaqueTokenIntrospectionSettings(SettingsModel):
    """Connection and trust policy for one external token introspection service."""

    endpoint: str
    expected_issuer: str = Field(min_length=1, max_length=512)
    timeout_seconds: float = Field(default=3.0, gt=0, le=30)
    max_response_bytes: int = Field(default=64 * 1024, ge=1024, le=1024 * 1024)
    require_expiration: bool = True
    default_principal_type: PrincipalType = PrincipalType.USER
    client_id: str = Field(default="", max_length=512)
    client_secret: SecretStr = Field(default_factory=lambda: SecretStr(""), repr=False)

    @model_validator(mode="after")
    def validate_endpoint_and_client_credentials(self) -> OpaqueTokenIntrospectionSettings:
        parts = urlsplit(self.endpoint)
        if (
            parts.scheme.lower() != "https"
            or not parts.hostname
            or parts.username is not None
            or parts.password is not None
            or parts.fragment
        ):
            raise ValueError("external authentication endpoint must be a credential-free HTTPS URL")
        if bool(self.client_id) != bool(self.client_secret.get_secret_value()):
            raise ValueError("external authentication client_id and client_secret must be paired")
        return self


class HttpxOpaqueTokenAuthenticator:
    """Authenticate bearer credentials with an RFC 7662-style introspection endpoint."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: OpaqueTokenIntrospectionSettings,
        *,
        clock: IdentityClock = utc_now,
    ) -> None:
        self._client = client
        self._settings = settings
        self._clock = clock

    async def authenticate(self, request: AuthenticationRequest) -> VerifiedIdentity:
        if request.scheme.strip().lower() != "bearer":
            raise _unauthenticated("Unsupported authentication scheme.")
        credential = request.credential.get_secret_value()
        if not credential:
            raise _unauthenticated("Credentials are missing.")

        auth = None
        if self._settings.client_id:
            auth = (
                self._settings.client_id,
                self._settings.client_secret.get_secret_value(),
            )
        try:
            async with self._client.stream(
                "POST",
                self._settings.endpoint,
                data={"token": credential},
                headers={"Accept": "application/json"},
                auth=auth,
                timeout=self._settings.timeout_seconds,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise _unavailable()
                body = await _read_bounded_body(
                    response,
                    max_bytes=self._settings.max_response_bytes,
                )
        except ApplicationError:
            raise
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            raise _unavailable() from exc

        payload = _decode_payload(body)
        return self._verified_identity(payload)

    def _verified_identity(self, payload: Mapping[str, object]) -> VerifiedIdentity:
        active = payload.get("active")
        if active is False:
            raise _unauthenticated("Credentials are invalid or expired.")
        if active is not True:
            raise _invalid_response()

        subject = _required_string(payload, "sub")
        issuer = _required_string(payload, "iss")
        if issuer != self._settings.expected_issuer:
            raise _unauthenticated("Credentials were issued by an untrusted authority.")

        expires_at = _expiration(payload.get("exp"), required=self._settings.require_expiration)
        now = self._clock()
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        if expires_at is not None and expires_at <= now.astimezone(UTC):
            raise _unauthenticated("Credentials are invalid or expired.")

        raw_type = payload.get("principal_type", self._settings.default_principal_type.value)
        try:
            principal_type = PrincipalType(str(raw_type))
        except ValueError as exc:
            raise _invalid_response() from exc

        credential_id = payload.get("jti", "")
        if not isinstance(credential_id, str):
            raise _invalid_response()
        return VerifiedIdentity(
            issuer=issuer,
            subject=subject,
            principal_type=principal_type,
            claims=dict(string_claims(payload.get("claims"))),
            credential_id=credential_id,
            expires_at=expires_at,
        )


async def _read_bounded_body(response: httpx.Response, *, max_bytes: int) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        if len(body) + len(chunk) > max_bytes:
            raise _invalid_response()
        body.extend(chunk)
    return bytes(body)


def _decode_payload(body: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _invalid_response() from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise _invalid_response()
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value.strip():
        raise _invalid_response()
    return value.strip()


def _expiration(value: object, *, required: bool) -> datetime | None:
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise _invalid_response()
    try:
        return datetime.fromtimestamp(float(value), tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise _invalid_response() from exc


def _unauthenticated(detail: str) -> ApplicationError:
    return ApplicationError(ApplicationErrorKind.UNAUTHENTICATED, detail)


def _invalid_response() -> ApplicationError:
    return ApplicationError(
        ApplicationErrorKind.UNAVAILABLE,
        "Authentication service returned an invalid response.",
    )


def _unavailable() -> ApplicationError:
    return ApplicationError(
        ApplicationErrorKind.UNAVAILABLE,
        "Authentication service is unavailable.",
    )
