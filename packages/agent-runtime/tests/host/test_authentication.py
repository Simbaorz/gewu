"""External subscriber authentication fails closed and maps only verified identity."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from gewu_agent_runtime import PrincipalRef, PrincipalType
from gewu_agent_runtime.host import (
    AuthenticationRequest,
    DirectSubjectPrincipalMapper,
    HttpxOpaqueTokenAuthenticator,
    OpaqueTokenIntrospectionSettings,
    VerifiedIdentity,
    authenticate_principal,
)
from gewu_core import ApplicationError, ApplicationErrorKind

NOW = datetime(2026, 8, 2, tzinfo=UTC)


def test_external_auth_configuration_requires_https_and_paired_client_credentials() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        _settings(endpoint="http://auth.example/introspect")
    with pytest.raises(ValidationError, match="must be paired"):
        _settings(client_id="host-only", client_secret=SecretStr(""))


async def test_external_auth_maps_verified_subject_without_trusting_request_identity() -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "active": True,
                "iss": "https://identity.subscriber-a.example",
                "sub": "external-user-42",
                "principal_type": "user",
                "exp": int((NOW + timedelta(minutes=5)).timestamp()),
                "jti": "credential-1",
                "claims": {"department": "finance"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        authenticator = HttpxOpaqueTokenAuthenticator(client, _settings(), clock=lambda: NOW)
        principal = await authenticate_principal(
            AuthenticationRequest(
                credential=SecretStr("opaque-secret"),
                request_id="request-1",
            ),
            authenticator=authenticator,
            mapper=DirectSubjectPrincipalMapper(
                "subscriber-a",
                allowed_issuers=("https://identity.subscriber-a.example",),
            ),
        )

    assert principal == PrincipalRef(
        subscriber_id="subscriber-a",
        principal_id="external-user-42",
        principal_type=PrincipalType.USER,
    )
    assert len(observed) == 1
    assert observed[0].url == httpx.URL("https://auth.subscriber-a.example/introspect")
    assert observed[0].headers["authorization"].startswith("Basic ")
    assert observed[0].content == b"token=opaque-secret"


def test_authentication_request_rejects_caller_supplied_identity_fields() -> None:
    with pytest.raises(ValidationError):
        AuthenticationRequest.model_validate(
            {
                "credential": "opaque-secret",
                "subscriber_id": "forged-subscriber",
                "principal_id": "forged-user",
            }
        )


@pytest.mark.parametrize(
    ("payload", "kind"),
    [
        ({"active": False}, ApplicationErrorKind.UNAUTHENTICATED),
        (
            {
                "active": True,
                "iss": "https://identity.subscriber-a.example",
                "sub": "user-1",
                "exp": int((NOW - timedelta(seconds=1)).timestamp()),
            },
            ApplicationErrorKind.UNAUTHENTICATED,
        ),
        (
            {
                "active": True,
                "iss": "https://identity.other.example",
                "sub": "user-1",
                "exp": int((NOW + timedelta(minutes=1)).timestamp()),
            },
            ApplicationErrorKind.UNAUTHENTICATED,
        ),
        ({"active": True, "sub": "user-1"}, ApplicationErrorKind.UNAVAILABLE),
    ],
)
async def test_external_auth_rejects_invalid_or_unverifiable_identity(
    payload: dict[str, object],
    kind: ApplicationErrorKind,
) -> None:
    async with _client(
        lambda request: httpx.Response(200, request=request, json=payload)
    ) as client:
        authenticator = HttpxOpaqueTokenAuthenticator(client, _settings(), clock=lambda: NOW)
        with pytest.raises(ApplicationError) as error:
            await authenticator.authenticate(_request())

    assert error.value.kind is kind


@pytest.mark.parametrize("failure", ["timeout", "status", "malformed", "oversized"])
async def test_external_auth_service_failures_are_unavailable(failure: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if failure == "timeout":
            raise httpx.ReadTimeout("timeout", request=request)
        if failure == "status":
            return httpx.Response(502, request=request)
        if failure == "malformed":
            return httpx.Response(200, request=request, content=b"not-json")
        return httpx.Response(200, request=request, content=b"x" * 1025)

    settings = _settings(max_response_bytes=1024)
    async with _client(handler) as client:
        authenticator = HttpxOpaqueTokenAuthenticator(client, settings, clock=lambda: NOW)
        with pytest.raises(ApplicationError) as error:
            await authenticator.authenticate(_request())

    assert error.value.kind is ApplicationErrorKind.UNAVAILABLE


async def test_verified_but_unmapped_identity_is_forbidden() -> None:
    identity = VerifiedIdentity(
        issuer="https://identity.subscriber-b.example",
        subject="user-1",
        principal_type=PrincipalType.USER,
    )

    class Authenticator:
        async def authenticate(self, request: AuthenticationRequest) -> VerifiedIdentity:
            del request
            return identity

    with pytest.raises(ApplicationError) as error:
        await authenticate_principal(
            _request(),
            authenticator=Authenticator(),
            mapper=DirectSubjectPrincipalMapper(
                "subscriber-a",
                allowed_issuers=("https://identity.subscriber-a.example",),
            ),
        )

    assert error.value.kind is ApplicationErrorKind.FORBIDDEN


def test_authentication_values_do_not_reveal_credentials_or_claims_in_repr() -> None:
    request = _request()
    identity = VerifiedIdentity(
        issuer="https://identity.subscriber-a.example",
        subject="user-1",
        principal_type=PrincipalType.USER,
        claims={"secret": "claim-secret"},
        credential_id="credential-secret",
    )

    assert "opaque-secret" not in repr(request)
    assert "claim-secret" not in repr(identity)
    assert "credential-secret" not in repr(identity)


def _request() -> AuthenticationRequest:
    return AuthenticationRequest(credential=SecretStr("opaque-secret"))


def _settings(**updates: object) -> OpaqueTokenIntrospectionSettings:
    values: dict[str, object] = {
        "endpoint": "https://auth.subscriber-a.example/introspect",
        "expected_issuer": "https://identity.subscriber-a.example",
        "client_id": "gewu-host",
        "client_secret": SecretStr("client-secret"),
    }
    values.update(updates)
    return OpaqueTokenIntrospectionSettings.model_validate(values)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))
