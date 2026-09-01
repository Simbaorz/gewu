"""Dedicated deployments can authenticate stable service principals with an SK."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import SecretStr

from gewu_agent_runtime import PrincipalRef, PrincipalType
from gewu_agent_runtime.host import (
    AuthenticationRequest,
    DirectSubjectPrincipalMapper,
    ServiceApiKeyAuthenticator,
    ServiceApiKeyRecord,
    authenticate_principal,
)
from gewu_core import ApplicationError, ApplicationErrorKind


async def test_service_api_key_authenticates_only_its_configured_service_subject() -> None:
    record = _record()
    principal = await authenticate_principal(
        AuthenticationRequest(
            scheme="api-key",
            credential=SecretStr("integration-a.service-secret"),
        ),
        authenticator=ServiceApiKeyAuthenticator((record,)),
        mapper=DirectSubjectPrincipalMapper(
            "subscriber-a",
            allowed_issuers=("gewu://subscriber-a/service-keys",),
        ),
    )

    assert principal == PrincipalRef(
        subscriber_id="subscriber-a",
        principal_id="integration-service",
        principal_type=PrincipalType.SERVICE,
    )


@pytest.mark.parametrize(
    "credential_request",
    [
        AuthenticationRequest(
            scheme="bearer", credential=SecretStr("integration-a.service-secret")
        ),
        AuthenticationRequest(scheme="api-key", credential=SecretStr("unknown.service-secret")),
        AuthenticationRequest(scheme="api-key", credential=SecretStr("integration-a.wrong")),
    ],
)
async def test_service_api_key_rejects_every_unverified_credential(
    credential_request: AuthenticationRequest,
) -> None:
    with pytest.raises(ApplicationError) as error:
        await ServiceApiKeyAuthenticator((_record(),)).authenticate(credential_request)

    assert error.value.kind is ApplicationErrorKind.UNAUTHENTICATED
    assert error.value.detail == "Service API key is invalid or expired."


async def test_service_api_key_rejects_disabled_or_expired_records() -> None:
    expired = _record(expires_at=datetime.now(UTC) - timedelta(seconds=1))
    disabled = _record(key_id="integration-b", enabled=False)
    authenticator = ServiceApiKeyAuthenticator((expired, disabled))

    for value in ("integration-a.service-secret", "integration-b.service-secret"):
        with pytest.raises(ApplicationError) as error:
            await authenticator.authenticate(
                AuthenticationRequest(scheme="api-key", credential=SecretStr(value))
            )
        assert error.value.kind is ApplicationErrorKind.UNAUTHENTICATED


def test_service_api_key_values_hide_secrets_in_repr() -> None:
    record = _record(claims={"private": "claim-secret"})

    assert "service-secret" not in repr(record)
    assert "claim-secret" not in repr(record)


def _record(**updates: object) -> ServiceApiKeyRecord:
    values: dict[str, object] = {
        "key_id": "integration-a",
        "secret": SecretStr("service-secret"),
        "issuer": "gewu://subscriber-a/service-keys",
        "subject": "integration-service",
    }
    values.update(updates)
    return ServiceApiKeyRecord.model_validate(values)
