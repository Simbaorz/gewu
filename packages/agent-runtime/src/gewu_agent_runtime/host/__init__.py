"""Host-side subscriber authentication and Runtime preparation boundaries."""

from gewu_agent_runtime.host.authentication import (
    AuthenticationRequest,
    DirectSubjectPrincipalMapper,
    PrincipalMapper,
    SubscriberAuthenticator,
    VerifiedIdentity,
    authenticate_principal,
)
from gewu_agent_runtime.host.external_auth import (
    HttpxOpaqueTokenAuthenticator,
    OpaqueTokenIntrospectionSettings,
)
from gewu_agent_runtime.host.provider import SubscriberRuntimeProvider, prepare_and_start_turn
from gewu_agent_runtime.host.service_keys import ServiceApiKeyAuthenticator, ServiceApiKeyRecord

__all__ = [
    "AuthenticationRequest",
    "DirectSubjectPrincipalMapper",
    "HttpxOpaqueTokenAuthenticator",
    "OpaqueTokenIntrospectionSettings",
    "PrincipalMapper",
    "ServiceApiKeyAuthenticator",
    "ServiceApiKeyRecord",
    "SubscriberAuthenticator",
    "SubscriberRuntimeProvider",
    "VerifiedIdentity",
    "authenticate_principal",
    "prepare_and_start_turn",
]
