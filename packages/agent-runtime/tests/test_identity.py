"""Neutral identity contract tests."""

from gewu_agent_runtime import PrincipalRef, PrincipalType


def test_principal_ref_contains_no_business_scope() -> None:
    principal = PrincipalRef(
        subscriber_id="subscriber-a",
        principal_id="user-42",
        principal_type=PrincipalType.USER,
    )

    assert principal.model_dump(mode="json") == {
        "subscriber_id": "subscriber-a",
        "principal_id": "user-42",
        "principal_type": "user",
    }
