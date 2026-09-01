"""Shared Agent Runtime test fixtures."""

from __future__ import annotations

import pytest

from gewu_agent_runtime.identity import PrincipalRef, PrincipalType
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)


@pytest.fixture
def principal() -> PrincipalRef:
    return PrincipalRef(
        subscriber_id="subscriber-a",
        principal_id="user-a",
        principal_type=PrincipalType.USER,
    )


@pytest.fixture
def workspace() -> WorkspaceSession:
    return WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=InMemoryWorkspaceBackend(),
            )
        ]
    )
