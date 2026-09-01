"""Tests for explicit per-turn Tool Set composition."""

from __future__ import annotations

import pytest

from gewu_agent_runtime.tools import ToolContext, ToolResult, ToolSet, ToolSetBuilder, tool
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)


@tool(description="Read a value.", allow_parallel=True)
def read_value() -> ToolResult:
    return ToolResult(output={"value": "read"})


@tool(description="Write a value.", writes_workspace=True)
def write_value() -> ToolResult:
    return ToolResult(output={"value": "written"})


def test_builder_composes_named_versioned_tool_set() -> None:
    tool_set = (
        ToolSetBuilder(name="subscriber-main", version="v1")
        .add(read_value)
        .extend((write_value,))
        .build()
    )

    assert tool_set.name == "subscriber-main"
    assert tool_set.version == "v1"
    assert tuple(tool.name for tool in tool_set.all()) == ("read_value", "write_value")
    assert tool_set.get("read_value") is read_value
    assert read_value.allow_parallel is True


def test_read_only_tool_set_removes_workspace_mutations() -> None:
    source = ToolSet((read_value, write_value), name="main", version="v1")

    selected = source.read_only()

    assert selected.name == "main:read-only"
    assert selected.version == "v1"
    assert selected.all() == (read_value,)


def test_duplicate_model_visible_names_are_rejected() -> None:
    duplicate = read_value.model_copy(update={"description": "duplicate"})

    with pytest.raises(ValueError, match="unique"):
        ToolSet((read_value, duplicate))


async def test_tool_failure_returns_unexpected_exception_body_like_subscriber() -> None:
    @tool(description="Fail unexpectedly.")
    def fail() -> ToolResult:
        raise RuntimeError("tool-private-secret")

    result = await fail.execute({}, _context())

    assert result.is_error is True
    assert result.output_payload() == {"error": "Tool execution error: tool-private-secret"}


def _context() -> ToolContext:
    return ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=WorkspaceSession(
            (
                WorkspaceMount(
                    mount_id="private",
                    mount_path="/workspace/private",
                    access_mode=AccessMode.READ_WRITE,
                    backend=InMemoryWorkspaceBackend(),
                ),
            ),
            default_root="/workspace/private",
        ),
    )
