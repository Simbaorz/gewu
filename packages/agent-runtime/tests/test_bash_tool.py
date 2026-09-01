"""Bash tool contract tests independent of a subscriber command policy."""

from __future__ import annotations

import hashlib
import json

from gewu_agent_runtime.builtins import BashOutput, bash
from gewu_agent_runtime.tools import ToolContext, ToolResult
from gewu_agent_runtime.workspace import WorkspaceSession


def _contract_digest() -> str:
    payload = {
        "description": bash.description,
        "input_schema": bash.input_schema,
        "category": bash.category,
        "writes": bash.writes_workspace,
        "allow_parallel": bash.allow_parallel,
        "retry": bash.retry_on_failure,
        "max_retries": bash.max_retries,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_bash_model_contract_exactly_matches_subscriber() -> None:
    assert _contract_digest() == "f12cfa59408e5d878f890ff55f25af61541f6215c18bd10860a349eb7957cc94"


async def test_bash_uses_host_executor_and_configured_default_timeout(
    workspace: WorkspaceSession,
) -> None:
    calls: list[tuple[str, int]] = []

    async def execute(command: str, timeout_ms: int) -> ToolResult:
        calls.append((command, timeout_ms))
        return ToolResult(output=BashOutput(stdout="ok"))

    runtime = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
        bash_executor=execute,
        bash_default_timeout_ms=4_321,
    )

    default = await bash.execute({"command": "date"}, runtime)
    capped = await bash.execute({"command": "date", "timeout": "900000"}, runtime)

    assert default.output_payload()["stdout"] == "ok"
    assert capped.is_error is False
    assert calls == [("date", 4_321), ("date", 600_000)]


async def test_bash_rejects_empty_command_and_missing_executor(
    workspace: WorkspaceSession,
) -> None:
    runtime = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )

    empty = await bash.execute({"command": "  "}, runtime)
    missing = await bash.execute({"command": "date"}, runtime)

    assert empty.output_payload()["error"] == "Command cannot be empty"
    assert missing.output_payload()["error"] == "No bash executor configured."
    assert empty.is_error is True
    assert missing.is_error is True
