"""Host-bound Bash tool contract."""

from __future__ import annotations

import inspect

from pydantic import BaseModel

from gewu_agent_runtime.tools import ToolContext, ToolResult, tool

MAX_TIMEOUT_MS = 600_000


class BashOutput(BaseModel):
    """Normalized output produced by the reference Bash tool."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    interrupted: bool = False
    error: str = ""


@tool(name="bash", category="execution", writes_workspace=True)
async def bash(
    command: str,
    description: str = "",  # noqa
    timeout: int | None = None,  # noqa: ASYNC109
    *,
    runtime: ToolContext,  # noqa
) -> ToolResult:
    """Execute an approved local command in the virtual workspace.

    Use this only when a dedicated tool cannot do the job.

    Important:
    - File search: use `glob`.
    - Content search: use `grep`.
    - Read files: use `read`.
    - Edit files: use `edit`.
    - Create or fully rewrite files: use `write`.
    - Append to files: use `append`.

    Keep commands non-interactive. Do not use shell redirection to write files.
    Use the `delete` tool instead of `rm`.

    Args:
        command: Shell command to execute. Use explicit paths when possible and
            quote paths containing spaces.
        description: Short active-voice summary of what the command does, such
            as `Run unit tests` or `Show working tree status`.
        timeout: Optional timeout in milliseconds. Defaults to the configured
            Bash timeout and is capped at 600000.

    Returns:
        Command stdout, stderr, exit code, timeout/interruption status, and any
        harness error.
    """

    del description
    if not command.strip():
        return ToolResult(
            output=BashOutput(exit_code=-1, error="Command cannot be empty"),
            is_error=True,
        )
    executor = runtime.bash_executor
    if executor is None:
        return ToolResult(
            output=BashOutput(exit_code=-1, error="No bash executor configured."),
            is_error=True,
        )
    result = executor(
        command,
        _normalize_timeout(timeout, runtime.bash_default_timeout_ms),
    )
    if inspect.isawaitable(result):
        result = await result
    return result


def _normalize_timeout(timeout: int | None, default_timeout_ms: int) -> int:
    """Normalize a model timeout to the supported millisecond range."""

    selected = default_timeout_ms if timeout is None else timeout
    return min(max(int(selected), 1_000), MAX_TIMEOUT_MS)
