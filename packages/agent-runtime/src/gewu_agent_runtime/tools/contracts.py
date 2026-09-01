"""Turn-bound tool definitions and execution results."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from gewu_agent_runtime.domain.file_state import FileStateCache
from gewu_agent_runtime.llm import ToolCall
from gewu_agent_runtime.workspace import WorkspaceSession


class PersistencePolicy(StrEnum):
    """How much of a tool result may be stored in runtime history."""

    FULL = "full"
    PROTECTED = "protected"
    SUMMARY = "summary"
    REFERENCE = "reference"
    NONE = "none"


class ToolResultMode(StrEnum):
    """Whether tool execution completed or suspended the turn."""

    NORMAL = "normal"
    SUSPENDED = "suspended"


class ToolError(BaseModel):
    """Tool failure returned to the model."""

    error: str


class AskSuspension(BaseModel):
    """A request to suspend execution until caller input is available."""

    ask_id: str
    questions: tuple[dict[str, Any], ...]
    timeout_seconds: int = 300


class AskResponse(BaseModel):
    """Answers returned immediately by a host Ask callback."""

    answers: dict[str, str | list[str]] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


AskCallback = Callable[
    [tuple[dict[str, Any], ...]],
    AskResponse | AskSuspension,
]


class ToolResult(BaseModel):
    """Normalized result produced by a tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    output: Any
    mode: ToolResultMode = ToolResultMode.NORMAL
    is_error: bool = False
    model_payload: dict[str, Any] | None = None
    persistence_payload: dict[str, Any] | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    new_messages: tuple[dict[str, Any], ...] = ()

    def output_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible model payload."""

        if self.model_payload is not None:
            return dict(self.model_payload)
        return self.raw_output_payload()

    def raw_output_payload(self) -> dict[str, Any]:
        """Return the complete result used by persistence and host transports."""

        if isinstance(self.output, BaseModel):
            value = self.output.model_dump(mode="json")
            return dict(value)
        if isinstance(self.output, Mapping):
            return dict(self.output)
        return {"value": self.output}


BashExecutor = Callable[[str, int], ToolResult | Awaitable[ToolResult]]


class ToolContext(BaseModel):
    """Neutral runtime services available to a bound tool."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    conversation_id: str
    run_id: str
    workspace: SkipValidation[WorkspaceSession]
    file_state_cache: SkipValidation[FileStateCache] = Field(default_factory=FileStateCache)
    bash_executor: SkipValidation[BashExecutor | None] = None
    bash_default_timeout_ms: int = Field(default=120_000, ge=1_000, le=600_000)
    ask_callback: SkipValidation[AskCallback | None] = None
    relative_file_not_found_hint: str = ""
    list_max_entries: int = Field(default=1_000, ge=1)
    file_search_max_entries: int = Field(default=10_000, ge=1)
    grep_max_scan_files: int = Field(default=200, ge=1)
    grep_max_scan_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    grep_max_result_lines: int = Field(default=1_000, ge=1)
    grep_max_result_bytes: int = Field(default=512 * 1024, ge=1)


class ToolRuntimeBindings(BaseModel):
    """Host-bound Tool capabilities and limits reused across one Runtime turn."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    bash_executor: SkipValidation[BashExecutor | None] = None
    bash_default_timeout_ms: int = Field(default=120_000, ge=1_000, le=600_000)
    ask_callback: SkipValidation[AskCallback | None] = None
    relative_file_not_found_hint: str = ""
    list_max_entries: int = Field(default=1_000, ge=1)
    file_search_max_entries: int = Field(default=10_000, ge=1)
    grep_max_scan_files: int = Field(default=200, ge=1)
    grep_max_scan_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    grep_max_result_lines: int = Field(default=1_000, ge=1)
    grep_max_result_bytes: int = Field(default=512 * 1024, ge=1)

    def create_context(
        self,
        *,
        conversation_id: str,
        run_id: str,
        workspace: WorkspaceSession,
        file_state_cache: FileStateCache | None = None,
    ) -> ToolContext:
        """Combine host bindings with Runtime-owned identifiers and mutable file state."""

        return ToolContext(
            conversation_id=conversation_id,
            run_id=run_id,
            workspace=workspace,
            file_state_cache=file_state_cache or FileStateCache(),
            bash_executor=self.bash_executor,
            bash_default_timeout_ms=self.bash_default_timeout_ms,
            ask_callback=self.ask_callback,
            relative_file_not_found_hint=self.relative_file_not_found_hint,
            list_max_entries=self.list_max_entries,
            file_search_max_entries=self.file_search_max_entries,
            grep_max_scan_files=self.grep_max_scan_files,
            grep_max_scan_bytes=self.grep_max_scan_bytes,
            grep_max_result_lines=self.grep_max_result_lines,
            grep_max_result_bytes=self.grep_max_result_bytes,
        )


ToolCallable = Callable[
    [dict[str, Any], ToolContext],
    ToolResult | Awaitable[ToolResult],
]


class Tool(BaseModel):
    """One already-authorized callable exposed to the model."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    name: str = Field(min_length=1, max_length=128)
    description: str
    input_schema: dict[str, Any]
    function: SkipValidation[ToolCallable]
    category: str = "general"
    writes_workspace: bool = False
    allow_parallel: bool = False
    retry_on_failure: bool = True
    max_retries: int = Field(default=2, ge=0)
    persistence_policy: PersistencePolicy = PersistencePolicy.FULL
    trace_result: bool = True

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        """Execute and normalize unexpected tool failures."""

        try:
            result = self.function(arguments, context)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, ToolResult):
                raise TypeError(f"Tool '{self.name}' returned {type(result).__name__}.")
            return result
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                output=ToolError(error=f"Tool execution error: {exc}"),
                is_error=True,
            )


class ToolSet:
    """Named immutable set of already-authorized tools bound to one turn."""

    def __init__(
        self,
        tools: Sequence[Tool] = (),
        *,
        name: str = "turn",
        version: str = "",
    ) -> None:
        """Build a tool set and reject duplicate model-visible names."""

        by_name = {item.name: item for item in tools}
        if len(by_name) != len(tools):
            raise ValueError("Tool names must be unique within one turn.")
        self.name = name
        self.version = version
        self._tools = by_name

    def all(self) -> tuple[Tool, ...]:
        """Return every bound tool."""

        return tuple(self._tools.values())

    def get(self, name: str) -> Tool | None:
        """Return one tool by name."""

        return self._tools.get(name)

    def read_only(self, *, name: str | None = None) -> ToolSet:
        """Return a set containing only tools that cannot mutate the workspace."""

        return ToolSet(
            tuple(tool for tool in self._tools.values() if not tool.writes_workspace),
            name=name or f"{self.name}:read-only",
            version=self.version,
        )


class ToolSetBuilder:
    """Composition-root builder for Runtime and subscriber-contributed tools."""

    def __init__(self, *, name: str = "turn", version: str = "") -> None:
        self._name = name
        self._version = version
        self._tools: list[Tool] = []

    def add(self, value: Tool) -> ToolSetBuilder:
        """Append one tool and return this builder."""

        self._tools.append(value)
        return self

    def extend(self, values: Sequence[Tool]) -> ToolSetBuilder:
        """Append a group of tools and return this builder."""

        self._tools.extend(values)
        return self

    def build(self) -> ToolSet:
        """Return an immutable tool set, validating duplicate names."""

        return ToolSet(self._tools, name=self._name, version=self._version)


ToolRegistry = ToolSet


class ToolExecutor:
    """Execute model calls against a turn-bound registry."""

    def __init__(self, registry: ToolSet, context: ToolContext) -> None:
        """Initialize the executor."""

        self._registry = registry
        self._context = context

    async def execute(self, call: ToolCall) -> ToolResult:
        """Execute one call or return a safe unknown-tool error."""

        selected = self._registry.get(call.name)
        if selected is None:
            return ToolResult(
                output=ToolError(error=f"Unknown tool '{call.name}'"),
                is_error=True,
            )
        return await selected.execute(call.arguments, self._context)
