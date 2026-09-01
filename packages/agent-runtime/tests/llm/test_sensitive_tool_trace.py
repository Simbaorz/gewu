"""Sensitive Tool-result trace policy across concrete model providers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Protocol

import pytest

from gewu_agent_runtime.adapters.llm import (
    AnthropicChatModel,
    ChinaUnicomOpenServiceChatModel,
    OpenAIChatModel,
)
from gewu_agent_runtime.engine import AgentEngine, AssistantFinal, ExecutionRequest
from gewu_agent_runtime.llm import (
    Message,
    ModelStreamChunk,
    ModelTool,
    ModelTracePayload,
    ScriptedChatModel,
    ToolCall,
)
from gewu_agent_runtime.tools import ToolContext, ToolExecutor, ToolResult, ToolSet, tool
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)


class TraceProvider(Protocol):
    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        """Build one provider-specific trace payload."""

    async def aclose(self) -> None:
        """Close provider-owned resources."""


class ProviderTraceScriptedModel(ScriptedChatModel):
    def __init__(self, provider: TraceProvider) -> None:
        super().__init__(
            (
                (
                    ModelStreamChunk(
                        tool_calls=(
                            ToolCall(
                                tool_call_id="business-1",
                                name="business_lookup",
                                arguments={},
                            ),
                        )
                    ),
                ),
                (ModelStreamChunk(content_delta="done"),),
            )
        )
        self._provider = provider

    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        return self._provider.trace_request(messages, tools)


class TraceSink:
    def __init__(self) -> None:
        self.payloads: list[ModelTracePayload] = []

    async def write(self, payload: ModelTracePayload) -> None:
        self.payloads.append(payload)


@pytest.mark.parametrize(
    "provider_factory",
    (
        lambda: OpenAIChatModel(
            model_ref="openai",
            model_name="gpt-test",
            api_key="test-key",
        ),
        lambda: AnthropicChatModel(
            model_ref="anthropic",
            model_name="claude-test",
            api_key="test-key",
        ),
        lambda: ChinaUnicomOpenServiceChatModel(
            model_ref="unicom",
            model_name="qwen-test",
            api_url="https://unicom.example.test/api",
            app_id="app-id",
            app_secret="app-secret",
        ),
    ),
    ids=("openai", "anthropic", "unicom"),
)
async def test_concrete_provider_trace_never_receives_sensitive_tool_result(
    provider_factory: Callable[[], TraceProvider],
) -> None:
    @tool(description="Return subscriber-sensitive data.", trace_result=False)
    def business_lookup() -> ToolResult:
        return ToolResult(output={"value": "sensitive-business-value"})

    provider = provider_factory()
    model = ProviderTraceScriptedModel(provider)
    sink = TraceSink()
    tool_set = ToolSet((business_lookup,))
    workspace = WorkspaceSession(
        (
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=InMemoryWorkspaceBackend(),
            ),
        )
    )
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            tool_set,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
        model_trace_sink=sink,
    )
    try:
        events = [
            event
            async for event in engine.execute(
                ExecutionRequest(messages=(Message.user("lookup"),), tools=tool_set.all())
            )
        ]
    finally:
        await provider.aclose()

    assert isinstance(events[-1], AssistantFinal)
    assert "sensitive-business-value" in model.requests[1][0][-1].content
    rendered_trace = json.dumps(sink.payloads[1].request, ensure_ascii=False)
    assert "sensitive-business-value" not in rendered_trace
    assert "sensitive tool result" in rendered_trace
