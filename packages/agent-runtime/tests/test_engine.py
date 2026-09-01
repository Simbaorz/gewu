"""Pure Agent Engine tests."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence

import pytest

from gewu_agent_runtime.engine import (
    AgentEngine,
    AskRequested,
    AssistantFinal,
    ExecutionError,
    ExecutionRequest,
    ToolResultEvent,
)
from gewu_agent_runtime.llm import (
    Message,
    ModelStreamChunk,
    ModelTool,
    ModelTracePayload,
    ScriptedChatModel,
    ToolCall,
)
from gewu_agent_runtime.tools import (
    AskSuspension,
    ToolContext,
    ToolExecutor,
    ToolRegistry,
    ToolResult,
    ToolResultMode,
    tool,
)
from gewu_agent_runtime.workspace import WorkspaceSession


class TraceableScriptedChatModel(ScriptedChatModel):
    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        return ModelTracePayload(
            provider="scripted",
            request={
                "messages": [message.model_dump(mode="json") for message in messages],
                "tools": [tool.name for tool in tools],
            },
        )


class TraceSink:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.payloads: list[ModelTracePayload] = []

    async def write(self, payload: ModelTracePayload) -> None:
        self.payloads.append(payload)
        if self.fail:
            raise OSError("trace-private-secret")


class SequencedModelProvider:
    def __init__(self, models: Sequence[ScriptedChatModel]) -> None:
        self.models = list(models)
        self.requests: list[tuple[Message, ...]] = []

    async def resolve(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ScriptedChatModel:
        del tools
        self.requests.append(tuple(messages))
        return self.models.pop(0)


def test_execution_request_accepts_subscriber_iteration_limit_above_100() -> None:
    request = ExecutionRequest(messages=(Message.user("run"),), max_iterations=101)

    assert request.max_iterations == 101


async def test_engine_executes_tool_then_finishes(workspace: WorkspaceSession) -> None:
    @tool(description="Add two integers.")
    def add(left: int, right: int) -> ToolResult:
        return ToolResult(output={"total": left + right})

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="call-1", name="add", arguments={"left": 2, "right": 3}
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="5")],
        ]
    )
    registry = ToolRegistry([add])
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("calculate"),), tools=registry.all())
        )
    ]

    assert isinstance(events[-1], AssistantFinal)
    assert events[-1].content == "5"
    assert model.requests[1][0][-1].content == '{"total": 5}'


async def test_engine_resolves_injected_model_provider_for_each_iteration(
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Return one value.")
    def lookup() -> ToolResult:
        return ToolResult(output={"value": 5})

    provider = SequencedModelProvider(
        (
            ScriptedChatModel(
                [
                    [
                        ModelStreamChunk(
                            tool_calls=(
                                ToolCall(tool_call_id="call-1", name="lookup", arguments={}),
                            )
                        )
                    ]
                ]
            ),
            ScriptedChatModel([[ModelStreamChunk(content_delta="done")]]),
        )
    )
    registry = ToolRegistry((lookup,))
    engine = AgentEngine(
        model_provider=provider,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("start"),), tools=registry.all())
        )
    ]

    assert len(provider.requests) == 2
    assert provider.models == []
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert isinstance(events[-1], AssistantFinal)
    assert events[-1].content == "done"


async def test_engine_executes_tool_calls_in_order_and_projects_exact_errors(
    workspace: WorkspaceSession,
) -> None:
    observed: list[str] = []

    @tool(description="Record one value.")
    async def record(value: str) -> ToolResult:
        observed.append(value)
        return ToolResult(output={"value": value})

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="record-1",
                            name="record",
                            arguments={"value": "first"},
                        ),
                        ToolCall(
                            tool_call_id="missing-1",
                            name="missing",
                            arguments={},
                        ),
                        ToolCall(
                            tool_call_id="record-2",
                            name="record",
                            arguments={"value": "last"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    registry = ToolRegistry((record,))
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("run"),), tools=registry.all())
        )
    ]

    assert observed == ["first", "last"]
    results = [event for event in events if isinstance(event, ToolResultEvent)]
    assert [event.call.tool_call_id for event in results] == [
        "record-1",
        "missing-1",
        "record-2",
    ]
    assert [event.result.is_error for event in results] == [False, True, False]
    assert [message.content for message in model.requests[1][0][-3:]] == [
        '{"value": "first"}',
        '{"error": "Unknown tool \'missing\'"}',
        '{"value": "last"}',
    ]


async def test_engine_traces_every_provider_request_without_owning_trace_policy(
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Return one value.")
    def lookup() -> ToolResult:
        return ToolResult(output={"value": 1})

    model = TraceableScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(ToolCall(tool_call_id="lookup-1", name="lookup", arguments={}),)
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    sink = TraceSink()
    registry = ToolRegistry((lookup,))
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
        model_trace_sink=sink,
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("lookup"),), tools=registry.all())
        )
    ]

    assert isinstance(events[-1], AssistantFinal)
    assert len(sink.payloads) == 2
    assert sink.payloads[0].request["tools"] == ["lookup"]
    assert sink.payloads[1].request["messages"][-1]["role"] == "tool"


async def test_engine_redacts_only_sensitive_tool_results_from_trace(
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Return sensitive business data.", trace_result=False)
    def sensitive_lookup() -> ToolResult:
        return ToolResult(output={"value": "sensitive-value"})

    @tool(description="Return public metadata.")
    def public_lookup() -> ToolResult:
        return ToolResult(output={"value": "public-value"})

    model = TraceableScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="sensitive-1",
                            name="sensitive_lookup",
                            arguments={},
                        ),
                        ToolCall(
                            tool_call_id="public-1",
                            name="public_lookup",
                            arguments={},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    sink = TraceSink()
    registry = ToolRegistry((sensitive_lookup, public_lookup))
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
        model_trace_sink=sink,
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("lookup"),), tools=registry.all())
        )
    ]

    assert isinstance(events[-1], AssistantFinal)
    actual_model_request = json.dumps(
        [message.model_dump(mode="json") for message in model.requests[1][0]],
        ensure_ascii=False,
    )
    traced_payload = sink.payloads[1].request
    traced_request = json.dumps(traced_payload, ensure_ascii=False)
    traced_results = {
        str(message["tool_call_id"]): str(message["content"])
        for message in traced_payload["messages"]
        if message["role"] == "tool"
    }
    assert "sensitive-value" in actual_model_request
    assert "public-value" in actual_model_request
    assert "sensitive-value" not in traced_request
    assert json.loads(traced_results["sensitive-1"]) == {"redacted": "sensitive tool result"}
    assert json.loads(traced_results["public-1"]) == {"value": "public-value"}
    assert "public-value" in traced_request


async def test_engine_preserves_historical_sensitive_trace_policy_after_tool_revocation(
    workspace: WorkspaceSession,
) -> None:
    model = TraceableScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    sink = TraceSink()
    registry = ToolRegistry()
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
        model_trace_sink=sink,
    )
    history = (
        Message.user("lookup"),
        Message.assistant(
            "",
            (
                ToolCall(
                    tool_call_id="sensitive-1",
                    name="revoked_sensitive_lookup",
                    arguments={},
                ),
            ),
        ),
        Message.tool(
            "sensitive-1",
            '{"value":"historical-sensitive-value"}',
            trace_result=False,
        ),
    )

    events = [event async for event in engine.execute(ExecutionRequest(messages=history, tools=()))]

    assert isinstance(events[-1], AssistantFinal)
    assert "historical-sensitive-value" in model.requests[0][0][-1].content
    traced_request = json.dumps(sink.payloads[0].request, ensure_ascii=False)
    assert "historical-sensitive-value" not in traced_request
    assert "sensitive tool result" in traced_request


async def test_engine_trace_failure_is_best_effort(
    workspace: WorkspaceSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger="gewu_agent_runtime.engine.engine")
    model = TraceableScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    sink = TraceSink(fail=True)
    registry = ToolRegistry()
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
        model_trace_sink=sink,
    )

    events = [
        event async for event in engine.execute(ExecutionRequest(messages=(Message.user("run"),)))
    ]

    assert isinstance(events[-1], AssistantFinal)
    assert len(model.requests) == 1
    assert "Failed to write model request trace snapshot" in caplog.text
    assert "exception_type=OSError" in caplog.text
    assert "trace-private-secret" not in caplog.text


async def test_engine_stops_on_ask_suspension(workspace: WorkspaceSession) -> None:
    from gewu_agent_runtime.builtins import ask_user_tool

    ask = ask_user_tool()
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-call",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ]
        ]
    )
    registry = ToolRegistry([ask])
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("ask me"),), tools=registry.all())
        )
    ]

    assert isinstance(events[-1], AskRequested)
    assert events[-1].questions[0]["question"] == "Continue?"


async def test_engine_rejects_invalid_suspended_tool_output(
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Return an invalid suspension.")
    def invalid_suspend() -> ToolResult:
        return ToolResult(
            mode=ToolResultMode.SUSPENDED,
            output={"unexpected": True},
        )

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="invalid-1",
                            name="invalid_suspend",
                            arguments={},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="recovered")],
        ]
    )
    registry = ToolRegistry([invalid_suspend])
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("run"),), tools=registry.all())
        )
    ]

    result = next(event for event in events if isinstance(event, ToolResultEvent))
    assert result.result.is_error is True
    assert result.result.output_payload() == {"error": "Suspended tool returned invalid output."}
    assert isinstance(events[-1], AssistantFinal)


async def test_engine_does_not_suspend_on_output_type_without_suspended_mode(
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Return a normal result whose payload resembles Ask.")
    def normal_result() -> ToolResult:
        return ToolResult(
            output=AskSuspension(
                ask_id="not-suspended",
                questions=({"question": "Ignored"},),
                timeout_seconds=60,
            )
        )

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="normal-1",
                            name="normal_result",
                            arguments={},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    registry = ToolRegistry([normal_result])
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(
            ExecutionRequest(messages=(Message.user("run"),), tools=registry.all())
        )
    ]

    assert not any(isinstance(event, AskRequested) for event in events)
    assert any(isinstance(event, ToolResultEvent) for event in events)
    assert isinstance(events[-1], AssistantFinal)


async def test_engine_stops_before_provider_at_safe_context_limit(
    workspace: WorkspaceSession,
) -> None:
    class SmallContextModel(ScriptedChatModel):
        model_name = "private-provider-model"
        context_window = 4_000

    model = SmallContextModel([[ModelStreamChunk(content_delta="must not run")]])
    registry = ToolRegistry()
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    events = [
        event
        async for event in engine.execute(ExecutionRequest(messages=(Message.user("x" * 30_000),)))
    ]

    assert len(events) == 1
    assert isinstance(events[0], ExecutionError)
    assert events[0].code == "context_limit_exceeded"
    assert events[0].message == (
        "Conversation context reached the safe model limit. "
        "Start a new Chat turn so the history can be compacted."
    )
    assert model.requests == []


async def test_engine_logs_estimate_against_provider_input_usage(
    workspace: WorkspaceSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="gewu_agent_runtime.engine.engine")
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    content_delta="ok",
                    usage={"input_tokens": 10, "output_tokens": 1, "total_tokens": 11},
                )
            ]
        ]
    )
    registry = ToolRegistry()
    engine = AgentEngine(
        model=model,
        tool_executor=ToolExecutor(
            registry,
            ToolContext(conversation_id="conversation", run_id="run", workspace=workspace),
        ),
    )

    _ = [
        event async for event in engine.execute(ExecutionRequest(messages=(Message.user("hello"),)))
    ]

    assert "model_ref=scripted:test" in caplog.text
    assert "input_tokens=10" in caplog.text


async def test_engine_uses_exact_subscriber_terminal_errors(workspace: WorkspaceSession) -> None:
    registry = ToolRegistry()
    context = ToolContext(conversation_id="conversation", run_id="run", workspace=workspace)
    empty = AgentEngine(
        model=ScriptedChatModel([[]]),
        tool_executor=ToolExecutor(registry, context),
    )
    empty_events = [
        event async for event in empty.execute(ExecutionRequest(messages=(Message.user("hello"),)))
    ]

    @tool(description="Complete one iteration with a tool call.")
    def noop() -> ToolResult:
        return ToolResult(output={"ok": True})

    tool_registry = ToolRegistry([noop])
    limited = AgentEngine(
        model=ScriptedChatModel(
            [
                [
                    ModelStreamChunk(
                        tool_calls=(ToolCall(tool_call_id="noop-1", name="noop", arguments={}),)
                    )
                ]
            ]
        ),
        tool_executor=ToolExecutor(tool_registry, context),
    )
    limited_events = [
        event
        async for event in limited.execute(
            ExecutionRequest(
                messages=(Message.user("hello"),),
                tools=tool_registry.all(),
                max_iterations=1,
            )
        )
    ]

    assert isinstance(empty_events[-1], ExecutionError)
    assert empty_events[-1].message == "LLM returned an empty response."
    assert empty_events[-1].message_id
    assert isinstance(limited_events[-1], ExecutionError)
    assert limited_events[-1].message == "Agent loop reached max LLM iterations."
