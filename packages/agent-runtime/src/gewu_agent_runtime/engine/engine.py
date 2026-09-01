"""Pure model to tool execution loop."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence

from gewu_agent_runtime.context_tokens import ContextTokenEstimator
from gewu_agent_runtime.engine.events import (
    AgentEvent,
    AskRequested,
    AssistantDelta,
    AssistantFinal,
    AssistantIntermediate,
    ExecutionError,
    ExecutionRequest,
    ToolResultEvent,
    ToolUse,
)
from gewu_agent_runtime.llm import (
    ChatModel,
    ChatModelProvider,
    ContentPart,
    ContentPartType,
    Message,
    MessageRole,
    ModelTool,
    ModelTraceSink,
    ToolCall,
)
from gewu_agent_runtime.tools import (
    AskSuspension,
    ToolError,
    ToolExecutor,
    ToolResult,
    ToolResultMode,
)
from gewu_core.ids import new_id

logger = logging.getLogger(__name__)
_REDACTED_TOOL_RESULT = json.dumps(
    {"redacted": "sensitive tool result"},
    separators=(",", ":"),
)


class AgentEngine:
    """Run a model/tool loop without persistence, authorization, or transport."""

    def __init__(
        self,
        *,
        model: ChatModel | None = None,
        model_provider: ChatModelProvider | None = None,
        tool_executor: ToolExecutor,
        model_trace_sink: ModelTraceSink | None = None,
    ) -> None:
        """Initialize one turn-bound engine."""

        if (model is None) == (model_provider is None):
            raise ValueError("Exactly one of model or model_provider is required.")
        self._model = model
        self._model_provider = model_provider
        self._tools = tool_executor
        self._model_trace_sink = model_trace_sink

    async def execute(self, request: ExecutionRequest) -> AsyncIterator[AgentEvent]:
        """Execute until a final response, failure, or ask suspension."""

        messages = list(request.messages)
        for _ in range(request.max_iterations):
            model = (
                self._model
                if self._model is not None
                else await self._resolve_model(messages, request.tools)
            )
            message_id = new_id()
            text_parts: list[str] = []
            tool_calls: list[ToolCall] = []
            if _messages_have_images(messages) and not model.support_vision:
                yield ExecutionError(code="vision_unsupported", message="当前模型不支持图片输入。")
                return
            model_name = model.model_name or "unknown"
            estimated_tokens = await ContextTokenEstimator(model_name).estimate_async(
                messages,
                request.tools,
            )
            hard_limit = int(model.context_window * 0.9)
            if estimated_tokens >= hard_limit:
                yield ExecutionError(
                    code="context_limit_exceeded",
                    message=(
                        "Conversation context reached the safe model limit. "
                        "Start a new Chat turn so the history can be compacted."
                    ),
                )
                return
            await self._trace_request(model, messages, request.tools)
            async for chunk in model.stream_chat(messages, request.tools):
                input_tokens = chunk.usage.get("input_tokens", 0)
                if input_tokens > 0:
                    logger.info(
                        "LLM input token observation. model_ref=%s provider=%s model=%s "
                        "estimated_tokens=%s input_tokens=%s error_ratio=%.6f",
                        model.model_ref,
                        model.provider,
                        model_name,
                        estimated_tokens,
                        input_tokens,
                        (estimated_tokens - input_tokens) / input_tokens,
                    )
                if chunk.content_delta:
                    text_parts.append(chunk.content_delta)
                    if "".join(text_parts).strip():
                        yield AssistantDelta(message_id=message_id, content=chunk.content_delta)
                tool_calls.extend(chunk.tool_calls)

            assistant_text = "".join(text_parts)
            if tool_calls:
                yield AssistantIntermediate(message_id=message_id, content=assistant_text)
                messages.append(Message.assistant(assistant_text, tool_calls))
                async for event in self._execute_tools(
                    messages,
                    tool_calls,
                    assistant_text,
                    request.tools,
                ):
                    yield event
                    if isinstance(event, AskRequested):
                        return
                continue
            if not assistant_text.strip():
                yield ExecutionError(
                    code="empty_response", message="LLM returned an empty response."
                )
                return
            yield AssistantFinal(message_id=message_id, content=assistant_text)
            return
        yield ExecutionError(
            code="iteration_limit", message="Agent loop reached max LLM iterations."
        )

    async def _resolve_model(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ChatModel:
        if self._model_provider is None:
            raise RuntimeError("Chat model provider is not configured.")
        return await self._model_provider.resolve(messages, tools)

    async def _trace_request(
        self,
        model: ChatModel,
        messages: list[Message],
        tools: Sequence[ModelTool],
    ) -> None:
        if self._model_trace_sink is None:
            return
        trace_request = getattr(model, "trace_request", None)
        if not callable(trace_request):
            return
        try:
            trace_messages = _redact_sensitive_tool_results(messages, tools)
            await self._model_trace_sink.write(trace_request(trace_messages, tools))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to write model request trace snapshot exception_type=%s",
                type(exc).__name__,
            )

    async def _execute_tools(
        self,
        messages: list[Message],
        calls: Sequence[ToolCall],
        assistant_text: str,
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[AgentEvent]:
        for call in calls:
            yield ToolUse(
                call=call, assistant_text=assistant_text if assistant_text.strip() else ""
            )
            result = await self._tools.execute(call)
            if result.mode is ToolResultMode.SUSPENDED:
                if not isinstance(result.output, AskSuspension):
                    result = ToolResult(
                        output=ToolError(error="Suspended tool returned invalid output."),
                        is_error=True,
                    )
                else:
                    yield AskRequested(
                        call=call,
                        ask_id=result.output.ask_id,
                        questions=result.output.questions,
                        timeout_seconds=result.output.timeout_seconds,
                    )
                    return
            yield ToolResultEvent(call=call, result=result)
            messages.append(
                Message.tool(
                    call.tool_call_id,
                    json.dumps(result.output_payload(), ensure_ascii=False),
                    trace_result=_tool_result_trace_enabled(call.name, tools),
                )
            )
            if result.new_messages:
                messages[:] = _normalize_messages(
                    [*messages, *(_extra_message(item) for item in result.new_messages)]
                )


def _messages_have_images(messages: Sequence[Message]) -> bool:
    return any(
        any(part.part_type is ContentPartType.IMAGE for part in message.content_parts)
        for message in messages
    )


def _redact_sensitive_tool_results(
    messages: Sequence[Message],
    tools: Sequence[ModelTool],
) -> tuple[Message, ...]:
    trace_result_by_name = {tool.name: bool(getattr(tool, "trace_result", True)) for tool in tools}
    sensitive_call_ids = {
        call.tool_call_id
        for message in messages
        for call in message.tool_calls
        if not trace_result_by_name.get(call.name, True)
    }
    sensitive_call_ids.update(
        message.tool_call_id
        for message in messages
        if message.role is MessageRole.TOOL and not message.trace_result
    )
    return tuple(
        (
            message.model_copy(update={"content": _REDACTED_TOOL_RESULT})
            if message.role is MessageRole.TOOL and message.tool_call_id in sensitive_call_ids
            else message
        )
        for message in messages
    )


def _tool_result_trace_enabled(name: str, tools: Sequence[ModelTool]) -> bool:
    return next(
        (bool(getattr(tool, "trace_result", True)) for tool in tools if tool.name == name),
        True,
    )


def _extra_message(value: dict[str, object]) -> Message:
    role = value.get("role")
    content = str(value.get("content") or "")
    if role == MessageRole.SYSTEM.value:
        return Message.system(content)
    if role == MessageRole.ASSISTANT.value:
        return Message.assistant(content)
    if role == MessageRole.TOOL.value:
        return Message.tool(str(value.get("tool_call_id") or ""), content)
    return Message.user(content)


def _normalize_messages(messages: Sequence[Message]) -> list[Message]:
    normalized: list[Message] = []
    for message in messages:
        if (
            message.role is MessageRole.USER
            and normalized
            and normalized[-1].role is MessageRole.USER
        ):
            if normalized[-1].content_parts or message.content_parts:
                normalized[-1] = Message.user_parts(
                    (
                        *_message_content_parts(normalized[-1]),
                        *_message_content_parts(message),
                    )
                )
                continue
            normalized[-1] = Message.user(
                "\n\n".join(part for part in (normalized[-1].content, message.content) if part)
            )
            continue
        normalized.append(message)
    return normalized


def _message_content_parts(message: Message) -> tuple[ContentPart, ...]:
    if message.content_parts:
        return message.content_parts
    if message.content:
        return (ContentPart(part_type=ContentPartType.TEXT, text=message.content),)
    return ()
