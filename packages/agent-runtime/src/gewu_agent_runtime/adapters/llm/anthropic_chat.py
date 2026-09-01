"""Anthropic Messages API adapter."""

from __future__ import annotations

import base64
import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import anthropic
import httpx
from anthropic import AsyncAnthropic

from gewu_agent_runtime.adapters.llm.generation import generation_kwargs, stream_enabled
from gewu_agent_runtime.llm import (
    ContentPart,
    ContentPartType,
    Message,
    MessageRole,
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRuntimeConfig,
    ModelStreamChunk,
    ModelTimeoutError,
    ModelTool,
    ModelTracePayload,
    ModelUnavailableError,
    ToolCall,
    model_error_for_status,
)
from gewu_core.blocking import run_cpu_task

ANTHROPIC_GENERATION_KEYS = {"temperature", "top_p", "max_tokens"}


class AnthropicChatModel:
    """Anthropic Messages streaming chat model."""

    def __init__(
        self,
        *,
        model_ref: str,
        model_name: str,
        api_key: str,
        base_url: str = "",
        timeout_seconds: int = 600,
        generation_config: dict[str, Any] | None = None,
        support_stream: bool = True,
        support_tools: bool = True,
        support_vision: bool = False,
        context_window: int = 32_768,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 5.0,
    ) -> None:
        self._owns_http_client = http_client is None
        self.client = AsyncAnthropic(
            api_key=api_key,
            base_url=base_url or None,
            timeout=httpx.Timeout(
                timeout_seconds,
                connect=connect_timeout_seconds,
                pool=pool_timeout_seconds,
            ),
            http_client=http_client,
        )
        self.model_ref = model_ref
        self.provider = "anthropic"
        self.model_name = model_name
        self.generation_config = dict(generation_config or {})
        self.support_stream = support_stream
        self.support_tools = support_tools
        self.support_vision = support_vision
        self.context_window = context_window

    @property
    def effective_context_window(self) -> int:
        return self.context_window

    @classmethod
    def from_runtime_config(
        cls,
        config: ModelRuntimeConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 5.0,
    ) -> AnthropicChatModel:
        return cls(
            model_ref=config.model_ref,
            model_name=config.model_name,
            api_key=config.api_key,
            base_url=config.endpoint_url,
            timeout_seconds=config.timeout_seconds,
            generation_config=config.generation_config,
            support_stream=config.support_stream,
            support_tools=config.support_tools,
            support_vision=config.support_vision,
            context_window=config.context_window,
            http_client=http_client,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self.client.close()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one completion with provider-neutral failure semantics."""

        try:
            async for chunk in self._stream_chat(messages, tools):
                yield chunk
        except anthropic.APITimeoutError as exc:
            raise ModelTimeoutError from exc
        except anthropic.AuthenticationError as exc:
            raise ModelAuthenticationError from exc
        except anthropic.PermissionDeniedError as exc:
            raise ModelPermissionDeniedError from exc
        except anthropic.RateLimitError as exc:
            raise ModelRateLimitError from exc
        except anthropic.APIConnectionError as exc:
            raise ModelUnavailableError from exc
        except anthropic.APIStatusError as exc:
            raise model_error_for_status(exc.status_code) from exc
        except anthropic.APIError as exc:
            raise ModelUnavailableError from exc
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError from exc

    async def _stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Execute one Anthropic Messages request."""

        request = await run_cpu_task(self._request_kwargs, messages, tools)
        if not stream_enabled(self.generation_config, self.support_stream):
            async for chunk in self._complete_chat(request):
                yield chunk
            return

        pending_tool_calls: dict[str, dict[str, Any]] = {}
        input_tokens = 0
        output_tokens = 0
        async with self.client.messages.stream(**request) as stream:
            async for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "message_start":
                    message = getattr(event, "message", None)
                    usage = getattr(message, "usage", None)
                    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
                elif event_type == "content_block_start":
                    block = getattr(event, "content_block", None)
                    if getattr(block, "type", "") == "tool_use":
                        index = str(getattr(event, "index", len(pending_tool_calls)))
                        pending_tool_calls[index] = {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input_json": "",
                            "input": getattr(block, "input", None),
                        }
                elif event_type == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    text = getattr(delta, "text", "")
                    if text:
                        yield ModelStreamChunk(content_delta=text)
                    partial_json = getattr(delta, "partial_json", "")
                    if partial_json:
                        index = str(getattr(event, "index", ""))
                        pending = pending_tool_calls.setdefault(
                            index,
                            {"id": "", "name": "", "input_json": "", "input": None},
                        )
                        pending["input_json"] += partial_json
                elif event_type == "message_delta":
                    delta = getattr(event, "delta", None)
                    usage = getattr(event, "usage", None)
                    output_tokens = int(getattr(usage, "output_tokens", 0) or output_tokens)
                    stop_reason = getattr(delta, "stop_reason", "")
                    if stop_reason:
                        yield ModelStreamChunk(
                            finish_reason=str(stop_reason),
                            tool_calls=tuple(
                                self._parse_tool_call(value)
                                for value in pending_tool_calls.values()
                            ),
                            usage=self._usage_dict(input_tokens, output_tokens),
                        )

    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        return ModelTracePayload(
            provider="anthropic",
            request=self._request_kwargs(messages, tools, redact_images=True),
        )

    async def _complete_chat(self, request: dict[str, Any]) -> AsyncIterator[ModelStreamChunk]:
        response = await self.client.messages.create(**request)
        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        for block in response.content:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(str(getattr(block, "text", "") or ""))
            elif block_type == "tool_use":
                raw_input = getattr(block, "input", None)
                arguments = raw_input if isinstance(raw_input, dict) else {}
                tool_calls.append(
                    ToolCall(
                        tool_call_id=str(getattr(block, "id", "") or ""),
                        name=str(getattr(block, "name", "") or ""),
                        arguments=arguments,
                    )
                )
        content = "".join(text_parts)
        if content:
            yield ModelStreamChunk(content_delta=content)
        yield ModelStreamChunk(
            finish_reason=str(getattr(response, "stop_reason", "") or "stop"),
            tool_calls=tuple(tool_calls),
            usage=self._usage_dict(
                int(getattr(getattr(response, "usage", None), "input_tokens", 0) or 0),
                int(getattr(getattr(response, "usage", None), "output_tokens", 0) or 0),
            ),
        )

    @staticmethod
    def _usage_dict(input_tokens: int, output_tokens: int) -> dict[str, int]:
        if input_tokens <= 0 and output_tokens <= 0:
            return {}
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def _request_kwargs(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
        *,
        redact_images: bool = False,
    ) -> dict[str, Any]:
        system, anthropic_messages = self._messages_to_anthropic(
            messages,
            redact_images=redact_images,
        )
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": anthropic_messages,
            "max_tokens": int(self.generation_config.get("max_tokens") or 1024),
            **_anthropic_generation_kwargs(self.generation_config),
        }
        if system:
            kwargs["system"] = system
        if tools and self.support_tools:
            kwargs["tools"] = [self._tool_to_anthropic_schema(tool) for tool in tools]
        return kwargs

    @classmethod
    def _messages_to_anthropic(
        cls,
        messages: Sequence[Message],
        *,
        redact_images: bool = False,
    ) -> tuple[str, list[dict[str, Any]]]:
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role is MessageRole.SYSTEM:
                if message.content:
                    system_parts.append(message.content)
                continue
            if message.role is MessageRole.USER:
                cls._append_anthropic_user_content(
                    converted,
                    cls._content_to_anthropic(message, redact_images=redact_images),
                )
                continue
            if message.role is MessageRole.ASSISTANT:
                content = cls._assistant_content_blocks(message)
                if content:
                    converted.append({"role": "assistant", "content": content})
                continue
            if message.role is MessageRole.TOOL:
                cls._append_anthropic_user_content(
                    converted,
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": message.tool_call_id,
                            "content": message.content,
                        }
                    ],
                )
        return "\n\n".join(system_parts), converted

    @staticmethod
    def _append_anthropic_user_content(
        messages: list[dict[str, Any]],
        content: str | list[dict[str, Any]],
    ) -> None:
        if not messages or messages[-1]["role"] != "user":
            messages.append({"role": "user", "content": content})
            return
        previous = messages[-1]["content"]
        if isinstance(previous, str) and isinstance(content, str):
            messages[-1]["content"] = f"{previous}\n\n{content}" if previous else content
            return
        previous_blocks = (
            previous if isinstance(previous, list) else [{"type": "text", "text": previous}]
        )
        content_blocks = (
            content if isinstance(content, list) else [{"type": "text", "text": content}]
        )
        messages[-1]["content"] = [*previous_blocks, *content_blocks]

    @staticmethod
    def _content_to_anthropic(
        message: Message,
        *,
        redact_images: bool = False,
    ) -> str | list[dict[str, Any]]:
        if not message.content_parts:
            return message.content
        return [
            _part_to_anthropic(part, redact_images=redact_images) for part in message.content_parts
        ]

    @staticmethod
    def _assistant_content_blocks(message: Message) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = []
        if message.content:
            content.append({"type": "text", "text": message.content})
        for tool_call in message.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": tool_call.tool_call_id,
                    "name": tool_call.name,
                    "input": tool_call.arguments,
                }
            )
        return content

    @staticmethod
    def _tool_to_anthropic_schema(tool: ModelTool) -> dict[str, Any]:
        return {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }

    @staticmethod
    def _parse_tool_call(raw: dict[str, Any]) -> ToolCall:
        parsed_input = raw.get("input")
        if not isinstance(parsed_input, dict):
            try:
                parsed_input = json.loads(str(raw.get("input_json") or "{}"))
            except json.JSONDecodeError:
                parsed_input = {"_raw_arguments": str(raw.get("input_json") or "")}
        return ToolCall(
            tool_call_id=str(raw.get("id") or ""),
            name=str(raw.get("name") or ""),
            arguments=parsed_input,
        )


def _anthropic_generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in generation_kwargs(config, ANTHROPIC_GENERATION_KEYS).items()
        if key != "max_tokens"
    }


def _part_to_anthropic(part: ContentPart, *, redact_images: bool) -> dict[str, Any]:
    if part.part_type is ContentPartType.IMAGE:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": part.mime_type,
                "data": (
                    "[redacted image base64]"
                    if redact_images
                    else part.base64_data or base64.b64encode(part.data).decode("ascii")
                ),
            },
        }
    return {"type": "text", "text": part.text}
