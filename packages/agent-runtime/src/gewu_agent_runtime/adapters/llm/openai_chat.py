"""OpenAI-compatible Chat Completions adapter."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from gewu_agent_runtime.adapters.llm.generation import generation_kwargs, stream_enabled
from gewu_agent_runtime.adapters.llm.openai_compat import (
    OPENAI_GENERATION_KEYS,
    openai_message_content,
    openai_tool_schema,
    parse_openai_tool_call,
)
from gewu_agent_runtime.llm import (
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
    model_error_for_status,
)
from gewu_core.blocking import run_cpu_task


class OpenAIChatModel:
    """OpenAI-compatible streaming chat model."""

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
        kwargs: dict[str, Any] = {
            "api_key": api_key or "none",
            "timeout": httpx.Timeout(
                timeout_seconds,
                connect=connect_timeout_seconds,
                pool=pool_timeout_seconds,
            ),
        }
        if base_url:
            kwargs["base_url"] = base_url
        if http_client is not None:
            kwargs["http_client"] = http_client
        self.client = AsyncOpenAI(**kwargs)
        self.model_ref = model_ref
        self.provider = "openai"
        self.model_name = model_name
        self.generation_config = dict(generation_config or {})
        self.support_stream = support_stream
        self.support_tools = support_tools
        self.support_vision = support_vision
        self.context_window = context_window

    @property
    def effective_context_window(self) -> int:
        """Expose the provider-facing effective context window."""

        return self.context_window

    @classmethod
    def from_runtime_config(
        cls,
        config: ModelRuntimeConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 5.0,
    ) -> OpenAIChatModel:
        """Create a client from one resolved runtime configuration."""

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
        """Close an internally owned provider transport."""

        if self._owns_http_client:
            await self.client.close()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one chat completion with provider-neutral failure semantics."""

        try:
            async for chunk in self._stream_chat(messages, tools):
                yield chunk
        except openai.APITimeoutError as exc:
            raise ModelTimeoutError from exc
        except openai.AuthenticationError as exc:
            raise ModelAuthenticationError from exc
        except openai.PermissionDeniedError as exc:
            raise ModelPermissionDeniedError from exc
        except openai.RateLimitError as exc:
            raise ModelRateLimitError from exc
        except openai.APIConnectionError as exc:
            raise ModelUnavailableError from exc
        except openai.APIStatusError as exc:
            raise model_error_for_status(exc.status_code) from exc
        except openai.APIError as exc:
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
        """Execute one OpenAI-compatible request."""

        kwargs = await run_cpu_task(self._request_kwargs, messages, tools)
        if not stream_enabled(self.generation_config, self.support_stream):
            async for chunk in self._complete_chat(kwargs):
                yield chunk
            return

        stream = await self.client.chat.completions.create(**kwargs)
        pending_tool_calls: dict[int, dict[str, str]] = {}
        async for chunk in stream:
            usage = self._usage_dict(getattr(chunk, "usage", None))
            choice = chunk.choices[0] if chunk.choices else None
            if choice is None:
                if usage:
                    yield ModelStreamChunk(usage=usage)
                continue
            delta = choice.delta
            if delta.content:
                yield ModelStreamChunk(content_delta=delta.content)

            for index, tool_call_delta in enumerate(delta.tool_calls or []):
                tool_index = tool_call_delta.index if tool_call_delta.index is not None else index
                pending = pending_tool_calls.setdefault(
                    tool_index,
                    {"id": "", "name": "", "arguments": ""},
                )
                if tool_call_delta.id:
                    pending["id"] = tool_call_delta.id
                if tool_call_delta.function:
                    if tool_call_delta.function.name:
                        pending["name"] = tool_call_delta.function.name
                    if tool_call_delta.function.arguments:
                        pending["arguments"] += tool_call_delta.function.arguments

            if choice.finish_reason:
                yield ModelStreamChunk(
                    finish_reason=str(choice.finish_reason),
                    tool_calls=tuple(
                        parse_openai_tool_call(value) for value in pending_tool_calls.values()
                    ),
                    usage=usage,
                )

    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        """Return a provider payload with image bytes redacted."""

        return ModelTracePayload(
            provider="openai",
            request=self._request_kwargs(messages, tools, redact_images=True),
        )

    def _request_kwargs(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
        *,
        redact_images: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages_to_openai(messages, redact_images=redact_images),
            **generation_kwargs(self.generation_config, OPENAI_GENERATION_KEYS),
        }
        if tools and self.support_tools:
            kwargs["tools"] = [openai_tool_schema(tool) for tool in tools]
        if stream_enabled(self.generation_config, self.support_stream):
            kwargs["stream"] = True
            kwargs["stream_options"] = {"include_usage": True}
        return kwargs

    async def _complete_chat(self, kwargs: dict[str, Any]) -> AsyncIterator[ModelStreamChunk]:
        response = await self.client.chat.completions.create(**kwargs)
        choice = response.choices[0] if response.choices else None
        if choice is None:
            return
        message = choice.message
        if message.content:
            yield ModelStreamChunk(content_delta=message.content)
        tool_calls = tuple(
            parse_openai_tool_call(
                {
                    "id": str(tool_call.id or ""),
                    "name": str(getattr(tool_call.function, "name", "") or ""),
                    "arguments": str(getattr(tool_call.function, "arguments", "") or ""),
                }
            )
            for tool_call in (message.tool_calls or [])
        )
        yield ModelStreamChunk(
            finish_reason=str(choice.finish_reason or "stop"),
            tool_calls=tool_calls,
            usage=self._usage_dict(getattr(response, "usage", None)),
        )

    @staticmethod
    def _usage_dict(usage: Any) -> dict[str, int]:
        if usage is None:
            return {}
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or input_tokens + output_tokens,
        }

    @classmethod
    def _messages_to_openai(
        cls,
        messages: Sequence[Message],
        *,
        redact_images: bool = False,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role is MessageRole.TOOL:
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message.tool_call_id,
                        "content": message.content,
                    }
                )
                continue
            item: dict[str, Any] = {
                "role": message.role.value,
                "content": openai_message_content(message, redact_images=redact_images),
            }
            if message.tool_calls:
                item["tool_calls"] = [
                    {
                        "id": tool_call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            converted.append(item)
        return converted
