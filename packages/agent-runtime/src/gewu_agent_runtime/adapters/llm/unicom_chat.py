"""China Unicom Open Service Chat adapter."""

from __future__ import annotations

import json
import random
import string
from collections.abc import AsyncIterator, Sequence
from datetime import datetime
from typing import Any

import httpx

from gewu_agent_runtime.adapters.llm.generation import generation_kwargs, stream_enabled
from gewu_agent_runtime.adapters.llm.openai_compat import (
    OPENAI_GENERATION_KEYS,
    openai_message_content,
    openai_tool_schema,
    parse_openai_tool_call,
)
from gewu_agent_runtime.adapters.llm.unicom import generate_unicom_token
from gewu_agent_runtime.llm import (
    Message,
    MessageRole,
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


class ChinaUnicomOpenServiceChatModel:
    """China Unicom Open Service streaming chat model."""

    def __init__(
        self,
        *,
        model_ref: str,
        model_name: str,
        api_url: str,
        app_id: str,
        app_secret: str,
        nlpt_authorization: str = "",
        timeout_seconds: int = 600,
        generation_config: dict[str, Any] | None = None,
        provider_config: dict[str, Any] | None = None,
        support_stream: bool = True,
        support_tools: bool = True,
        support_vision: bool = False,
        context_window: int = 32_768,
        http_client: httpx.AsyncClient | None = None,
        connect_timeout_seconds: float = 5.0,
        pool_timeout_seconds: float = 5.0,
    ) -> None:
        self.model_ref = model_ref
        self.provider = "unicom"
        self.model_name = model_name
        self.api_url = api_url
        self.app_id = app_id
        self.app_secret = app_secret
        self.nlpt_authorization = nlpt_authorization
        self.timeout_seconds = timeout_seconds
        self.generation_config = dict(generation_config or {})
        self.provider_config = dict(provider_config or {})
        self.support_stream = support_stream
        self.support_tools = support_tools
        self.support_vision = support_vision
        self.context_window = context_window
        self._client = http_client
        self._owns_client = http_client is None
        self._connect_timeout_seconds = connect_timeout_seconds
        self._pool_timeout_seconds = pool_timeout_seconds
        self.req_key = _unicom_req_key(self.provider_config)
        self.role_reflect = _unicom_role_reflect(self.provider_config)

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
    ) -> ChinaUnicomOpenServiceChatModel:
        credentials = dict(config.credentials)
        return cls(
            model_ref=config.model_ref,
            model_name=config.model_name,
            api_url=config.endpoint_url,
            app_id=_string_value(credentials.get("app_id")),
            app_secret=_string_value(credentials.get("app_secret")),
            nlpt_authorization=_string_value(
                credentials.get("nlpt_authorization") or credentials.get("nlpt-Authorization")
            ),
            timeout_seconds=config.timeout_seconds,
            generation_config=config.generation_config,
            provider_config=config.provider_config,
            support_stream=config.support_stream,
            support_tools=config.support_tools,
            support_vision=config.support_vision,
            context_window=config.context_window,
            http_client=http_client,
            connect_timeout_seconds=connect_timeout_seconds,
            pool_timeout_seconds=pool_timeout_seconds,
        )

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one completion with provider-neutral failure semantics."""

        try:
            async for chunk in self._stream_chat(messages, tools):
                yield chunk
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError from exc
        except httpx.HTTPStatusError as exc:
            raise model_error_for_status(exc.response.status_code) from exc
        except httpx.TransportError as exc:
            raise ModelUnavailableError from exc

    async def _stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Execute one China Unicom Open Service request."""

        use_stream = stream_enabled(self.generation_config, self.support_stream)
        if not use_stream:
            async for chunk in self._complete_chat(messages, tools):
                yield chunk
            return

        payload = await run_cpu_task(self._request_payload, messages, tools, stream=True)
        pending_tool_calls: dict[int, dict[str, str]] = {}
        async with self._get_client().stream(
            "POST",
            self.api_url,
            json=payload,
            headers=self._headers(),
            timeout=self._request_timeout(),
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                parsed_chunk = self._parse_stream_line(line, pending_tool_calls)
                if parsed_chunk is None:
                    continue
                if parsed_chunk.finish_reason == "[DONE]":
                    break
                yield parsed_chunk

    def trace_request(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ModelTracePayload:
        use_stream = stream_enabled(self.generation_config, self.support_stream)
        return ModelTracePayload(
            provider="unicom",
            request={
                "url": self.api_url,
                "req_key": self.req_key,
                "body": self._body_params(
                    messages,
                    tools,
                    stream=use_stream,
                    redact_images=True,
                ),
            },
        )

    async def _complete_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        payload = await run_cpu_task(self._request_payload, messages, tools, stream=False)
        async with self._get_client().stream(
            "POST",
            self.api_url,
            json=payload,
            headers=self._headers(),
            timeout=self._request_timeout(),
        ) as response:
            response.raise_for_status()
            raw_body = await response.aread()
        decoded = await run_cpu_task(json.loads, raw_body)
        body = self._unwrap_body(decoded)
        parsed = self._parse_complete_body(body)
        if parsed.content_delta:
            yield ModelStreamChunk(content_delta=parsed.content_delta)
        yield ModelStreamChunk(
            finish_reason=parsed.finish_reason,
            tool_calls=parsed.tool_calls,
            usage=parsed.usage,
        )

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._request_timeout())
        return self._client

    def _request_timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            timeout=self.timeout_seconds,
            connect=self._connect_timeout_seconds,
            pool=self._pool_timeout_seconds,
        )

    def _request_payload(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
        *,
        stream: bool,
        redact_images: bool = False,
    ) -> dict[str, Any]:
        return _build_unicom_payload(
            app_id=self.app_id,
            app_secret=self.app_secret,
            req_key=self.req_key,
            body_params=self._body_params(
                messages,
                tools,
                stream=stream,
                redact_images=redact_images,
            ),
        )

    def _body_params(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
        *,
        stream: bool,
        redact_images: bool = False,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": self._messages_to_unicom(messages, redact_images=redact_images),
            "stream": stream,
            "chat_template_kwargs": _dict_value(self.provider_config.get("chat_template_kwargs")),
            **generation_kwargs(self.generation_config, OPENAI_GENERATION_KEYS),
        }
        if tools and self.support_tools:
            body["tools"] = [openai_tool_schema(tool) for tool in tools]
        return body

    def _messages_to_unicom(
        self,
        messages: Sequence[Message],
        *,
        redact_images: bool = False,
    ) -> list[dict[str, Any]]:
        converted: list[dict[str, Any]] = []
        for message in messages:
            if message.role is MessageRole.TOOL:
                if self.support_tools:
                    converted.append(
                        {
                            "role": "tool",
                            "tool_call_id": message.tool_call_id,
                            "content": message.content,
                        }
                    )
                else:
                    converted.append(
                        {
                            "role": self.role_reflect["user"],
                            "content": f"Tool result {message.tool_call_id}:\n{message.content}",
                        }
                    )
                continue
            role = self.role_reflect.get(message.role.value, message.role.value)
            item: dict[str, Any] = {
                "role": role,
                "content": openai_message_content(message, redact_images=redact_images),
            }
            if message.tool_calls and self.support_tools:
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

    def _headers(self) -> dict[str, str]:
        headers = _unicom_extra_headers(self.provider_config)
        if self.nlpt_authorization:
            headers["nlpt-Authorization"] = self.nlpt_authorization
        return headers

    def _unwrap_body(self, data: dict[str, Any]) -> dict[str, Any]:
        body = data.get("UNI_BSS_BODY")
        if isinstance(body, dict):
            nested = body.get(self.req_key)
            if isinstance(nested, dict):
                return nested
        return data

    def _parse_complete_body(self, body: dict[str, Any]) -> ModelStreamChunk:
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices:
            return ModelStreamChunk(
                finish_reason="stop",
                usage=self._usage_dict(body.get("usage")),
            )
        choice = choices[0]
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        content = _string_value(message.get("content")) if isinstance(message, dict) else ""
        tool_calls = self._parse_openai_tool_calls(message.get("tool_calls", []))
        return ModelStreamChunk(
            content_delta=content,
            finish_reason=_string_value(choice.get("finish_reason") or "stop"),
            tool_calls=tool_calls,
            usage=self._usage_dict(body.get("usage")),
        )

    def _parse_stream_line(
        self,
        line: str,
        pending_tool_calls: dict[int, dict[str, str]],
    ) -> ModelStreamChunk | None:
        text = line.strip()
        if not text:
            return None
        if text.startswith("data:"):
            text = text[len("data:") :].strip()
        if text == "[DONE]":
            return ModelStreamChunk(finish_reason="[DONE]")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        body = self._unwrap_body(data)
        choices = body.get("choices", []) if isinstance(body, dict) else []
        if not choices:
            usage = self._usage_dict(body.get("usage"))
            return ModelStreamChunk(usage=usage) if usage else None
        choice = choices[0]
        delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
        content = _string_value(delta.get("content")) if isinstance(delta, dict) else ""
        self._accumulate_tool_deltas(delta.get("tool_calls", []), pending_tool_calls)
        finish_reason = _string_value(choice.get("finish_reason"))
        if finish_reason:
            return ModelStreamChunk(
                content_delta=content,
                finish_reason=finish_reason,
                tool_calls=tuple(
                    parse_openai_tool_call(raw) for raw in pending_tool_calls.values()
                ),
                usage=self._usage_dict(body.get("usage")),
            )
        if content:
            return ModelStreamChunk(content_delta=content)
        return None

    @staticmethod
    def _usage_dict(value: Any) -> dict[str, int]:
        if not isinstance(value, dict):
            return {}
        input_tokens = int(value.get("prompt_tokens") or value.get("input_tokens") or 0)
        output_tokens = int(value.get("completion_tokens") or value.get("output_tokens") or 0)
        total_tokens = int(value.get("total_tokens") or 0)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens or input_tokens + output_tokens,
        }

    @staticmethod
    def _parse_openai_tool_calls(tool_calls: Any) -> tuple[ToolCall, ...]:
        if not isinstance(tool_calls, list):
            return ()
        parsed: list[ToolCall] = []
        for item in tool_calls:
            if not isinstance(item, dict):
                continue
            function = item.get("function", {})
            if not isinstance(function, dict):
                function = {}
            parsed.append(
                parse_openai_tool_call(
                    {
                        "id": _string_value(item.get("id")),
                        "name": _string_value(function.get("name")),
                        "arguments": _string_value(function.get("arguments")),
                    }
                )
            )
        return tuple(parsed)

    @staticmethod
    def _accumulate_tool_deltas(
        tool_call_deltas: Any,
        pending_tool_calls: dict[int, dict[str, str]],
    ) -> None:
        if not isinstance(tool_call_deltas, list):
            return
        for fallback_index, item in enumerate(tool_call_deltas):
            if not isinstance(item, dict):
                continue
            raw_index = item.get("index", fallback_index)
            tool_index = raw_index if isinstance(raw_index, int) else fallback_index
            pending = pending_tool_calls.setdefault(
                tool_index,
                {"id": "", "name": "", "arguments": ""},
            )
            if item.get("id"):
                pending["id"] = _string_value(item.get("id"))
            function = item.get("function", {})
            if not isinstance(function, dict):
                continue
            if function.get("name"):
                pending["name"] = _string_value(function.get("name"))
            if function.get("arguments"):
                pending["arguments"] += _string_value(function.get("arguments"))


def _build_unicom_payload(
    *,
    app_id: str,
    app_secret: str,
    req_key: str,
    body_params: dict[str, Any],
) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S %f")[:-3]
    trans_id = datetime.now().strftime("%Y%m%d%H%M%S%f")[:-3] + "".join(
        random.choices(string.digits, k=6)
    )
    head: dict[str, Any] = {
        "APP_ID": app_id,
        "TIMESTAMP": timestamp,
        "TRANS_ID": trans_id,
    }
    head["TOKEN"] = generate_unicom_token(head, app_secret)
    return {
        "UNI_BSS_HEAD": head,
        "UNI_BSS_BODY": {req_key: body_params},
        "UNI_BSS_ATTACHED": {"MEDIA_INFO": ""},
    }


def _unicom_req_key(provider_config: dict[str, Any]) -> str:
    return _string_value(provider_config.get("req_key") or provider_config.get("app_req_key"))


def _unicom_role_reflect(provider_config: dict[str, Any]) -> dict[str, str]:
    value = provider_config.get("role_reflect")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    mapping = _dict_value(value)
    return {
        "system": _string_value(mapping.get("system") or "system"),
        "assistant": _string_value(mapping.get("assistant") or "assistant"),
        "user": _string_value(mapping.get("user") or "user"),
    }


def _unicom_extra_headers(provider_config: dict[str, Any]) -> dict[str, str]:
    return {
        key: value
        for key, value in _dict_string_values(provider_config.get("extra_headers")).items()
        if key.lower() not in {"nlpt-authorization", "authorization"}
    }


def _dict_string_values(value: Any) -> dict[str, str]:
    return {
        str(key): str(item)
        for key, item in _dict_value(value).items()
        if item is not None and str(item) != ""
    }


def _dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else "" if value is None else str(value)
