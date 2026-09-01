"""OpenAI-compatible Provider parity tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from gewu_agent_runtime.adapters.llm import (
    DefaultProviderChatModelFactory,
    OpenAIChatModel,
    build_chat_model,
)
from gewu_agent_runtime.adapters.llm.generation import generation_kwargs
from gewu_agent_runtime.adapters.llm.openai_compat import (
    OPENAI_GENERATION_KEYS,
    parse_openai_tool_call,
)
from gewu_agent_runtime.llm import (
    ContentPart,
    Message,
    ModelPermissionDeniedError,
    ModelRuntimeConfig,
    ModelTimeoutError,
    ModelUnavailableError,
    ToolCall,
)


class _Tool:
    name = "lookup"
    description = "Look up one value."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def _config(**updates: Any) -> ModelRuntimeConfig:
    values: dict[str, Any] = {
        "model_ref": "config-1",
        "provider": "openai",
        "protocol": "openai-chat",
        "model_name": "gpt-test",
        "api_key": "sk-test",
    }
    values.update(updates)
    return ModelRuntimeConfig.model_validate(values)


def test_openai_request_preserves_tools_generation_images_and_usage_options() -> None:
    model = OpenAIChatModel.from_runtime_config(
        _config(
            support_vision=True,
            context_window=128_000,
            generation_config={
                "temperature": 0.2,
                "top_p": 0.8,
                "max_tokens": "512",
                "ignored": "value",
            },
        )
    )
    messages = [
        Message.system("system prompt"),
        Message.user_parts(
            (
                ContentPart.text_part("describe"),
                ContentPart.image(
                    mime_type="image/png",
                    data=b"abc",
                    resource_id="image-1",
                    name="a.png",
                ),
            )
        ),
        Message.assistant(
            "checking",
            (ToolCall(tool_call_id="call-1", name="lookup", arguments={"query": "x"}),),
        ),
        Message.tool("call-1", '{"value":"ok"}'),
    ]

    request = model._request_kwargs(messages, [_Tool()])

    assert request["model"] == "gpt-test"
    assert request["temperature"] == 0.2
    assert request["top_p"] == 0.8
    assert request["max_tokens"] == 512
    assert "ignored" not in request
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["messages"][1]["content"] == [
        {"type": "text", "text": "describe"},
        {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64,YWJj"},
        },
    ]
    assert request["messages"][2]["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "lookup", "arguments": '{"query": "x"}'},
    }
    assert request["messages"][3] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"value":"ok"}',
    }
    assert request["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Look up one value.",
                "parameters": _Tool.input_schema,
            },
        }
    ]
    assert model.model_ref == "config-1"
    assert model.provider == "openai"
    assert model.model_name == "gpt-test"
    assert model.context_window == 128_000
    assert model.effective_context_window == 128_000


def test_openai_trace_redacts_image_payload_without_changing_request_shape() -> None:
    model = OpenAIChatModel.from_runtime_config(_config(support_vision=True))

    trace = model.trace_request(
        [
            Message.user_parts(
                (
                    ContentPart.image(
                        mime_type="image/jpeg",
                        data=b"secret image",
                        resource_id="image-1",
                    ),
                )
            )
        ],
        [],
    )

    assert trace.provider == "openai"
    assert trace.request["messages"][0]["content"][0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,[redacted image base64]"},
    }


def test_openai_non_stream_request_hides_tools_when_capability_is_disabled() -> None:
    model = OpenAIChatModel.from_runtime_config(
        _config(
            support_stream=False,
            support_tools=False,
            generation_config={"stream": True},
        )
    )

    request = model._request_kwargs([Message.user("hello")], [_Tool()])

    assert "stream" not in request
    assert "stream_options" not in request
    assert "tools" not in request


def test_openai_tool_call_and_usage_normalization_match_subscriber_contract() -> None:
    parsed = parse_openai_tool_call(
        {"id": "call-1", "name": "lookup", "arguments": '{"query":"x"}'}
    )
    malformed = parse_openai_tool_call({"id": "call-2", "name": "lookup", "arguments": "not-json"})
    usage = OpenAIChatModel._usage_dict(
        SimpleNamespace(prompt_tokens=120, completion_tokens=30, total_tokens=0)
    )

    assert parsed.arguments == {"query": "x"}
    assert malformed.arguments == {"_raw_arguments": "not-json"}
    assert usage == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}


def test_generation_filter_rejects_invalid_integer_values() -> None:
    assert generation_kwargs(
        {"max_tokens": 0, "seed": "bad", "temperature": 0.1, "unknown": 1},
        OPENAI_GENERATION_KEYS,
    ) == {"temperature": 0.1}


async def test_factory_reuses_external_pool_and_rejects_use_after_close() -> None:
    shared = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    )
    factory = DefaultProviderChatModelFactory(http_client=shared)

    first = factory.create(_config())
    second = factory.create(_config(model_ref="config-2"))
    await factory.aclose()

    assert isinstance(first, OpenAIChatModel)
    assert isinstance(second, OpenAIChatModel)
    assert first is not second
    assert shared.is_closed is False
    with pytest.raises(RuntimeError, match="closed"):
        factory.create(_config())
    await shared.aclose()


def test_factory_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported runtime LLM provider"):
        build_chat_model(_config(provider="unknown"))


async def test_openai_translates_provider_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    model = OpenAIChatModel.from_runtime_config(_config(support_stream=False))
    request = httpx.Request("POST", "https://model.example.test")

    async def fail(**kwargs: Any) -> None:
        del kwargs
        raise openai.APITimeoutError(request=request)

    monkeypatch.setattr(model.client.chat.completions, "create", fail)

    with pytest.raises(ModelTimeoutError):
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    await model.aclose()


async def test_openai_translates_permission_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    model = OpenAIChatModel.from_runtime_config(_config(support_stream=False))
    request = httpx.Request("POST", "https://model.example.test")
    response = httpx.Response(403, request=request)

    async def fail(**kwargs: Any) -> None:
        del kwargs
        raise openai.PermissionDeniedError(
            "private provider detail",
            response=response,
            body={"secret": "private"},
        )

    monkeypatch.setattr(model.client.chat.completions, "create", fail)

    with pytest.raises(ModelPermissionDeniedError) as captured:
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    assert "private provider detail" not in str(captured.value)
    await model.aclose()


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (
            httpx.ReadTimeout(
                "private stream timeout",
                request=httpx.Request("POST", "https://model.example.test"),
            ),
            ModelTimeoutError,
        ),
        (
            httpx.ReadError(
                "private stream failure",
                request=httpx.Request("POST", "https://model.example.test"),
            ),
            ModelUnavailableError,
        ),
        (
            openai.APIError(
                "private SSE error",
                request=httpx.Request("POST", "https://model.example.test"),
                body={"secret": "private"},
            ),
            ModelUnavailableError,
        ),
    ],
)
async def test_openai_translates_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    error_type: type[Exception],
) -> None:
    model = OpenAIChatModel.from_runtime_config(_config())

    async def broken_stream() -> Any:
        raise failure
        yield None  # pragma: no cover

    async def create(**kwargs: Any) -> Any:
        del kwargs
        return broken_stream()

    monkeypatch.setattr(model.client.chat.completions, "create", create)

    with pytest.raises(error_type) as captured:
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    assert "private" not in str(captured.value)
    await model.aclose()
