"""Anthropic Messages Provider parity tests."""

from __future__ import annotations

from typing import Any

import anthropic
import httpx
import pytest

from gewu_agent_runtime.adapters.llm import AnthropicChatModel, build_chat_model
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
    name = "read"
    description = "Read one file."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }


def _config(**updates: Any) -> ModelRuntimeConfig:
    values: dict[str, Any] = {
        "model_ref": "config-claude",
        "provider": "anthropic",
        "protocol": "anthropic-messages",
        "model_name": "claude-test",
        "api_key": "sk-test",
    }
    values.update(updates)
    return ModelRuntimeConfig.model_validate(values)


def test_anthropic_converts_system_tools_results_and_adjacent_user_content() -> None:
    model = AnthropicChatModel.from_runtime_config(_config())
    system, converted = model._messages_to_anthropic(
        [
            Message.system("system one"),
            Message.system("system two"),
            Message.user("hello"),
            Message.user("more"),
            Message.assistant(
                "checking",
                (ToolCall(tool_call_id="toolu-1", name="read", arguments={"path": "a.md"}),),
            ),
            Message.tool("toolu-1", '{"content":"ok"}'),
            Message.user("continue"),
        ]
    )

    assert system == "system one\n\nsystem two"
    assert converted == [
        {"role": "user", "content": "hello\n\nmore"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "checking"},
                {
                    "type": "tool_use",
                    "id": "toolu-1",
                    "name": "read",
                    "input": {"path": "a.md"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu-1",
                    "content": '{"content":"ok"}',
                },
                {"type": "text", "text": "continue"},
            ],
        },
    ]


def test_anthropic_request_preserves_schema_generation_and_image_shape() -> None:
    model = AnthropicChatModel.from_runtime_config(
        _config(
            support_vision=True,
            context_window=200_000,
            generation_config={"temperature": 0.2, "top_p": 0.8, "max_tokens": 4096},
        )
    )
    request = model._request_kwargs(
        [
            Message.system("system"),
            Message.user_parts(
                (
                    ContentPart.text_part("describe"),
                    ContentPart.image(
                        mime_type="image/jpeg",
                        data=b"abc",
                        resource_id="image-1",
                    ),
                )
            ),
        ],
        [_Tool()],
    )

    assert request == {
        "model": "claude-test",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": "YWJj",
                        },
                    },
                ],
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.2,
        "top_p": 0.8,
        "system": "system",
        "tools": [
            {
                "name": "read",
                "description": "Read one file.",
                "input_schema": _Tool.input_schema,
            }
        ],
    }
    assert model.context_window == 200_000


def test_anthropic_trace_redacts_images_and_omits_credentials() -> None:
    model = AnthropicChatModel.from_runtime_config(_config(support_vision=True))

    trace = model.trace_request(
        [
            Message.user_parts(
                (
                    ContentPart.image(
                        mime_type="image/png",
                        data=b"raw-secret-image",
                        resource_id="image-1",
                    ),
                )
            )
        ],
        [],
    )

    assert trace.provider == "anthropic"
    assert trace.request["messages"][0]["content"][0]["source"]["data"] == (
        "[redacted image base64]"
    )
    assert "api_key" not in trace.request


def test_anthropic_stream_tool_parser_and_usage_match_subscriber_contract() -> None:
    parsed = AnthropicChatModel._parse_tool_call(
        {"id": "toolu-1", "name": "read", "input": None, "input_json": '{"path":"a"}'}
    )
    malformed = AnthropicChatModel._parse_tool_call(
        {"id": "toolu-2", "name": "read", "input": None, "input_json": "bad"}
    )

    assert parsed.arguments == {"path": "a"}
    assert malformed.arguments == {"_raw_arguments": "bad"}
    assert AnthropicChatModel._usage_dict(120, 30) == {
        "input_tokens": 120,
        "output_tokens": 30,
        "total_tokens": 150,
    }
    assert AnthropicChatModel._usage_dict(0, 0) == {}


async def test_factory_builds_anthropic_without_closing_borrowed_pool() -> None:
    shared = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, request=request))
    )
    model = build_chat_model(_config(), http_client=shared)

    assert isinstance(model, AnthropicChatModel)
    await model.aclose()
    assert shared.is_closed is False
    await shared.aclose()


async def test_anthropic_translates_provider_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    model = AnthropicChatModel.from_runtime_config(_config(support_stream=False))
    request = httpx.Request("POST", "https://model.example.test")

    async def fail(**kwargs: Any) -> None:
        del kwargs
        raise anthropic.APITimeoutError(request=request)

    monkeypatch.setattr(model.client.messages, "create", fail)

    with pytest.raises(ModelTimeoutError):
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    await model.aclose()


async def test_anthropic_translates_permission_denial(monkeypatch: pytest.MonkeyPatch) -> None:
    model = AnthropicChatModel.from_runtime_config(_config(support_stream=False))
    request = httpx.Request("POST", "https://model.example.test")
    response = httpx.Response(403, request=request)

    async def fail(**kwargs: Any) -> None:
        del kwargs
        raise anthropic.PermissionDeniedError(
            "private provider detail",
            response=response,
            body={"secret": "private"},
        )

    monkeypatch.setattr(model.client.messages, "create", fail)

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
    ],
)
async def test_anthropic_translates_stream_failure(
    monkeypatch: pytest.MonkeyPatch,
    failure: Exception,
    error_type: type[Exception],
) -> None:
    model = AnthropicChatModel.from_runtime_config(_config())

    async def broken_stream() -> Any:
        raise failure
        yield None  # pragma: no cover

    class BrokenStreamContext:
        async def __aenter__(self) -> Any:
            return broken_stream()

        async def __aexit__(self, *args: object) -> None:
            del args

    def stream(**kwargs: Any) -> BrokenStreamContext:
        del kwargs
        return BrokenStreamContext()

    monkeypatch.setattr(model.client.messages, "stream", stream)

    with pytest.raises(error_type) as captured:
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    assert "private" not in str(captured.value)
    await model.aclose()
