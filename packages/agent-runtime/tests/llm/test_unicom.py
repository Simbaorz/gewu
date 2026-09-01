"""China Unicom Open Service Provider parity tests."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from gewu_agent_runtime.adapters.llm import ChinaUnicomOpenServiceChatModel, build_chat_model
from gewu_agent_runtime.adapters.llm.unicom import generate_unicom_token
from gewu_agent_runtime.llm import (
    ContentPart,
    Message,
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelRuntimeConfig,
    ModelTimeoutError,
    ModelUnavailableError,
    ToolCall,
)


class _Tool:
    name = "lookup"
    description = "Lookup data."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }


def _config(**updates: Any) -> ModelRuntimeConfig:
    values: dict[str, Any] = {
        "model_ref": "config-unicom",
        "provider": "unicom",
        "protocol": "chinaunicom-open-service",
        "model_name": "Qwen3",
        "endpoint_url": "https://unicom.example.test/api",
        "credentials": {
            "app_id": "app",
            "app_secret": "secret",
            "nlpt_authorization": "nlpt-token",
        },
        "provider_config": {"req_key": "YUANJING_MODEL_REQ"},
    }
    values.update(updates)
    return ModelRuntimeConfig.model_validate(values)


def test_unicom_builds_signed_envelope_headers_roles_and_generation() -> None:
    model = ChinaUnicomOpenServiceChatModel.from_runtime_config(
        _config(
            generation_config={"temperature": 0.3, "max_tokens": 256, "stream": False},
            provider_config={
                "req_key": "YUANJING_MODEL_REQ",
                "role_reflect": {
                    "system": "sys",
                    "user": "human",
                    "assistant": "bot",
                },
                "chat_template_kwargs": {"enable_thinking": False},
                "extra_headers": {
                    "scene_code": "IT-01-0002",
                    "Authorization": "must-not-leak",
                    "nlpt-Authorization": "must-not-override",
                },
            },
        )
    )

    payload = model._request_payload(
        [Message.system("system prompt"), Message.user("hello"), Message.assistant("hi")],
        [],
        stream=False,
    )
    body = payload["UNI_BSS_BODY"]["YUANJING_MODEL_REQ"]

    assert payload["UNI_BSS_HEAD"]["APP_ID"] == "app"
    assert len(payload["UNI_BSS_HEAD"]["TOKEN"]) == 32
    assert payload["UNI_BSS_ATTACHED"] == {"MEDIA_INFO": ""}
    assert body == {
        "model": "Qwen3",
        "messages": [
            {"role": "sys", "content": "system prompt"},
            {"role": "human", "content": "hello"},
            {"role": "bot", "content": "hi"},
        ],
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.3,
        "max_tokens": 256,
    }
    assert model._headers() == {
        "scene_code": "IT-01-0002",
        "nlpt-Authorization": "nlpt-token",
    }


def test_unicom_token_excludes_existing_token_and_is_order_independent() -> None:
    first = generate_unicom_token(
        {"APP_ID": "app", "TIMESTAMP": "time", "TOKEN": "ignored"}, "secret"
    )
    second = generate_unicom_token({"TIMESTAMP": "time", "APP_ID": "app"}, "secret")

    assert first == second
    assert len(first) == 32


def test_unicom_converts_tools_and_downgrades_history_when_tools_disabled() -> None:
    supported = ChinaUnicomOpenServiceChatModel.from_runtime_config(_config())
    messages = [
        Message.assistant(
            "checking",
            (
                ToolCall(
                    tool_call_id="call-1",
                    name="read",
                    arguments={"file_path": "a.md"},
                ),
            ),
        ),
        Message.tool("call-1", '{"content":"ok"}'),
    ]
    converted = supported._messages_to_unicom(messages)

    assert converted[0]["tool_calls"][0] == {
        "id": "call-1",
        "type": "function",
        "function": {"name": "read", "arguments": '{"file_path": "a.md"}'},
    }
    assert converted[1] == {
        "role": "tool",
        "tool_call_id": "call-1",
        "content": '{"content":"ok"}',
    }

    disabled = ChinaUnicomOpenServiceChatModel.from_runtime_config(_config(support_tools=False))
    assert disabled._messages_to_unicom([messages[1]]) == [
        {"role": "user", "content": 'Tool result call-1:\n{"content":"ok"}'}
    ]


def test_unicom_parses_wrapped_complete_response_and_usage() -> None:
    model = build_chat_model(_config())
    assert isinstance(model, ChinaUnicomOpenServiceChatModel)
    body = model._unwrap_body(
        {
            "UNI_BSS_BODY": {
                "YUANJING_MODEL_REQ": {
                    "choices": [
                        {
                            "message": {
                                "content": "hello",
                                "tool_calls": [
                                    {
                                        "id": "call-1",
                                        "function": {
                                            "name": "lookup",
                                            "arguments": '{"query":"x"}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "tool_calls",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 4,
                        "total_tokens": 7,
                    },
                }
            }
        }
    )
    chunk = model._parse_complete_body(body)

    assert chunk.content_delta == "hello"
    assert chunk.finish_reason == "tool_calls"
    assert chunk.tool_calls[0].arguments == {"query": "x"}
    assert chunk.usage == {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7}


def test_unicom_stream_parser_accumulates_tool_arguments_and_done_marker() -> None:
    model = ChinaUnicomOpenServiceChatModel.from_runtime_config(_config())
    pending: dict[int, dict[str, str]] = {}
    first = model._parse_stream_line(
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call-1",
                                    "function": {"name": "lookup", "arguments": '{"query":'},
                                }
                            ]
                        }
                    }
                ]
            }
        ),
        pending,
    )
    final = model._parse_stream_line(
        "data: "
        + json.dumps(
            {
                "choices": [
                    {
                        "delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"x"}'}}]},
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            }
        ),
        pending,
    )

    assert first is None
    assert final is not None
    assert final.finish_reason == "tool_calls"
    assert final.tool_calls[0].arguments == {"query": "x"}
    assert final.usage["total_tokens"] == 7
    assert model._parse_stream_line("data: [DONE]", pending).finish_reason == "[DONE]"  # type: ignore[union-attr]
    assert model._parse_stream_line("not-json", pending) is None


def test_unicom_trace_omits_envelope_credentials_and_redacts_images() -> None:
    model = ChinaUnicomOpenServiceChatModel.from_runtime_config(_config(support_vision=True))
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
        [_Tool()],
    )

    rendered = str(trace.request)
    assert trace.provider == "unicom"
    assert trace.request["url"] == "https://unicom.example.test/api"
    assert trace.request["req_key"] == "YUANJING_MODEL_REQ"
    assert trace.request["body"]["tools"][0]["function"]["name"] == "lookup"
    assert "UNI_BSS_HEAD" not in trace.request
    assert "secret" not in rendered
    assert "nlpt-token" not in rendered
    assert "[redacted image base64]" in rendered


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    [
        (401, ModelAuthenticationError),
        (403, ModelPermissionDeniedError),
        (408, ModelTimeoutError),
        (429, ModelRateLimitError),
        (500, ModelUnavailableError),
        (400, ModelRequestRejectedError),
    ],
)
async def test_unicom_translates_http_status(
    status_code: int,
    error_type: type[Exception],
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status_code, request=request))
    )
    model = ChinaUnicomOpenServiceChatModel.from_runtime_config(
        _config(),
        http_client=client,
    )

    with pytest.raises(error_type):
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    await client.aclose()


@pytest.mark.parametrize(
    ("failure_type", "error_type"),
    [
        (httpx.ReadTimeout, ModelTimeoutError),
        (httpx.ConnectError, ModelUnavailableError),
    ],
)
async def test_unicom_translates_transport_failure(
    failure_type: type[httpx.TransportError],
    error_type: type[Exception],
) -> None:
    def fail(request: httpx.Request) -> httpx.Response:
        raise failure_type("private upstream detail", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(fail))
    model = ChinaUnicomOpenServiceChatModel.from_runtime_config(
        _config(),
        http_client=client,
    )

    with pytest.raises(error_type) as captured:
        [chunk async for chunk in model.stream_chat([Message.user("hello")], [])]

    assert "private upstream detail" not in str(captured.value)
    await client.aclose()
