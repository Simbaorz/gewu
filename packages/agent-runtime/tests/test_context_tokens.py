"""Conservative model-request token estimation behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import pytest

from gewu_agent_runtime import context_tokens as context_tokens_module
from gewu_agent_runtime.context_tokens import (
    IMAGE_ESTIMATE_TOKENS,
    ContextTokenEstimator,
    initialize_context_token_encodings,
)
from gewu_agent_runtime.llm import ContentPart, Message, ToolCall
from gewu_agent_runtime.tools import Tool, ToolResult, tool


def test_estimator_counts_messages_tool_calls_and_schema() -> None:
    estimator = ContextTokenEstimator("gpt-4o")
    messages = (
        Message.system("system"),
        Message.user("中文问题"),
        Message.assistant(
            "checking",
            (
                ToolCall(
                    tool_call_id="call-1",
                    name="query",
                    arguments={"filters": {"地区": ["南京", "苏州"]}},
                ),
            ),
        ),
        Message.tool("call-1", '{"rows":[1,2,3]}'),
    )

    @tool(description="Query business rows.")
    def query(filters: dict[str, object]) -> ToolResult:
        return ToolResult(output={"filters": filters})

    without_tool = estimator.estimate(messages)
    with_tool = estimator.estimate(messages, (query,))

    assert with_tool > without_tool > estimator.count_text("中文问题")


def test_estimator_uses_conservative_image_budget() -> None:
    estimator = ContextTokenEstimator("gpt-4o")
    text_only = estimator.estimate((Message.user("describe"),))
    image = estimator.estimate(
        (
            Message.user_parts(
                (
                    ContentPart.text_part("describe"),
                    ContentPart.image(
                        mime_type="image/png",
                        data=b"binary-data" * 100,
                        resource_id="image-1",
                    ),
                )
            ),
        )
    )

    assert image - text_only >= IMAGE_ESTIMATE_TOKENS


def test_unknown_model_and_special_literals_remain_available_offline() -> None:
    estimator = ContextTokenEstimator("private-provider-model")

    assert estimator.count_text("hello 世界") > 0
    assert estimator.estimate((Message.user("literal <|endoftext|> value"),)) > 0


async def test_required_complete_cache_rejects_missing_local_tokenizers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(context_tokens_module, "_REGISTRY", None)

    with pytest.raises(RuntimeError, match="Required tiktoken cache is incomplete"):
        await initialize_context_token_encodings(require_complete_cache=True)


async def test_optional_complete_cache_allows_offline_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(context_tokens_module, "_REGISTRY", None)

    def unexpected_loader(name: str) -> object:
        raise AssertionError(f"unexpected tokenizer load: {name}")

    monkeypatch.setattr(context_tokens_module.tiktoken, "get_encoding", unexpected_loader)

    await initialize_context_token_encodings(require_complete_cache=False)

    assert ContextTokenEstimator("gpt-4o").count_text("hello 世界") > 0


async def test_async_estimate_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_estimate = ContextTokenEstimator.estimate
    heartbeat = threading.Event()
    heartbeat_seen_during_estimate: list[bool] = []

    def slow_estimate(
        estimator: ContextTokenEstimator,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
    ) -> int:
        time.sleep(0.05)
        heartbeat_seen_during_estimate.append(heartbeat.is_set())
        return original_estimate(estimator, messages, tools)

    async def pulse() -> None:
        await asyncio.sleep(0.01)
        heartbeat.set()

    monkeypatch.setattr(ContextTokenEstimator, "estimate", slow_estimate)

    await asyncio.gather(
        ContextTokenEstimator("gpt-4o").estimate_async((Message.user("hello"),)),
        pulse(),
    )

    assert heartbeat_seen_during_estimate == [True]
