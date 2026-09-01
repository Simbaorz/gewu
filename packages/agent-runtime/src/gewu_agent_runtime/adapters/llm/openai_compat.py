"""OpenAI-compatible message and tool payload mapping."""

from __future__ import annotations

import base64
import json
from typing import Any

from gewu_agent_runtime.llm import (
    ContentPart,
    ContentPartType,
    Message,
    ModelTool,
    ToolCall,
)

OPENAI_GENERATION_KEYS = {
    "temperature",
    "top_p",
    "max_tokens",
    "presence_penalty",
    "frequency_penalty",
    "seed",
}


def openai_message_content(
    message: Message,
    *,
    redact_images: bool = False,
) -> str | list[dict[str, Any]] | None:
    """Convert text or multimodal content to OpenAI-compatible message content."""

    if not message.content_parts:
        return message.content or None
    return [_part_to_openai(part, redact_images=redact_images) for part in message.content_parts]


def openai_tool_schema(tool: ModelTool) -> dict[str, Any]:
    """Convert an internal Tool to OpenAI-compatible function format."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def parse_openai_tool_call(raw: dict[str, str]) -> ToolCall:
    """Parse an OpenAI-compatible Tool Call accumulator."""

    try:
        arguments = json.loads(raw["arguments"] or "{}")
    except json.JSONDecodeError:
        arguments = {"_raw_arguments": raw["arguments"]}
    return ToolCall(
        tool_call_id=raw["id"],
        name=raw["name"],
        arguments=arguments,
    )


def _part_to_openai(part: ContentPart, *, redact_images: bool) -> dict[str, Any]:
    if part.part_type is ContentPartType.IMAGE:
        encoded = (
            "[redacted image base64]"
            if redact_images
            else part.base64_data or base64.b64encode(part.data).decode("ascii")
        )
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{part.mime_type};base64,{encoded}"},
        }
    return {"type": "text", "text": part.text}
