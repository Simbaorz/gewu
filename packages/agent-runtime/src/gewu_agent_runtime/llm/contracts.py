"""Provider-neutral model message and streaming contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from enum import StrEnum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class MessageRole(StrEnum):
    """Roles exchanged with a chat model."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentPartType(StrEnum):
    """Kinds of structured model content."""

    TEXT = "text"
    IMAGE = "image"


class ContentPart(BaseModel):
    """One provider-neutral multimodal content part."""

    model_config = ConfigDict(frozen=True)

    part_type: ContentPartType
    text: str = ""
    mime_type: str = ""
    data: bytes = b""
    base64_data: str = ""
    resource_id: str = ""
    name: str = ""

    @classmethod
    def text_part(cls, value: str) -> ContentPart:
        """Build a text part."""

        return cls(part_type=ContentPartType.TEXT, text=value)

    @classmethod
    def image(
        cls,
        *,
        mime_type: str,
        data: bytes,
        resource_id: str,
        name: str = "",
    ) -> ContentPart:
        """Build an image part whose bytes are ready for an adapter."""

        return cls(
            part_type=ContentPartType.IMAGE,
            mime_type=mime_type,
            data=data,
            resource_id=resource_id,
            name=name,
        )

    @classmethod
    def encoded_image(
        cls,
        *,
        mime_type: str,
        base64_data: str,
        resource_id: str,
        name: str = "",
    ) -> ContentPart:
        """Build an image part already encoded for provider adapters."""

        return cls(
            part_type=ContentPartType.IMAGE,
            mime_type=mime_type,
            base64_data=base64_data,
            resource_id=resource_id,
            name=name,
        )


class ToolCall(BaseModel):
    """One tool invocation requested by a model."""

    model_config = ConfigDict(frozen=True)

    tool_call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One normalized chat message."""

    model_config = ConfigDict(frozen=True)

    role: MessageRole
    content: str = ""
    content_parts: tuple[ContentPart, ...] = ()
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""
    trace_result: bool = Field(default=True, exclude=True)

    @classmethod
    def system(cls, content: str) -> Message:
        """Build a system message."""

        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """Build a user message."""

        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def user_parts(cls, parts: Sequence[ContentPart]) -> Message:
        """Build a user message from structured content."""

        content_parts = tuple(parts)
        text = "\n".join(
            part.text for part in content_parts if part.part_type is ContentPartType.TEXT
        )
        return cls(role=MessageRole.USER, content=text, content_parts=content_parts)

    @classmethod
    def assistant(cls, content: str, tool_calls: Sequence[ToolCall] = ()) -> Message:
        """Build an assistant message."""

        return cls(role=MessageRole.ASSISTANT, content=content, tool_calls=tuple(tool_calls))

    @classmethod
    def tool(
        cls,
        tool_call_id: str,
        content: str,
        *,
        trace_result: bool = True,
    ) -> Message:
        """Build a tool-result message."""

        return cls(
            role=MessageRole.TOOL,
            tool_call_id=tool_call_id,
            content=content,
            trace_result=trace_result,
        )


class ModelStreamChunk(BaseModel):
    """One normalized streamed response from a chat model."""

    model_config = ConfigDict(frozen=True)

    content_delta: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    finish_reason: str = ""
    usage: dict[str, int] = Field(default_factory=dict)


class ModelRuntimeConfig(BaseModel):
    """Already-authorized provider settings used to build one model client."""

    model_config = ConfigDict(frozen=True)

    model_ref: str = Field(min_length=1)
    provider: str
    protocol: str
    model_name: str
    endpoint_url: str = ""
    api_key: str = Field(default="", exclude=True, repr=False)
    timeout_seconds: int = 600
    support_stream: bool = True
    support_tools: bool = True
    support_vision: bool = False
    context_window: int = Field(default=32_768, ge=1)
    generation_config: dict[str, Any] = Field(default_factory=dict)
    provider_config: dict[str, Any] = Field(default_factory=dict)
    credentials: dict[str, Any] = Field(default_factory=dict, exclude=True, repr=False)


class ModelTracePayload(BaseModel):
    """Provider request payload captured before a model call."""

    model_config = ConfigDict(frozen=True)

    provider: str = Field(min_length=1)
    request: dict[str, Any]


class ModelTraceSink(Protocol):
    """Optional best-effort observer for provider request payloads."""

    async def write(self, payload: ModelTracePayload) -> None:
        """Observe one already-redacted provider request."""


class ModelTool(Protocol):
    """Minimal tool definition required by model adapters."""

    @property
    def name(self) -> str:
        """Return the model-visible tool name."""

    @property
    def description(self) -> str:
        """Return the model-visible tool description."""

    @property
    def input_schema(self) -> dict[str, Any]:
        """Return the model-visible JSON schema."""


class ChatModel(Protocol):
    """A model capability already authorized and bound for one turn."""

    model_ref: str
    provider: str
    model_name: str
    support_vision: bool
    context_window: int

    def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        """Stream one completion."""


class ChatModelProvider(Protocol):
    """Resolve the authorized model for each model-loop iteration."""

    async def resolve(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> ChatModel:
        """Return the subscriber-selected model for the current request."""
