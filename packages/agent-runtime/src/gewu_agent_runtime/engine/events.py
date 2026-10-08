"""Events emitted by the pure Agent Engine."""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.llm import Message, ToolCall
from gewu_agent_runtime.tools import Tool, ToolResult
from gewu_core.ids import new_id


class EventType(StrEnum):
    """Stable event discriminator values."""

    ASSISTANT_DELTA = "assistant_delta"
    ASSISTANT_INTERMEDIATE = "assistant_intermediate"
    ASSISTANT_FINAL = "assistant_final"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    ASK_REQUESTED = "ask_requested"
    ERROR = "error"


class ExecutionRequest(BaseModel):
    """Complete provider-neutral input to one model/tool loop."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    messages: tuple[Message, ...]
    tools: tuple[Tool, ...] = ()
    max_iterations: int = Field(default=20, ge=1)


class AssistantDelta(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.ASSISTANT_DELTA] = EventType.ASSISTANT_DELTA
    message_id: str
    content: str


class AssistantIntermediate(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.ASSISTANT_INTERMEDIATE] = EventType.ASSISTANT_INTERMEDIATE
    message_id: str
    content: str


class AssistantFinal(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.ASSISTANT_FINAL] = EventType.ASSISTANT_FINAL
    message_id: str
    content: str


class ToolUse(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.TOOL_USE] = EventType.TOOL_USE
    message_id: str = Field(default_factory=new_id)
    assistant_message_id: str = Field(default_factory=new_id, min_length=1)
    call: ToolCall
    assistant_text: str = ""


class ToolResultEvent(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
    type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    message_id: str = Field(default_factory=new_id)
    call: ToolCall
    result: ToolResult


class AskRequested(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.ASK_REQUESTED] = EventType.ASK_REQUESTED
    message_id: str = Field(default_factory=new_id)
    call: ToolCall
    ask_id: str
    questions: tuple[dict[str, Any], ...]
    timeout_seconds: int


class ExecutionError(BaseModel):
    model_config = ConfigDict(frozen=True)
    type: Literal[EventType.ERROR] = EventType.ERROR
    code: str
    message: str
    error_id: str = ""
    message_id: str = Field(default_factory=new_id)


AgentEvent = Annotated[
    AssistantDelta
    | AssistantIntermediate
    | AssistantFinal
    | ToolUse
    | ToolResultEvent
    | AskRequested
    | ExecutionError,
    Field(discriminator="type"),
]
