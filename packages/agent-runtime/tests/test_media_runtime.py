"""Persistent Runtime integration for image turns and Ask resume."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from gewu_agent_runtime import AttachmentRef, InvocationTarget, InvocationTargetKind
from gewu_agent_runtime.builtins import ask_user_tool
from gewu_agent_runtime.domain import MessageKind
from gewu_agent_runtime.engine import AskRequested, AssistantFinal, ExecutionError
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import ModelStreamChunk, ScriptedChatModel, ToolCall
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.runtime import AgentRuntime, AskAnswer, TurnBindings, TurnRequest
from gewu_agent_runtime.tools import ToolSet
from gewu_agent_runtime.workspace import WorkspaceSession


class RecordingLoader:
    def __init__(self, data: bytes = b"abc") -> None:
        self.data = data
        self.keys: list[str] = []

    async def read(self, resource_key: str) -> bytes:
        self.keys.append(resource_key)
        return self.data


class NoVisionModel(ScriptedChatModel):
    support_vision = False


def _attachment() -> AttachmentRef:
    return AttachmentRef(
        attachment_id="img-1",
        resource_key="private/storage/key",
        original_name="evidence.png",
        mime_type="image/png",
        size_bytes=3,
    )


def test_turn_request_requires_content_attachment_or_invocation_target(
    principal: PrincipalRef,
) -> None:
    with pytest.raises(ValidationError, match="content or attachments is required"):
        TurnRequest(invoker=principal, content="")

    assert TurnRequest(invoker=principal, content="", attachments=(_attachment(),)).attachments
    assert TurnRequest(
        invoker=principal,
        content="",
        invocation_target=InvocationTarget(
            kind=InvocationTargetKind.SCENE,
            resource_id="scene-1",
        ),
    ).invocation_target


async def test_runtime_persists_only_public_attachment_metadata_and_sends_current_image(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    loader = RecordingLoader()
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="inspect", attachments=(_attachment(),)),
        TurnBindings(model=model, workspace=workspace, attachment_loader=loader),
    )

    events = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    persisted_input = next(value for value in messages if value.kind is MessageKind.INPUT)
    serialized_run = json.dumps((await store.get_run(session.run_id)).model_dump(mode="json"))

    assert isinstance(events[-1], AssistantFinal)
    assert loader.keys == ["private/storage/key"]
    assert persisted_input.payload["attachments"] == [
        {
            "attachment_id": "img-1",
            "original_name": "evidence.png",
            "mime_type": "image/png",
            "size_bytes": 3,
        }
    ]
    assert "private/storage/key" not in json.dumps(persisted_input.payload)
    assert "private/storage/key" not in serialized_run
    assert "YWJj" not in serialized_run
    assert model.requests[0][0][-1].content_parts[-1].base64_data == "YWJj"


async def test_non_vision_model_fails_before_model_call_but_keeps_public_input(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    loader = RecordingLoader()
    model = NoVisionModel([[ModelStreamChunk(content_delta="must not run")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="", attachments=(_attachment(),)),
        TurnBindings(model=model, workspace=workspace, attachment_loader=loader),
    )

    events = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    persisted_error = next(value for value in messages if value.kind is MessageKind.ERROR)

    assert isinstance(events[-1], ExecutionError)
    assert events[-1].code == "vision_unsupported"
    assert events[-1].message == "当前模型不支持图片输入。"
    assert events[-1].message_id == persisted_error.message_id
    assert model.requests == []
    assert (
        next(value for value in messages if value.kind is MessageKind.INPUT).payload["attachments"][
            0
        ]["attachment_id"]
        == "img-1"
    )


async def test_ask_resume_rebuilds_image_as_placeholder_without_reloading_bytes(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=60)
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-1",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Which target?",
                                        "header": "Target",
                                        "options": [{"label": "A"}, {"label": "B"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="selected A")],
        ]
    )
    store = InMemoryRuntimeStore()
    loader = RecordingLoader()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((ask,)),
        attachment_loader=loader,
    )
    first = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            content="analyze this image",
            attachments=(_attachment(),),
        ),
        bindings,
    )
    suspended = [event async for event in first.stream()]
    ask_event = suspended[-1]
    assert isinstance(ask_event, AskRequested)
    pending = await store.get_state(first.conversation_id, "pending_ask")
    assert pending is not None
    serialized_pending = json.dumps(pending.model_dump(mode="json"))
    assert "private/storage/key" not in serialized_pending
    assert "YWJj" not in serialized_pending

    resumed = await runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Which target?": "A"}),
        bindings=bindings,
    )
    resumed_events = [event async for event in resumed.stream()]

    assert isinstance(resumed_events[-1], AssistantFinal)
    assert loader.keys == ["private/storage/key"]
    assert all(not message.content_parts for message in model.requests[-1][0])
    assert any(
        "[Image: evidence.png, image/png, attachment_id=img-1]" in message.content
        for message in model.requests[-1][0]
    )
