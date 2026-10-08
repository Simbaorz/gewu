"""Conversation context reconstruction parity."""

from __future__ import annotations

import pytest

from gewu_agent_runtime.context import ConversationContextBuilder
from gewu_agent_runtime.domain import (
    Conversation,
    ConversationMessage,
    FileState,
    FileStateCache,
    MessageKind,
    NewConversationMessage,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import Message, MessageRole, ToolCall
from gewu_agent_runtime.persistence import InMemoryRuntimeStore


class TrackingStore(InMemoryRuntimeStore):
    def __init__(self) -> None:
        super().__init__()
        self.list_messages_calls = 0
        self.page_calls: list[tuple[int, int | None, int]] = []

    async def list_messages(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        through_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        self.list_messages_calls += 1
        return await super().list_messages(
            conversation_id,
            after_sequence=after_sequence,
            through_sequence=through_sequence,
        )

    async def list_message_page(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> tuple[ConversationMessage, ...]:
        self.page_calls.append((after_sequence, before_sequence, limit))
        return await super().list_message_page(
            conversation_id,
            after_sequence=after_sequence,
            before_sequence=before_sequence,
            limit=limit,
        )


def _message(
    sequence: int,
    kind: MessageKind,
    role: MessageRole,
    content: str = "",
    *,
    payload: dict[str, object] | None = None,
    run_id: str = "",
) -> ConversationMessage:
    return ConversationMessage(
        conversation_id="conversation-1",
        sequence=sequence,
        role=role,
        kind=kind,
        content=content,
        payload=payload or {},
        run_id=run_id,
    )


def _tool_result(
    sequence: int,
    result: dict[str, object],
    *,
    tool_name: str = "read",
    is_error: bool = False,
    trace_result: bool = True,
) -> ConversationMessage:
    return _message(
        sequence,
        MessageKind.TOOL_RESULT,
        MessageRole.TOOL,
        payload={
            "tool_call_id": f"call-{sequence}",
            "tool_name": tool_name,
            "result": result,
            "is_error": is_error,
            "trace_result": trace_result,
        },
    )


async def test_context_pages_complete_history_without_full_range_query(
    principal: PrincipalRef,
) -> None:
    store = TrackingStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content=f"message-{sequence}",
            )
            for sequence in range(1, 6)
        ),
    )

    messages = await ConversationContextBuilder(store, history_page_size=2).build(
        conversation.conversation_id
    )

    assert [message.content for message in messages] == [
        "message-1\n\nmessage-2\n\nmessage-3\n\nmessage-4\n\nmessage-5"
    ]
    assert store.list_messages_calls == 0
    assert store.page_calls == [
        (0, None, 2),
        (2, None, 2),
        (4, None, 2),
    ]


def test_context_rejects_non_positive_history_page_size() -> None:
    with pytest.raises(ValueError, match="history_page_size must be greater than zero"):
        ConversationContextBuilder(InMemoryRuntimeStore(), history_page_size=0)


def test_context_converts_persisted_turn_events_to_model_messages() -> None:
    converted = ConversationContextBuilder.convert_messages(
        (
            _message(1, MessageKind.INPUT, MessageRole.USER, "hi"),
            _message(
                2,
                MessageKind.TOOL_USE,
                MessageRole.ASSISTANT,
                "checking",
                payload={
                    "tool_call_id": "call-1",
                    "tool_name": "echo",
                    "arguments": {"text": "hi"},
                },
            ),
            _tool_result(3, {"stdout": "hi"}, tool_name="echo"),
            _message(4, MessageKind.ASSISTANT, MessageRole.ASSISTANT, "done"),
        )
    )

    assert converted == [
        Message.user("hi"),
        Message.assistant(
            "checking",
            (
                ToolCall(
                    tool_call_id="call-1",
                    name="echo",
                    arguments={"text": "hi"},
                ),
            ),
        ),
        Message.tool("call-3", '{"stdout": "hi"}'),
        Message.assistant("done"),
    ]


def test_context_groups_only_explicit_assistant_response_ids() -> None:
    history = []
    for index, group in enumerate(("response-A", "response-A", "response-B"), start=1):
        history.extend(
            (
                _message(
                    index * 2,
                    MessageKind.TOOL_USE,
                    MessageRole.ASSISTANT,
                    "same text",
                    payload={
                        "assistant_message_id": group,
                        "tool_call_id": f"call-{index}",
                        "tool_name": "read",
                        "arguments": {"path": f"/{index}"},
                    },
                ),
                _message(
                    index * 2 + 1,
                    MessageKind.TOOL_RESULT,
                    MessageRole.TOOL,
                    payload={
                        "tool_call_id": f"call-{index}",
                        "result": {"value": index},
                        "trace_result": False,
                    },
                ),
            )
        )
    converted = ConversationContextBuilder.convert_messages(history)
    assert [
        tuple(call.tool_call_id for call in message.tool_calls)
        for message in converted
        if message.tool_calls
    ] == [("call-1", "call-2"), ("call-3",)]
    assert [message.role.value for message in converted] == [
        "assistant",
        "tool",
        "tool",
        "assistant",
        "tool",
    ]
    assert all(
        message.trace_result is False for message in converted if message.role.value == "tool"
    )


def test_context_restores_trace_policy_without_serializing_it_to_provider_payload() -> None:
    converted = ConversationContextBuilder.convert_messages(
        (_tool_result(1, {"value": "sensitive"}, trace_result=False),)
    )

    assert converted[0].trace_result is False
    assert "trace_result" not in converted[0].model_dump(mode="json")


def test_context_includes_meta_and_skips_llm_ignored_messages() -> None:
    converted = ConversationContextBuilder.convert_messages(
        (
            _message(
                1,
                MessageKind.META,
                MessageRole.USER,
                "<command-name>/review</command-name>",
                payload={
                    "is_meta": True,
                    "attachment_type": "skill_command_metadata",
                },
            ),
            _message(
                2,
                MessageKind.INPUT,
                MessageRole.USER,
                "/review diff",
                payload={"llm_ignore": True},
            ),
        )
    )

    assert converted == [Message.user("<command-name>/review</command-name>")]


def test_context_skips_only_current_run_input_when_rebuilding_history() -> None:
    converted = ConversationContextBuilder.convert_messages(
        (
            _message(
                1,
                MessageKind.INPUT,
                MessageRole.USER,
                "current",
                run_id="run-current",
            ),
            _message(
                2,
                MessageKind.META,
                MessageRole.USER,
                "<system-reminder>scene</system-reminder>",
                run_id="run-current",
            ),
            _message(
                3,
                MessageKind.INPUT,
                MessageRole.USER,
                "previous",
                run_id="run-previous",
            ),
        ),
        skip_input_run_id="run-current",
    )

    assert converted == [
        Message.user("<system-reminder>scene</system-reminder>"),
        Message.user("previous"),
    ]


def test_context_retains_file_state_only_with_real_read_evidence() -> None:
    cache = FileStateCache()
    expected = FileState(version="1", offset=0, limit=1)
    cache.set("/workspace/private/docs/a.md", expected)
    cache.set(
        "/workspace/private/docs/stale.md",
        FileState(version="1", offset=0, limit=1),
    )

    ConversationContextBuilder.reconcile_file_state(
        (
            _tool_result(
                1,
                {
                    "type": "text",
                    "content": "1: hello",
                    "file_path": "\\workspace\\private\\docs\\a.md",
                    "start_line": 0,
                    "num_lines": 1,
                },
            ),
        ),
        cache,
    )

    assert cache.get("/workspace/private/docs/a.md") == expected
    assert cache.get("/workspace/private/docs/stale.md") is None


@pytest.mark.parametrize(
    "result",
    (
        {
            "type": "text",
            "content": "1: hello",
            "file_path": "/workspace/private/docs/big.md",
            "start_line": 0,
            "num_lines": 1,
            "truncated": True,
        },
        {
            "type": "text",
            "content": "1: hello\n...(truncated at 25000 chars)\nUse offset=1 and limit=2000.",
            "file_path": "/workspace/private/docs/big.md",
            "start_line": 0,
            "num_lines": 1,
        },
    ),
)
def test_context_drops_file_state_for_truncated_read_evidence(
    result: dict[str, object],
) -> None:
    cache = FileStateCache()
    cache.set(
        "/workspace/private/docs/big.md",
        FileState(version="1", offset=0, limit=1),
    )

    ConversationContextBuilder.reconcile_file_state(
        (_tool_result(1, result),),
        cache,
    )

    assert cache.get("/workspace/private/docs/big.md") is None
