"""Cumulative context compaction tests."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

import gewu_agent_runtime.compaction.service as compaction_service_module
from gewu_agent_runtime.compaction import (
    CompactionPolicy,
    CompactionService,
    ContextCompactionFailedError,
    FullCompactProgress,
    FullCompactResult,
    HeuristicTokenEstimator,
    ScriptedCompactionModel,
)
from gewu_agent_runtime.context import CLEARED_TOOL_RESULT_CONTENT, ConversationContextBuilder
from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationCompaction,
    ConversationCompactionCommit,
    ConversationMessage,
    ConversationState,
    FileState,
    FileStateCache,
    MessageKind,
    NewConversationMessage,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import (
    Message,
    MessageRole,
    ModelStreamChunk,
    ScriptedChatModel,
)
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.runtime import AgentRuntime, TurnBindings, TurnRequest
from gewu_agent_runtime.tools import Tool
from gewu_agent_runtime.workspace import WorkspaceSession


class _TrackingRuntimeStore(InMemoryRuntimeStore):
    def __init__(self, *, recent_delay: float = 0.0) -> None:
        super().__init__()
        self.recent_delay = recent_delay
        self.recent_limits: list[int] = []
        self.page_limits: list[int] = []
        self.compaction_commits: list[ConversationCompactionCommit] = []

    async def list_recent_messages(
        self,
        conversation_id: str,
        *,
        limit: int,
        before_sequence: int | None = None,
    ) -> tuple[ConversationMessage, ...]:
        self.recent_limits.append(limit)
        if self.recent_delay:
            await asyncio.sleep(self.recent_delay)
        return await super().list_recent_messages(
            conversation_id,
            limit=limit,
            before_sequence=before_sequence,
        )

    async def list_message_page(
        self,
        conversation_id: str,
        *,
        after_sequence: int = 0,
        before_sequence: int | None = None,
        limit: int = 100,
    ) -> tuple[ConversationMessage, ...]:
        self.page_limits.append(limit)
        return await super().list_message_page(
            conversation_id,
            after_sequence=after_sequence,
            before_sequence=before_sequence,
            limit=limit,
        )

    async def commit_compaction_for_run(
        self,
        commit: ConversationCompactionCommit,
        run_id: str,
    ) -> bool:
        self.compaction_commits.append(commit)
        return await super().commit_compaction_for_run(commit, run_id)


class _FailingDeleteStateCache:
    def __init__(self) -> None:
        self.delete_calls: list[tuple[str, str]] = []

    async def get(self, conversation_id: str, kind: str) -> ConversationState | None:
        del conversation_id, kind
        return None

    async def set(self, state: ConversationState) -> None:
        del state

    async def delete(self, conversation_id: str, kind: str) -> None:
        self.delete_calls.append((conversation_id, kind))
        raise RuntimeError("cache refresh unavailable")


class _RawCompactionModel:
    model_ref = "raw:compaction"
    model_name = "raw"
    context_window = 20_000
    max_output_tokens = 2_000

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = list(responses)
        self.requests: list[tuple[Message, ...]] = []

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool],
    ) -> AsyncIterator[ModelStreamChunk]:
        assert not tools
        self.requests.append(tuple(messages))
        yield ModelStreamChunk(
            content_delta=self._responses.pop(0),
            finish_reason="stop",
        )


class _CountingContextBuilder(ConversationContextBuilder):
    compact_calls = 0

    @classmethod
    def compact_history_messages(
        cls,
        messages: Sequence[ConversationMessage],
        *,
        keep_recent_tool_results: int = 5,
    ) -> Sequence[ConversationMessage]:
        cls.compact_calls += 1
        return super().compact_history_messages(
            messages,
            keep_recent_tool_results=keep_recent_tool_results,
        )


async def _running_conversation(
    store: InMemoryRuntimeStore,
    principal: PrincipalRef,
    messages: Sequence[NewConversationMessage],
) -> tuple[Conversation, AgentRun]:
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(conversation.conversation_id, messages)
    run = await store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    return conversation, run


async def _start_run(
    store: InMemoryRuntimeStore,
    conversation: Conversation,
    principal: PrincipalRef,
) -> AgentRun:
    run = await store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    return await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )


async def test_default_compaction_estimator_uses_the_selected_model_name(
    principal: PrincipalRef,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_models: list[str] = []

    class RecordingEstimator(HeuristicTokenEstimator):
        def __init__(self, model_name: str) -> None:
            selected_models.append(model_name)

    monkeypatch.setattr(
        compaction_service_module,
        "ContextTokenEstimator",
        RecordingEstimator,
        raising=False,
    )
    store = InMemoryRuntimeStore()
    conversation, run = await _running_conversation(store, principal, ())

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=None,
        tools=(),
        policy=CompactionPolicy(),
        model=ScriptedCompactionModel((), model_name="subscriber-compaction-model"),
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    assert result.compacted is False
    assert selected_models == ["subscriber-compaction-model"]


def _tool_result(
    sequence: int,
    name: str,
    result: dict[str, object],
    *,
    is_error: bool = False,
) -> ConversationMessage:
    return ConversationMessage(
        conversation_id="conversation-1",
        sequence=sequence,
        role=MessageRole.TOOL,
        kind=MessageKind.TOOL_RESULT,
        content=json.dumps(result, ensure_ascii=False),
        payload={
            "tool_call_id": f"call-{sequence}",
            "tool_name": name,
            "result": result,
            "is_error": is_error,
        },
    )


async def test_compaction_replaces_prefix_with_cumulative_summary(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="long historical message " * 20,
            )
            for _ in range(8)
        ),
    )
    run = await store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    model = ScriptedCompactionModel(["Cumulative history."] * 20, context_window=1_500)
    events = [
        event
        async for event in CompactionService(store).prepare_events(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(trigger_percent=50, target_percent=50),
            model=model,
            file_state_cache=FileStateCache(),
            skill_state={},
        )
    ]
    context = await ConversationContextBuilder(store).build(conversation.conversation_id)
    compacted = await store.get_latest_compaction(conversation.conversation_id)

    assert [event.phase for event in events if isinstance(event, FullCompactProgress)] == [
        "started",
        "completed",
    ]
    assert isinstance(events[-1], FullCompactResult) and events[-1].compacted is True
    assert compacted is not None and compacted.through_sequence == 6
    assert "Cumulative history." in context[0].content
    assert len(context) == 1
    assert context[0].content.count("long historical message") == 40
    lifecycle = [
        value
        for value in await store.list_messages(conversation.conversation_id)
        if value.kind is MessageKind.MEMORY_COMPACTION
    ]
    assert [value.payload["phase"] for value in lifecycle] == ["started", "completed"]
    assert len(store.compaction_commits) == 1
    lifecycle_commit = store.compaction_commits[0]
    assert isinstance(events[0], FullCompactProgress)
    assert isinstance(events[1], FullCompactProgress)
    assert [value.message_id for value in lifecycle_commit.messages] == [
        events[0].message_id,
        events[1].message_id,
    ]
    assert all(value.role is MessageRole.SYSTEM for value in lifecycle_commit.messages)
    assert all(value.payload["llm_ignore"] is True for value in lifecycle_commit.messages)
    assert all(
        value.payload["compaction_id"] == lifecycle_commit.compaction.compaction_id
        for value in lifecycle_commit.messages
    )
    assert model.requests[0][0].content.startswith("You are a conversation compaction assistant")


def test_summary_serialization_skips_ignored_messages_without_losing_turn_boundaries() -> None:
    ignored_input = ConversationMessage(
        conversation_id="conversation-1",
        sequence=1,
        role=MessageRole.USER,
        kind=MessageKind.INPUT,
        content="/review diff",
        payload={"llm_ignore": True},
    )
    skill_meta = ConversationMessage(
        conversation_id="conversation-1",
        sequence=2,
        role=MessageRole.USER,
        kind=MessageKind.META,
        content="<command-args>diff</command-args>",
        payload={"attachment_type": "skill_command_metadata"},
    )
    ignored_assistant = ConversationMessage(
        conversation_id="conversation-1",
        sequence=3,
        role=MessageRole.ASSISTANT,
        kind=MessageKind.ASSISTANT,
        content="duplicate intermediate text",
        payload={"llm_ignore": True},
    )
    next_input = ConversationMessage(
        conversation_id="conversation-1",
        sequence=4,
        role=MessageRole.USER,
        kind=MessageKind.INPUT,
        content="next question",
    )

    groups = CompactionService._turn_groups(
        (ignored_input, skill_meta, ignored_assistant, next_input)
    )
    segments = [CompactionService._serialize_summary_messages(group) for group in groups]

    assert len(groups) == 2
    assert "/review diff" not in segments[0]
    assert "duplicate intermediate text" not in segments[0]
    assert "<command-args>diff</command-args>" in segments[0]
    assert "next question" in segments[1]


async def test_full_compact_message_budget_stops_before_an_extra_page(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation, run = await _running_conversation(
        store,
        principal,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=f"message-{index}",
            )
            for index in range(5)
        ),
    )

    with pytest.raises(ContextCompactionFailedError, match="online maintenance capacity"):
        await CompactionService(store).prepare(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(
                enabled=False,
                history_page_size=2,
                max_history_messages=3,
                max_history_pages=10,
            ),
            model=ScriptedCompactionModel([]),
            file_state_cache=FileStateCache(),
            skill_state={},
        )

    assert store.recent_limits == [2, 1]
    assert store.page_limits == []
    assert await store.get_latest_compaction(conversation.conversation_id) is None


async def test_full_compact_page_budget_stops_database_scanning(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation, run = await _running_conversation(
        store,
        principal,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=f"message-{index}",
            )
            for index in range(3)
        ),
    )

    with pytest.raises(ContextCompactionFailedError, match="online maintenance capacity"):
        await CompactionService(store).prepare(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(
                enabled=False,
                history_page_size=1,
                max_history_messages=100,
                max_history_pages=2,
            ),
            model=ScriptedCompactionModel([]),
            file_state_cache=FileStateCache(),
            skill_state={},
        )

    assert store.recent_limits == [1, 1]
    assert await store.get_latest_compaction(conversation.conversation_id) is None


async def test_full_compact_wall_time_cancels_slow_history_query(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore(recent_delay=0.05)
    conversation, run = await _running_conversation(
        store,
        principal,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="current",
            ),
        ),
    )

    with pytest.raises(ContextCompactionFailedError, match="online maintenance capacity"):
        await CompactionService(store).prepare(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(enabled=False, wall_time_seconds=0.01),
            model=ScriptedCompactionModel([]),
            file_state_cache=FileStateCache(),
            skill_state={},
        )

    assert store.recent_limits == [500]
    assert await store.get_latest_compaction(conversation.conversation_id) is None


async def test_full_compact_page_projection_does_not_block_event_loop(
    principal: PrincipalRef,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryRuntimeStore()
    conversation, run = await _running_conversation(
        store,
        principal,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="current",
            ),
        ),
    )
    service = CompactionService(store)
    original_projection = service._project_history_page
    heartbeat = threading.Event()
    heartbeat_seen_during_projection: list[bool] = []

    def slow_projection(*args: Any, **kwargs: Any) -> Any:
        time.sleep(0.05)
        heartbeat_seen_during_projection.append(heartbeat.is_set())
        return original_projection(*args, **kwargs)

    async def pulse() -> None:
        await asyncio.sleep(0.01)
        heartbeat.set()

    monkeypatch.setattr(service, "_project_history_page", slow_projection)

    await asyncio.gather(
        service.prepare(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(enabled=False),
            model=ScriptedCompactionModel([]),
            file_state_cache=FileStateCache(),
            skill_state={},
        ),
        pulse(),
    )

    assert heartbeat_seen_during_projection == [True]


async def test_invalid_compaction_pointer_falls_back_without_skipping_history(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(
        Conversation(owner=principal, latest_compaction_id="missing")
    )
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="must remain",
            ),
        ),
    )

    history = await ConversationContextBuilder(store).build(conversation.conversation_id)

    assert [message.content for message in history] == ["must remain"]


async def test_full_compact_uses_complete_input_boundary_and_rebuilds_state(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    long_old = "旧甲" * 7_000
    long_recent = "近期乙" * 3_400
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=long_old,
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content="old listing",
                payload={"attachment_type": "skill_listing", "skill_names": ["old"]},
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content="old skill",
                payload={"attachment_type": "skill_content", "skill_name": "old"},
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.TOOL_USE,
                payload={"tool_call_id": "read-old", "tool_name": "read"},
            ),
            NewConversationMessage(
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                payload={
                    "tool_call_id": "read-old",
                    "tool_name": "read",
                    "result": {
                        "type": "text",
                        "content": "old",
                        "file_path": "/old.md",
                        "start_line": 0,
                        "num_lines": 1,
                    },
                },
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="old answer",
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=long_old,
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="second answer",
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content=long_recent,
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content="recent listing",
                payload={"attachment_type": "skill_listing", "skill_names": ["recent"]},
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.META,
                content="recent skill",
                payload={"attachment_type": "skill_content", "skill_name": "recent"},
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.TOOL_USE,
                payload={"tool_call_id": "read-recent", "tool_name": "read"},
            ),
            NewConversationMessage(
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                payload={
                    "tool_call_id": "read-recent",
                    "tool_name": "read",
                    "result": {
                        "type": "text",
                        "content": "recent",
                        "file_path": "/recent.md",
                        "start_line": 0,
                        "num_lines": 1,
                    },
                },
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="recent answer",
            ),
        ),
    )
    run = await _start_run(store, conversation, principal)
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="当前问题",
                run_id=run.run_id,
            ),
        ),
    )
    file_state = FileStateCache()
    file_state.set("/old.md", FileState(version="1", offset=0, limit=1))
    file_state.set("/recent.md", FileState(version="1", offset=0, limit=1))
    file_state.mark_clean()
    skill_state = {
        "sent_skill_names": ["old", "recent"],
        "invoked_skills": {
            "old": {"name": "old"},
            "recent": {"name": "recent"},
        },
    }
    model = ScriptedCompactionModel(["第一代中文摘要"] * 20)

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=Message.user("当前问题"),
        tools=(),
        policy=CompactionPolicy(),
        model=model,
        file_state_cache=file_state,
        skill_state=skill_state,
    )

    assert result.compacted is True
    assert len(store.compaction_commits) == 1
    commit = store.compaction_commits[0]
    assert commit.compaction.source_from_sequence == 1
    assert commit.compaction.through_sequence == 8
    assert commit.compaction.summary == "第一代中文摘要"
    assert commit.compaction.post_compaction_tokens < 25_000
    assert commit.state_payloads["file"] == {
        "files": {"recent.md": {"version": "1", "offset": 0, "limit": 1}}
    }
    assert commit.state_payloads["skill"]["sent_skill_names"] == ["recent"]
    assert set(commit.state_payloads["skill"]["invoked_skills"]) == {"recent"}
    assert file_state.has("/recent.md")
    assert not file_state.has("/old.md")
    assert any("<conversation-summary>" in message.content for message in result.messages)
    assert model.requests[0][0].content.startswith("You are a conversation compaction assistant")


async def test_invalid_summary_near_hard_limit_is_repaired_once_and_not_committed(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation, run = await _running_conversation(
        store,
        principal,
        tuple(
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="very long context " * 2_000,
            )
            for _ in range(3)
        ),
    )
    model = _RawCompactionModel(("invalid envelope", "still invalid"))

    events: list[FullCompactProgress | FullCompactResult] = []
    with pytest.raises(ContextCompactionFailedError):
        async for event in CompactionService(store).prepare_events(
            conversation.conversation_id,
            run_id=run.run_id,
            request_id=run.request_id,
            system_prompt="system",
            current_input_message=None,
            tools=(),
            policy=CompactionPolicy(),
            model=model,
            file_state_cache=FileStateCache(),
            skill_state={},
        ):
            events.append(event)

    assert len(model.requests) == 2
    assert "invalid envelope" in model.requests[1][1].content
    assert await store.get_latest_compaction(conversation.conversation_id) is None
    persisted = await store.list_messages(conversation.conversation_id)
    assert all(message.kind is not MessageKind.MEMORY_COMPACTION for message in persisted)
    assert [event.phase for event in events if isinstance(event, FullCompactProgress)] == [
        "started",
        "cancelled",
    ]
    assert isinstance(events[0], FullCompactProgress)
    assert isinstance(events[1], FullCompactProgress)
    assert events[0].message_id == events[1].message_id
    assert store.compaction_commits == []


async def test_post_commit_cache_refresh_failure_does_not_fail_compaction(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="long historical message " * 3_000,
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="old answer",
            ),
        ),
    )
    await store.save_state(
        ConversationState(
            conversation_id=conversation.conversation_id,
            kind="skill",
            revision=1,
            payload={"sent_skill_names": ["stale"]},
        ),
        expected_revision=0,
    )
    cache = _FailingDeleteStateCache()
    runtime = AgentRuntime(store=store, state_cache=cache)
    session = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=conversation.conversation_id,
            content="current question",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="done")]]),
            workspace=workspace,
            compaction_policy=CompactionPolicy(trigger_percent=50, target_percent=50),
            compaction_model=ScriptedCompactionModel(
                ["summary"] * 20,
                context_window=20_000,
            ),
        ),
    )

    _ = [event async for event in session.stream()]

    run = await store.get_run(session.run_id)
    assert run is not None and run.status is AgentRunStatus.COMPLETED
    assert await store.get_latest_compaction(conversation.conversation_id) is not None
    assert await store.get_state(conversation.conversation_id, "skill") is None
    assert cache.delete_calls == [(conversation.conversation_id, "skill")]


async def test_reserved_context_can_trigger_full_compact(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation, run = await _running_conversation(
        store,
        principal,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="history " * 2_000,
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="old answer",
            ),
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="current question",
            ),
        ),
    )
    candidate = await ConversationContextBuilder(store).build(
        conversation.conversation_id,
        system_prompt="system",
    )
    base_tokens = HeuristicTokenEstimator().estimate(candidate)
    context_window = ((base_tokens + 256) * 100 + 74) // 75
    trigger_tokens = context_window * 75 // 100
    assert base_tokens < trigger_tokens <= base_tokens + 512

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=None,
        tools=(),
        policy=CompactionPolicy(),
        model=ScriptedCompactionModel(
            ["summary"] * 20,
            context_window=context_window,
        ),
        file_state_cache=FileStateCache(),
        skill_state={},
        reserved_context_tokens=512,
    )

    assert result.compacted is True
    assert result.estimated_tokens < context_window * 90 // 100


async def test_next_generation_uses_previous_summary_and_only_new_messages(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="old question",
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="old answer",
            ),
        ),
    )
    previous = ConversationCompaction(
        compaction_id="compact-1",
        conversation_id=conversation.conversation_id,
        generation=1,
        source_from_sequence=1,
        through_sequence=2,
        summary="first generation summary",
    )
    await store.save_compaction(previous, expected_previous_id="")
    await store.append_messages(
        conversation.conversation_id,
        tuple(
            NewConversationMessage(
                role=(MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT),
                kind=(MessageKind.INPUT if index % 2 == 0 else MessageKind.ASSISTANT),
                content=(f"new fact {index} " * 100),
            )
            for index in range(6)
        ),
    )
    run = await store.create_run(
        AgentRun(conversation_id=conversation.conversation_id, invoker=principal)
    )
    await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    model = ScriptedCompactionModel(["second generation summary"] * 20, context_window=4_000)

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=None,
        tools=(),
        policy=CompactionPolicy(trigger_percent=50, target_percent=50),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    committed = await store.get_latest_compaction(conversation.conversation_id)
    assert result.compacted is True
    assert committed is not None
    assert committed.generation == 2
    assert committed.previous_compaction_id == previous.compaction_id
    assert committed.source_from_sequence == 3
    assert any("first generation summary" in request[1].content for request in model.requests)
    assert any("new fact" in request[1].content for request in model.requests)
    assert all("old question" not in request[1].content for request in model.requests)


async def test_micro_compact_runs_once_and_can_avoid_full_compact(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    messages: list[NewConversationMessage] = [
        NewConversationMessage(
            role=MessageRole.USER,
            kind=MessageKind.INPUT,
            content="old turn",
        )
    ]
    for index in range(5):
        call_id = f"read-{index}"
        messages.extend(
            (
                NewConversationMessage(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.TOOL_USE,
                    payload={"tool_call_id": call_id, "tool_name": "read"},
                ),
                NewConversationMessage(
                    role=MessageRole.TOOL,
                    kind=MessageKind.TOOL_RESULT,
                    payload={
                        "tool_call_id": call_id,
                        "tool_name": "read",
                        "result": {"type": "text", "content": "x" * 60_000},
                    },
                ),
            )
        )
    await store.append_messages(conversation.conversation_id, tuple(messages))
    run = await _start_run(store, conversation, principal)
    persisted = await store.list_messages(conversation.conversation_id)
    builder = _CountingContextBuilder(store, keep_recent_tool_results=0)
    raw_request = builder.build_messages(
        Message.system("system"),
        summary="",
        history=persisted,
    )
    assert HeuristicTokenEstimator().estimate(raw_request) > 37_500
    _CountingContextBuilder.compact_calls = 0
    model = ScriptedCompactionModel([])
    service = CompactionService(store, keep_recent_tool_results=0)
    service._context = builder

    result = await service.prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=None,
        tools=(),
        policy=CompactionPolicy(),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    assert result.compacted is False
    assert result.estimated_tokens < 37_500
    assert _CountingContextBuilder.compact_calls == 1
    assert store.compaction_commits == []
    assert model.requests == []


async def test_smaller_model_recompacts_summary_without_advancing_boundary(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal, next_sequence=81))
    previous = ConversationCompaction(
        compaction_id="compact-large",
        conversation_id=conversation.conversation_id,
        generation=4,
        source_from_sequence=1,
        through_sequence=80,
        summary="既有超长摘要" * 7_000,
        model_ref="large:model",
        model_name="large-model",
        context_window=200_000,
        pre_compaction_tokens=140_000,
        post_compaction_tokens=90_000,
    )
    await store.save_compaction(previous, expected_previous_id="")
    run = await _start_run(store, conversation, principal)
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="当前问题",
                run_id=run.run_id,
            ),
        ),
    )
    model = ScriptedCompactionModel(["缩短后的摘要"] * 10)

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=Message.user("当前问题"),
        tools=(),
        policy=CompactionPolicy(),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    assert result.compacted is True
    commit = store.compaction_commits[0].compaction
    assert commit.generation == 5
    assert commit.source_from_sequence is None
    assert commit.through_sequence == 80
    assert commit.summary == "缩短后的摘要"


async def test_saturated_summary_still_compacts_post_boundary_messages(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal, next_sequence=21))
    previous = ConversationCompaction(
        compaction_id="compact-1",
        conversation_id=conversation.conversation_id,
        generation=1,
        source_from_sequence=1,
        through_sequence=20,
        summary="既有累计摘要" * 2_000,
        model_ref="large:model",
        model_name="large-model",
        context_window=200_000,
        pre_compaction_tokens=140_000,
        post_compaction_tokens=90_000,
    )
    await store.save_compaction(previous, expected_previous_id="")
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="必须带入新摘要的近期事实",
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="近期回答",
            ),
        ),
    )
    run = await _start_run(store, conversation, principal)
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="当前问题",
                run_id=run.run_id,
            ),
        ),
    )
    estimator = HeuristicTokenEstimator()
    fixed_candidate = ConversationContextBuilder.build_messages(
        Message.system("system"),
        summary=previous.summary,
        history=(),
        current_input_message=Message.user("当前问题"),
    )
    fixed_tokens = estimator.estimate(fixed_candidate)
    context_window = (fixed_tokens * 100 + 75) // 76
    assert fixed_tokens >= context_window * 75 // 100
    assert fixed_tokens < context_window * 90 // 100
    model = ScriptedCompactionModel(
        ["包含近期事实的新摘要"] * 10,
        context_window=context_window,
    )

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=Message.user("当前问题"),
        tools=(),
        policy=CompactionPolicy(),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    assert result.compacted is True
    commit = store.compaction_commits[0].compaction
    assert commit.source_from_sequence == 21
    assert commit.through_sequence == 22
    assert "必须带入新摘要的近期事实" in model.requests[0][1].content


async def test_full_compact_pages_history_without_materializing_all_rows(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    messages: list[NewConversationMessage] = []
    for index in range(70):
        messages.extend(
            (
                NewConversationMessage(
                    role=MessageRole.USER,
                    kind=MessageKind.INPUT,
                    content=f"第{index}轮" + "历史" * 300,
                ),
                NewConversationMessage(
                    role=MessageRole.ASSISTANT,
                    kind=MessageKind.ASSISTANT,
                    content="answer",
                ),
            )
        )
    await store.append_messages(conversation.conversation_id, tuple(messages))
    run = await _start_run(store, conversation, principal)
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="当前问题",
                run_id=run.run_id,
            ),
        ),
    )
    model = ScriptedCompactionModel(
        [f"滚动摘要{index}" for index in range(40)],
        context_window=8_000,
    )

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=Message.user("当前问题"),
        tools=(),
        policy=CompactionPolicy(history_page_size=50),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    assert result.compacted is True
    assert store.recent_limits
    assert store.page_limits
    assert max((*store.recent_limits, *store.page_limits)) <= 50
    assert store.compaction_commits[0].compaction.source_from_sequence == 1


async def test_summary_fragments_each_fit_small_model_input_budget(
    principal: PrincipalRef,
) -> None:
    store = _TrackingRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="齉" * 20_000,
            ),
            NewConversationMessage(
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content="old answer",
            ),
        ),
    )
    run = await _start_run(store, conversation, principal)
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage(
                role=MessageRole.USER,
                kind=MessageKind.INPUT,
                content="当前问题",
                run_id=run.run_id,
            ),
        ),
    )
    model = ScriptedCompactionModel(
        [f"摘要{index}" for index in range(40)],
        context_window=14_000,
    )

    result = await CompactionService(store).prepare(
        conversation.conversation_id,
        run_id=run.run_id,
        request_id=run.request_id,
        system_prompt="system",
        current_input_message=Message.user("当前问题"),
        tools=(),
        policy=CompactionPolicy(),
        model=model,
        file_state_cache=FileStateCache(),
        skill_state={},
    )

    input_budget = model.context_window - model.max_output_tokens
    estimator = HeuristicTokenEstimator()
    assert result.compacted is True
    assert len(model.requests) >= 2
    assert all(estimator.estimate(request) <= input_budget for request in model.requests)


def test_micro_compact_clears_only_old_successful_low_value_results() -> None:
    messages = (
        _tool_result(1, "read", {"content": "old read"}),
        _tool_result(2, "bash", {"stdout": "old bash"}),
        _tool_result(3, "read", {"error": "missing"}, is_error=True),
        _tool_result(4, "skill", {"skill_name": "review"}),
        _tool_result(5, "ask_user", {"answers": {"q": "a"}}),
        _tool_result(6, "grep", {"content": "recent grep"}),
    )

    compacted = ConversationContextBuilder.compact_history_messages(
        messages,
        keep_recent_tool_results=1,
    )
    converted = ConversationContextBuilder.convert_messages(compacted)

    assert CLEARED_TOOL_RESULT_CONTENT in converted[0].content
    assert CLEARED_TOOL_RESULT_CONTENT in converted[1].content
    assert "missing" in converted[2].content
    assert "review" in converted[3].content
    assert "answers" in converted[4].content
    assert "recent grep" in converted[5].content
    assert messages[0].payload["result"] == {"content": "old read"}


def test_micro_compact_zero_clears_every_eligible_result() -> None:
    messages = (
        _tool_result(1, "read", {"content": "old read"}),
        _tool_result(2, "bash", {"stdout": "new bash"}),
    )

    compacted = ConversationContextBuilder.compact_history_messages(
        messages,
        keep_recent_tool_results=0,
    )
    converted = ConversationContextBuilder.convert_messages(compacted)

    assert all(CLEARED_TOOL_RESULT_CONTENT in value.content for value in converted)
    assert all("old read" not in value.content for value in converted)
    assert all("new bash" not in value.content for value in converted)


async def test_context_micro_compact_removes_stale_file_read_evidence(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    conversation = await store.create_conversation(Conversation(owner=principal))
    read_result = _tool_result(
        1,
        "read",
        {
            "type": "text",
            "content": "1: hello",
            "file_path": "/workspace/private/docs/a.md",
            "start_line": 0,
            "num_lines": 1,
        },
    ).model_copy(update={"conversation_id": conversation.conversation_id})
    await store.append_messages(
        conversation.conversation_id,
        (
            NewConversationMessage.model_validate(
                read_result.model_dump(exclude={"conversation_id", "sequence"})
            ),
        ),
    )
    cache = FileStateCache()
    cache.set(
        "/workspace/private/docs/a.md",
        FileState(version="1", offset=0, limit=1),
    )

    await ConversationContextBuilder(store, keep_recent_tool_results=0).build(
        conversation.conversation_id,
        file_state_cache=cache,
    )

    assert cache.get("/workspace/private/docs/a.md") is None
