"""Persistent Agent Runtime behavior tests."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncGenerator, AsyncIterator, Sequence
from datetime import timedelta
from typing import Literal, cast
from uuid import UUID

import httpx
import pytest

from gewu_agent_runtime.builtins import (
    BashOutput,
    SkillCapacityExceededError,
    ask_user_tool,
    bash,
    read,
    write,
)
from gewu_agent_runtime.compaction import (
    CompactionPolicy,
    ContextCompactionFailedError,
    ContextLimitExceededError,
    ScriptedCompactionModel,
)
from gewu_agent_runtime.coordination import InMemoryRunLease, RunLeaseStatus
from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationCompaction,
    ConversationMessage,
    ConversationState,
    ConversationStatus,
    MessageKind,
    NewConversationMessage,
)
from gewu_agent_runtime.engine import (
    AskRequested,
    AssistantDelta,
    AssistantFinal,
    AssistantIntermediate,
    ExecutionError,
    ToolResultEvent,
    ToolUse,
)
from gewu_agent_runtime.identity import PrincipalRef, PrincipalType
from gewu_agent_runtime.llm import (
    Message,
    ModelAuthenticationError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelStreamChunk,
    ModelTimeoutError,
    ModelTool,
    ModelUnavailableError,
    ScriptedChatModel,
    ToolCall,
)
from gewu_agent_runtime.media import ImageCapacityExceededError
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.prompts import build_system_prompt
from gewu_agent_runtime.runtime import (
    AgentRuntime,
    AskAnswer,
    AskExpiredError,
    AskNotPendingError,
    PreparedAgentTurn,
    PrincipalMismatchError,
    RuntimeEvent,
    SafeExecutionError,
    SubscriberMismatchError,
    TurnBindings,
    TurnRequest,
    TurnSession,
)
from gewu_agent_runtime.tools import (
    PersistencePolicy,
    ToolResult,
    ToolRuntimeBindings,
    ToolSet,
    tool,
)
from gewu_agent_runtime.workspace import WorkspaceSession
from gewu_core.secrets import JsonSecretCipher
from gewu_core.time import utc_now


class BlockingChatModel:
    """Hold one model stream open until a test releases process capacity."""

    model_ref = "blocking:test"
    provider = "blocking"
    model_name = "blocking"
    support_vision = True
    context_window = 1_000_000

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        del messages, tools
        self.started.set()
        await self.release.wait()
        yield ModelStreamChunk(content_delta="released")


class FailingChatModel:
    """Raise one private provider failure from the model stream."""

    model_ref = "failing:test"
    provider = "failing"
    model_name = "failing"
    support_vision = True
    context_window = 1_000_000
    error_message = "private-provider-secret"

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        del messages, tools
        raise RuntimeError(self.error_message)
        yield ModelStreamChunk()  # pragma: no cover


class RaisingChatModel:
    """Raise one caller-selected failure from the model stream."""

    model_ref = "raising:test"
    provider = "raising"
    model_name = "raising"
    support_vision = True
    context_window = 1_000_000

    def __init__(self, error: BaseException) -> None:
        self.error = error

    async def stream_chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ModelTool],
    ) -> AsyncIterator[ModelStreamChunk]:
        del messages, tools
        raise self.error
        yield ModelStreamChunk()  # pragma: no cover


class CredentialBearingChatModel(ScriptedChatModel):
    """Expose provider credentials to prove Runtime snapshots use an allowlist."""

    api_key = "model-api-secret"
    credentials = {"token": "provider-token-secret"}


class FailingStatusRunLease(InMemoryRunLease):
    """Simulate a temporary shared coordination backend failure."""

    async def status(self, conversation_id: str, run_id: str) -> RunLeaseStatus:
        del conversation_id, run_id
        raise RuntimeError("coordination-private-secret")


class LostStatusRunLease(InMemoryRunLease):
    """Lose the shared lease after the database run has been claimed."""

    async def status(self, conversation_id: str, run_id: str) -> RunLeaseStatus:
        del conversation_id, run_id
        return RunLeaseStatus.LOST


class PollingInMemoryRunLease(InMemoryRunLease):
    @property
    def monitor_interval_seconds(self) -> float:
        return 0.001


class ConversationChangesAfterAdmissionStore(InMemoryRuntimeStore):
    """Return a changed conversation snapshot only inside the acquired RunLease."""

    def __init__(self, *, missing: bool) -> None:
        super().__init__()
        self.missing = missing
        self.get_conversation_calls = 0

    async def get_conversation(self, conversation_id: str) -> Conversation | None:
        self.get_conversation_calls += 1
        conversation = await super().get_conversation(conversation_id)
        if self.get_conversation_calls < 2 or conversation is None:
            return conversation
        if self.missing:
            return None
        return conversation.model_copy(update={"status": ConversationStatus.ARCHIVED})


class OwnershipLostOnTerminalCommitStore(InMemoryRuntimeStore):
    """Revoke database ownership immediately before one terminal message commit."""

    def __init__(self) -> None:
        super().__init__()
        self.terminal_commit_attempts = 0

    async def append_messages_for_run(
        self,
        run_id: str,
        messages: Sequence[NewConversationMessage],
        *,
        finish_status: AgentRunStatus | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> tuple[ConversationMessage, ...] | None:
        if finish_status is not None:
            self.terminal_commit_attempts += 1
            await super().transition_run(
                run_id,
                expected=(AgentRunStatus.RUNNING,),
                status=AgentRunStatus.FAILED,
                error_code="superseded",
                error_message="ownership transferred",
            )
        return await super().append_messages_for_run(
            run_id,
            messages,
            finish_status=finish_status,
            error_code=error_code,
            error_message=error_message,
        )


async def _collect_turn(session: TurnSession) -> list[RuntimeEvent]:
    return [event async for event in session.stream()]


async def _wait_for_admission(
    runtime: AgentRuntime,
    *,
    active: int,
    queued: int,
) -> None:
    for _ in range(100):
        stats = await runtime.admission_stats()
        if stats.active == active and stats.queued == queued:
            return
        await asyncio.sleep(0)
    pytest.fail(f"Admission state did not reach active={active}, queued={queued}.")


async def test_runtime_rejects_cross_subscriber_conversation_access(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    foreign_invoker = principal.model_copy(update={"subscriber_id": "subscriber-b"})
    runtime = AgentRuntime(store=store)

    with pytest.raises(
        SubscriberMismatchError,
        match="Runtime data cannot cross subscribers",
    ):
        await runtime.start_turn(
            TurnRequest(
                invoker=foreign_invoker,
                conversation_id="c1",
                content="must not run",
            ),
            TurnBindings(
                model=ScriptedChatModel([[ModelStreamChunk(content_delta="unexpected")]]),
                workspace=workspace,
            ),
        )

    assert await store.get_latest_run("c1") is None
    assert await store.list_messages("c1") == ()


async def test_runtime_rejects_implicit_cross_principal_conversation_access(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    other_invoker = principal.model_copy(update={"principal_id": "other-user"})
    runtime = AgentRuntime(store=store)

    with pytest.raises(
        PrincipalMismatchError,
        match="Runtime data cannot cross principals",
    ):
        await runtime.start_turn(
            TurnRequest(
                invoker=other_invoker,
                conversation_id="c1",
                content="must not run",
            ),
            TurnBindings(
                model=ScriptedChatModel([[ModelStreamChunk(content_delta="unexpected")]]),
                workspace=workspace,
            ),
        )

    assert await store.get_latest_run("c1") is None
    assert await store.list_messages("c1") == ()

    run = await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        run.run_id,
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    with pytest.raises(PrincipalMismatchError, match="Runtime data cannot cross principals"):
        await runtime.resume_ask(
            run_id=run.run_id,
            invoker=other_invoker,
            answer=AskAnswer(ask_id="ask-1"),
            bindings=TurnBindings(
                model=ScriptedChatModel([[ModelStreamChunk(content_delta="unexpected")]]),
                workspace=workspace,
            ),
        )
    with pytest.raises(PrincipalMismatchError, match="Runtime data cannot cross principals"):
        await runtime.interrupt_conversation("c1", other_invoker)
    with pytest.raises(PrincipalMismatchError, match="Runtime data cannot cross principals"):
        await runtime.inspect_conversation("c1", other_invoker)

    unchanged = await store.get_run(run.run_id)
    assert unchanged is not None and unchanged.status is AgentRunStatus.WAITING_INPUT


async def test_runtime_allows_host_authorized_delegation_within_subscriber(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    service_invoker = PrincipalRef(
        subscriber_id=principal.subscriber_id,
        principal_id="integration-service",
        principal_type=PrincipalType.SERVICE,
    )
    store = InMemoryRuntimeStore()
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="delegated")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(
            invoker=service_invoker,
            owner=principal,
            content="authorized by the host",
        ),
        TurnBindings(model=model, workspace=workspace),
    )

    events = await _collect_turn(session)

    conversation = await store.get_conversation(session.conversation_id)
    assert conversation is not None and conversation.owner == principal
    assert session.run.invoker == service_invoker
    assert isinstance(events[-1], AssistantFinal)
    assert events[-1].content == "delegated"

    continued = await AgentRuntime(store=store).start_turn(
        TurnRequest(
            invoker=service_invoker,
            owner=principal,
            conversation_id=session.conversation_id,
            content="continue explicit delegation",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="continued")]]),
            workspace=workspace,
        ),
    )
    continued_events = await _collect_turn(continued)
    assert isinstance(continued_events[-1], AssistantFinal)
    assert continued_events[-1].content == "continued"
    snapshot = await AgentRuntime(store=store).inspect_conversation(
        session.conversation_id,
        service_invoker,
        owner=principal,
    )
    assert snapshot.run is not None
    assert snapshot.run.invoker == service_invoker
    assert (
        await AgentRuntime(store=store).interrupt_conversation(
            session.conversation_id,
            service_invoker,
            owner=principal,
        )
        is True
    )


async def test_runtime_starts_one_complete_host_prepared_turn(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    prepared = PreparedAgentTurn(
        request=TurnRequest(invoker=principal, content="prepared by host"),
        bindings=TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="complete")]]),
            workspace=workspace,
        ),
    )

    session = await AgentRuntime(store=store).start_prepared_turn(prepared)
    events = await _collect_turn(session)

    assert isinstance(events[-1], AssistantFinal)
    assert events[-1].content == "complete"
    assert session.run.invoker == principal


def test_prepared_turn_requires_exactly_one_capability_source(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    request = TurnRequest(invoker=principal, content="prepared by host")
    bindings = TurnBindings(
        model=ScriptedChatModel([[ModelStreamChunk(content_delta="complete")]]),
        workspace=workspace,
    )

    async def bindings_factory() -> TurnBindings:
        return bindings

    with pytest.raises(ValueError, match="Exactly one"):
        PreparedAgentTurn(request=request)
    with pytest.raises(ValueError, match="Exactly one"):
        PreparedAgentTurn(
            request=request,
            bindings=bindings,
            bindings_factory=bindings_factory,
        )


async def test_runtime_uses_the_host_preallocated_run_id(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(
            run_id="host-run-1",
            invoker=principal,
            content="work",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="done")]]),
            workspace=workspace,
        ),
    )

    assert session.run_id == "host-run-1"
    assert await store.get_run("host-run-1") == session.run


async def test_runtime_fails_a_duplicate_caller_assigned_input_message_id(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    model = ScriptedChatModel(
        [
            [ModelStreamChunk(content_delta="first")],
            [ModelStreamChunk(content_delta="must not run")],
        ]
    )
    runtime = AgentRuntime(store=store)
    first = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id="conversation-1",
            content="first input",
            request_id="client-request-1",
            input_message_id="client-request-1",
        ),
        TurnBindings(model=model, workspace=workspace),
    )
    first_events = await _collect_turn(first)
    assert isinstance(first_events[-1], AssistantFinal)

    duplicate = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id="conversation-1",
            content="duplicate input",
            request_id="client-request-1",
            input_message_id="client-request-1",
        ),
        TurnBindings(model=model, workspace=workspace),
    )
    duplicate_events = await _collect_turn(duplicate)

    assert len(model.requests) == 1
    assert len(duplicate_events) == 1
    assert isinstance(duplicate_events[0], ExecutionError)
    assert duplicate_events[0].code == "message_write_conflict"
    messages = await store.list_messages("conversation-1")
    assert [message.kind for message in messages] == [
        MessageKind.INPUT,
        MessageKind.ASSISTANT,
        MessageKind.ERROR,
    ]
    assert messages[0].message_id == "client-request-1"
    assert messages[-1].request_id == "client-request-1"
    failed_run = await store.get_run(duplicate.run_id)
    assert failed_run is not None
    assert failed_run.status is AgentRunStatus.FAILED
    assert failed_run.error_code == "message_write_conflict"
    conversation = await store.get_conversation("conversation-1")
    assert conversation is not None
    assert conversation.active_run_id is None


async def test_runtime_defers_subscriber_bindings_until_run_is_admitted(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    factory_calls = 0
    observed_statuses: list[AgentRunStatus] = []

    async def bindings_factory() -> TurnBindings:
        nonlocal factory_calls
        factory_calls += 1
        latest = await store.get_latest_run("conversation-1")
        assert latest is not None
        observed_statuses.append(latest.status)
        return TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="done")]]),
            workspace=workspace,
        )

    session = await runtime.start_prepared_turn(
        PreparedAgentTurn(
            request=TurnRequest(
                invoker=principal,
                conversation_id="conversation-1",
                content="hello",
            ),
            bindings_factory=bindings_factory,
        )
    )

    assert factory_calls == 0
    events = await _collect_turn(session)
    assert factory_calls == 1
    assert observed_statuses == [AgentRunStatus.RUNNING]
    assert isinstance(events[-1], AssistantFinal)


async def test_runtime_inspection_marks_confirmed_lost_run_as_failed(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )

    snapshot = await AgentRuntime(store=store).inspect_conversation("c1", principal)

    assert snapshot.run is not None
    assert snapshot.run.status is AgentRunStatus.FAILED
    assert snapshot.run.error_code == "run_lease_lost"


async def test_runtime_exposes_business_neutral_run_lease_status() -> None:
    lease = InMemoryRunLease()
    runtime = AgentRuntime(store=InMemoryRuntimeStore(), run_lease=lease)
    assert await lease.acquire("c1", "run-1") is True

    assert await runtime.run_lease_status("c1", "run-1") is RunLeaseStatus.ACTIVE

    await lease.release("c1", "run-1")
    assert await runtime.run_lease_status("c1", "run-1") is RunLeaseStatus.LOST


async def test_runtime_inspection_exposes_only_the_latest_compaction_boundary(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    conversation = await store.create_conversation(
        Conversation(conversation_id="c1", owner=principal)
    )
    await store.save_compaction(
        ConversationCompaction(
            compaction_id="compact-0",
            conversation_id=conversation.conversation_id,
            generation=1,
            through_sequence=8,
            summary="older private summary",
        ),
        expected_previous_id="",
    )
    await store.save_compaction(
        ConversationCompaction(
            compaction_id="compact-1",
            conversation_id=conversation.conversation_id,
            generation=2,
            previous_compaction_id="compact-0",
            through_sequence=17,
            summary="private compacted conversation summary",
        ),
        expected_previous_id="compact-0",
    )

    snapshot = await runtime.inspect_conversation("c1", principal)

    assert snapshot.compaction_boundary is not None
    assert snapshot.compaction_boundary.model_dump() == {
        "compaction_id": "compact-1",
        "generation": 2,
        "through_sequence": 17,
    }
    assert "private compacted conversation summary" not in snapshot.model_dump_json()


async def test_runtime_inspection_preserves_running_state_when_coordination_fails(
    principal: PrincipalRef,
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )
    runtime = AgentRuntime(store=store, run_lease=FailingStatusRunLease())

    with caplog.at_level(logging.ERROR):
        snapshot = await runtime.inspect_conversation("c1", principal)

    assert snapshot.run is not None
    assert snapshot.run.status is AgentRunStatus.RUNNING
    assert snapshot.run.error_code == ""
    assert "Unable to reconcile Agent run coordination state" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert "coordination-private-secret" not in caplog.text


async def test_runtime_inspection_repairs_waiting_run_without_pending_ask(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )

    snapshot = await AgentRuntime(store=store).inspect_conversation("c1", principal)

    assert snapshot.run is not None
    assert snapshot.run.status is AgentRunStatus.CANCELLED
    assert snapshot.run.error_code == "ask_state_missing"
    assert snapshot.pending_ask is None


async def test_runtime_inspection_repairs_malformed_pending_ask(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    await store.save_state(
        ConversationState(
            conversation_id="c1",
            kind="pending_ask",
            revision=1,
            payload={"run_id": "run-1", "ask_id": "ask-1"},
            expires_at=utc_now() + timedelta(minutes=5),
        ),
        expected_revision=0,
    )

    snapshot = await AgentRuntime(store=store).inspect_conversation("c1", principal)

    assert snapshot.run is not None
    assert snapshot.run.status is AgentRunStatus.CANCELLED
    assert snapshot.run.error_code == "ask_state_missing"
    assert snapshot.pending_ask is None
    assert await store.get_state("c1", "pending_ask", include_expired=True) is None


async def test_runtime_cleanup_resolves_expired_pending_asks(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    await store.save_state(
        ConversationState(
            conversation_id="c1",
            kind="pending_ask",
            revision=1,
            payload={
                "run_id": "run-1",
                "ask_id": "ask-1",
                "tool_call_id": "call-1",
                "tool_name": "ask_user",
            },
            expires_at=utc_now() - timedelta(seconds=1),
        ),
        expected_revision=0,
    )
    runtime = AgentRuntime(store=store)

    resolved_count = await runtime.cleanup_expired_pending_asks()
    repeated_count = await runtime.cleanup_expired_pending_asks()

    assert resolved_count == 1
    assert repeated_count == 0
    run = await store.get_run("run-1")
    assert run is not None
    assert run.status is AgentRunStatus.CANCELLED
    assert run.error_code == "ask_expired"
    assert await store.get_state("c1", "pending_ask", include_expired=True) is None
    messages = await store.list_messages("c1")
    assert len(messages) == 1
    assert messages[0].kind is MessageKind.TOOL_RESULT
    assert messages[0].payload["result"]["metadata"]["reason"] == "expired"


async def test_runtime_inspection_can_defer_expired_ask_resolution(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    await store.save_state(
        ConversationState(
            conversation_id="c1",
            kind="pending_ask",
            revision=1,
            payload={
                "run_id": "run-1",
                "ask_id": "ask-1",
                "tool_call_id": "call-1",
                "tool_name": "ask_user",
            },
            expires_at=utc_now() - timedelta(seconds=1),
        ),
        expected_revision=0,
    )
    runtime = AgentRuntime(store=store)

    deferred = await runtime.inspect_conversation(
        "c1",
        principal,
        resolve_expired_ask=False,
    )

    assert deferred.run is not None
    assert deferred.run.status is AgentRunStatus.WAITING_INPUT
    assert deferred.pending_ask is not None
    assert deferred.pending_ask["ask_id"] == "ask-1"
    assert await store.list_messages("c1") == ()

    reconciled = await runtime.inspect_conversation("c1", principal)

    assert reconciled.run is not None
    assert reconciled.run.status is AgentRunStatus.CANCELLED
    assert reconciled.run.error_code == "ask_expired"
    assert reconciled.pending_ask is None
    messages = await store.list_messages("c1")
    assert len(messages) == 1
    assert messages[0].payload["result"]["metadata"]["reason"] == "expired"


async def test_runtime_cleanup_does_not_expire_ask_during_owned_resume(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    lease = InMemoryRunLease()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.WAITING_INPUT,
    )
    await store.save_state(
        ConversationState(
            conversation_id="c1",
            kind="pending_ask",
            revision=1,
            payload={"run_id": "run-1", "ask_id": "ask-1"},
            expires_at=utc_now() - timedelta(seconds=1),
        ),
        expected_revision=0,
    )
    assert await lease.acquire("c1", "run-1") is True
    runtime = AgentRuntime(store=store, run_lease=lease)

    while_owned = await runtime.cleanup_expired_pending_asks()
    await lease.release("c1", "run-1")
    after_release = await runtime.cleanup_expired_pending_asks()

    assert while_owned == 0
    assert after_release == 1
    messages = await store.list_messages("c1")
    assert len(messages) == 1


async def test_runtime_pending_ask_cleanup_interval_has_floor() -> None:
    runtime = AgentRuntime(
        store=InMemoryRuntimeStore(),
        pending_ask_cleanup_interval_seconds=1,
    )

    assert runtime.pending_ask_cleanup_interval_seconds == 60
    runtime.start()
    runtime.start()
    await runtime.stop()


async def test_runtime_persists_final_and_replays_idempotent_run(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    request = TurnRequest(
        invoker=principal,
        content="work",
        request_id="request-1",
        idempotency_key="idem-1",
    )
    first = await runtime.start_turn(request, TurnBindings(model=model, workspace=workspace))
    first_events = [event async for event in first.stream()]
    second = await runtime.start_turn(
        request.model_copy(update={"conversation_id": first.conversation_id}),
        TurnBindings(model=model, workspace=workspace),
    )
    replayed_events = [event async for event in second.stream()]

    assert isinstance(first_events[-1], AssistantFinal)
    assert first.run_id != request.request_id
    assert second.run_id == first.run_id
    assert second.replayed is True
    assert isinstance(replayed_events[-1], AssistantFinal)
    assert len(model.requests) == 1


@pytest.mark.parametrize("missing", [False, True], ids=["archived", "missing"])
async def test_runtime_rechecks_active_conversation_inside_run_lease(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    missing: bool,
) -> None:
    store = ConversationChangesAfterAdmissionStore(missing=missing)
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="must not run")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, conversation_id="c1", content="work"),
        TurnBindings(model=model, workspace=workspace),
    )

    events = await _collect_turn(session)

    assert store.get_conversation_calls == 2
    assert len(events) == 1
    assert isinstance(events[0], ExecutionError)
    assert events[0].code == "conversation_not_active"
    assert events[0].message == "Conversation does not exist or is not active for this actor."
    assert model.requests == []
    assert await store.list_messages("c1") == ()
    run = await store.get_run(session.run_id)
    assert run is not None and run.status is AgentRunStatus.FAILED


async def test_runtime_inspection_prefers_active_run_over_newer_rejected_run(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    blocking_model = BlockingChatModel()
    active = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    active_task = asyncio.create_task(_collect_turn(active))
    await blocking_model.started.wait()

    rejected = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=active.conversation_id,
            content="second",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="unused")]]),
            workspace=workspace,
        ),
    )
    rejected_events = await _collect_turn(rejected)
    snapshot = await runtime.inspect_conversation(active.conversation_id, principal)

    assert isinstance(rejected_events[-1], ExecutionError)
    assert rejected_events[-1].code == "concurrent_run"
    assert snapshot.run is not None
    assert snapshot.run.run_id == active.run_id
    assert snapshot.run.status is AgentRunStatus.RUNNING

    blocking_model.release.set()
    await active_task


async def test_runtime_projects_mid_turn_ownership_loss_as_the_legacy_terminal_error(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store, run_lease=LostStatusRunLease())
    session = await runtime.start_turn(
        TurnRequest(invoker=principal, content="work"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="must not escape")]]),
            workspace=workspace,
        ),
    )

    events = await _collect_turn(session)

    assert len(events) == 1
    assert isinstance(events[0], ExecutionError)
    assert events[0].code == ""
    assert events[0].message == "Agent run ownership was lost."
    messages = await store.list_messages(session.conversation_id)
    assert [message.kind for message in messages] == [MessageKind.INPUT]


async def test_runtime_projects_terminal_commit_ownership_loss_without_final_message(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = OwnershipLostOnTerminalCommitStore()
    runtime = AgentRuntime(store=store)
    session = await runtime.start_turn(
        TurnRequest(invoker=principal, content="work"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="must not finalize")]]),
            workspace=workspace,
        ),
    )

    events = await _collect_turn(session)

    assert store.terminal_commit_attempts == 1
    assert not any(isinstance(event, AssistantFinal) for event in events)
    assert isinstance(events[-1], ExecutionError)
    assert events[-1].code == ""
    assert events[-1].message == "Agent run ownership was lost."
    messages = await store.list_messages(session.conversation_id)
    assert [message.kind for message in messages] == [MessageKind.INPUT]


async def test_runtime_correlates_internal_failure_without_exposing_details(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="gewu_agent_runtime.runtime.runtime")
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    model = FailingChatModel()
    request = TurnRequest(
        invoker=principal,
        content="work",
        request_id="request-failure",
        idempotency_key="idem-failure",
    )

    first = await runtime.start_turn(request, TurnBindings(model=model, workspace=workspace))
    first_events = await _collect_turn(first)

    assert len(first_events) == 1
    error = first_events[0]
    assert isinstance(error, ExecutionError)
    assert error.code == "agent_internal_error"
    UUID(error.error_id)
    assert error.message == (
        "Agent execution failed. Please try again later. " f"Reference ID: {error.error_id}."
    )
    assert model.error_message not in error.model_dump_json()
    assert model.error_message not in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert "FailingChatModel.stream_chat" in caplog.text
    assert error.error_id in caplog.text

    messages = await store.list_messages(first.conversation_id)
    persisted = next(message for message in messages if message.kind is MessageKind.ERROR)
    assert persisted.content == error.message
    assert persisted.payload == {
        "error": error.message,
        "code": "agent_internal_error",
        "error_id": error.error_id,
    }
    assert model.error_message not in persisted.model_dump_json()

    failed_run = await store.get_run(first.run_id)
    assert failed_run is not None
    assert failed_run.status is AgentRunStatus.FAILED
    assert failed_run.error_code == "agent_internal_error"

    replay = await runtime.start_turn(
        request.model_copy(update={"conversation_id": first.conversation_id}),
        TurnBindings(model=model, workspace=workspace),
    )
    replayed_events = await _collect_turn(replay)

    assert replay.replayed is True
    assert replayed_events == first_events


@pytest.mark.parametrize(
    ("failure", "code", "message"),
    [
        (
            SafeExecutionError(
                code="subscriber_capability_unavailable",
                message="The required subscriber capability is unavailable.",
            ),
            "subscriber_capability_unavailable",
            "The required subscriber capability is unavailable.",
        ),
        (
            ContextCompactionFailedError("Conversation compaction failed."),
            "context_compaction_failed",
            "Conversation compaction failed.",
        ),
        (
            ContextLimitExceededError("Conversation context is too large."),
            "context_limit_exceeded",
            "Conversation context is too large.",
        ),
        (
            SkillCapacityExceededError("Too many Skills are visible."),
            "skill_capacity_exceeded",
            "Too many Skills are visible.",
        ),
        (
            ImageCapacityExceededError("Image processing capacity is exhausted."),
            "image_capacity_exceeded",
            "Image processing capacity is exhausted.",
        ),
        (
            ModelAuthenticationError(),
            "model_authentication_failed",
            "The configured model credentials were rejected.",
        ),
        (
            ModelPermissionDeniedError(),
            "model_permission_denied",
            "The configured model credentials cannot access the requested model.",
        ),
        (
            ModelRateLimitError(),
            "model_rate_limited",
            "The model service rate limit was reached. Please try again later.",
        ),
        (
            ModelTimeoutError(),
            "model_timeout",
            "The model service timed out. Please try again later.",
        ),
        (
            ModelUnavailableError(),
            "model_unavailable",
            "The model service is temporarily unavailable. Please try again later.",
        ),
        (
            ModelRequestRejectedError(),
            "model_request_rejected",
            "The model service rejected the request.",
        ),
    ],
)
async def test_runtime_projects_known_execution_failure(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    failure: BaseException,
    code: str,
    message: str,
) -> None:
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="work"),
        TurnBindings(model=RaisingChatModel(failure), workspace=workspace),
    )

    events = await _collect_turn(session)

    assert len(events) == 1
    error = events[0]
    assert isinstance(error, ExecutionError)
    assert error.code == code
    assert error.message == message
    assert error.error_id == ""
    run = await store.get_run(session.run_id)
    assert run is not None
    assert run.status is AgentRunStatus.FAILED
    assert run.error_code == code
    assert run.error_message == message


@pytest.mark.parametrize("log_level", [logging.DEBUG, logging.INFO])
@pytest.mark.parametrize("failure_kind", ["http", "connect", "read", "timeout"])
async def test_runtime_logs_model_failure_diagnostics_only_at_debug(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    caplog: pytest.LogCaptureFixture,
    failure_kind: str,
    log_level: int,
) -> None:
    request = httpx.Request(
        "POST", "https://user:password-secret@model.test/chat?token=query-secret"
    )
    if failure_kind == "http":
        cause: BaseException = httpx.HTTPStatusError(
            "private response-secret",
            request=request,
            response=httpx.Response(503, request=request, text="private response-secret"),
        )
    else:
        failure_types = {
            "connect": httpx.ConnectError,
            "read": httpx.RemoteProtocolError,
            "timeout": httpx.ReadTimeout,
        }
        cause = failure_types[failure_kind]("private network-secret", request=request)
    failure = ModelTimeoutError() if failure_kind == "timeout" else ModelUnavailableError()
    failure.__cause__ = cause
    store = InMemoryRuntimeStore()
    with caplog.at_level(log_level, logger="gewu_agent_runtime.runtime.runtime"):
        session = await AgentRuntime(store=store).start_turn(
            TurnRequest(
                invoker=principal, content="private prompt-secret", request_id="diagnostic-request"
            ),
            TurnBindings(model=RaisingChatModel(failure), workspace=workspace),
        )
        events = await _collect_turn(session)

    assert isinstance(events[0], ExecutionError)
    expected_code = "model_timeout" if failure_kind == "timeout" else "model_unavailable"
    assert events[0].code == expected_code
    records = [record for record in caplog.records if "MODEL_CALL_FAILED" in record.getMessage()]
    if log_level == logging.INFO:
        assert records == []
        return
    assert len(records) == 1
    record = records[0]
    assert record.levelno == logging.DEBUG
    message = record.getMessage()
    assert f"run_id={session.run_id}" in message
    assert f"conversation_id={session.conversation_id}" in message
    assert "request_id=diagnostic-request" in message
    assert f"error_code={expected_code}" in message
    assert f"http_status={503 if failure_kind == 'http' else 'none'}" in message
    assert type(cause).__name__ in message
    assert record.exc_info is None
    for secret in (
        "password-secret",
        "query-secret",
        "response-secret",
        "network-secret",
        "prompt-secret",
    ):
        assert secret not in caplog.text


async def test_runtime_model_snapshot_excludes_provider_credentials(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    model = CredentialBearingChatModel([[ModelStreamChunk(content_delta="done")]])

    turn = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="work"),
        TurnBindings(model=model, workspace=workspace),
    )
    run = await store.get_run(turn.run_id)

    assert run is not None
    assert run.model_snapshot == {
        "model_ref": "scripted:test",
        "provider": "scripted",
        "model_name": "scripted",
        "context_window": 1_000_000,
        "support_vision": True,
    }
    assert "model-api-secret" not in run.model_dump_json()
    assert "provider-token-secret" not in run.model_dump_json()


async def test_runtime_rejects_immediately_when_turn_capacity_has_no_queue(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(
        store=store,
        max_concurrent_turns=1,
        queue_capacity=0,
        admission_timeout_seconds=1,
    )
    blocking_model = BlockingChatModel()
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    first_task = asyncio.create_task(_collect_turn(first))
    await blocking_model.started.wait()

    second = await runtime.start_turn(
        TurnRequest(invoker=principal, content="second"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="unused")]]),
            workspace=workspace,
        ),
    )
    second_events = await _collect_turn(second)

    assert len(second_events) == 1
    assert isinstance(second_events[0], ExecutionError)
    assert second_events[0].code == "runtime_capacity"
    assert second_events[0].message == "Agent runtime is temporarily unavailable."
    assert second_events[0].message_id
    rejected_run = await store.get_run(second.run_id)
    assert rejected_run is not None
    assert rejected_run.status is AgentRunStatus.FAILED
    assert rejected_run.error_code == "runtime_capacity"
    stats = await runtime.admission_stats()
    assert stats.model_dump() == {
        "active": 1,
        "queued": 0,
        "admitted_total": 1,
        "rejected_total": 1,
    }

    blocking_model.release.set()
    await first_task
    assert (await runtime.admission_stats()).active == 0


async def test_runtime_publishes_terminal_event_after_releasing_run_and_process_capacity(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(
        store=store,
        max_concurrent_turns=1,
        queue_capacity=0,
        admission_timeout_seconds=1,
    )
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="first done")]]),
            workspace=workspace,
        ),
    )
    first_stream = first.stream()
    terminal = None
    async for event in first_stream:
        if isinstance(event, AssistantFinal):
            terminal = event
            break

    assert terminal is not None
    stored = await store.get_run(first.run_id)
    assert stored is not None and stored.status is AgentRunStatus.COMPLETED
    assert (await runtime.admission_stats()).active == 0

    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="second",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="second done")]]),
            workspace=workspace,
        ),
    )
    second_events = await _collect_turn(second)

    assert isinstance(second_events[-1], AssistantFinal)
    assert second_events[-1].content == "second done"
    await cast(AsyncGenerator[RuntimeEvent, None], first_stream).aclose()


async def test_stale_disconnect_interrupt_does_not_cancel_newer_run(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="first done")]]),
            workspace=workspace,
        ),
    )
    await _collect_turn(first)
    blocking_model = BlockingChatModel()
    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="second",
        ),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    second_task = asyncio.create_task(_collect_turn(second))
    await blocking_model.started.wait()

    interrupted = await runtime.interrupt_conversation(
        first.conversation_id,
        principal,
        expected_run_id=first.run_id,
    )
    running = await store.get_run(second.run_id)

    assert interrupted is True
    assert running is not None
    assert running.status is AgentRunStatus.RUNNING
    blocking_model.release.set()
    second_events = await second_task
    assert isinstance(second_events[-1], AssistantFinal)
    completed = await store.get_run(second.run_id)
    assert completed is not None
    assert completed.status is AgentRunStatus.COMPLETED


async def test_expected_interrupt_without_live_lease_does_not_rewrite_run(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))
    await store.create_run(AgentRun(run_id="run-1", conversation_id="c1", invoker=principal))
    await store.transition_run(
        "run-1",
        expected=(AgentRunStatus.PENDING,),
        status=AgentRunStatus.RUNNING,
    )

    interrupted = await AgentRuntime(store=store).interrupt_conversation(
        "c1",
        principal,
        expected_run_id="run-1",
    )

    run = await store.get_run("run-1")
    assert interrupted is True
    assert run is not None and run.status is AgentRunStatus.RUNNING


async def test_interrupt_can_leave_idle_projection_to_the_host(
    principal: PrincipalRef,
) -> None:
    store = InMemoryRuntimeStore()
    await store.create_conversation(Conversation(conversation_id="c1", owner=principal))

    interrupted = await AgentRuntime(store=store).interrupt_conversation(
        "c1",
        principal,
        record_idle_interrupt=False,
    )

    assert interrupted is True
    assert await store.get_latest_run("c1") is None


async def test_runtime_coordinates_and_interrupts_across_instances(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    lease = PollingInMemoryRunLease()
    first_runtime = AgentRuntime(store=store, run_lease=lease)
    second_runtime = AgentRuntime(store=store, run_lease=lease)
    blocking_model = BlockingChatModel()
    first = await first_runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    first_task = asyncio.create_task(_collect_turn(first))
    await blocking_model.started.wait()

    competing = await second_runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="competing",
        ),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="must not run")]]),
            workspace=workspace,
        ),
    )
    competing_events = await _collect_turn(competing)

    assert len(competing_events) == 1
    assert isinstance(competing_events[0], ExecutionError)
    assert competing_events[0].message == "Conversation already has an active turn."
    assert await second_runtime.interrupt_conversation(first.conversation_id, principal) is True
    first_events = await asyncio.wait_for(first_task, timeout=1)

    assert isinstance(first_events[-1], ExecutionError)
    assert first_events[-1].code == "cancelled"
    assert first_events[-1].message == "Conversation run interrupted."
    first_run = await store.get_run(first.run_id)
    assert first_run is not None and first_run.status is AgentRunStatus.CANCELLED
    messages = await store.list_messages(first.conversation_id)
    assert [message.kind for message in messages if message.run_id == first.run_id] == [
        MessageKind.INPUT
    ]


async def test_interrupt_persists_cancelled_state_before_returning(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    blocking_model = BlockingChatModel()
    session = await runtime.start_turn(
        TurnRequest(invoker=principal, content="block"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    stream_task = asyncio.create_task(_collect_turn(session))
    await blocking_model.started.wait()

    assert await runtime.interrupt_conversation(session.conversation_id, principal) is True

    interrupted = await store.get_run(session.run_id)
    conversation = await store.get_conversation(session.conversation_id)
    assert interrupted is not None
    assert interrupted.status is AgentRunStatus.CANCELLED
    assert conversation is not None
    assert conversation.active_run_id is None

    events = await asyncio.wait_for(stream_task, timeout=1)
    assert isinstance(events[-1], ExecutionError)
    assert events[-1].code == "cancelled"


async def test_runtime_runs_a_queued_turn_after_capacity_is_released(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    runtime = AgentRuntime(
        store=InMemoryRuntimeStore(),
        max_concurrent_turns=1,
        queue_capacity=1,
        admission_timeout_seconds=1,
    )
    blocking_model = BlockingChatModel()
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    first_task = asyncio.create_task(_collect_turn(first))
    await blocking_model.started.wait()
    second = await runtime.start_turn(
        TurnRequest(invoker=principal, content="second"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="queued")]]),
            workspace=workspace,
        ),
    )
    second_task = asyncio.create_task(_collect_turn(second))
    await _wait_for_admission(runtime, active=1, queued=1)

    blocking_model.release.set()
    first_events, second_events = await asyncio.gather(first_task, second_task)

    assert isinstance(first_events[-1], AssistantFinal)
    assert isinstance(second_events[-1], AssistantFinal)
    assert second_events[-1].content == "queued"
    stats = await runtime.admission_stats()
    assert stats.model_dump() == {
        "active": 0,
        "queued": 0,
        "admitted_total": 2,
        "rejected_total": 0,
    }


async def test_runtime_fails_a_turn_when_queue_admission_times_out(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(
        store=store,
        max_concurrent_turns=1,
        queue_capacity=1,
        admission_timeout_seconds=0.01,
    )
    blocking_model = BlockingChatModel()
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="first"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    first_task = asyncio.create_task(_collect_turn(first))
    await blocking_model.started.wait()
    second = await runtime.start_turn(
        TurnRequest(invoker=principal, content="second"),
        TurnBindings(
            model=ScriptedChatModel([[ModelStreamChunk(content_delta="unused")]]),
            workspace=workspace,
        ),
    )
    second_task = asyncio.create_task(_collect_turn(second))
    await _wait_for_admission(runtime, active=1, queued=1)

    second_events = await second_task

    assert len(second_events) == 1
    assert isinstance(second_events[0], ExecutionError)
    assert second_events[0].code == "runtime_capacity"
    rejected_run = await store.get_run(second.run_id)
    assert rejected_run is not None and rejected_run.status is AgentRunStatus.FAILED
    stats = await runtime.admission_stats()
    assert stats.active == 1
    assert stats.queued == 0
    assert stats.rejected_total == 1

    blocking_model.release.set()
    await first_task


async def test_runtime_uses_the_assembled_system_prompt(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    prompt = build_system_prompt(assistant_name="Bound Agent")
    session = await AgentRuntime(store=InMemoryRuntimeStore()).start_turn(
        TurnRequest(invoker=principal, content="work"),
        TurnBindings(model=model, workspace=workspace, prompt=prompt),
    )

    _ = [event async for event in session.stream()]

    assert model.requests[0][0][0].content == prompt.full
    assert "You are Bound Agent" in model.requests[0][0][0].content


async def test_runtime_injects_host_bound_bash_capability(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    observed: list[tuple[str, int]] = []

    async def execute(command: str, timeout_ms: int) -> ToolResult:
        observed.append((command, timeout_ms))
        return ToolResult(output=BashOutput(stdout="bound\n"))

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="bash-1",
                            name="bash",
                            arguments={"command": "pwd"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    session = await AgentRuntime(store=InMemoryRuntimeStore()).start_turn(
        TurnRequest(invoker=principal, content="run"),
        TurnBindings(
            model=model,
            workspace=workspace,
            tool_set=ToolSet((bash,)),
            tool_runtime=ToolRuntimeBindings(
                bash_executor=execute,
                bash_default_timeout_ms=90_000,
            ),
        ),
    )

    events = [event async for event in session.stream()]
    result = next(event for event in events if isinstance(event, ToolResultEvent))

    assert observed == [("pwd", 90_000)]
    assert result.result.output_payload()["stdout"] == "bound\n"


async def test_runtime_does_not_persist_reference_read_content(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    await workspace.write_text("/secret.txt", "sensitive-value")
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/secret.txt"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="read")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="read it"),
        TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((read,))),
    )

    _ = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    result = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)

    assert "sensitive-value" in model.requests[1][0][-1].content
    assert "sensitive-value" not in result.model_dump_json()
    assert result.payload["result"]["content_omitted"] is True


async def test_runtime_full_policy_persists_raw_result_with_subscriber_message_content(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    await workspace.write_text("/notes.txt", "persisted text")
    full_read = read.model_copy(update={"persistence_policy": PersistencePolicy.FULL})
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/notes.txt"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="read it"),
        TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((full_read,))),
    )

    _ = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    result = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)

    assert result.payload["result"]["content"] == "1: persisted text"
    assert result.content == ""


async def test_runtime_protected_policy_encrypts_storage_and_restores_later_context(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    @tool(
        description="Return protected business data.",
        persistence_policy=PersistencePolicy.PROTECTED,
        trace_result=False,
    )
    def lookup() -> ToolResult:
        return ToolResult(output={"rows": [{"customer": "sensitive-customer"}], "count": 1})

    first_model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(ToolCall(tool_call_id="lookup-1", name="lookup", arguments={}),)
                )
            ],
            [ModelStreamChunk(content_delta="first done")],
        ]
    )
    store = InMemoryRuntimeStore(
        protected_payload_cipher=JsonSecretCipher("protected-runtime-test-key"),
        encrypt_protected_payloads=True,
    )
    runtime = AgentRuntime(store=store)
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="look it up"),
        TurnBindings(model=first_model, workspace=workspace, tool_set=ToolSet((lookup,))),
    )
    _ = [event async for event in first.stream()]

    raw_history = store._messages[first.conversation_id]
    assert "sensitive-customer" not in "".join(message.model_dump_json() for message in raw_history)
    restored = await store.list_messages(first.conversation_id)
    result = next(message for message in restored if message.kind is MessageKind.TOOL_RESULT)
    assert result.payload["result"] == {
        "rows": [{"customer": "sensitive-customer"}],
        "count": 1,
    }
    assert result.payload["trace_result"] is False

    continuation_model = ScriptedChatModel([[ModelStreamChunk(content_delta="second done")]])
    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="continue",
        ),
        TurnBindings(
            model=continuation_model,
            workspace=workspace,
            tool_set=ToolSet((lookup,)),
        ),
    )
    _ = [event async for event in second.stream()]

    historical_tool_result = next(
        message
        for message in continuation_model.requests[0][0]
        if message.tool_call_id == "lookup-1"
    )
    assert "sensitive-customer" in historical_tool_result.content


async def test_runtime_separates_model_payload_from_full_persistence_payload(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    @tool(
        description="Return a complete result and a compact model projection.",
        trace_result=False,
    )
    def lookup() -> ToolResult:
        return ToolResult(
            output={"rows": [{"secret": "full-result"}], "count": 1},
            model_payload={"count": 1, "summary": "one row"},
        )

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(ToolCall(tool_call_id="lookup-1", name="lookup", arguments={}),)
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="look it up"),
        TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((lookup,))),
    )

    _ = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    result = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)

    assert model.requests[1][0][-1].content == '{"count": 1, "summary": "one row"}'
    assert result.payload["result"] == {
        "rows": [{"secret": "full-result"}],
        "count": 1,
    }
    assert result.payload["trace_result"] is False


async def test_runtime_persists_subscriber_tool_error_as_message_content(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    full_read = read.model_copy(update={"persistence_policy": PersistencePolicy.FULL})
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/missing.txt"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="read it"),
        TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((full_read,))),
    )

    _ = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    result = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)

    assert result.content == "File does not exist or is not readable."
    assert result.payload["result"]["error"] == result.content


async def test_runtime_persists_tool_preamble_without_replaying_it_as_final(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    await workspace.write_text("/notes.txt", "one")
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(content_delta="checking. "),
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/notes.txt"},
                        ),
                    )
                ),
            ],
            [ModelStreamChunk(content_delta="done.")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    request = TurnRequest(
        invoker=principal,
        content="inspect",
        request_id="request-preamble",
        idempotency_key="idem-preamble",
    )
    bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((read,)),
    )

    first = await runtime.start_turn(request, bindings)
    first_events = await _collect_turn(first)
    messages = await store.list_messages(first.conversation_id)

    assert [message.kind for message in messages] == [
        MessageKind.INPUT,
        MessageKind.ASSISTANT,
        MessageKind.TOOL_USE,
        MessageKind.TOOL_RESULT,
        MessageKind.ASSISTANT,
    ]
    assistant_messages = [message for message in messages if message.kind is MessageKind.ASSISTANT]
    assert [(message.content, message.payload) for message in assistant_messages] == [
        ("checking. ", {"final": True, "llm_ignore": True}),
        ("done.", {"final": True}),
    ]
    assert model.requests[1][0][-2].content == "checking. "
    assert model.requests[1][0][-2].tool_calls[0].name == "read"

    replay = await runtime.start_turn(
        request.model_copy(update={"conversation_id": first.conversation_id}),
        bindings,
    )
    replayed_events = await _collect_turn(replay)

    assert replay.replayed is True
    assert not any(
        isinstance(event, AssistantFinal) and event.content == "checking. "
        for event in replayed_events
    )
    assert isinstance(first_events[-1], AssistantFinal)
    assert isinstance(replayed_events[-1], AssistantFinal)
    assert replayed_events[-1].content == "done."


async def test_runtime_filters_whitespace_only_tool_preamble(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    @tool(description="Echo text.")
    def echo(text: str) -> ToolResult:
        return ToolResult(output={"text": text})

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(content_delta=" \n"),
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="echo-1",
                            name="echo",
                            arguments={"text": "ready"},
                        ),
                    )
                ),
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="echo"),
        TurnBindings(
            model=model,
            workspace=workspace,
            tool_set=ToolSet((echo,)),
        ),
    )

    events = await _collect_turn(session)

    assert not any(
        isinstance(event, AssistantDelta) and not event.content.strip() for event in events
    )
    use = next(event for event in events if isinstance(event, ToolUse))
    assert use.assistant_text == ""
    messages = await store.list_messages(session.conversation_id)
    assistant_messages = [message for message in messages if message.kind is MessageKind.ASSISTANT]
    assert [message.content for message in assistant_messages] == ["done"]


@pytest.mark.parametrize("answer_status", ["answered", "skipped"])
async def test_runtime_suspends_and_resumes_ask(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    answer_status: Literal["answered", "skipped"],
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="continued")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,)))
    session = await runtime.start_turn(
        TurnRequest(invoker=principal, content="start"),
        bindings,
    )
    first_events = [event async for event in session.stream()]
    ask_event = first_events[-1]
    assert isinstance(ask_event, AskRequested)
    waiting = await store.get_run(session.run_id)
    assert waiting is not None and waiting.status is AgentRunStatus.WAITING_INPUT

    resumed = await runtime.resume_ask(
        run_id=session.run_id,
        invoker=principal,
        answer=AskAnswer(
            ask_id=ask_event.ask_id,
            answers={"Continue?": "Yes"},
            status=answer_status,
            metadata={"transport_only": "must-not-reach-model"},
        ),
        bindings=bindings,
    )
    resumed_events = [event async for event in resumed.stream()]

    assert isinstance(resumed_events[0], ToolResultEvent)
    assert resumed_events[0].call.tool_call_id == "ask-1"
    assert resumed_events[0].call.name == "ask_user"
    assert resumed_events[0].result.output_payload() == {
        "answers": {"Continue?": "Yes"},
        "annotations": {},
        "metadata": {"status": answer_status, "skipped": answer_status == "skipped"},
        "error": "",
    }
    assert isinstance(resumed_events[-1], AssistantFinal)
    assert resumed_events[-1].content == "continued"
    completed = await store.get_run(session.run_id)
    assert completed is not None and completed.status is AgentRunStatus.COMPLETED
    assert await store.get_state(session.conversation_id, "pending_ask") is None
    tool_result = next(
        message
        for message in await store.list_messages(session.conversation_id)
        if message.kind is MessageKind.TOOL_RESULT
    )
    assert tool_result.payload["result"] == {
        "answers": {"Continue?": "Yes"},
        "annotations": {},
        "metadata": {"status": answer_status, "skipped": answer_status == "skipped"},
        "error": "",
    }
    assert "transport_only" not in model.requests[-1][0][-2].content


async def test_runtime_resume_resolves_model_after_answer_commit(
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ]
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    suspended = await runtime.start_turn(
        TurnRequest(invoker=principal, content="start"),
        TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,))),
    )
    first_events = await _collect_turn(suspended)
    ask_event = first_events[-1]
    assert isinstance(ask_event, AskRequested)
    model_resolver_calls = 0

    class FailingModelProvider:
        async def resolve(
            self,
            messages: Sequence[Message],
            tools: Sequence[ModelTool],
        ) -> ScriptedChatModel:
            del messages, tools
            nonlocal model_resolver_calls
            model_resolver_calls += 1
            persisted = await store.list_messages(suspended.conversation_id)
            assert any(message.kind is MessageKind.TOOL_RESULT for message in persisted)
            raise RuntimeError("private subscriber model failure")

    async def resolve_bindings() -> TurnBindings:
        return TurnBindings(model_provider=FailingModelProvider(), workspace=workspace)

    resumed = await runtime.resume_ask(
        run_id=suspended.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings_factory=resolve_bindings,
    )
    resumed_events = await _collect_turn(resumed)

    assert model_resolver_calls == 1
    assert isinstance(resumed_events[0], ToolResultEvent)
    assert resumed_events[0].result.output_payload()["answers"] == {"Continue?": "Yes"}
    assert isinstance(resumed_events[1], ExecutionError)
    assert resumed_events[1].code == "agent_internal_error"
    run = await store.get_run(suspended.run_id)
    assert run is not None and run.status is AgentRunStatus.FAILED
    assert run.error_code == "agent_internal_error"
    assert await store.get_state(suspended.conversation_id, "pending_ask") is None


async def test_runtime_ask_resume_does_not_run_new_turn_compaction(
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="continued")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    initial_bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((ask,)),
    )
    suspended = await runtime.start_turn(
        TurnRequest(invoker=principal, content="start"),
        initial_bindings,
    )
    first_events = await _collect_turn(suspended)
    ask_event = first_events[-1]
    assert isinstance(ask_event, AskRequested)
    compaction_model = ScriptedCompactionModel(
        ["must not be used"],
        context_window=1_000,
    )

    resumed = await runtime.resume_ask(
        run_id=suspended.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings=initial_bindings.model_copy(
            update={
                "compaction_policy": CompactionPolicy(trigger_percent=1, target_percent=1),
                "compaction_model": compaction_model,
            }
        ),
    )
    resumed_events = await _collect_turn(resumed)

    assert isinstance(resumed_events[-1], AssistantFinal)
    assert resumed_events[-1].content == "continued"
    assert compaction_model.requests == []


async def test_runtime_ask_resume_shares_normal_turn_capacity(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=60)
    ask_model = ScriptedChatModel(
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="continued")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(
        store=store,
        max_concurrent_turns=1,
        queue_capacity=1,
        admission_timeout_seconds=1,
    )
    ask_bindings = TurnBindings(
        model=ask_model,
        workspace=workspace,
        tool_set=ToolSet((ask,)),
    )
    suspended = await runtime.start_turn(
        TurnRequest(invoker=principal, content="ask first"),
        ask_bindings,
    )
    suspended_events = await _collect_turn(suspended)
    ask_event = suspended_events[-1]
    assert isinstance(ask_event, AskRequested)

    blocking_model = BlockingChatModel()
    blocking = await runtime.start_turn(
        TurnRequest(invoker=principal, content="occupy capacity"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    blocking_task = asyncio.create_task(_collect_turn(blocking))
    await blocking_model.started.wait()

    resumed = await runtime.resume_ask(
        run_id=suspended.run_id,
        invoker=principal,
        answer=AskAnswer(
            ask_id=ask_event.ask_id,
            answers={"Continue?": "Yes"},
        ),
        bindings=ask_bindings,
    )
    resume_task = asyncio.create_task(_collect_turn(resumed))
    await asyncio.sleep(0)

    stats = await runtime.admission_stats()
    assert stats.active == 1
    assert stats.queued == 1
    assert len(ask_model.requests) == 1

    blocking_model.release.set()
    blocking_events, resumed_events = await asyncio.gather(blocking_task, resume_task)

    assert isinstance(blocking_events[-1], AssistantFinal)
    assert isinstance(resumed_events[-1], AssistantFinal)
    assert resumed_events[-1].content == "continued"
    assert (await runtime.admission_stats()).model_dump() == {
        "active": 0,
        "queued": 0,
        "admitted_total": 3,
        "rejected_total": 0,
    }


async def test_runtime_capacity_rejection_keeps_pending_ask_retryable(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=60)
    ask_model = ScriptedChatModel(
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ]
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store, max_concurrent_turns=1, queue_capacity=0)
    suspended = await runtime.start_turn(
        TurnRequest(invoker=principal, content="ask first"),
        TurnBindings(model=ask_model, workspace=workspace, tool_set=ToolSet((ask,))),
    )
    suspended_events = await _collect_turn(suspended)
    ask_event = suspended_events[-1]
    assert isinstance(ask_event, AskRequested)

    blocking_model = BlockingChatModel()
    blocking = await runtime.start_turn(
        TurnRequest(invoker=principal, content="occupy capacity"),
        TurnBindings(model=blocking_model, workspace=workspace),
    )
    blocking_task = asyncio.create_task(_collect_turn(blocking))
    await blocking_model.started.wait()
    bindings_factory_called = False

    async def bindings_factory() -> TurnBindings:
        nonlocal bindings_factory_called
        bindings_factory_called = True
        raise AssertionError("rejected Ask must not prepare capabilities")

    resumed = await runtime.resume_ask(
        run_id=suspended.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings_factory=bindings_factory,
    )
    resumed_events = await _collect_turn(resumed)

    assert len(resumed_events) == 1
    assert isinstance(resumed_events[0], ExecutionError)
    assert resumed_events[0].code == "runtime_capacity"
    assert bindings_factory_called is False
    waiting = await store.get_run(suspended.run_id)
    assert waiting is not None and waiting.status is AgentRunStatus.WAITING_INPUT
    assert await store.get_state(suspended.conversation_id, "pending_ask") is not None

    blocking_model.release.set()
    blocking_events = await blocking_task
    assert isinstance(blocking_events[-1], AssistantFinal)


async def test_runtime_restores_pending_ask_from_store_and_complete_history(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=60)

    @tool(description="Return durable early evidence.")
    def evidence() -> ToolResult:
        return ToolResult(output={"stdout": "early evidence"})

    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="evidence-1",
                            name="evidence",
                            arguments={},
                        ),
                    )
                )
            ],
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-1",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((ask, evidence)),
    )
    first_runtime = AgentRuntime(store=store)
    first = await first_runtime.start_turn(
        TurnRequest(invoker=principal, content="keep the original question"),
        bindings,
    )
    first_events = [event async for event in first.stream()]
    ask_event = first_events[-1]
    assert isinstance(ask_event, AskRequested)
    pending = await store.get_state(first.conversation_id, "pending_ask")
    assert pending is not None
    assert "checkpoint_messages" not in pending.payload
    assert "attachment_refs" not in pending.payload

    restored_runtime = AgentRuntime(store=store)
    wrong = await restored_runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id="wrong", answers={"Continue?": "No"}),
        bindings=bindings,
    )
    with pytest.raises(AskNotPendingError, match="ask_id does not match"):
        _ = [event async for event in wrong.stream()]
    still_waiting = await store.get_run(first.run_id)
    assert still_waiting is not None
    assert still_waiting.status is AgentRunStatus.WAITING_INPUT
    assert await store.get_state(first.conversation_id, "pending_ask") is not None

    resumed = await restored_runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings=bindings,
    )
    resumed_events = [event async for event in resumed.stream()]

    assert isinstance(resumed_events[-1], AssistantFinal)
    assert resumed_events[-1].content == "done"
    restored_messages = model.requests[-1][0]
    assert any(message.content == "keep the original question" for message in restored_messages)
    assert any(
        message.role.value == "tool" and "early evidence" in message.content
        for message in restored_messages
    )
    assert any(
        message.role.value == "tool" and "Continue?" in message.content
        for message in restored_messages
    )


async def test_pending_ask_compare_and_set_rejects_a_different_ask(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    """An obsolete answer must not consume a newer pending Ask."""
    ask = ask_user_tool(timeout_seconds=60)
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-new",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ]
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,)))
    suspended = await runtime.start_turn(
        TurnRequest(invoker=principal, content="ask first"),
        bindings,
    )
    suspended_events = await _collect_turn(suspended)
    ask_event = suspended_events[-1]
    assert isinstance(ask_event, AskRequested)
    before = await store.get_state(suspended.conversation_id, "pending_ask")
    assert before is not None and before.payload["ask_id"] == ask_event.ask_id

    wrong = await runtime.resume_ask(
        run_id=suspended.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id="ask-old", answers={"Continue?": "No"}),
        bindings=bindings,
    )
    with pytest.raises(AskNotPendingError, match="ask_id does not match"):
        await _collect_turn(wrong)

    run = await store.get_run(suspended.run_id)
    after = await store.get_state(suspended.conversation_id, "pending_ask")
    assert run is not None and run.status is AgentRunStatus.WAITING_INPUT
    assert after == before


async def test_runtime_repeated_ask_uses_independent_assistant_messages(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=60)
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(content_delta="first reply. "),
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-1",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "First?",
                                        "header": "First",
                                        "options": [{"label": "A"}, {"label": "B"}],
                                    }
                                ]
                            },
                        ),
                    )
                ),
            ],
            [
                ModelStreamChunk(content_delta="second reply. "),
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-2",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Second?",
                                        "header": "Second",
                                        "options": [{"label": "A"}, {"label": "B"}],
                                    }
                                ]
                            },
                        ),
                    )
                ),
            ],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,)))
    first = await runtime.start_turn(TurnRequest(invoker=principal, content="ask twice"), bindings)
    first_events = [event async for event in first.stream()]
    first_ask = first_events[-1]
    assert isinstance(first_ask, AskRequested)

    resumed = await runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=first_ask.ask_id, answers={"First?": "A"}),
        bindings=bindings,
    )
    resumed_events = [event async for event in resumed.stream()]
    second_ask = resumed_events[-1]
    assert isinstance(second_ask, AskRequested)

    first_intermediate = next(
        event for event in first_events if isinstance(event, AssistantIntermediate)
    )
    second_intermediate = next(
        event for event in resumed_events if isinstance(event, AssistantIntermediate)
    )
    assert first_intermediate.content == "first reply. "
    assert second_intermediate.content == "second reply. "
    assert first_intermediate.message_id != second_intermediate.message_id
    persisted = await store.list_messages(first.conversation_id)
    assistant_messages = [message for message in persisted if message.kind is MessageKind.ASSISTANT]
    assert [message.content for message in assistant_messages] == [
        "first reply. ",
        "second reply. ",
    ]
    assert len({message.message_id for message in assistant_messages}) == 2


async def test_runtime_ask_resume_rebuilds_file_read_state(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    await workspace.write_text("/notes.txt", "hello\n")
    ask = ask_user_tool(timeout_seconds=60)
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/notes.txt", "offset": 0, "limit": 1},
                        ),
                    )
                )
            ],
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="ask-1",
                            name="ask_user",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ],
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-2",
                            name="read",
                            arguments={"file_path": "/notes.txt", "offset": 0, "limit": 1},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((ask, read)),
    )
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="read then ask"),
        bindings,
    )
    first_events = [event async for event in first.stream()]
    ask_event = first_events[-1]
    assert isinstance(ask_event, AskRequested)

    resumed = await runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings=bindings,
    )
    resumed_events = [event async for event in resumed.stream()]

    repeated_read = next(
        event
        for event in resumed_events
        if isinstance(event, ToolResultEvent) and event.call.tool_call_id == "read-2"
    )
    assert repeated_read.result.output_payload()["type"] == "file_unchanged"
    assert repeated_read.result.output_payload()["unchanged"] is True
    assert isinstance(resumed_events[-1], AssistantFinal)


@pytest.mark.parametrize(
    ("expire_pending", "reason", "error"),
    [
        (
            False,
            "new_user_input",
            "Pending ask_user request cancelled by new user input.",
        ),
        (
            True,
            "expired",
            "Pending ask_user request expired before it was answered.",
        ),
    ],
)
async def test_runtime_new_input_resolves_pending_ask_before_model_call(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
    expire_pending: bool,
    reason: str,
    error: str,
) -> None:
    ask = ask_user_tool(timeout_seconds=300)
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
            [ModelStreamChunk(content_delta="continued with new input")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,)))
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="need details"),
        bindings,
    )
    first_events = [event async for event in first.stream()]
    assert isinstance(first_events[-1], AskRequested)
    if expire_pending:
        state = await store.get_state(first.conversation_id, "pending_ask")
        assert state is not None
        await store.save_state(
            ConversationState(
                conversation_id=state.conversation_id,
                kind=state.kind,
                revision=state.revision + 1,
                payload=state.payload,
                expires_at=utc_now() - timedelta(seconds=1),
            ),
            expected_revision=state.revision,
        )

    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="ignore that",
        ),
        bindings,
    )
    second_events = [event async for event in second.stream()]

    assert isinstance(second_events[-1], AssistantFinal)
    messages = await store.list_messages(first.conversation_id)
    resolution = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)
    assert resolution.payload["is_error"] is False
    assert resolution.payload["result"] == {
        "answers": {},
        "annotations": {},
        "metadata": {
            "status": "skipped",
            "skipped": True,
            "reason": reason,
        },
        "error": error,
    }
    prior_run = await store.get_run(first.run_id)
    assert prior_run is not None and prior_run.status is AgentRunStatus.CANCELLED
    assert await store.get_state(first.conversation_id, "pending_ask") is None
    assert any(
        message.role.value == "tool" and reason in message.content
        for message in model.requests[-1][0]
    )


async def test_runtime_expired_ask_persists_resolution_before_rejecting_resume(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    ask = ask_user_tool(timeout_seconds=300)
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
                                        "question": "Continue?",
                                        "header": "Next",
                                        "options": [{"label": "Yes"}, {"label": "No"}],
                                    }
                                ]
                            },
                        ),
                    )
                )
            ]
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(model=model, workspace=workspace, tool_set=ToolSet((ask,)))
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="start"),
        bindings,
    )
    events = [event async for event in first.stream()]
    ask_event = events[-1]
    assert isinstance(ask_event, AskRequested)
    state = await store.get_state(first.conversation_id, "pending_ask")
    assert state is not None
    await store.save_state(
        ConversationState(
            conversation_id=state.conversation_id,
            kind=state.kind,
            revision=state.revision + 1,
            payload=state.payload,
            expires_at=utc_now() - timedelta(seconds=1),
        ),
        expected_revision=state.revision,
    )

    resumed = await runtime.resume_ask(
        run_id=first.run_id,
        invoker=principal,
        answer=AskAnswer(ask_id=ask_event.ask_id, answers={"Continue?": "Yes"}),
        bindings=bindings,
    )
    with pytest.raises(AskExpiredError, match="pending Ask request has expired"):
        _ = [event async for event in resumed.stream()]

    messages = await store.list_messages(first.conversation_id)
    resolution = next(message for message in messages if message.kind is MessageKind.TOOL_RESULT)
    assert resolution.payload["result"]["metadata"]["reason"] == "expired"
    assert resolution.payload["result"]["error"] == (
        "Pending ask_user request expired before it was answered."
    )
    run = await store.get_run(first.run_id)
    assert run is not None and run.status is AgentRunStatus.CANCELLED
    assert (
        await store.get_state(
            first.conversation_id,
            "pending_ask",
            include_expired=True,
        )
        is None
    )


async def test_runtime_persists_and_reconciles_read_before_write_state(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    await workspace.write_text("/tracked.txt", "before")
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="read-1",
                            name="read",
                            arguments={"file_path": "/tracked.txt"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="read complete")],
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="write-1",
                            name="write",
                            arguments={"file_path": "/tracked.txt", "content": "second"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="write complete")],
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="write-2",
                            name="write",
                            arguments={"file_path": "/tracked.txt", "content": "third"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="write rejected")],
        ]
    )
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    bindings = TurnBindings(
        model=model,
        workspace=workspace,
        tool_set=ToolSet((read, write)),
    )

    first = await runtime.start_turn(TurnRequest(invoker=principal, content="read"), bindings)
    _ = [event async for event in first.stream()]
    assert await store.get_state(first.conversation_id, "file") is not None

    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="write",
        ),
        bindings,
    )
    second_events = [event async for event in second.stream()]
    second_result = next(event for event in second_events if isinstance(event, ToolResultEvent))

    third = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="write again",
        ),
        bindings,
    )
    third_events = [event async for event in third.stream()]
    third_result = next(event for event in third_events if isinstance(event, ToolResultEvent))

    assert second_result.result.is_error is False
    assert third_result.result.is_error is True
    assert "has not been read" in third_result.result.output_payload()["error"]
    assert await workspace.read_text("/tracked.txt") == "second"
    assert await store.get_state(first.conversation_id, "file") is None
