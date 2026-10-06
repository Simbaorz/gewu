"""Persistent turn orchestration over the pure Agent Engine."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, suppress
from copy import deepcopy
from datetime import timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, model_validator
from pydantic_core import to_jsonable_python

from gewu_agent_runtime.builtins.skills import (
    RuntimeVariableProvider,
    SceneCatalog,
    SkillCapacityExceededError,
    SkillCatalog,
    SkillRegistry,
    bind_skill_tool,
    prepare_skill_turn,
    record_skill_invocation,
)
from gewu_agent_runtime.compaction import (
    CompactionModel,
    CompactionModelProvider,
    CompactionPolicy,
    CompactionService,
    ContextCompactionFailedError,
    ContextLimitExceededError,
    FullCompactProgress,
    FullCompactResult,
)
from gewu_agent_runtime.context import ConversationContextBuilder
from gewu_agent_runtime.coordination import InMemoryRunLease, RunLease, RunLeaseStatus
from gewu_agent_runtime.domain import (
    AgentRun,
    AgentRunStatus,
    Conversation,
    ConversationState,
    ConversationStatus,
    FileStateCache,
    MessageKind,
    NewConversationMessage,
    ProtectedMessageBody,
)
from gewu_agent_runtime.engine import (
    AgentEngine,
    AgentEvent,
    AskRequested,
    AssistantDelta,
    AssistantFinal,
    AssistantIntermediate,
    ExecutionError,
    ExecutionRequest,
    ToolResultEvent,
    ToolUse,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.invocation import InvocationTarget, InvocationTargetKind
from gewu_agent_runtime.llm import (
    ChatModel,
    ChatModelProvider,
    Message,
    MessageRole,
    ModelAuthenticationError,
    ModelInvocationError,
    ModelPermissionDeniedError,
    ModelRateLimitError,
    ModelRequestRejectedError,
    ModelTimeoutError,
    ModelTraceSink,
    ModelUnavailableError,
    ToolCall,
)
from gewu_agent_runtime.media import (
    AttachmentLoader,
    AttachmentRef,
    ImageCapacityExceededError,
    ImagePayloadManager,
)
from gewu_agent_runtime.persistence import (
    ConcurrentWriteError,
    EntityNotFoundError,
    MessageWriteConflictError,
    RuntimeStore,
    StateCache,
)
from gewu_agent_runtime.prompts import SystemPrompt, build_system_prompt
from gewu_agent_runtime.runtime.errors import (
    AskExpiredError,
    AskNotPendingError,
    ConcurrentRunError,
    PrincipalMismatchError,
    SafeExecutionError,
    SubscriberMismatchError,
)
from gewu_agent_runtime.tools import (
    PersistencePolicy,
    ToolExecutor,
    ToolResult,
    ToolRuntimeBindings,
    ToolSet,
)
from gewu_agent_runtime.workspace import WorkspaceSession
from gewu_core.concurrency import (
    AsyncAdmissionCapacityExceededError,
    AsyncAdmissionStats,
    FairAsyncCapacityLimiter,
)
from gewu_core.ids import new_id, new_uuid4_id
from gewu_core.time import utc_now

PENDING_ASK_STATE = "pending_ask"
FILE_STATE = "file"
SKILL_STATE = "skill"
SELECTED_SCENE_TOKEN_RESERVE = 512
INTERNAL_AGENT_ERROR_CODE = "agent_internal_error"
INTERNAL_AGENT_ERROR_MESSAGE = "Agent execution failed. Please try again later."
PENDING_ASK_CLEANUP_BATCH_SIZE = 100

logger = logging.getLogger(__name__)

RuntimeEvent = AgentEvent | FullCompactProgress


class TurnRequest(BaseModel):
    """Caller input for a new runtime turn."""

    model_config = ConfigDict(frozen=True)

    run_id: str | None = Field(default=None, min_length=1, max_length=64)
    invoker: PrincipalRef
    content: str = ""
    attachments: tuple[AttachmentRef, ...] = ()
    invocation_target: InvocationTarget | None = None
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    owner: PrincipalRef | None = None
    request_id: str = Field(default_factory=new_id, min_length=1, max_length=128)
    input_message_id: str | None = Field(default=None, min_length=1, max_length=64)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=128)
    conversation_title: str = Field(default="", max_length=256)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_input(self) -> TurnRequest:
        """Require text, an image, or a structured Skill/Scene target."""

        if not self.content.strip() and not self.attachments and self.invocation_target is None:
            raise ValueError("content or attachments is required.")
        return self


class AskAnswer(BaseModel):
    """Caller response that resumes one suspended Ask tool call."""

    model_config = ConfigDict(frozen=True)

    ask_id: str = Field(min_length=1, max_length=128)
    answers: dict[str, str | list[str]] = Field(default_factory=dict)
    status: Literal["answered", "skipped"] = "answered"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TurnBindings(BaseModel):
    """Already-authorized capabilities injected by the host for one turn."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    model: SkipValidation[ChatModel | None] = None
    model_provider: SkipValidation[ChatModelProvider | None] = None
    workspace: SkipValidation[WorkspaceSession]
    tool_set: SkipValidation[ToolSet] = Field(default_factory=ToolSet)
    tool_runtime: ToolRuntimeBindings = Field(default_factory=ToolRuntimeBindings)
    prompt: SystemPrompt = Field(default_factory=build_system_prompt)
    skill_catalog: SkipValidation[SkillCatalog | None] = None
    scene_catalog: SkipValidation[SceneCatalog | None] = None
    skill_runtime_variables: SkipValidation[RuntimeVariableProvider | None] = None
    attachment_loader: SkipValidation[AttachmentLoader | None] = None
    max_visible_skills: int = Field(default=256, ge=1)
    max_skill_listing_bytes: int = Field(default=256 * 1024, ge=1)
    max_iterations: int = Field(default=20, ge=1)
    compaction_policy: CompactionPolicy | None = None
    compaction_model: SkipValidation[CompactionModel | None] = None
    compaction_model_provider: SkipValidation[CompactionModelProvider | None] = None
    model_trace_sink: SkipValidation[ModelTraceSink | None] = None

    @model_validator(mode="after")
    def validate_model_bindings(self) -> TurnBindings:
        """Require one main model source and at most one compaction model source."""

        if (self.model is None) == (self.model_provider is None):
            raise ValueError("Exactly one of model or model_provider is required.")
        if self.compaction_model is not None and self.compaction_model_provider is not None:
            raise ValueError(
                "compaction_model and compaction_model_provider cannot both be configured."
            )
        return self


type TurnBindingsFactory = Callable[[], Awaitable[TurnBindings]]


class PreparedAgentTurn(BaseModel):
    """Caller input plus eager or deferred capabilities prepared by one host."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    request: TurnRequest
    bindings: TurnBindings | None = None
    bindings_factory: SkipValidation[TurnBindingsFactory | None] = None

    @model_validator(mode="after")
    def validate_bindings(self) -> PreparedAgentTurn:
        """Require one eager or deferred authorized capability binding."""

        if (self.bindings is None) == (self.bindings_factory is None):
            raise ValueError("Exactly one of bindings or bindings_factory is required.")
        return self


class RuntimeCompactionBoundary(BaseModel):
    """Public identity and sequence boundary of the latest compacted history."""

    model_config = ConfigDict(frozen=True)

    compaction_id: str = Field(min_length=1, max_length=64)
    generation: int = Field(ge=1)
    through_sequence: int = Field(ge=1)


class RuntimeConversationSnapshot(BaseModel):
    """Current durable run, Ask and compacted-history projections."""

    model_config = ConfigDict(frozen=True)

    run: AgentRun | None = None
    pending_ask: dict[str, Any] | None = None
    compaction_boundary: RuntimeCompactionBoundary | None = None


class TurnSession:
    """Single-consumption handle returned before streamed execution begins."""

    def __init__(
        self,
        *,
        run: AgentRun,
        replayed: bool,
        stream_factory: Callable[[], AsyncIterator[RuntimeEvent]],
    ) -> None:
        self.run = run
        self.replayed = replayed
        self._stream_factory = stream_factory
        self._consumed = False

    @property
    def run_id(self) -> str:
        return self.run.run_id

    @property
    def conversation_id(self) -> str:
        return self.run.conversation_id

    async def stream(self) -> AsyncIterator[RuntimeEvent]:
        """Stream this session exactly once."""

        if self._consumed:
            raise RuntimeError("A TurnSession stream can only be consumed once.")
        self._consumed = True
        async for event in self._stream_factory():
            yield event


class AgentRuntime:
    """Own conversation persistence and execute host-bound Agent turns."""

    def __init__(
        self,
        *,
        store: RuntimeStore,
        run_lease: RunLease | None = None,
        state_cache: StateCache | None = None,
        image_payload_manager: ImagePayloadManager | None = None,
        micro_compact_keep_recent_tool_results: int = 5,
        max_concurrent_turns: int = 32,
        queue_capacity: int = 128,
        admission_timeout_seconds: float = 5.0,
        pending_ask_cleanup_interval_seconds: int = 60,
    ) -> None:
        self._store = store
        self._run_lease = run_lease or InMemoryRunLease()
        self._state_cache = state_cache
        self._context = ConversationContextBuilder(
            store,
            keep_recent_tool_results=micro_compact_keep_recent_tool_results,
        )
        self._compaction = CompactionService(
            store,
            keep_recent_tool_results=micro_compact_keep_recent_tool_results,
        )
        self._image_payload_manager = image_payload_manager or ImagePayloadManager()
        self._admission = FairAsyncCapacityLimiter(
            capacity_name="Agent turn",
            max_active=max_concurrent_turns,
            queue_capacity=queue_capacity,
            admission_timeout_seconds=admission_timeout_seconds,
        )
        self._pending_ask_cleanup_interval_seconds = max(
            pending_ask_cleanup_interval_seconds,
            60,
        )
        self._pending_ask_cleanup_task: asyncio.Task[None] | None = None
        self._active_turns: dict[
            tuple[str, str],
            tuple[asyncio.Task[Any], asyncio.Event, asyncio.Event],
        ] = {}

    @property
    def pending_ask_cleanup_interval_seconds(self) -> int:
        """Return the maintenance interval with its minimum supported floor."""

        return self._pending_ask_cleanup_interval_seconds

    def start(self) -> None:
        """Start process-local Runtime maintenance."""

        if self._pending_ask_cleanup_task is not None:
            return
        self._pending_ask_cleanup_task = asyncio.create_task(self._run_pending_ask_cleanup())

    async def stop(self) -> None:
        """Stop process-local Runtime maintenance."""

        task = self._pending_ask_cleanup_task
        self._pending_ask_cleanup_task = None
        if task is None:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def admission_stats(self) -> AsyncAdmissionStats:
        """Return process-local Agent execution capacity state."""

        return await self._admission.snapshot()

    async def run_lease_status(
        self,
        conversation_id: str,
        run_id: str,
    ) -> RunLeaseStatus:
        """Return the shared coordination status for one observed run identity."""

        return await self._run_lease.status(conversation_id, run_id)

    async def cleanup_expired_pending_asks(self) -> int:
        """Resolve one cutoff's expired Ask states in stable keyset pages."""

        expires_before = utc_now()
        after: ConversationState | None = None
        resolved_count = 0
        while True:
            states = await self._store.list_expired_states(
                PENDING_ASK_STATE,
                expires_before,
                after=after,
                limit=PENDING_ASK_CLEANUP_BATCH_SIZE,
            )
            if not states:
                return resolved_count
            for state in states:
                run_id = str(state.payload.get("run_id") or "")
                run = await self._store.get_run(run_id) if run_id else None
                if run is None or run.status is not AgentRunStatus.WAITING_INPUT:
                    continue
                if not await self._waiting_run_is_unowned(run):
                    continue
                if await self._resolve_pending_ask(run, state, reason="expired") is not None:
                    resolved_count += 1
            after = states[-1]
            if len(states) < PENDING_ASK_CLEANUP_BATCH_SIZE:
                return resolved_count

    async def _run_pending_ask_cleanup(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.pending_ask_cleanup_interval_seconds)
                await self.cleanup_expired_pending_asks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Pending Ask cleanup failed exception_type=%s",
                    type(exc).__name__,
                )

    async def start_prepared_turn(self, prepared: PreparedAgentTurn) -> TurnSession:
        """Start one host-authorized turn without interpreting subscriber policy."""

        return await self.start_turn(
            prepared.request,
            prepared.bindings,
            bindings_factory=prepared.bindings_factory,
        )

    async def start_turn(
        self,
        request: TurnRequest,
        bindings: TurnBindings | None = None,
        *,
        bindings_factory: TurnBindingsFactory | None = None,
    ) -> TurnSession:
        """Persist an input and return a handle for its model/tool execution."""

        if (bindings is None) == (bindings_factory is None):
            raise ValueError("Exactly one of bindings or bindings_factory is required.")

        conversation = await self._resolve_conversation(request)
        expected_owner = request.owner or request.invoker
        self._require_subscriber(request.invoker, expected_owner)
        self._require_principal(expected_owner, conversation.owner)
        proposed = AgentRun(
            run_id=request.run_id or new_id(),
            conversation_id=conversation.conversation_id,
            invoker=request.invoker,
            request_id=request.request_id,
            idempotency_key=request.idempotency_key,
            input_snapshot={
                "content": request.content,
                "input_message_id": request.input_message_id,
                "metadata": request.metadata,
                "invocation_target": (
                    request.invocation_target.model_dump(mode="json")
                    if request.invocation_target is not None
                    else None
                ),
                "attachments": [value.to_message_payload() for value in request.attachments],
            },
            model_snapshot=(
                _model_snapshot(bindings.model)
                if bindings is not None and bindings.model is not None
                else {}
            ),
        )
        run = await self._store.create_run(proposed)
        replayed = run.run_id != proposed.run_id and run.status is not AgentRunStatus.PENDING

        def factory() -> AsyncIterator[RuntimeEvent]:
            if replayed:
                return self._replay(run)
            return self._execute(
                run,
                bindings,
                bindings_factory=bindings_factory,
                expected=(AgentRunStatus.PENDING,),
                append_input=True,
                attachments=request.attachments,
            )

        return TurnSession(run=run, replayed=replayed, stream_factory=factory)

    async def resume_ask(
        self,
        *,
        run_id: str,
        invoker: PrincipalRef,
        answer: AskAnswer,
        bindings: TurnBindings | None = None,
        bindings_factory: TurnBindingsFactory | None = None,
        request_id: str | None = None,
    ) -> TurnSession:
        """Return a handle that atomically resumes one pending Ask."""

        if (bindings is None) == (bindings_factory is None):
            raise ValueError("Exactly one of bindings or bindings_factory is required.")

        run = await self._store.get_run(run_id)
        if run is None:
            raise EntityNotFoundError("Run does not exist.")
        self._require_principal(invoker, run.invoker)
        if run.status is not AgentRunStatus.WAITING_INPUT:
            raise AskNotPendingError("Run is not waiting for Ask input.")
        if request_id is not None:
            run = run.model_copy(update={"request_id": request_id})

        async def stream() -> AsyncIterator[RuntimeEvent]:
            async for event in self._execute(
                run,
                bindings,
                bindings_factory=bindings_factory,
                expected=(AgentRunStatus.WAITING_INPUT,),
                answer=answer,
            ):
                yield event

        return TurnSession(run=run, replayed=False, stream_factory=stream)

    async def inspect_conversation(
        self,
        conversation_id: str,
        invoker: PrincipalRef,
        *,
        owner: PrincipalRef | None = None,
        resolve_expired_ask: bool = True,
    ) -> RuntimeConversationSnapshot:
        """Reconcile an owned conversation, allowing explicit same-subscriber delegation."""

        expected_owner = owner or invoker
        self._require_subscriber(invoker, expected_owner)
        conversation = await self._store.get_conversation(conversation_id)
        if conversation is None:
            return RuntimeConversationSnapshot()
        self._require_principal(expected_owner, conversation.owner)
        latest_compaction = await self._store.get_latest_compaction(conversation_id)
        compaction_boundary = (
            RuntimeCompactionBoundary(
                compaction_id=latest_compaction.compaction_id,
                generation=latest_compaction.generation,
                through_sequence=latest_compaction.through_sequence,
            )
            if latest_compaction is not None
            else None
        )

        run = None
        if conversation.active_run_id is not None:
            active_run = await self._store.get_run(conversation.active_run_id)
            if (
                active_run is not None
                and active_run.conversation_id == conversation_id
                and active_run.status is AgentRunStatus.RUNNING
            ):
                run = active_run
        if run is None:
            run = await self._store.get_latest_run(conversation_id)
        if run is not None and run.status is AgentRunStatus.RUNNING:
            try:
                lease_status = await self._run_lease.status(conversation_id, run.run_id)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Unable to reconcile Agent run coordination state exception_type=%s",
                    type(exc).__name__,
                )
            else:
                if lease_status is RunLeaseStatus.LOST:
                    try:
                        run = await self._store.transition_run(
                            run.run_id,
                            expected=(AgentRunStatus.RUNNING,),
                            status=AgentRunStatus.FAILED,
                            error_code="run_lease_lost",
                            error_message="Agent run coordination was lost.",
                        )
                    except ConcurrentWriteError:
                        run = await self._store.get_latest_run(conversation_id) or run
        state = await self._get_state(
            conversation_id,
            PENDING_ASK_STATE,
            include_expired=True,
        )
        if (
            run is not None
            and run.status is AgentRunStatus.WAITING_INPUT
            and not _valid_pending_ask_state(state, run.run_id)
        ):
            repaired = False
            try:
                run = await self._store.transition_run(
                    run.run_id,
                    expected=(AgentRunStatus.WAITING_INPUT,),
                    status=AgentRunStatus.CANCELLED,
                    error_code="ask_state_missing",
                    error_message="Pending Ask state does not exist for the waiting run.",
                )
                repaired = True
            except ConcurrentWriteError:
                run = await self._store.get_latest_run(conversation_id) or run
            if repaired and state is not None:
                try:
                    await self._delete_state(
                        conversation_id,
                        PENDING_ASK_STATE,
                        expected_revision=state.revision,
                    )
                except ConcurrentWriteError:
                    logger.warning(
                        "Pending Ask state changed while repairing an invalid waiting run."
                    )
            return RuntimeConversationSnapshot(
                run=run,
                compaction_boundary=compaction_boundary,
            )
        if (
            resolve_expired_ask
            and state is not None
            and state.expires_at is not None
            and state.expires_at <= utc_now()
        ):
            pending_run_id = str(state.payload.get("run_id") or "")
            pending_run = await self._store.get_run(pending_run_id) if pending_run_id else None
            if (
                pending_run is not None
                and pending_run.status is AgentRunStatus.WAITING_INPUT
                and await self._waiting_run_is_unowned(pending_run)
            ):
                resolved = await self._resolve_pending_ask(
                    pending_run,
                    state,
                    reason="expired",
                )
                if resolved is not None:
                    return RuntimeConversationSnapshot(
                        run=await self._store.get_latest_run(conversation_id),
                        compaction_boundary=compaction_boundary,
                    )
            run = await self._store.get_latest_run(conversation_id)
            state = await self._get_state(
                conversation_id,
                PENDING_ASK_STATE,
                include_expired=True,
            )
        pending_ask = None
        if (
            state is not None
            and run is not None
            and run.status is AgentRunStatus.WAITING_INPUT
            and state.payload.get("run_id") == run.run_id
        ):
            pending_ask = {key: value for key, value in state.payload.items() if key != "run_id"}
            if state.expires_at is not None:
                pending_ask["expires_at"] = state.expires_at.isoformat()
        return RuntimeConversationSnapshot(
            run=run,
            pending_ask=pending_ask,
            compaction_boundary=compaction_boundary,
        )

    async def _waiting_run_is_unowned(self, run: AgentRun) -> bool:
        try:
            status = await self._run_lease.status(run.conversation_id, run.run_id)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Unable to reconcile Pending Ask coordination state exception_type=%s",
                type(exc).__name__,
            )
            return False
        return status is RunLeaseStatus.LOST

    async def interrupt_conversation(
        self,
        conversation_id: str,
        invoker: PrincipalRef,
        *,
        owner: PrincipalRef | None = None,
        expected_run_id: str | None = None,
        record_idle_interrupt: bool = True,
    ) -> bool:
        """Interrupt an owned run, allowing only explicit same-subscriber delegation."""

        conversation = await self._store.get_conversation(conversation_id)
        if conversation is None:
            return False
        expected_owner = owner or invoker
        self._require_subscriber(invoker, expected_owner)
        self._require_principal(expected_owner, conversation.owner)
        cancelled_run_id = await self._request_run_cancel(
            conversation_id,
            expected_run_id=expected_run_id,
        )
        if cancelled_run_id is not None:
            cancelled_run = await self._store.get_run(cancelled_run_id)
            if cancelled_run is not None and cancelled_run.status in {
                AgentRunStatus.PENDING,
                AgentRunStatus.RUNNING,
            }:
                with suppress(ConcurrentWriteError):
                    await self._store.transition_run(
                        cancelled_run_id,
                        expected=(cancelled_run.status,),
                        status=AgentRunStatus.CANCELLED,
                    )
            elif cancelled_run is not None and cancelled_run.status is AgentRunStatus.WAITING_INPUT:
                await self._cancel_run(cancelled_run_id)
            return True
        if expected_run_id is not None:
            return True
        run = await self._store.get_latest_run(conversation_id)
        if run is not None and run.status in {
            AgentRunStatus.PENDING,
            AgentRunStatus.RUNNING,
            AgentRunStatus.WAITING_INPUT,
        }:
            if await self._cancel_run(run.run_id):
                return True
            with suppress(ConcurrentWriteError):
                await self._store.transition_run(
                    run.run_id,
                    expected=(run.status,),
                    status=AgentRunStatus.CANCELLED,
                )
            return True
        if not record_idle_interrupt:
            return True
        marker = await self._store.create_run(
            AgentRun(
                conversation_id=conversation_id,
                invoker=invoker,
                input_snapshot={"interrupted_without_active_run": True},
            )
        )
        await self._store.transition_run(
            marker.run_id,
            expected=(AgentRunStatus.PENDING,),
            status=AgentRunStatus.CANCELLED,
        )
        return True

    async def _cancel_run(self, run_id: str) -> bool:
        """Cancel a run after its owning conversation has been authorized."""

        run = await self._store.get_run(run_id)
        if run is None:
            return False
        if run.status is AgentRunStatus.WAITING_INPUT:
            await self._store.transition_run(
                run_id,
                expected=(AgentRunStatus.WAITING_INPUT,),
                status=AgentRunStatus.CANCELLED,
            )
            state = await self._get_state(run.conversation_id, PENDING_ASK_STATE)
            if state is not None:
                await self._delete_state(
                    run.conversation_id,
                    PENDING_ASK_STATE,
                    expected_revision=state.revision,
                )
            return True
        owner = await self._request_run_cancel(
            run.conversation_id,
            expected_run_id=run_id,
        )
        return owner is not None

    async def _request_run_cancel(
        self,
        conversation_id: str,
        *,
        expected_run_id: str | None,
    ) -> str | None:
        owner = await self._run_lease.request_cancel(
            conversation_id,
            expected_run_id=expected_run_id,
        )
        if owner is None:
            return None
        active = self._active_turns.get((conversation_id, owner))
        if active is not None:
            task, cancel_requested, _ = active
            cancel_requested.set()
            task.cancel()
        return owner

    async def _resolve_conversation(self, request: TurnRequest) -> Conversation:
        if request.conversation_id is not None:
            existing = await self._store.get_conversation(request.conversation_id)
            if existing is not None:
                return existing
        owner = request.owner or request.invoker
        self._require_subscriber(request.invoker, owner)
        conversation = Conversation(
            conversation_id=request.conversation_id or new_id(),
            owner=owner,
            title=request.conversation_title,
        )
        return await self._store.create_conversation(conversation)

    async def _execute(
        self,
        run: AgentRun,
        bindings: TurnBindings | None,
        *,
        bindings_factory: TurnBindingsFactory | None = None,
        expected: Sequence[AgentRunStatus],
        answer: AskAnswer | None = None,
        append_input: bool = False,
        attachments: Sequence[AttachmentRef] = (),
    ) -> AsyncIterator[RuntimeEvent]:
        try:
            lease = await self._admission.acquire()
        except AsyncAdmissionCapacityExceededError:
            error = ExecutionError(
                code="runtime_capacity",
                message="Agent runtime is temporarily unavailable.",
            )
            if answer is None:
                with suppress(ConcurrentWriteError):
                    await self._store.transition_run(
                        run.run_id,
                        expected=expected,
                        status=AgentRunStatus.FAILED,
                        error_code=error.code,
                        error_message=error.message,
                    )
            yield error
            return
        terminal_event: RuntimeEvent | None = None
        async with lease:
            async for event in self._execute_admitted(
                run,
                bindings,
                bindings_factory=bindings_factory,
                expected=expected,
                answer=answer,
                append_input=append_input,
                attachments=attachments,
            ):
                if isinstance(event, (AskRequested, AssistantFinal, ExecutionError)):
                    if terminal_event is not None:
                        raise RuntimeError("Agent Runtime emitted more than one terminal event.")
                    terminal_event = event
                else:
                    yield event
        if terminal_event is not None:
            yield terminal_event

    async def _execute_admitted(
        self,
        run: AgentRun,
        bindings: TurnBindings | None,
        *,
        bindings_factory: TurnBindingsFactory | None = None,
        expected: Sequence[AgentRunStatus],
        answer: AskAnswer | None = None,
        append_input: bool = False,
        attachments: Sequence[AttachmentRef] = (),
    ) -> AsyncIterator[RuntimeEvent]:
        if not await self._run_lease.acquire(run.conversation_id, run.run_id):
            error = ExecutionError(
                code="concurrent_run",
                message="Conversation already has an active turn.",
            )
            if answer is None:
                with suppress(ConcurrentWriteError):
                    await self._store.transition_run(
                        run.run_id,
                        expected=expected,
                        status=AgentRunStatus.FAILED,
                        error_code=error.code,
                        error_message=error.message,
                    )
            yield error
            return
        conversation = await self._store.get_conversation(run.conversation_id)
        if conversation is None or conversation.status is not ConversationStatus.ACTIVE:
            error = ExecutionError(
                code="conversation_not_active",
                message="Conversation does not exist or is not active for this actor.",
            )
            with suppress(ConcurrentWriteError):
                await self._store.transition_run(
                    run.run_id,
                    expected=expected,
                    status=AgentRunStatus.FAILED,
                    error_code=error.code,
                    error_message=error.message,
                )
            await self._run_lease.release(run.conversation_id, run.run_id)
            yield error
            return
        owner_task = asyncio.current_task()
        if owner_task is None:
            await self._run_lease.release(run.conversation_id, run.run_id)
            raise RuntimeError("Agent turn must execute inside an asyncio task.")
        cancel_requested = asyncio.Event()
        renewal_lost = asyncio.Event()
        active_key = (run.conversation_id, run.run_id)
        self._active_turns[active_key] = (owner_task, cancel_requested, renewal_lost)
        renewal_task = self._start_renewal(run, renewal_lost, owner_task)
        monitor_task = self._start_lease_monitor(
            run,
            cancel_requested,
            renewal_lost,
            owner_task,
        )
        image_stack = AsyncExitStack()  # noqa
        terminal_event: RuntimeEvent | None = None
        try:
            pending_state = None
            if answer is not None:
                pending_state = await self._validate_ask_answer(run, answer)
            await self._store.transition_run(
                run.run_id,
                expected=expected,
                status=AgentRunStatus.RUNNING,
                expected_active_run_id=conversation.active_run_id,
            )
            if append_input:
                await self._resolve_pending_ask_before_new_input(run)
            if bindings is None:
                if bindings_factory is None:
                    raise RuntimeError("Turn bindings are not configured.")
                bindings = await bindings_factory()
            if answer is not None and pending_state is not None:
                yield await self._apply_ask_answer(run, answer, pending_state)
            skill_state = await self._get_state(run.conversation_id, SKILL_STATE)
            skill_payload = _dict(skill_state.payload) if skill_state is not None else {}
            skill_registry: SkillRegistry | None = None
            current_input_message = None
            if append_input:
                preparation = await prepare_skill_turn(
                    metadata=_dict(run.input_snapshot.get("metadata")),
                    invocation_target=_invocation_target(
                        run.input_snapshot.get("invocation_target")
                    ),
                    tool_context=bindings.tool_runtime.create_context(
                        conversation_id=run.conversation_id,
                        run_id=run.run_id,
                        workspace=bindings.workspace,
                    ),
                    skill_catalog=bindings.skill_catalog,
                    scene_catalog=bindings.scene_catalog,
                    persisted_state=skill_payload,
                    max_visible_skills=bindings.max_visible_skills,
                    max_skill_listing_bytes=bindings.max_skill_listing_bytes,
                    runtime_variables=bindings.skill_runtime_variables,
                )
                skill_registry = preparation.registry
                input_values: dict[str, Any] = {
                    "role": MessageRole.USER,
                    "kind": MessageKind.INPUT,
                    "content": str(run.input_snapshot.get("content") or ""),
                    "payload": {
                        **preparation.input_metadata,
                        **(
                            {"attachments": [value.to_message_payload() for value in attachments]}
                            if attachments
                            else {}
                        ),
                    },
                    "run_id": run.run_id,
                    "request_id": run.request_id,
                }
                input_message_id = run.input_snapshot.get("input_message_id")
                if isinstance(input_message_id, str) and input_message_id:
                    input_values["message_id"] = input_message_id
                input_message = NewConversationMessage.model_validate(input_values)
                prepared_messages = tuple(
                    value.model_copy(update={"run_id": run.run_id, "request_id": run.request_id})
                    for value in (
                        input_message,
                        *(_new_meta_message(value) for value in preparation.messages),
                    )
                )
                persisted_input = await self._store.append_messages_for_run(
                    run.run_id,
                    prepared_messages,
                )
                if persisted_input is None:
                    raise ConcurrentWriteError("Run ownership was lost before input persistence.")
                skill_payload = preparation.state
                if preparation.state_dirty:
                    skill_state = await self._persist_skill_state(
                        run,
                        skill_payload,
                        skill_state,
                    )
                if preparation.input_metadata.get("llm_ignore") is not True:
                    content = str(run.input_snapshot.get("content") or "")
                    if attachments:
                        if bindings.attachment_loader is None:
                            raise RuntimeError("Chat attachment loader is not configured.")
                        current_input_message = await image_stack.enter_async_context(
                            self._image_payload_manager.prepare(
                                content,
                                attachments,
                                bindings.attachment_loader,
                            )
                        )
                    else:
                        current_input_message = Message.user(content)
            elif bindings.skill_catalog is not None:
                skill_registry = await SkillRegistry.load(
                    bindings.skill_catalog,
                    max_visible_skills=bindings.max_visible_skills,
                )
            registry = bind_skill_tool(
                bindings.tool_set,
                skill_registry,
                runtime_variables=bindings.skill_runtime_variables,
            )
            file_state = await self._get_state(run.conversation_id, FILE_STATE)
            file_state_cache = (
                FileStateCache.from_payload(file_state.payload)
                if file_state is not None
                else FileStateCache()
            )
            if (
                append_input
                and bindings.compaction_policy is not None
                and (
                    bindings.compaction_model is not None
                    or bindings.compaction_model_provider is not None
                )
            ):
                compaction_model = bindings.compaction_model
                if compaction_model is None:
                    assert bindings.compaction_model_provider is not None
                    compaction_model = await bindings.compaction_model_provider.resolve()
                skill_before_compaction = deepcopy(skill_payload)
                compact_result: FullCompactResult | None = None
                async for compact_event in self._compaction.prepare_events(
                    run.conversation_id,
                    run_id=run.run_id,
                    request_id=run.request_id,
                    system_prompt=bindings.prompt.full,
                    current_input_message=current_input_message,
                    tools=registry.all(),
                    policy=bindings.compaction_policy,
                    model=compaction_model,
                    file_state_cache=file_state_cache,
                    skill_state=skill_payload,
                    reserved_context_tokens=(
                        SELECTED_SCENE_TOKEN_RESERVE
                        if append_input and _has_selected_scene(run.input_snapshot)
                        else 0
                    ),
                ):
                    if isinstance(compact_event, FullCompactProgress):
                        yield compact_event
                    else:
                        compact_result = compact_event
                if compact_result is None:
                    raise RuntimeError("Full Compact completed without a result.")
                messages = list(compact_result.messages)
                if compact_result.compacted:
                    await self._invalidate_cached_states(
                        run.conversation_id,
                        compact_result.changed_state_kinds,
                    )
                    file_state = await self._store.get_state(run.conversation_id, FILE_STATE)
                    skill_state = await self._store.get_state(run.conversation_id, SKILL_STATE)
                else:
                    file_state = await self._persist_file_state(
                        run,
                        file_state_cache,
                        file_state,
                    )
                    if skill_payload != skill_before_compaction:
                        skill_state = await self._persist_skill_state(
                            run,
                            skill_payload,
                            skill_state,
                        )
            else:
                messages = await self._context.build(
                    run.conversation_id,
                    system_prompt=bindings.prompt.full,
                    file_state_cache=file_state_cache,
                    skip_input_run_id=run.run_id if append_input else "",
                    current_input_message=current_input_message,
                )
                file_state = await self._persist_file_state(
                    run,
                    file_state_cache,
                    file_state,
                )
            executor = ToolExecutor(
                registry,
                bindings.tool_runtime.create_context(
                    conversation_id=run.conversation_id,
                    run_id=run.run_id,
                    workspace=bindings.workspace,
                    file_state_cache=file_state_cache,
                ),
            )
            engine = AgentEngine(
                model=bindings.model,
                model_provider=bindings.model_provider,
                tool_executor=executor,
                model_trace_sink=bindings.model_trace_sink,
            )
            terminal = False
            async for event in engine.execute(
                ExecutionRequest(
                    messages=tuple(messages),
                    tools=registry.all(),
                    max_iterations=bindings.max_iterations,
                )
            ):
                if renewal_lost.is_set():
                    raise ConcurrentRunError("Run lease was lost during execution.")
                lease_status = await self._run_lease.status(run.conversation_id, run.run_id)
                if lease_status is RunLeaseStatus.CANCEL_REQUESTED:
                    cancelled = ExecutionError(
                        code="cancelled",
                        message="Conversation run interrupted.",
                    )
                    await self._store.transition_run(
                        run.run_id,
                        expected=(AgentRunStatus.RUNNING,),
                        status=AgentRunStatus.CANCELLED,
                    )
                    terminal_event = cancelled
                    terminal = True
                    break
                if lease_status is RunLeaseStatus.LOST:
                    raise ConcurrentRunError("Run lease was lost during execution.")
                if isinstance(event, ToolResultEvent):
                    file_state = await self._persist_file_state(
                        run,
                        file_state_cache,
                        file_state,
                    )
                finish_status = None
                finish_error_code = ""
                finish_error_message = ""
                if isinstance(event, AssistantFinal):
                    finish_status = AgentRunStatus.COMPLETED
                elif isinstance(event, ExecutionError):
                    finish_status = AgentRunStatus.FAILED
                    finish_error_code = event.code
                    finish_error_message = event.message
                await self._persist_event(
                    run,
                    event,
                    registry,
                    finish_status=finish_status,
                    error_code=finish_error_code,
                    error_message=finish_error_message,
                )
                if isinstance(event, ToolResultEvent):
                    invocation = event.result.extra.get("skill_invocation")
                    if isinstance(invocation, dict) and record_skill_invocation(
                        skill_payload,
                        invocation,
                    ):
                        skill_state = await self._persist_skill_state(
                            run,
                            skill_payload,
                            skill_state,
                        )
                if isinstance(event, AskRequested):
                    terminal = True
                    terminal_event = event
                    break
                if isinstance(event, AssistantFinal):
                    terminal = True
                    terminal_event = event
                    break
                if isinstance(event, ExecutionError):
                    terminal = True
                    terminal_event = event
                    break
                yield event
            if not terminal:
                raise RuntimeError("Agent Engine ended without a terminal event.")
        except asyncio.CancelledError:
            if cancel_requested.is_set():
                cancelled = ExecutionError(
                    code="cancelled",
                    message="Conversation run interrupted.",
                )
                with suppress(ConcurrentWriteError):
                    await self._store.transition_run(
                        run.run_id,
                        expected=(AgentRunStatus.PENDING, AgentRunStatus.RUNNING),
                        status=AgentRunStatus.CANCELLED,
                    )
                terminal_event = cancelled
            elif renewal_lost.is_set():
                error = ExecutionError(
                    code="run_lease_lost",
                    message="Agent run coordination was lost.",
                )
                with suppress(ConcurrentWriteError):
                    await self._persist_event(
                        run,
                        error,
                        ToolSet(),
                        finish_status=AgentRunStatus.FAILED,
                        error_code=error.code,
                        error_message=error.message,
                    )
                terminal_event = error
            else:
                raise
        except (
            SafeExecutionError,
            ContextCompactionFailedError,
            ContextLimitExceededError,
            SkillCapacityExceededError,
            ImageCapacityExceededError,
            ModelAuthenticationError,
            ModelPermissionDeniedError,
            ModelRateLimitError,
            ModelRequestRejectedError,
            ModelTimeoutError,
            ModelUnavailableError,
            MessageWriteConflictError,
        ) as exc:
            error = _known_execution_error(exc)
            if isinstance(exc, ModelInvocationError) and logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MODEL_CALL_FAILED run_id=%s conversation_id=%s request_id=%s "
                    "error_code=%s http_status=%s exception_chain=%s",
                    run.run_id,
                    run.conversation_id,
                    run.request_id,
                    error.code,
                    _model_failure_http_status(exc),
                    _safe_exception_chain(exc),
                )
            with suppress(ConcurrentWriteError):
                await self._persist_event(
                    run,
                    error,
                    ToolSet(),
                    finish_status=AgentRunStatus.FAILED,
                    error_code=error.code,
                    error_message=error.message,
                )
            terminal_event = error
        except (ConcurrentRunError, ConcurrentWriteError):
            terminal_event = ExecutionError(
                code="",
                message="Agent run ownership was lost.",
            )
        except (AskExpiredError, AskNotPendingError):
            raise
        except Exception as exc:
            error = _internal_error("Agent turn failed.")
            with suppress(ConcurrentWriteError):
                await self._persist_event(
                    run,
                    error,
                    ToolSet(),
                    finish_status=AgentRunStatus.FAILED,
                    error_code=error.code,
                    error_message=type(exc).__name__,
                )
            terminal_event = error
        finally:
            await image_stack.aclose()
            if renewal_task is not None:
                renewal_task.cancel()
                with suppress(asyncio.CancelledError):
                    await renewal_task
            if monitor_task is not None:
                monitor_task.cancel()
                with suppress(asyncio.CancelledError):
                    await monitor_task
            self._active_turns.pop(active_key, None)
            await self._run_lease.release(run.conversation_id, run.run_id)
        if terminal_event is not None:
            yield terminal_event

    async def _validate_ask_answer(
        self,
        run: AgentRun,
        answer: AskAnswer,
    ) -> ConversationState:
        state = await self._get_state(
            run.conversation_id,
            PENDING_ASK_STATE,
            include_expired=True,
        )
        if state is None or state.payload.get("run_id") != run.run_id:
            raise AskNotPendingError("Pending Ask state does not exist.")
        if state.expires_at is not None and state.expires_at <= utc_now():
            resolved = await self._resolve_pending_ask(
                run,
                state,
                reason="expired",
            )
            if resolved is None:
                raise AskNotPendingError("Pending Ask state changed before expiration.")
            raise AskExpiredError("The pending Ask request has expired.")
        if state.payload.get("ask_id") != answer.ask_id:
            raise AskNotPendingError("ask_id does not match the pending request.")
        return state

    async def _resolve_pending_ask_before_new_input(self, run: AgentRun) -> None:
        state = await self._get_state(
            run.conversation_id,
            PENDING_ASK_STATE,
            include_expired=True,
        )
        if state is None:
            return
        pending_run_id = str(state.payload.get("run_id") or "")
        if not pending_run_id or pending_run_id == run.run_id:
            await self._delete_state_for_run(
                run.run_id,
                run.conversation_id,
                PENDING_ASK_STATE,
                expected_revision=state.revision,
            )
            return
        pending_run = await self._store.get_run(pending_run_id)
        reason: Literal["expired", "new_user_input"] = (
            "expired"
            if state.expires_at is not None and state.expires_at <= utc_now()
            else "new_user_input"
        )
        if pending_run is not None and pending_run.status is AgentRunStatus.WAITING_INPUT:
            await self._resolve_pending_ask(
                pending_run,
                state,
                reason=reason,
            )
            return
        await self._delete_state_for_run(
            run.run_id,
            run.conversation_id,
            PENDING_ASK_STATE,
            expected_revision=state.revision,
        )

    async def _resolve_pending_ask(
        self,
        run: AgentRun,
        state: ConversationState,
        *,
        reason: Literal["expired", "new_user_input"],
    ) -> AgentRun | None:
        payload = _pending_ask_resolution_payload(reason)
        resolved = await self._store.resolve_pending_ask(
            run.run_id,
            state_revision=state.revision,
            message=NewConversationMessage(
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                payload={
                    "tool_call_id": str(state.payload.get("tool_call_id") or ""),
                    "tool_name": str(state.payload.get("tool_name") or "ask_user"),
                    "result": payload,
                    "is_error": False,
                },
                run_id=run.run_id,
                request_id=run.request_id,
            ),
            error_code="ask_expired" if reason == "expired" else "ask_cancelled",
            error_message=payload["error"],
        )
        if resolved is not None and self._state_cache is not None:
            with suppress(Exception):
                await self._state_cache.delete(run.conversation_id, PENDING_ASK_STATE)
        return resolved

    async def _apply_ask_answer(
        self,
        run: AgentRun,
        answer: AskAnswer,
        state: ConversationState,
    ) -> ToolResultEvent:
        skipped = answer.status == "skipped"
        payload = {
            "answers": answer.answers,
            "annotations": {},
            "metadata": {"status": answer.status, "skipped": skipped},
            "error": "",
        }
        call = ToolCall(
            tool_call_id=str(state.payload.get("tool_call_id") or ""),
            name=str(state.payload.get("tool_name") or "ask_user"),
        )
        result = ToolResult(output=payload)
        committed = await self._store.commit_ask_answer(
            run.run_id,
            state_revision=state.revision,
            message=NewConversationMessage(
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                content=_json(result.output_payload()),
                payload={
                    "tool_call_id": call.tool_call_id,
                    "tool_name": call.name,
                    "result": result.output_payload(),
                    "is_error": False,
                },
                run_id=run.run_id,
                request_id=run.request_id,
            ),
        )
        if not committed:
            raise ConcurrentWriteError("Pending Ask state changed before answer was committed.")
        if self._state_cache is not None:
            with suppress(Exception):
                await self._state_cache.delete(run.conversation_id, PENDING_ASK_STATE)
        return ToolResultEvent(call=call, result=result)

    async def _persist_event(
        self,
        run: AgentRun,
        event: AgentEvent,
        registry: ToolSet,
        *,
        finish_status: AgentRunStatus | None = None,
        error_code: str = "",
        error_message: str = "",
    ) -> None:
        message: NewConversationMessage | None = None
        additional_messages: tuple[NewConversationMessage, ...] = ()
        if isinstance(event, AssistantDelta):
            return
        if isinstance(event, AssistantIntermediate):
            if not event.content.strip():
                return
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content=event.content,
                payload={"final": True, "llm_ignore": True},
            )
        elif isinstance(event, AssistantFinal):
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASSISTANT,
                content=event.content,
                payload={"final": True},
            )
        elif isinstance(event, ToolUse):
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.ASSISTANT,
                kind=MessageKind.TOOL_USE,
                content=event.assistant_text,
                payload={
                    "tool_call_id": event.call.tool_call_id,
                    "tool_name": event.call.name,
                    "arguments": event.call.arguments,
                },
            )
        elif isinstance(event, ToolResultEvent):
            persisted = _persisted_tool_result(event.call, event.result, registry)
            selected_tool = registry.get(event.call.name)
            content = str(persisted.get("error") or "") if event.result.is_error else ""
            payload = {
                "tool_call_id": event.call.tool_call_id,
                "tool_name": event.call.name,
                "result": persisted,
                "is_error": event.result.is_error,
                "extra": event.result.extra,
                "trace_result": (selected_tool.trace_result if selected_tool is not None else True),
            }
            protected = (
                selected_tool is not None
                and selected_tool.persistence_policy is PersistencePolicy.PROTECTED
            )
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.TOOL,
                kind=MessageKind.TOOL_RESULT,
                content="" if protected else content,
                payload={} if protected else payload,
                protected_body=(
                    ProtectedMessageBody(content=content, payload=payload) if protected else None
                ),
            )
            additional_messages = tuple(
                _new_meta_message(value) for value in event.result.new_messages
            )
        elif isinstance(event, AskRequested):
            expires_at = utc_now() + timedelta(seconds=event.timeout_seconds)
            state = ConversationState(
                conversation_id=run.conversation_id,
                kind=PENDING_ASK_STATE,
                revision=1,
                payload={
                    "run_id": run.run_id,
                    "ask_id": event.ask_id,
                    "request_id": run.request_id,
                    "tool_call_id": event.call.tool_call_id,
                    "tool_name": event.call.name,
                    "questions": event.questions,
                    "timeout_seconds": event.timeout_seconds,
                },
                expires_at=expires_at,
            )
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ASK,
                payload={**state.payload, "expires_at": expires_at.isoformat(), "llm_ignore": True},
                run_id=run.run_id,
                request_id=run.request_id,
            )
            committed = await self._store.commit_pending_ask(
                run.run_id,
                state=state,
                message=message,
            )
            if committed is None:
                raise ConcurrentWriteError("Run or Pending Ask state changed before suspension.")
            if self._state_cache is not None:
                with suppress(Exception):
                    await self._state_cache.set(state)
            return
        elif isinstance(event, ExecutionError):
            payload = {"error": event.message, "code": event.code}
            if event.error_id:
                payload["error_id"] = event.error_id
            message = NewConversationMessage(
                message_id=event.message_id,
                role=MessageRole.ASSISTANT,
                kind=MessageKind.ERROR,
                content=event.message,
                payload=payload,
            )
        if message is not None:
            persisted_messages = await self._store.append_messages_for_run(
                run.run_id,
                tuple(
                    item.model_copy(update={"run_id": run.run_id, "request_id": run.request_id})
                    for item in (message, *additional_messages)
                ),
                finish_status=finish_status,
                error_code=error_code,
                error_message=error_message,
            )
            if persisted_messages is None:
                raise ConcurrentWriteError("Run ownership was lost before message persistence.")

    async def _replay(self, run: AgentRun) -> AsyncIterator[AgentEvent]:
        messages = await self._store.list_messages_for_run(run.run_id)
        for message in messages:
            if (
                message.kind is MessageKind.ASSISTANT
                and message.payload.get("final") is True
                and message.payload.get("llm_ignore") is not True
            ):
                yield AssistantFinal(message_id=message.message_id, content=message.content)
            elif message.kind is MessageKind.TOOL_USE:
                yield ToolUse(
                    call=ToolCall(
                        tool_call_id=str(message.payload.get("tool_call_id") or ""),
                        name=str(message.payload.get("tool_name") or ""),
                        arguments=_dict(message.payload.get("arguments")),
                    ),
                    assistant_text=message.content,
                )
            elif message.kind is MessageKind.ERROR:
                yield ExecutionError(
                    code=str(message.payload.get("code") or "runtime_error"),
                    message=message.content,
                    error_id=str(message.payload.get("error_id") or ""),
                    message_id=message.message_id,
                )

    def _start_renewal(
        self,
        run: AgentRun,
        lost: asyncio.Event,
        owner_task: asyncio.Task[Any],
    ) -> asyncio.Task[None] | None:
        interval = self._run_lease.renewal_interval_seconds
        if interval is None:
            return None

        async def renew() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    renewed = await self._run_lease.renew(run.conversation_id, run.run_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Agent run lease renewal failed; retrying exception_type=%s",
                        type(exc).__name__,
                    )
                    continue
                if renewed:
                    continue
                lost.set()
                owner_task.cancel()
                return

        return asyncio.create_task(renew())

    def _start_lease_monitor(
        self,
        run: AgentRun,
        cancel_requested: asyncio.Event,
        lost: asyncio.Event,
        owner_task: asyncio.Task[Any],
    ) -> asyncio.Task[None] | None:
        interval = self._run_lease.monitor_interval_seconds
        if interval is None:
            return None

        async def monitor() -> None:
            while True:
                await asyncio.sleep(interval)
                try:
                    status = await self._run_lease.status(run.conversation_id, run.run_id)
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Agent run lease monitor failed; retrying exception_type=%s",
                        type(exc).__name__,
                    )
                    continue
                if status is RunLeaseStatus.ACTIVE:
                    continue
                if status is RunLeaseStatus.CANCEL_REQUESTED:
                    cancel_requested.set()
                else:
                    lost.set()
                owner_task.cancel()
                return

        return asyncio.create_task(monitor())

    async def _get_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        include_expired: bool = False,
    ) -> ConversationState | None:
        if self._state_cache is not None and not include_expired:
            try:
                cached = await self._state_cache.get(conversation_id, kind)
            except Exception:  # noqa: BLE001
                cached = None
            if cached is not None:
                return cached
        state = await self._store.get_state(
            conversation_id,
            kind,
            include_expired=include_expired,
        )
        if state is not None and self._state_cache is not None and not include_expired:
            with suppress(Exception):
                await self._state_cache.set(state)
        return state

    async def _save_state(
        self,
        run_id: str,
        state: ConversationState,
        *,
        expected_revision: int,
    ) -> ConversationState:
        stored = await self._store.save_state_for_run(
            run_id,
            state,
            expected_revision=expected_revision,
        )
        if stored is None:
            raise ConcurrentWriteError("Run ownership was lost before state persistence.")
        if self._state_cache is not None:
            with suppress(Exception):
                await self._state_cache.set(stored)
        return stored

    async def _delete_state_for_run(
        self,
        run_id: str,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        deleted = await self._store.delete_state_for_run(
            run_id,
            conversation_id,
            kind,
            expected_revision=expected_revision,
        )
        if not deleted:
            raise ConcurrentWriteError("Run ownership was lost before state deletion.")
        if self._state_cache is not None:
            with suppress(Exception):
                await self._state_cache.delete(conversation_id, kind)

    async def _delete_state(
        self,
        conversation_id: str,
        kind: str,
        *,
        expected_revision: int | None = None,
    ) -> None:
        await self._store.delete_state(
            conversation_id,
            kind,
            expected_revision=expected_revision,
        )
        if self._state_cache is not None:
            with suppress(Exception):
                await self._state_cache.delete(conversation_id, kind)

    async def _invalidate_cached_states(
        self,
        conversation_id: str,
        kinds: Sequence[str],
    ) -> None:
        if self._state_cache is None:
            return
        for kind in kinds:
            with suppress(Exception):
                await self._state_cache.delete(conversation_id, kind)

    async def _persist_file_state(
        self,
        run: AgentRun,
        cache: FileStateCache,
        current: ConversationState | None,
    ) -> ConversationState | None:
        """Persist one dirty FileState working copy under the active run lease."""

        if not cache.dirty:
            return current
        if cache.items():
            revision = current.revision if current is not None else 0
            saved = await self._save_state(
                run.run_id,
                ConversationState(
                    conversation_id=run.conversation_id,
                    kind=FILE_STATE,
                    revision=revision + 1,
                    payload=cache.to_payload(),
                ),
                expected_revision=revision,
            )
            cache.mark_clean()
            return saved
        if current is not None:
            await self._delete_state_for_run(
                run.run_id,
                run.conversation_id,
                FILE_STATE,
                expected_revision=current.revision,
            )
        cache.mark_clean()
        return None

    async def _persist_skill_state(
        self,
        run: AgentRun,
        payload: dict[str, Any],
        current: ConversationState | None,
    ) -> ConversationState:
        """Persist compact Skill markers under the active run lease."""

        revision = current.revision if current is not None else 0
        return await self._save_state(
            run.run_id,
            ConversationState(
                conversation_id=run.conversation_id,
                kind=SKILL_STATE,
                revision=revision + 1,
                payload=payload,
            ),
            expected_revision=revision,
        )

    @staticmethod
    def _require_subscriber(left: PrincipalRef, right: PrincipalRef) -> None:
        if left.subscriber_id != right.subscriber_id:
            raise SubscriberMismatchError("Runtime data cannot cross subscribers.")

    @classmethod
    def _require_principal(cls, left: PrincipalRef, right: PrincipalRef) -> None:
        cls._require_subscriber(left, right)
        if left != right:
            raise PrincipalMismatchError("Runtime data cannot cross principals.")


def _internal_error(log_message: str) -> ExecutionError:
    """Log one private failure and return its correlated safe projection."""

    error_id = new_uuid4_id()
    error = sys.exception()
    logger.error(
        "%s error_id=%s exception_type=%s exception_chain=%s",
        log_message,
        error_id,
        type(error).__name__ if error is not None else "unknown",
        _safe_exception_chain(error),
    )
    return ExecutionError(
        code=INTERNAL_AGENT_ERROR_CODE,
        message=f"{INTERNAL_AGENT_ERROR_MESSAGE} Reference ID: {error_id}.",
        error_id=error_id,
    )


def _known_execution_error(error: BaseException) -> ExecutionError:
    """Project one audited Runtime failure without assigning an internal reference ID."""

    if isinstance(error, SafeExecutionError):
        code = error.code
        message = error.message
    elif isinstance(error, ContextCompactionFailedError):
        code = "context_compaction_failed"
        message = str(error)
    elif isinstance(error, ContextLimitExceededError):
        code = "context_limit_exceeded"
        message = str(error)
    elif isinstance(error, SkillCapacityExceededError):
        code = "skill_capacity_exceeded"
        message = str(error)
    elif isinstance(error, ImageCapacityExceededError):
        code = "image_capacity_exceeded"
        message = str(error)
    elif isinstance(error, ModelAuthenticationError):
        code = "model_authentication_failed"
        message = "The configured model credentials were rejected."
    elif isinstance(error, ModelPermissionDeniedError):
        code = "model_permission_denied"
        message = "The configured model credentials cannot access the requested model."
    elif isinstance(error, ModelRateLimitError):
        code = "model_rate_limited"
        message = "The model service rate limit was reached. Please try again later."
    elif isinstance(error, ModelTimeoutError):
        code = "model_timeout"
        message = "The model service timed out. Please try again later."
    elif isinstance(error, ModelUnavailableError):
        code = "model_unavailable"
        message = "The model service is temporarily unavailable. Please try again later."
    elif isinstance(error, ModelRequestRejectedError):
        code = "model_request_rejected"
        message = "The model service rejected the request."
    elif isinstance(error, MessageWriteConflictError):
        code = "message_write_conflict"
        message = "A conversation message with the same identifier already exists."
    else:  # pragma: no cover - guarded by the caller's exception tuple
        raise TypeError(f"Unsupported known execution error: {type(error).__name__}")
    return ExecutionError(code=code, message=message)


def _model_failure_http_status(error: BaseException) -> int | Literal["none"]:
    """Inspect chained provider failures without logging request or response contents."""

    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        status = getattr(current, "status_code", None)
        if not isinstance(status, int):
            status = getattr(getattr(current, "response", None), "status_code", None)
        if isinstance(status, int) and 100 <= status <= 599:
            return status
        next_error = current.__cause__
        if next_error is None and not current.__suppress_context__:
            next_error = current.__context__
        current = next_error
    return "none"


def _safe_exception_chain(error: BaseException | None) -> str:
    """Return exception types and code locations without messages or frame values."""

    if error is None:
        return "unknown"
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        traceback = current.__traceback__
        if traceback is None:
            origin = "unknown"
        else:
            while traceback.tb_next is not None:
                traceback = traceback.tb_next
            frame = traceback.tb_frame
            module = str(frame.f_globals.get("__name__") or "unknown")
            origin = f"{module}:{traceback.tb_lineno}:{frame.f_code.co_qualname}"
        parts.append(f"{type(current).__name__}@{origin}")
        next_error = current.__cause__
        if next_error is None and not current.__suppress_context__:
            next_error = current.__context__
        current = next_error
    return " <- ".join(parts)


def _model_snapshot(model: ChatModel) -> dict[str, Any]:
    return {
        "model_ref": model.model_ref,
        "provider": model.provider,
        "model_name": model.model_name,
        "context_window": model.context_window,
        "support_vision": model.support_vision,
    }


def _new_meta_message(value: dict[str, Any]) -> NewConversationMessage:
    raw_role = str(value.get("role") or MessageRole.USER.value)
    try:
        role = MessageRole(raw_role)
    except ValueError:
        role = MessageRole.USER
    payload = {key: item for key, item in value.items() if key not in {"role", "content"}}
    return NewConversationMessage(
        role=role,
        kind=MessageKind.META,
        content=str(value.get("content") or ""),
        payload=payload,
    )


def _persisted_tool_result(
    call: ToolCall,
    result: ToolResult,
    registry: ToolSet,
) -> dict[str, Any]:
    tool = registry.get(call.name)
    policy = tool.persistence_policy if tool is not None else PersistencePolicy.NONE
    explicit = result.persistence_payload
    if policy in {PersistencePolicy.FULL, PersistencePolicy.PROTECTED}:
        return _dict(to_jsonable_python(result.raw_output_payload()))
    if explicit is not None:
        return _dict(to_jsonable_python(explicit))
    if policy is PersistencePolicy.SUMMARY:
        return {"omitted": True, "reason": "summary_not_provided"}
    if policy is PersistencePolicy.REFERENCE:
        return {"omitted": True, "reason": "reference_not_provided"}
    return {"omitted": True}


def _json(value: object) -> str:
    return json.dumps(to_jsonable_python(value), ensure_ascii=False, separators=(",", ":"))


def _dict(value: object) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _pending_ask_resolution_payload(
    reason: Literal["expired", "new_user_input"],
) -> dict[str, Any]:
    error = (
        "Pending ask_user request expired before it was answered."
        if reason == "expired"
        else "Pending ask_user request cancelled by new user input."
    )
    return {
        "answers": {},
        "annotations": {},
        "metadata": {
            "status": "skipped",
            "skipped": True,
            "reason": reason,
        },
        "error": error,
    }


def _valid_pending_ask_state(
    state: ConversationState | None,
    run_id: str,
) -> bool:
    if state is None or state.payload.get("run_id") != run_id:
        return False
    expires_at = state.expires_at
    return bool(
        state.payload.get("ask_id")
        and state.payload.get("tool_call_id")
        and expires_at is not None
        and expires_at.utcoffset() is not None
    )


def _invocation_target(value: object) -> InvocationTarget | None:
    if not isinstance(value, dict):
        return None
    return InvocationTarget.model_validate(value)


def _has_selected_scene(input_snapshot: dict[str, Any]) -> bool:
    target = _invocation_target(input_snapshot.get("invocation_target"))
    return target is not None and target.kind is InvocationTargetKind.SCENE
