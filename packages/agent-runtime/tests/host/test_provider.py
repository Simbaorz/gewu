"""Subscriber providers hand complete authorized turns to the neutral Runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from gewu_agent_runtime import (
    AgentRuntime,
    PreparedAgentTurn,
    PrincipalRef,
    PrincipalType,
    TurnBindings,
    TurnRequest,
)
from gewu_agent_runtime.builtins import SkillCatalog, SkillDocument, skill
from gewu_agent_runtime.engine import AssistantFinal
from gewu_agent_runtime.host import prepare_and_start_turn
from gewu_agent_runtime.llm import ModelStreamChunk, ScriptedChatModel, ToolCall
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.tools import ToolSet
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)
from gewu_core import ApplicationError, ApplicationErrorKind


@dataclass(frozen=True)
class SubscriberCommand:
    content: str


class SubscriberProvider:
    def __init__(self, *, forged_invoker: PrincipalRef | None = None) -> None:
        self.forged_invoker = forged_invoker

    async def prepare_turn(
        self,
        principal: PrincipalRef,
        command: SubscriberCommand,
    ) -> PreparedAgentTurn:
        workspace = WorkspaceSession(
            (
                WorkspaceMount(
                    mount_id="root",
                    mount_path="/",
                    access_mode=AccessMode.READ_WRITE,
                    backend=InMemoryWorkspaceBackend(),
                ),
            )
        )
        return PreparedAgentTurn(
            request=TurnRequest(
                invoker=self.forged_invoker or principal,
                content=command.content,
            ),
            bindings=TurnBindings(
                model=ScriptedChatModel([[ModelStreamChunk(content_delta="subscriber result")]]),
                workspace=workspace,
            ),
        )


class DepartmentFirstSkillCatalog(SkillCatalog):
    """Example subscriber policy resolved before crossing the Runtime boundary."""

    def __init__(self) -> None:
        company = SkillDocument(
            asset_key="company-review",
            name="review",
            description="Company review policy",
            content="Follow the company review policy.",
        )
        department = SkillDocument(
            asset_key="department-review",
            name="review",
            description="Department review policy",
            content="Follow the department review policy.",
        )
        candidates = {"company": company, "department": department}
        self._winner = candidates["department"]

    async def list_skills(self, *, limit: int | None = None) -> Sequence[SkillDocument]:
        values = (self._winner.model_copy(update={"content": ""}),)
        return values[:limit] if limit is not None else values

    async def get_skill(self, name: str) -> SkillDocument | None:
        return self._winner if name == self._winner.name else None

    async def get_skill_by_asset_key(self, asset_key: str) -> SkillDocument | None:
        return self._winner if asset_key == self._winner.asset_key else None


class CustomHierarchySubscriberProvider:
    async def prepare_turn(
        self,
        principal: PrincipalRef,
        command: SubscriberCommand,
    ) -> PreparedAgentTurn:
        workspace = WorkspaceSession(
            (
                WorkspaceMount(
                    mount_id="knowledge",
                    mount_path="/knowledge",
                    access_mode=AccessMode.READ_WRITE,
                    backend=InMemoryWorkspaceBackend(),
                ),
            ),
            default_root="/knowledge",
        )
        model = ScriptedChatModel(
            [
                [
                    ModelStreamChunk(
                        tool_calls=(
                            ToolCall(
                                tool_call_id="skill-1",
                                name="skill",
                                arguments={"name": "review", "args": "change"},
                            ),
                        )
                    )
                ],
                [ModelStreamChunk(content_delta="subscriber result")],
            ]
        )
        return PreparedAgentTurn(
            request=TurnRequest(invoker=principal, content=command.content),
            bindings=TurnBindings(
                model=model,
                workspace=workspace,
                tool_set=ToolSet((skill,)),
                skill_catalog=DepartmentFirstSkillCatalog(),
            ),
        )


async def test_new_subscriber_provider_runs_without_runtime_changes() -> None:
    principal = _principal("subscriber-a", "external-user-1")
    store = InMemoryRuntimeStore()

    session = await prepare_and_start_turn(
        AgentRuntime(store=store),
        SubscriberProvider(),
        principal=principal,
        command=SubscriberCommand(content="analyze"),
    )
    events = [event async for event in session.stream()]

    assert isinstance(events[-1], AssistantFinal)
    assert events[-1].content == "subscriber result"
    assert session.run.invoker == principal


async def test_subscriber_owns_skill_hierarchy_and_runtime_receives_unique_winner() -> None:
    principal = _principal("subscriber-a", "external-user-1")
    store = InMemoryRuntimeStore()

    session = await prepare_and_start_turn(
        AgentRuntime(store=store),
        CustomHierarchySubscriberProvider(),
        principal=principal,
        command=SubscriberCommand(content="review this change"),
    )
    events = [event async for event in session.stream()]

    assert isinstance(events[-1], AssistantFinal)
    messages = await store.list_messages(session.conversation_id)
    skill_content = next(
        message for message in messages if message.payload.get("attachment_type") == "skill_content"
    )
    assert skill_content.payload["skill_name"] == "review"
    assert "department review policy" in skill_content.content
    assert "company review policy" not in skill_content.content


async def test_host_rejects_provider_that_replaces_authenticated_invoker() -> None:
    principal = _principal("subscriber-a", "external-user-1")
    forged = _principal("subscriber-b", "forged-user")
    store = InMemoryRuntimeStore()

    with pytest.raises(ApplicationError) as error:
        await prepare_and_start_turn(
            AgentRuntime(store=store),
            SubscriberProvider(forged_invoker=forged),
            principal=principal,
            command=SubscriberCommand(content="must not run"),
        )

    assert error.value.kind is ApplicationErrorKind.FORBIDDEN
    assert (await store.list_conversations(principal, limit=10)).items == ()


def _principal(subscriber_id: str, principal_id: str) -> PrincipalRef:
    return PrincipalRef(
        subscriber_id=subscriber_id,
        principal_id=principal_id,
        principal_type=PrincipalType.USER,
    )
