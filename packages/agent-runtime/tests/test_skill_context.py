"""Subscriber-compatible Skill and Scene context over neutral authorized catalogs."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from gewu_agent_runtime.builtins import (
    SceneCatalog,
    SceneDocument,
    SkillCapacityExceededError,
    SkillCatalog,
    SkillCatalogContractError,
    SkillDocument,
    SkillRegistry,
    prepare_skill_turn,
    reconcile_skill_state,
    skill,
)
from gewu_agent_runtime.domain import (
    Conversation,
    ConversationMessage,
    ConversationState,
    MessageKind,
)
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.invocation import InvocationTarget, InvocationTargetKind
from gewu_agent_runtime.llm import MessageRole, ModelStreamChunk, ScriptedChatModel, ToolCall
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.runtime import AgentRuntime, TurnBindings, TurnRequest, TurnSession
from gewu_agent_runtime.tools import ToolContext, ToolSet
from gewu_agent_runtime.workspace import WorkspaceSession


class MemorySkillCatalog(SkillCatalog):
    def __init__(self, *skills: SkillDocument) -> None:
        self.skills = tuple(skills)
        self.requested_limits: list[int | None] = []

    async def list_skills(self, *, limit: int | None = None) -> Sequence[SkillDocument]:
        self.requested_limits.append(limit)
        values = self.skills[:limit] if limit is not None else self.skills
        return tuple(value.model_copy(update={"content": ""}) for value in values)

    async def get_skill(self, name: str) -> SkillDocument | None:
        return next((value for value in self.skills if value.name == name), None)

    async def get_skill_by_asset_key(self, asset_key: str) -> SkillDocument | None:
        return next(
            (value for value in self.skills if (value.asset_key or value.name) == asset_key),
            None,
        )


class MemorySceneCatalog(SceneCatalog):
    def __init__(self, *scenes: SceneDocument) -> None:
        self.scenes = {value.asset_key: value for value in scenes}

    async def get_scene(self, asset_key: str) -> SceneDocument | None:
        return self.scenes.get(asset_key)


async def _consume(session: TurnSession) -> list[object]:
    return [event async for event in session.stream()]


async def test_runtime_lists_skills_once_and_tracks_model_invocation(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="skill_review",
            name="review",
            description="Review code",
            when_to_use="Use for review",
            content="Review ${target}",
            base_path="/workspace/private/.skills/review",
            arguments=["target"],
        )
    )
    model = ScriptedChatModel(
        [
            [
                ModelStreamChunk(
                    tool_calls=(
                        ToolCall(
                            tool_call_id="skill-1",
                            name="skill",
                            arguments={"name": "review", "args": "target=diff"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    first = await runtime.start_turn(
        TurnRequest(invoker=principal, content="use skill"),
        TurnBindings(
            model=model,
            workspace=workspace,
            tool_set=ToolSet((skill,)),
            skill_catalog=catalog,
        ),
    )

    await _consume(first)
    messages = await store.list_messages(first.conversation_id)
    state = await store.get_state(first.conversation_id, "skill")

    meta = [value for value in messages if value.kind is MessageKind.META]
    assert [value.payload["attachment_type"] for value in meta] == [
        "skill_listing",
        "skill_content",
    ]
    assert "- review: Review code - Use for review" in meta[0].content
    assert state is not None
    assert state.payload == {
        "sent_skill_names": ["review"],
        "invoked_skills": {
            "review": {
                "name": "review",
                "args": "target=diff",
                "base_path": "/workspace/private/.skills/review",
                "source": "model",
            }
        },
    }
    assert catalog.requested_limits == [257]

    second_model = ScriptedChatModel([[ModelStreamChunk(content_delta="again")]])
    second = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=first.conversation_id,
            content="continue",
        ),
        TurnBindings(
            model=second_model,
            workspace=workspace,
            tool_set=ToolSet((skill,)),
            skill_catalog=catalog,
        ),
    )
    await _consume(second)
    messages = await store.list_messages(first.conversation_id)
    listings = [
        value
        for value in messages
        if value.kind is MessageKind.META
        and value.payload.get("attachment_type") == "skill_listing"
    ]
    assert len(listings) == 1


async def test_runtime_consumes_user_slash_skill_but_preserves_unknown_slash(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="skill_review",
            name="review",
            description="Review code",
            content="Review $ARGUMENTS",
            base_path="/workspace/private/.skills/review",
        )
    )
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    session = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            content="/review diff",
            invocation_target=InvocationTarget(
                kind=InvocationTargetKind.SKILL,
                name="review",
                arguments="diff",
            ),
        ),
        TurnBindings(model=model, workspace=workspace, skill_catalog=catalog),
    )

    await _consume(session)
    messages = await store.list_messages(session.conversation_id)
    assert messages[0].content == "/review diff"
    assert messages[0].payload["llm_ignore"] is True
    assert messages[0].payload["skill_command"] == {
        "asset_key": "skill_review",
        "skill_name": "review",
        "command_name": "review",
        "args": "diff",
    }
    assert [value.payload.get("attachment_type") for value in messages[1:4]] == [
        "skill_command_metadata",
        "skill_content",
        "skill_listing",
    ]
    observed = "\n\n".join(
        value.content for value in model.requests[0][0] if value.role.value == "user"
    )
    assert "/review diff" not in observed
    assert "<command-name>/review</command-name>" in observed
    assert "Review diff" in observed

    unknown_model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    unknown = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=session.conversation_id,
            content="/unknown args",
            invocation_target=InvocationTarget(
                kind=InvocationTargetKind.SKILL,
                name="unknown",
                arguments="args",
            ),
        ),
        TurnBindings(model=unknown_model, workspace=workspace, skill_catalog=catalog),
    )
    await _consume(unknown)
    observed = "\n\n".join(
        value.content for value in unknown_model.requests[0][0] if value.role.value == "user"
    )
    assert "/unknown args" in observed


async def test_structured_skill_target_falls_back_to_visible_name_winner(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="winner-review",
            name="review",
            description="Visible review",
            content="Winner $ARGUMENTS",
            base_path="/workspace/skills/review",
        )
    )
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(
            invoker=principal,
            content="diff",
            invocation_target=InvocationTarget(
                kind=InvocationTargetKind.SKILL,
                resource_id="shadowed-review",
                name="review",
                arguments="diff",
            ),
        ),
        TurnBindings(model=model, workspace=workspace, skill_catalog=catalog),
    )

    await _consume(session)

    messages = await store.list_messages(session.conversation_id)
    assert messages[0].payload["skill_command"]["asset_key"] == "winner-review"
    skill_content = next(
        value for value in messages if value.payload.get("attachment_type") == "skill_content"
    )
    assert "Winner diff" in skill_content.content


async def test_runtime_uses_host_scene_path_and_bound_skill_state(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    runtime = AgentRuntime(store=store)
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="skill_fault",
            name="fault_workflow",
            description="Troubleshoot faults",
            content="Follow the fault workflow",
            base_path="/workspace/company/acme/.skills/fault",
        )
    )
    scene_catalog = MemorySceneCatalog(
        SceneDocument(
            asset_key="scene_fault",
            name="Fault Scene",
            description="  Fault   troubleshooting  ",
            workspace_path="/workspace/company/acme/department/ops/.scenes/fault",
            required_skill_asset_key="skill_fault",
        )
    )
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    session = await runtime.start_turn(
        TurnRequest(
            invoker=principal,
            content="help",
            invocation_target=InvocationTarget(
                kind=InvocationTargetKind.SCENE,
                resource_id="scene_fault",
                name="Fault Scene",
            ),
        ),
        TurnBindings(
            model=model,
            workspace=workspace,
            skill_catalog=catalog,
            scene_catalog=scene_catalog,
        ),
    )

    await _consume(session)
    messages = await store.list_messages(session.conversation_id)
    scene = next(
        value for value in messages if value.payload.get("attachment_type") == "scene_reminder"
    )
    assert scene.content == (
        "<system-reminder>\n"
        "User-selected Scene:\n"
        "- Scene name: Fault Scene\n"
        "- Scene entry path: /workspace/company/acme/department/ops/.scenes/fault\n"
        "- Bound skill: fault_workflow\n"
        "</system-reminder>"
    )
    assert "tenant" not in scene.content
    assert messages[0].payload.get("llm_ignore") is None
    system_message = model.requests[0][0][0]
    assert system_message.role.value == "system"
    assert "## Scenes" in system_message.content
    assert "load that Skill" in system_message.content
    assert "Do not discover or switch Scenes autonomously" in system_message.content


async def test_scene_reminder_detects_already_invoked_bound_skill(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    store = InMemoryRuntimeStore()
    conversation_id = "conversation-1"
    await store.create_conversation(Conversation(conversation_id=conversation_id, owner=principal))
    await store.save_state(
        ConversationState(
            conversation_id=conversation_id,
            kind="skill",
            revision=1,
            payload={
                "invoked_skills": {
                    "fault_workflow": {
                        "name": "fault_workflow",
                        "source": "model",
                    }
                }
            },
        ),
        expected_revision=0,
    )
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="skill_fault",
            name="fault_workflow",
            description="Troubleshoot faults",
        )
    )
    scene_catalog = MemorySceneCatalog(
        SceneDocument(
            asset_key="scene_fault",
            name="Fault Scene",
            workspace_path="/workspace/department/ops/.scenes/fault",
            required_skill_asset_key="skill_fault",
        )
    )
    model = ScriptedChatModel([[ModelStreamChunk(content_delta="done")]])
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(
            invoker=principal,
            conversation_id=conversation_id,
            content="help",
            invocation_target=InvocationTarget(
                kind=InvocationTargetKind.SCENE,
                resource_id="scene_fault",
                name="Fault Scene",
            ),
        ),
        TurnBindings(
            model=model,
            workspace=workspace,
            skill_catalog=catalog,
            scene_catalog=scene_catalog,
        ),
    )

    await _consume(session)
    messages = await store.list_messages(conversation_id)
    scene = next(
        value for value in messages if value.payload.get("attachment_type") == "scene_reminder"
    )
    assert "- Bound skill: fault_workflow" in scene.content
    assert "Required workflow:" not in scene.content


async def test_skill_turn_state_does_not_alias_persisted_nested_payload(
    workspace: WorkspaceSession,
) -> None:
    persisted = {
        "invoked_skills": {
            "existing": {
                "name": "existing",
                "source": "model",
            }
        }
    }

    preparation = await prepare_skill_turn(
        metadata={},
        invocation_target=None,
        tool_context=ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
        ),
        skill_catalog=None,
        scene_catalog=None,
        persisted_state=persisted,
    )
    invoked = preparation.state["invoked_skills"]
    invoked["new"] = {"name": "new", "source": "model"}

    assert "new" not in persisted["invoked_skills"]


async def test_skill_listing_matches_subscriber_shape_and_exact_utf8_budget(
    workspace: WorkspaceSession,
) -> None:
    catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="skill-review",
            name="review",
            description="检查代码",
            when_to_use="用于审查",
        )
    )
    context = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )
    expected = (
        "<system-reminder>\n"
        "The following skills are available for use with the skill tool:\n\n"
        "- review: 检查代码 - 用于审查\n"
        "</system-reminder>"
    )
    exact_bytes = len(expected.encode("utf-8"))

    preparation = await prepare_skill_turn(
        metadata={},
        invocation_target=None,
        tool_context=context,
        skill_catalog=catalog,
        scene_catalog=None,
        persisted_state=None,
        max_skill_listing_bytes=exact_bytes,
    )

    assert [message["content"] for message in preparation.messages] == [expected]
    with pytest.raises(
        SkillCapacityExceededError,
        match=f"Skill listing exceeds configured limit of {exact_bytes - 1} UTF-8 bytes",
    ):
        await prepare_skill_turn(
            metadata={},
            invocation_target=None,
            tool_context=context,
            skill_catalog=catalog,
            scene_catalog=None,
            persisted_state=None,
            max_skill_listing_bytes=exact_bytes - 1,
        )


async def test_scene_reminder_omits_unbound_skill(
    workspace: WorkspaceSession,
) -> None:
    scene_catalog = MemorySceneCatalog(
        SceneDocument(
            asset_key="scene-private",
            name="Private Scene",
            workspace_path="/workspace/private/.scenes/Private Scene",
        )
    )

    preparation = await prepare_skill_turn(
        metadata={},
        invocation_target=InvocationTarget(
            kind=InvocationTargetKind.SCENE,
            resource_id="scene-private",
            name="Private Scene",
        ),
        tool_context=ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
        ),
        skill_catalog=MemorySkillCatalog(),
        scene_catalog=scene_catalog,
        persisted_state=None,
    )

    reminder = preparation.messages[-1]
    assert reminder["attachment_type"] == "scene_reminder"
    assert reminder["content"] == (
        "<system-reminder>\n"
        "User-selected Scene:\n"
        "- Scene name: Private Scene\n"
        "- Scene entry path: /workspace/private/.scenes/Private Scene\n"
        "</system-reminder>"
    )


async def test_scene_shadowed_required_skill_key_remains_unbound(
    workspace: WorkspaceSession,
) -> None:
    skill_catalog = MemorySkillCatalog(
        SkillDocument(
            asset_key="winner-review",
            name="review",
            description="Visible review",
        )
    )
    scene_catalog = MemorySceneCatalog(
        SceneDocument(
            asset_key="scene-review",
            name="Review Scene",
            workspace_path="/workspace/scenes/review",
            required_skill_asset_key="shadowed-review",
        )
    )

    preparation = await prepare_skill_turn(
        metadata={},
        invocation_target=InvocationTarget(
            kind=InvocationTargetKind.SCENE,
            resource_id="scene-review",
            name="Review Scene",
        ),
        tool_context=ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
        ),
        skill_catalog=skill_catalog,
        scene_catalog=scene_catalog,
        persisted_state=None,
    )

    reminder = preparation.messages[-1]["content"]
    assert "Bound skill:" not in reminder


async def test_scene_reminder_does_not_expand_description(
    workspace: WorkspaceSession,
) -> None:
    long_description = "  " + " \n ".join(["Fault"] * 80) + "  "
    scene_catalog = MemorySceneCatalog(
        SceneDocument(
            asset_key="scene-fault",
            name="Fault Scene",
            description=long_description,
            workspace_path="/workspace/shared/tenant/.scenes/Fault Scene",
        )
    )

    preparation = await prepare_skill_turn(
        metadata={},
        invocation_target=InvocationTarget(
            kind=InvocationTargetKind.SCENE,
            resource_id="scene-fault",
            name="Fault Scene",
        ),
        tool_context=ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
        ),
        skill_catalog=MemorySkillCatalog(),
        scene_catalog=scene_catalog,
        persisted_state=None,
    )

    assert "Scene description:" not in preparation.messages[-1]["content"]
    assert long_description not in preparation.messages[-1]["content"]


async def test_skill_registry_detects_capacity_without_silent_truncation() -> None:
    catalog = MemorySkillCatalog(
        *(SkillDocument(name=f"skill_{index}", description="Skill") for index in range(3))
    )

    with pytest.raises(
        SkillCapacityExceededError,
        match="visible Skill count exceeds configured limit of 2",
    ):
        await SkillRegistry.load(catalog, max_visible_skills=2)

    assert catalog.requested_limits == [3]


async def test_skill_registry_rejects_duplicate_subscriber_names() -> None:
    catalog = MemorySkillCatalog(
        SkillDocument(asset_key="private-review", name="review", description="Private"),
        SkillDocument(asset_key="shared-review", name="review", description="Shared"),
    )

    with pytest.raises(
        SkillCatalogContractError,
        match="Authorized Skill catalog returned duplicate name: review",
    ):
        await SkillRegistry.load(catalog, max_visible_skills=10)


async def test_skill_registry_rejects_duplicate_subscriber_asset_keys() -> None:
    catalog = MemorySkillCatalog(
        SkillDocument(asset_key="same-key", name="review", description="Review"),
        SkillDocument(asset_key="same-key", name="search", description="Search"),
    )

    with pytest.raises(
        SkillCatalogContractError,
        match="Authorized Skill catalog returned duplicate asset key: same-key",
    ):
        await SkillRegistry.load(catalog, max_visible_skills=10)


async def test_skill_registry_accepts_an_independent_subscriber_resolved_catalog() -> None:
    candidates = (
        (
            "company",
            SkillDocument(asset_key="company-review", name="review", description="Company"),
        ),
        (
            "department",
            SkillDocument(
                asset_key="department-review",
                name="review",
                description="Department",
            ),
        ),
        ("company", SkillDocument(asset_key="company-search", name="search", description="Search")),
    )
    priority = {"department": 0, "company": 1}
    winners: dict[str, SkillDocument] = {}
    for _, descriptor in sorted(candidates, key=lambda value: priority[value[0]]):
        winners.setdefault(descriptor.name, descriptor)

    registry = await SkillRegistry.load(
        MemorySkillCatalog(*winners.values()),
        max_visible_skills=10,
    )

    assert [(skill.name, skill.asset_key) for skill in registry.all()] == [
        ("review", "department-review"),
        ("search", "company-search"),
    ]


async def test_skill_registry_rejects_body_with_different_catalog_identity() -> None:
    descriptor = SkillDocument(asset_key="winner-review", name="review", description="Review")

    class InconsistentCatalog(MemorySkillCatalog):
        async def get_skill(self, name: str) -> SkillDocument | None:
            return SkillDocument(
                asset_key="shadow-review",
                name=name,
                description="Shadow",
                content="shadow body",
            )

        async def get_skill_by_asset_key(self, asset_key: str) -> SkillDocument | None:
            return SkillDocument(
                asset_key="shadow-review",
                name="review",
                description="Shadow",
                content="shadow body",
            )

    registry = await SkillRegistry.load(
        InconsistentCatalog(descriptor),
        max_visible_skills=10,
    )

    assert await registry.get_skill("review") is None
    assert await registry.get_skill_by_asset_key("winner-review") is None


def test_skill_state_reconciles_against_retained_meta_evidence() -> None:
    state = {
        "sent_skill_names": ["review"],
        "invoked_skills": {
            "review": {"name": "review", "source": "model"},
        },
    }
    retained_listing = ConversationMessage(
        conversation_id="conversation-1",
        sequence=1,
        role=MessageRole.USER,
        kind=MessageKind.META,
        payload={
            "attachment_type": "skill_listing",
            "skill_names": ["review"],
        },
    )

    assert reconcile_skill_state((retained_listing,), state) is True
    assert state == {"sent_skill_names": ["review"]}
    assert reconcile_skill_state((), state) is True
    assert state == {}
