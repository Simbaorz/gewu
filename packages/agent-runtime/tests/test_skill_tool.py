"""Business-neutral Skill invocation with exact Subscriber Tool behavior."""

from __future__ import annotations

import hashlib
import json

from gewu_agent_runtime.builtins import SkillDocument, skill, skill_tool
from gewu_agent_runtime.domain import MessageKind
from gewu_agent_runtime.identity import PrincipalRef
from gewu_agent_runtime.llm import ModelStreamChunk, ScriptedChatModel, ToolCall
from gewu_agent_runtime.persistence import InMemoryRuntimeStore
from gewu_agent_runtime.runtime import AgentRuntime, TurnBindings, TurnRequest
from gewu_agent_runtime.tools import ToolContext, ToolSet
from gewu_agent_runtime.workspace import WorkspaceSession


class MemorySkillCatalog:
    def __init__(self, *skills: SkillDocument) -> None:
        self._skills = {item.name: item for item in skills}

    async def get_skill(self, name: str) -> SkillDocument | None:
        return self._skills.get(name)


def test_skill_model_contract_exactly_matches_subscriber() -> None:
    payload = {
        "description": skill.description,
        "input_schema": skill.input_schema,
        "category": skill.category,
        "writes": skill.writes_workspace,
        "allow_parallel": skill.allow_parallel,
        "retry": skill.retry_on_failure,
        "max_retries": skill.max_retries,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert digest == "ba5a0d00aa3b4353a065ed927c4cdce20b1b7000b7abe19ebb8435894e91eff0"


async def test_skill_expands_arguments_and_host_runtime_variables(
    workspace: WorkspaceSession,
) -> None:
    catalog = MemorySkillCatalog(
        SkillDocument(
            name="review",
            description="Review code",
            content="Review ${target} in ${HOST_SESSION}",
            base_path="/workspace/private/.skills/review",
            arguments=["target"],
        )
    )
    tool_value = skill_tool(
        catalog,
        runtime_variables=lambda runtime, _skill: {"HOST_SESSION": runtime.conversation_id},
    )
    runtime = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )

    result = await tool_value.execute(
        {"name": "review", "args": "target=diff"},
        runtime,
    )

    assert result.is_error is False
    assert result.output_payload() == {
        "skill_name": "review",
        "description": "Review code",
        "invoked": True,
        "error": "",
    }
    assert result.new_messages == (
        {
            "role": "user",
            "content": (
                "Base directory for this skill: /workspace/private/.skills/review\n\n"
                "Review diff in conversation-1"
            ),
            "is_meta": True,
            "attachment_type": "skill_content",
            "skill_name": "review",
        },
    )
    assert result.extra["skill_invocation"] == {
        "name": "review",
        "args": "target=diff",
        "base_path": "/workspace/private/.skills/review",
        "source": "model",
    }


async def test_skill_rejects_missing_and_model_disabled_skill(
    workspace: WorkspaceSession,
) -> None:
    tool_value = skill_tool(
        MemorySkillCatalog(
            SkillDocument(
                name="manual",
                description="Manual only",
                disable_model_invocation=True,
            )
        )
    )
    runtime = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )

    missing = await tool_value.execute({"name": "missing"}, runtime)
    disabled = await tool_value.execute({"name": "manual"}, runtime)

    assert missing.output_payload()["error"] == "Skill 'missing' not found"
    assert disabled.output_payload()["error"] == ("Skill 'manual' cannot be invoked by the model")
    assert missing.is_error is True
    assert disabled.is_error is True


async def test_runtime_persists_skill_content_as_meta_user_message(
    principal: PrincipalRef,
    workspace: WorkspaceSession,
) -> None:
    tool_value = skill_tool(
        MemorySkillCatalog(
            SkillDocument(
                name="review",
                description="Review code",
                content="Review carefully",
                base_path="/skills/review",
            )
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
                            arguments={"name": "review"},
                        ),
                    )
                )
            ],
            [ModelStreamChunk(content_delta="done")],
        ]
    )
    store = InMemoryRuntimeStore()
    session = await AgentRuntime(store=store).start_turn(
        TurnRequest(invoker=principal, content="review this"),
        TurnBindings(
            model=model,
            workspace=workspace,
            tool_set=ToolSet((tool_value,)),
        ),
    )

    _ = [event async for event in session.stream()]
    messages = await store.list_messages(session.conversation_id)
    meta = next(item for item in messages if item.kind is MessageKind.META)
    tool_result = next(item for item in messages if item.kind is MessageKind.TOOL_RESULT)

    assert meta.content == "Base directory for this skill: /skills/review\n\nReview carefully"
    assert meta.payload == {
        "is_meta": True,
        "attachment_type": "skill_content",
        "skill_name": "review",
    }
    assert tool_result.payload["extra"]["skill_invocation"]["name"] == "review"
    assert model.requests[1][0][-1].content == meta.content
