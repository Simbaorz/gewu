"""Exact Ask model contract and host callback behavior."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from gewu_agent_runtime.builtins import ask_user, ask_user_tool
from gewu_agent_runtime.tools import AskResponse, AskSuspension, ToolContext
from gewu_agent_runtime.workspace import WorkspaceSession


def test_ask_model_contract_exactly_matches_subscriber() -> None:
    payload = {
        "description": ask_user.description,
        "input_schema": ask_user.input_schema,
        "category": ask_user.category,
        "writes": ask_user.writes_workspace,
        "allow_parallel": ask_user.allow_parallel,
        "retry": ask_user.retry_on_failure,
        "max_retries": ask_user.max_retries,
    }
    digest = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    assert digest == "3a27f6ebcc22ab3dd5fb843e1c6398aab78cf86da2d24c9991be36c619028dd7"


async def test_ask_tolerantly_parses_bounded_questions_and_returns_host_answer(
    workspace: WorkspaceSession,
) -> None:
    observed: list[tuple[dict[str, Any], ...]] = []

    def answer(questions: tuple[dict[str, Any], ...]) -> AskResponse:
        observed.append(questions)
        return AskResponse(answers={"Choose?": "One"}, metadata={"source": "host"})

    runtime = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
        ask_callback=answer,
    )
    result = await ask_user.execute(
        {
            "questions": [
                {"question": "invalid"},
                {
                    "question": "Choose?",
                    "header": "A header longer than twelve",
                    "options": [
                        {"label": "One", "preview": "first"},
                        {"label": "Two", "description": "second"},
                        {"label": "Three"},
                        {"label": "Four"},
                        {"label": "Ignored"},
                    ],
                    "multiSelect": False,
                },
            ]
        },
        runtime,
    )

    assert result.output_payload() == {
        "answers": {"Choose?": "One"},
        "annotations": {},
        "metadata": {"source": "host"},
        "error": "",
    }
    assert len(observed) == 1
    assert observed[0][0]["header"] == "A header lon"
    assert [option["label"] for option in observed[0][0]["options"]] == [
        "One",
        "Two",
        "Three",
        "Four",
    ]
    assert observed[0][0]["options"][0]["preview"] == "first"


async def test_ask_preserves_missing_callback_and_suspension_semantics(
    workspace: WorkspaceSession,
) -> None:
    questions = [
        {
            "question": "Continue?",
            "header": "Next",
            "options": [{"label": "Yes"}, {"label": "No"}],
        }
    ]
    unbound = ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
    )

    missing = await ask_user.execute({"questions": questions}, unbound)

    def suspend(values: tuple[dict[str, Any], ...]) -> AskSuspension:
        return AskSuspension(
            ask_id="ask-1",
            questions=values,
            timeout_seconds=60,
        )

    suspended = await ask_user.execute(
        {"questions": questions},
        unbound.model_copy(update={"ask_callback": suspend}),
    )

    assert missing.is_error is False
    assert missing.output_payload()["error"] == "No ask callback configured"
    assert suspended.is_error is False
    assert isinstance(suspended.output, AskSuspension)
    assert suspended.output.ask_id == "ask-1"


async def test_ask_callback_failure_returns_exception_body_like_subscriber(
    workspace: WorkspaceSession,
) -> None:
    def fail(_questions: tuple[dict[str, Any], ...]) -> AskResponse:
        raise RuntimeError("ask-callback-private-secret")

    result = await ask_user.execute(
        {
            "questions": [
                {
                    "question": "Continue?",
                    "header": "Next",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
            ]
        },
        ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
            ask_callback=fail,
        ),
    )

    assert result.is_error is True
    assert result.output_payload()["error"] == ("Failed to ask user: ask-callback-private-secret")


async def test_ask_preserves_legacy_non_positive_timeout(
    workspace: WorkspaceSession,
) -> None:
    bound = ask_user_tool(timeout_seconds=0)
    result = await bound.execute(
        {
            "questions": [
                {
                    "question": "Continue?",
                    "header": "Next",
                    "options": [{"label": "Yes"}, {"label": "No"}],
                }
            ]
        },
        ToolContext(
            conversation_id="conversation-1",
            run_id="run-1",
            workspace=workspace,
        ),
    )

    assert isinstance(result.output, AskSuspension)
    assert result.output.timeout_seconds == 0
