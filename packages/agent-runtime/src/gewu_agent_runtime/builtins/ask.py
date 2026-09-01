"""Host-neutral interactive clarification tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.tools import (
    AskSuspension,
    PersistencePolicy,
    Tool,
    ToolContext,
    ToolResult,
    ToolResultMode,
    tool,
)
from gewu_core.ids import new_id


class AskOption(BaseModel):
    """One option passed to a host Ask callback."""

    model_config = ConfigDict(frozen=True)

    label: str
    description: str = ""
    preview: str | None = None


class AskQuestion(BaseModel):
    """One validated multiple-choice question."""

    model_config = ConfigDict(frozen=True)

    question: str
    header: str
    options: list[AskOption]
    multiSelect: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return the model-visible question representation."""

        return self.model_dump(mode="json")


class AskOutput(BaseModel):
    """Immediate result produced by the reference Ask tool."""

    answers: dict[str, str | list[str]] = Field(default_factory=dict)
    annotations: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str = ""


@tool(
    name="ask_user",
    category="interaction",
    retry_on_failure=False,
    persistence_policy=PersistencePolicy.NONE,
)
def ask_user(
    questions: list[dict[str, Any]],
    *,
    runtime: ToolContext,  # noqa
) -> ToolResult:
    """Ask the user multiple-choice questions.

    Use this tool when you need to ask the user questions during execution. This
    allows you to gather preferences or requirements, clarify ambiguous
    instructions, get decisions on implementation choices, or offer choices
    about what direction to take.

    Usage notes:
    - Prefer one question; never ask more than four.
    - Each question must include a short header, a clear question, and at least
      two options.
    - Users will always be able to select "Other" to provide custom text input.
    - Use `multiSelect: true` only when multiple answers are valid.
    - If you recommend a specific option, make that the first option and add
      `(Recommended)` at the end of the label.

    Preview feature:
    Use the optional `preview` field on options when presenting concrete
    artifacts that users need to visually compare, such as code snippets,
    diagrams, configuration examples, or ASCII mockups. Preview content is
    rendered as markdown in a monospace box. Do not use previews for simple
    preference questions where labels and descriptions are sufficient.

    Args:
        questions: List of 1-4 question objects. Each question must have
            `question`, `header`, and `options`. `header` is a short label, max
            12 characters. `options` is a list of 2-4 options, each with
            `label`, optional `description`, and optional `preview`.
            `multiSelect` controls whether multiple selections are allowed.

    Returns:
        dict with:
            - answers: Mapping question text to selected label or labels.
            - annotations: Empty dict, reserved for future use.
            - metadata: Optional metadata dict.
            - error: Error message if failed, empty on success.

        If `multiSelect` is true, the answer value is a list of labels.
    """

    parsed_questions = _parse_questions(questions)
    if not parsed_questions:
        return ToolResult(
            output=AskOutput(error="No valid questions provided"),
            is_error=True,
        )
    callback = runtime.ask_callback
    if callback is None:
        return ToolResult(
            output=AskOutput(error="No ask callback configured"),
            is_error=False,
        )
    serialized = tuple(question.to_dict() for question in parsed_questions)
    try:
        result = callback(serialized)
        if isinstance(result, AskSuspension):
            return ToolResult(
                mode=ToolResultMode.SUSPENDED,
                output=result,
                is_error=False,
            )
        return ToolResult(
            output=AskOutput(
                answers=result.answers,
                annotations={},
                metadata=result.metadata,
            ),
            is_error=False,
        )
    except Exception as exc:  # noqa: BLE001
        return ToolResult(
            output=AskOutput(error=f"Failed to ask user: {exc}"),
            is_error=True,
        )


def ask_user_tool(*, timeout_seconds: int = 300) -> Tool:
    """Bind the exact Ask contract to a Runtime-managed suspension callback."""

    def suspend(questions: tuple[dict[str, Any], ...]) -> AskSuspension:
        return AskSuspension(
            ask_id=new_id(),
            questions=questions,
            timeout_seconds=timeout_seconds,
        )

    async def execute(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        bound = context.model_copy(update={"ask_callback": suspend})
        return await ask_user.execute(arguments, bound)

    return ask_user.model_copy(update={"function": execute})


def _parse_questions(questions: list[dict[str, Any]]) -> list[AskQuestion]:
    """Apply tolerant parsing and bounded question/option counts."""

    parsed: list[AskQuestion] = []
    for raw_question in questions[:4]:
        if not isinstance(raw_question, dict):
            continue
        if not {"question", "header", "options"}.issubset(raw_question):
            continue
        raw_options = raw_question.get("options")
        if not isinstance(raw_options, list):
            continue
        options = [
            AskOption(
                label=str(option["label"]),
                description=str(option.get("description", "")),
                preview=option.get("preview"),
            )
            for option in raw_options[:4]
            if isinstance(option, dict) and "label" in option
        ]
        if len(options) < 2:
            continue
        parsed.append(
            AskQuestion(
                question=str(raw_question["question"]),
                header=str(raw_question["header"])[:12],
                options=options,
                multiSelect=bool(raw_question.get("multiSelect", False)),
            )
        )
    return parsed
