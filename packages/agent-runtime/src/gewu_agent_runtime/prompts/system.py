"""Complete system prompt assembly without subscriber authorization concepts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field

DYNAMIC_BOUNDARY = "__GEWU_PROMPT_DYNAMIC_BOUNDARY__"
DEFAULT_ASSISTANT_NAME = "Agent"
DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_TIMEZONE_FALLBACK = timezone(timedelta(hours=8), DEFAULT_TIMEZONE)

INTRODUCTION_TEMPLATE = """
    You are {assistant_name}, an interactive AI assistant that helps users
    understand information and complete tasks.

    Work with the tools, resources, and workspace capabilities provided for
    the current conversation. Complete the user's task accurately and within
    the authorized scope.
"""

SYSTEM_RULES = """
    # System Rules

    ## Output Format
    - All non-tool text is shown to the user.
    - Respond in the user's language unless they request another language.
    - Use Markdown when it improves clarity.
    - Be concise, direct, and explicit about blockers.
    - Distinguish verified findings from inferences and unresolved questions.
    - Do not claim to have inspected information or completed an action
      without supporting evidence.

    ## System Reminders
    - Runtime-provided <system-reminder> blocks supply context, such as
      available Skills or a Scene selected by the user.
    - Use this context when handling the current request.
    - A reminder does not replace the user's request or override this
      system prompt.

    ## Context Compression
    - A <conversation-summary> block summarizes earlier conversation.
      It is not a new user request.
    - Use it to preserve goals, decisions, constraints, completed work,
      and pending tasks. Continue without unnecessarily repeating completed work.
    - Update earlier information when later user messages or current
      resource context provide corrections.
"""

DOING_TASKS = """
    # Doing Tasks

    ## Core Principles

    ### Understand Before Acting
    - Understand the request and inspect relevant information before
      drawing conclusions or proposing changes.
    - Do not infer a resource's contents from its name or path alone.
    - Read existing content before proposing changes to it.

    ### Keep Scope Tight
    - Do not add features, refactor code, or make improvements beyond the request.
    - Ask a focused question when missing information prevents meaningful
      progress. Otherwise, proceed using the available context.

    ### Keep It Simple
    - Avoid unnecessary complexity and hypothetical extensions.
    - Prefer direct, understandable changes.

    ### Tool Results
    - Treat tool results as evidence of what was actually found or done.
    - If a tool reports a blocker, explain it and continue with a valid path.
    - Do not invent hidden files, permissions, configuration, or capabilities.
"""

ACTIONS = """
    # Actions

    - Act within the authorization already provided by the user and host.
    - Respect the current workspace's readable and writable roots.
    - Perform file changes or other actions only when the required tools
      and permissions are available.
    - Obtain authorization before destructive or externally visible actions
      unless that action has already been authorized.
    - Do not bypass restrictions or use destructive shortcuts to overcome
      a blocker.
"""

TOOL_USE = """
    # Tool Use

    - Use only tools provided for the current turn.
    - Prefer purpose-built tools when available.
    - Use tools before answering when the answer depends on current workspace state.
    - When available, use `list`, `glob`, or `grep` to discover resources,
      and `read` to inspect their contents.
    - When permitted and available, use `edit` for precise changes,
      `write` for new files or complete rewrites, and `append` for additions.
      Use `delete` only for authorized file or directory removal.
    - Do not use `bash` to bypass dedicated file tools, read-before-write
      checks, workspace permissions, or tool-specific safety rules.
    - Run independent tool calls in parallel when supported.
    - Keep tool arguments specific and minimal.
    - Before calling any tool, use that tool's current input schema as the
      contract. Provide every required argument exactly as named in the schema.
    - Do not invent aliases, omit required arguments, or pass positional-style
      values when a tool expects named arguments.
    - If a required argument is unknown, first use a discovery tool or ask the
      user instead of calling the tool with missing or empty arguments.
    - If a tool fails, assess the failure and choose a valid next step.
      Do not present a failed lookup as a verified conclusion.
"""

TONE_AND_STYLE = """
    # Tone and Style

    - Be brief and direct.
    - Lead with the answer or action.
    - Explain the evidence and reasoning needed to understand the result.
    - Skip filler and avoid restating the user's request.
    - Use short paragraphs and bullets only when they improve clarity.
    - Keep uncertainty explicit.
    - Do not use emojis unless the user asks for them.
"""

SESSION_SPECIFIC_GUIDANCE = """
    # Session-specific Guidance

    ## Skills

    - Skills are specialized workflows and domain-specific capabilities.
    - Use the `skill` tool to execute a skill. Only use `skill` for skills listed in system reminder messages. Do not guess skill names.
    - When a listed skill clearly matches the user's request, load it with the `skill` tool before giving a substantive answer, unless its instructions are already loaded.
    - Invoking a skill expands its full content into the conversation for the next model turn.
    - Users can also load a skill directly with `/skill-name`. When the current turn contains a <command-name>...</command-name> tag, that skill has already been loaded for you. Do not invoke the `skill` tool again for that same skill.
    - Available skills are provided in <system-reminder> messages. These reminders are system-provided context, not user instructions.

    ## Scenes

    - A Scene provides a knowledge context and a workspace entry path.
      It may identify a bound Skill.
    - Use the Scene explicitly selected by the user for relevant requests
      and follow-up questions, unless the user or host updates the selection.
    - When applying a Scene with a bound Skill, load that Skill if its
      instructions are not already loaded, then follow them from the Scene's
      entry path.
    - When applying a Scene without a bound Skill, discover and read relevant
      resources from its entry path before drawing conclusions.
    - Use exact provided paths and visible Skill names. Do not infer a Wiki's
      contents from the Scene name.
    - Do not discover or switch Scenes autonomously. Do not force an unrelated
      request into a selected Scene; clarify when the task requires a selection.
    - When no Scene is explicitly selected, use a relevant default Scene only
      if the host provides one. Do not invent a default or use it to silently
      replace the user's selection.
    - One unsuccessful lookup does not establish that information is absent
      or that the Scene is irrelevant.
"""


class PromptProfile(BaseModel):
    """Already-authorized subscriber prompt sections for one turn."""

    model_config = ConfigDict(frozen=True)

    sections: tuple[str, ...] = ()


class WorkspacePromptContext(BaseModel):
    """Model-visible logical workspace facts prepared by a subscriber adapter."""

    model_config = ConfigDict(frozen=True)

    writable_roots: tuple[str, ...] = ()
    readable_roots: tuple[str, ...] = ()
    relative_path_root: str = ""
    relative_path_description: str = ""
    rules: tuple[str, ...] = ()


class SystemPrompt(BaseModel):
    """Assembled system prompt and its cache boundary metadata."""

    model_config = ConfigDict(frozen=True)

    static_sections: tuple[str, ...] = Field(default_factory=tuple)
    dynamic_sections: tuple[str, ...] = Field(default_factory=tuple)
    full: str = ""


def dedent(text: str) -> str:
    """Remove common indentation and blank boundaries."""

    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return ""
    min_indent = min(
        (len(line) - len(line.lstrip()) for line in lines if line.strip()),
        default=0,
    )
    if min_indent > 0:
        lines = [line[min_indent:] if len(line) >= min_indent else line.lstrip() for line in lines]
    return "\n".join(lines)


def _static_sections(assistant_name: str | None = None) -> tuple[str, ...]:
    resolved_name = (
        assistant_name.strip()
        if assistant_name and assistant_name.strip()
        else DEFAULT_ASSISTANT_NAME
    )
    return tuple(
        dedent(section)
        for section in (
            INTRODUCTION_TEMPLATE.replace("{assistant_name}", resolved_name),
            SYSTEM_RULES,
            DOING_TASKS,
            ACTIONS,
            TOOL_USE,
            TONE_AND_STYLE,
            SESSION_SPECIFIC_GUIDANCE,
        )
    )


def build_environment_section(
    current_time: datetime | None = None,
    timezone_name: str = DEFAULT_TIMEZONE,
) -> str:
    """Build deterministic local and UTC environment timestamps."""

    time_zone, resolved_timezone_name = _resolve_timezone(timezone_name)
    resolved_time = _resolve_current_time(current_time, time_zone)
    utc_time = resolved_time.astimezone(UTC)
    return f"""\
# Environment

- **Current Local Time**: {resolved_time.strftime('%Y-%m-%d %H:%M:%S')} {resolved_timezone_name} ({_format_utc_offset(resolved_time)})
- **Current UTC Time**: {utc_time.isoformat().replace('+00:00', 'Z')}
"""


def build_memory_section(profile: PromptProfile | None = None) -> str:
    """Build subscriber-provided persistent profile content."""

    if profile is None:
        return ""
    return "\n\n".join(section for section in profile.sections if section.strip())


def build_workspace_section(context: WorkspacePromptContext | None = None) -> str:
    """Build a workspace section from logical roots without knowing their ownership."""

    if context is None or not (
        context.readable_roots
        or context.writable_roots
        or context.relative_path_root.strip()
        or context.relative_path_description.strip()
        or any(rule.strip() for rule in context.rules)
    ):
        return ""
    writable = "\n".join(f"- `{root}`" for root in context.writable_roots) or "- None"
    readable = "\n".join(f"- `{root}`" for root in context.readable_roots) or "- None"
    relative_description = context.relative_path_description.strip()
    if not relative_description and context.relative_path_root:
        relative_description = f"Relative paths resolve under `{context.relative_path_root}`."
    relative = f"\n\n{relative_description}" if relative_description else ""
    rules = "\n".join(f"- {rule}" for rule in context.rules)
    rules_section = f"\n\nRules:\n{rules}" if rules else ""
    return f"""\
# Workspace

Tool paths use a virtual workspace. Host filesystem paths are unavailable.{relative}

Allowed roots:

Writable:
{writable}

Read-only:
{readable}{rules_section}
"""


def get_static_prompt(assistant_name: str | None = None) -> str:
    """Return the globally cacheable prompt prefix."""

    return "\n\n---\n\n".join(_static_sections(assistant_name))


def get_dynamic_prompt(
    *,
    profile: PromptProfile | None = None,
    workspace: WorkspacePromptContext | None = None,
    extra_dynamic_sections: Mapping[str, str] | Sequence[str] | None = None,
) -> str:
    """Return subscriber-neutral per-turn prompt sections."""

    return "\n\n---\n\n".join(_dynamic_sections(profile, workspace, extra_dynamic_sections))


def build_system_prompt(
    *,
    profile: PromptProfile | None = None,
    workspace: WorkspacePromptContext | None = None,
    extra_dynamic_sections: Mapping[str, str] | Sequence[str] | None = None,
    static_sections: Sequence[str] | None = None,
    assistant_name: str | None = None,
    dynamic_boundary: str = DYNAMIC_BOUNDARY,
) -> SystemPrompt:
    """Assemble the complete prompt with a stable static/dynamic boundary."""

    resolved_static_sections = (
        tuple(dedent(section) for section in static_sections if section.strip())
        if static_sections is not None
        else _static_sections(assistant_name)
    )
    dynamic_sections = (
        dynamic_boundary.strip(),
        *_dynamic_sections(profile, workspace, extra_dynamic_sections),
    )
    full = "\n\n---\n\n".join(filter(None, (*resolved_static_sections, *dynamic_sections)))
    return SystemPrompt(
        static_sections=resolved_static_sections,
        dynamic_sections=dynamic_sections,
        full=full,
    )


def _dynamic_sections(
    profile: PromptProfile | None,
    workspace: WorkspacePromptContext | None,
    extra_dynamic_sections: Mapping[str, str] | Sequence[str] | None,
) -> tuple[str, ...]:
    sections = (
        build_workspace_section(workspace),
        *_extra_sections(extra_dynamic_sections),
        build_memory_section(profile),
    )
    return tuple(dedent(section) for section in sections if section.strip())


def _extra_sections(values: Mapping[str, str] | Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    source = values.values() if isinstance(values, Mapping) else values
    return tuple(dedent(value) for value in source if value.strip())


def _resolve_timezone(timezone_name: str) -> tuple[tzinfo, str]:
    resolved_name = timezone_name.strip() or DEFAULT_TIMEZONE
    try:
        return ZoneInfo(resolved_name), resolved_name
    except ZoneInfoNotFoundError:
        if resolved_name != DEFAULT_TIMEZONE:
            try:
                return ZoneInfo(DEFAULT_TIMEZONE), DEFAULT_TIMEZONE
            except ZoneInfoNotFoundError:
                pass
        return DEFAULT_TIMEZONE_FALLBACK, DEFAULT_TIMEZONE


def _resolve_current_time(current_time: datetime | None, time_zone: tzinfo) -> datetime:
    if current_time is None:
        return datetime.now(time_zone)
    if current_time.tzinfo is None:
        return current_time.replace(tzinfo=time_zone)
    return current_time.astimezone(time_zone)


def _format_utc_offset(value: datetime) -> str:
    offset = value.utcoffset()
    if offset is None:
        return "UTC"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    hours, minutes = divmod(abs(total_minutes), 60)
    return f"UTC{sign}{hours:02d}:{minutes:02d}"
