"""Subscriber-authorized Skill contracts and invocation mechanics."""

from __future__ import annotations

import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field, SkipValidation

from gewu_agent_runtime.domain import ConversationMessage, MessageKind
from gewu_agent_runtime.invocation import InvocationTarget, InvocationTargetKind
from gewu_agent_runtime.skill_contracts import SkillDescriptor
from gewu_agent_runtime.tools import Tool, ToolContext, ToolResult, ToolSet, tool

SENT_SKILL_NAMES_KEY = "sent_skill_names"
INVOKED_SKILLS_KEY = "invoked_skills"
SKILL_INVOCATION_ARGS_MAX_BYTES = 4 * 1024
MAX_LISTING_DESC_CHARS = 250
SKILL_LISTING_LEAD = "The following skills are available for use with the skill tool:"


class SkillCapacityExceededError(ValueError):
    """An authorized Skill catalog cannot fit in one Runtime turn."""


class SkillCatalogContractError(ValueError):
    """An authorized Skill catalog violated the Runtime boundary contract."""


class SkillDocument(SkillDescriptor):
    """Business-neutral Skill content returned by an authorized catalog."""

    model_config = ConfigDict(frozen=True)

    content: str = ""
    asset_key: str = ""
    base_path: str = ""
    source_path: str = ""
    content_hash: str = ""


class SkillCatalog(Protocol):
    """Subscriber-resolved Skill catalog authorized for one turn.

    Implementations own authorization and any business-specific same-name
    precedence. Returned descriptors must have unique model-visible names and
    stable asset keys; the Runtime never interprets subscriber hierarchy.
    """

    async def list_skills(self, *, limit: int | None = None) -> Sequence[SkillDocument]:
        """List authorized descriptors after subscriber-defined name resolution."""

    async def get_skill(self, name: str) -> SkillDocument | None:
        """Return one readable Skill by its model-visible name."""

    async def get_skill_by_asset_key(self, asset_key: str) -> SkillDocument | None:
        """Load one authorized Skill body by its stable opaque key."""


class SceneDocument(BaseModel):
    """Authorized Scene metadata with a subscriber-defined logical path."""

    model_config = ConfigDict(frozen=True)

    asset_key: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=256)
    description: str = ""
    workspace_path: str = Field(min_length=1)
    required_skill_asset_key: str = ""
    recommended_skill_asset_keys: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)


class SceneCatalog(Protocol):
    """Catalog already filtered to Scenes authorized for one turn."""

    async def get_scene(self, asset_key: str) -> SceneDocument | None:
        """Return one authorized Scene by opaque asset key."""


class SkillOutput(BaseModel):
    """Normalized result produced by the reference Skill tool."""

    skill_name: str = ""
    description: str = ""
    invoked: bool = True
    error: str = ""


class SkillInvocation(BaseModel):
    """Expanded Skill content and compact state metadata."""

    model_config = ConfigDict(frozen=True)

    skill_name: str
    args: str
    base_path: str
    content: str
    source: str

    def metadata(self) -> dict[str, str]:
        return {
            "name": self.skill_name,
            "args": self.args,
            "base_path": self.base_path,
            "source": self.source,
        }


class UserSkillCommand(BaseModel):
    """A slash command resolved against the authorized Skill registry."""

    model_config = ConfigDict(frozen=True)

    skill: SkillDocument
    args: str
    command_name: str
    skill_name: str


class SkillRegistry:
    """Per-turn index over an already-authorized subscriber catalog."""

    def __init__(
        self,
        catalog: SkillCatalog,
        descriptors: Sequence[SkillDocument],
    ) -> None:
        self._catalog = catalog
        self._by_name: dict[str, SkillDocument] = {}
        self._by_asset_key: dict[str, SkillDocument] = {}
        for descriptor in descriptors:
            name = descriptor.name.strip()
            asset_key = _skill_key(descriptor)
            if not name:
                raise SkillCatalogContractError(
                    "Authorized Skill catalog returned an empty model-visible name."
                )
            if name in self._by_name:
                raise SkillCatalogContractError(
                    f"Authorized Skill catalog returned duplicate name: {name}."
                )
            if not asset_key:
                raise SkillCatalogContractError(
                    f"Authorized Skill catalog returned an empty asset key for: {name}."
                )
            if asset_key in self._by_asset_key:
                raise SkillCatalogContractError(
                    f"Authorized Skill catalog returned duplicate asset key: {asset_key}."
                )
            self._by_name[name] = descriptor
            self._by_asset_key[asset_key] = descriptor

    @classmethod
    async def load(
        cls,
        catalog: SkillCatalog,
        *,
        max_visible_skills: int,
    ) -> SkillRegistry:
        """Load descriptors with explicit overflow detection."""

        descriptors = tuple(await catalog.list_skills(limit=max_visible_skills + 1))
        if len(descriptors) > max_visible_skills:
            raise SkillCapacityExceededError(
                "Agent visible Skill count exceeds configured limit of " f"{max_visible_skills}."
            )
        return cls(catalog, descriptors)

    def all(self) -> tuple[SkillDocument, ...]:
        """Return the subscriber-resolved model-visible descriptors."""

        return tuple(self._by_name.values())

    async def list_skills(self, *, limit: int | None = None) -> Sequence[SkillDocument]:
        """Expose the already-loaded descriptors when used as a bound catalog."""

        values = self.all()
        return values[:limit] if limit is not None else values

    async def get_skill(self, name: str) -> SkillDocument | None:
        """Load a full Skill by its model-visible name."""

        descriptor = self._by_name.get(name)
        if descriptor is None:
            return None
        loaded = await self._catalog.get_skill_by_asset_key(_skill_key(descriptor))
        if _matches_descriptor_identity(loaded, descriptor):
            return loaded
        fallback = await self._catalog.get_skill(name)
        return fallback if _matches_descriptor_identity(fallback, descriptor) else None

    async def get_skill_by_asset_key(self, asset_key: str) -> SkillDocument | None:
        """Load a full Skill only if its asset is the visible name winner."""

        descriptor = self._by_asset_key.get(asset_key)
        if descriptor is None:
            return None
        loaded = await self._catalog.get_skill_by_asset_key(asset_key)
        return loaded if _matches_descriptor_identity(loaded, descriptor) else None

    def skill_name_for_asset_key(self, asset_key: str) -> str:
        """Return a visible Skill name without loading its body."""

        descriptor = self._by_asset_key.get(asset_key)
        return descriptor.name if descriptor is not None else ""


class SkillTurnPreparation(BaseModel):
    """Prepared input, attachments and mutable Skill state for one turn."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    registry: SkipValidation[SkillRegistry | None] = None
    input_metadata: dict[str, Any] = Field(default_factory=dict)
    messages: tuple[dict[str, Any], ...] = ()
    state: dict[str, Any] = Field(default_factory=dict)
    state_dirty: bool = False


RuntimeVariableProvider = Callable[[ToolContext, SkillDocument], Mapping[str, str]]


async def prepare_skill_turn(
    *,
    metadata: Mapping[str, Any],
    invocation_target: InvocationTarget | None,
    tool_context: ToolContext,
    skill_catalog: SkillCatalog | None,
    scene_catalog: SceneCatalog | None,
    persisted_state: Mapping[str, Any] | None,
    max_visible_skills: int = 256,
    max_skill_listing_bytes: int = 256 * 1024,
    runtime_variables: RuntimeVariableProvider | None = None,
) -> SkillTurnPreparation:
    """Prepare Skill and Scene context for one Runtime turn."""

    registry = (
        await SkillRegistry.load(
            skill_catalog,
            max_visible_skills=max_visible_skills,
        )
        if skill_catalog is not None
        else None
    )
    state = deepcopy(dict(persisted_state or {}))
    input_metadata = deepcopy(dict(metadata))
    messages: list[dict[str, Any]] = []
    dirty = False
    command = await _resolve_user_skill_command(invocation_target, registry)
    listing = _new_skill_listing(
        registry,
        state,
        max_bytes=max_skill_listing_bytes,
    )

    if command is not None:
        variables = runtime_variables(tool_context, command.skill) if runtime_variables else {}
        invocation = expand_skill_invocation(
            command.skill,
            command.args,
            source="user",
            runtime_variables=variables,
        )
        input_metadata = {
            **input_metadata,
            "llm_ignore": True,
            "skill_command": {
                "asset_key": _skill_key(command.skill),
                "skill_name": command.skill_name,
                "command_name": command.command_name,
                "args": command.args,
            },
        }
        messages.extend(
            (
                _skill_command_message(command.command_name, command.args),
                _skill_content_message(command.skill_name, invocation.content),
            )
        )
        if listing is not None:
            messages.append(listing)
            _mark_listing_sent(state, registry)
        record_skill_invocation(state, invocation.metadata())
        dirty = True
    elif listing is not None:
        messages.append(listing)
        _mark_listing_sent(state, registry)
        dirty = True

    scene_message = await _selected_scene_message(
        invocation_target=invocation_target,
        scene_catalog=scene_catalog,
        registry=registry,
    )
    if scene_message is not None:
        messages.append(scene_message)

    return SkillTurnPreparation(
        registry=registry,
        input_metadata=input_metadata,
        messages=tuple(messages),
        state=state,
        state_dirty=dirty,
    )


def record_skill_invocation(state: dict[str, Any], invocation: Mapping[str, Any]) -> bool:
    """Record compact invocation metadata without retaining expanded Skill text."""

    name = invocation.get("name") or invocation.get("skill_name")
    if not isinstance(name, str) or not name:
        return False
    invoked = state.get(INVOKED_SKILLS_KEY)
    values = dict(invoked) if isinstance(invoked, dict) else {}
    normalized = dict(invocation)
    if values.get(name) == normalized:
        return False
    values[name] = normalized
    state[INVOKED_SKILLS_KEY] = values
    return True


def reconcile_skill_state(
    messages: Sequence[ConversationMessage],
    state: dict[str, Any],
) -> bool:
    """Retain Skill markers only while their META evidence remains in context."""

    listing_names: set[str] = set()
    content_names: set[str] = set()
    for message in messages:
        if message.kind is not MessageKind.META:
            continue
        attachment_type = message.payload.get("attachment_type")
        if attachment_type == "skill_listing":
            raw_names = message.payload.get("skill_names")
            if isinstance(raw_names, (list, tuple)):
                listing_names.update(
                    value for value in raw_names if isinstance(value, str) and value
                )
        elif attachment_type == "skill_content":
            name = message.payload.get("skill_name")
            if isinstance(name, str) and name:
                content_names.add(name)

    changed = False
    retained_sent = sorted(_sent_skill_names(state) & listing_names)
    if retained_sent:
        if state.get(SENT_SKILL_NAMES_KEY) != retained_sent:
            state[SENT_SKILL_NAMES_KEY] = retained_sent
            changed = True
    elif SENT_SKILL_NAMES_KEY in state:
        state.pop(SENT_SKILL_NAMES_KEY, None)
        changed = True

    raw_invoked = state.get(INVOKED_SKILLS_KEY)
    invoked = raw_invoked if isinstance(raw_invoked, dict) else {}
    retained_invoked = {name: value for name, value in invoked.items() if name in content_names}
    if retained_invoked:
        if state.get(INVOKED_SKILLS_KEY) != retained_invoked:
            state[INVOKED_SKILLS_KEY] = retained_invoked
            changed = True
    elif INVOKED_SKILLS_KEY in state:
        state.pop(INVOKED_SKILLS_KEY, None)
        changed = True
    return changed


def bind_skill_tool(
    tool_set: ToolSet,
    registry: SkillRegistry | None,
    *,
    runtime_variables: RuntimeVariableProvider | None = None,
) -> ToolSet:
    """Replace the authorized `skill` Tool with its per-turn catalog binding."""

    if registry is None or tool_set.get(skill.name) is None:
        return tool_set
    tools = tuple(
        (
            skill_tool(registry, runtime_variables=runtime_variables)
            if value.name == skill.name
            else value
        )
        for value in tool_set.all()
    )
    return ToolSet(tools, name=tool_set.name, version=tool_set.version)


@tool(name="skill", category="execution")
async def skill(
    name: str,
    args: str | None = None,
    *,
    runtime: ToolContext,
) -> ToolResult:
    """Invoke a skill by name.

    Skills provide specialized capabilities and domain knowledge for specific
    tasks. When you invoke a skill, the skill content is loaded from the current
    readable skill set and injected into the conversation as a user message,
    effectively activating that expert workflow.

    Usage:
    - Use this tool when the available skill list or user request indicates a
      matching skill should handle the task.
    - Use the skill name and optional arguments.
    - Only skills readable in the current context can be invoked.
    - Do not invoke a skill when the current turn already contains a
      <command-name>...</command-name> tag for that skill. That means the user
      already loaded it directly.

    Args:
        name: The skill name to invoke.
        args: Optional arguments for the skill as a string.

    Returns:
        dict with:
            - skill_name: Name of the skill, empty on error.
            - description: Skill description, empty on error.
            - invoked: True when successfully loaded, false on error.
            - error: Error message if failed, empty on success.

        The skill's full content is injected into the conversation as a user
        message for the next model turn.
    """

    del name, args, runtime
    return ToolResult(
        output=SkillOutput(error="No skill resolver configured", invoked=False),
        is_error=True,
    )


def skill_tool(
    catalog: SkillCatalog,
    *,
    runtime_variables: RuntimeVariableProvider | None = None,
) -> Tool:
    """Bind the exact Skill contract to an already-authorized host catalog."""

    async def execute(arguments: dict[str, Any], context: ToolContext) -> ToolResult:
        name = str(arguments.get("name") or "")
        raw_args = arguments.get("args")
        args = None if raw_args is None else str(raw_args)
        skill_document = await catalog.get_skill(name)
        if skill_document is None:
            return ToolResult(
                output=SkillOutput(error=f"Skill '{name}' not found", invoked=False),
                is_error=True,
            )
        if skill_document.disable_model_invocation:
            return ToolResult(
                output=SkillOutput(
                    error=f"Skill '{name}' cannot be invoked by the model",
                    invoked=False,
                ),
                is_error=True,
            )
        variables = runtime_variables(context, skill_document) if runtime_variables else {}
        invocation = expand_skill_invocation(
            skill_document,
            args,
            source="model",
            runtime_variables=variables,
        )
        return ToolResult(
            output=SkillOutput(
                skill_name=name,
                description=skill_document.description,
                invoked=True,
            ),
            extra={"skill_invocation": invocation.metadata()},
            new_messages=(
                {
                    "role": "user",
                    "content": invocation.content,
                    "is_meta": True,
                    "attachment_type": "skill_content",
                    "skill_name": invocation.skill_name,
                },
            ),
        )

    return skill.model_copy(update={"function": execute})


def load_skill_tool(catalog: SkillCatalog) -> Tool:
    """Compatibility alias for callers migrating to the exact `skill` Tool."""

    return skill_tool(catalog)


def expand_skill_invocation(
    skill_document: SkillDocument,
    args: str | None = None,
    *,
    source: str = "model",
    runtime_variables: Mapping[str, str] | None = None,
) -> SkillInvocation:
    """Expand Claude-style arguments and host-provided runtime variables."""

    raw_args = (args or "").strip()
    content, replaced_args = _apply_argument_substitutions(
        skill_document.content,
        skill_document,
        raw_args,
    )
    for key, value in (runtime_variables or {}).items():
        content = content.replace(f"${{{key}}}", value)
    if raw_args and not replaced_args:
        content = f"{content}\n\nARGUMENTS: {raw_args}"
    return SkillInvocation(
        skill_name=skill_document.name,
        args=raw_args,
        base_path=skill_document.base_path,
        content=f"Base directory for this skill: {skill_document.base_path}\n\n{content}",
        source=source,
    )


def _apply_argument_substitutions(
    content: str,
    skill_document: SkillDocument,
    raw_args: str,
) -> tuple[str, bool]:
    replacements = _argument_replacements(skill_document, raw_args)
    if not replacements:
        return content, False
    replaced = False

    def replace_dollar(match: re.Match[str]) -> str:
        nonlocal replaced
        key = match.group(1)
        if key not in replacements:
            return match.group(0)
        replaced = True
        return replacements[key]

    content = re.sub(
        r"\$(ARGUMENTS(?:\[\d+])?|\d+|[A-Za-z_][A-Za-z0-9_]*)",
        replace_dollar,
        content,
    )
    for key, value in replacements.items():
        for variant in (key, key.lower(), key.upper()):
            for template in (f"${{{variant}}}", f"{{{{{variant}}}}}"):
                if template in content:
                    replaced = True
                    content = content.replace(template, value)
    return content, replaced


def _argument_replacements(
    skill_document: SkillDocument,
    raw_args: str,
) -> dict[str, str]:
    if not raw_args:
        return {}
    try:
        tokens = shlex.split(raw_args)
    except ValueError:
        tokens = raw_args.split()
    replacements = {"ARGUMENTS": raw_args}
    for index, token in enumerate(tokens):
        replacements[str(index)] = token
        replacements[f"ARGUMENTS[{index}]"] = token
    key_values: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue
        key, parsed_value = token.split("=", 1)
        if key:
            key_values[key] = parsed_value
    for index, name in enumerate(skill_document.arguments):
        argument_value = key_values.get(name)
        if argument_value is None and index < len(tokens):
            argument_value = tokens[index]
        if argument_value is not None:
            replacements[name] = argument_value
    replacements.update(key_values)
    return replacements


async def _resolve_user_skill_command(
    invocation_target: InvocationTarget | None,
    registry: SkillRegistry | None,
) -> UserSkillCommand | None:
    if registry is None:
        return None
    if invocation_target is None or invocation_target.kind is not InvocationTargetKind.SKILL:
        return None
    skill_name = invocation_target.name.strip()
    args = invocation_target.arguments.strip()
    skill_document = None
    if invocation_target.resource_id:
        skill_document = await registry.get_skill_by_asset_key(invocation_target.resource_id)
    if skill_document is None and skill_name:
        skill_document = await registry.get_skill(skill_name)
    if skill_document is None or not skill_document.user_invocable:
        return None
    if len(args.encode()) > SKILL_INVOCATION_ARGS_MAX_BYTES:
        raise ValueError(f"Skill slash arguments exceed {SKILL_INVOCATION_ARGS_MAX_BYTES} bytes.")
    return UserSkillCommand(
        skill=skill_document,
        args=args,
        command_name=skill_name,
        skill_name=skill_document.name,
    )


def _new_skill_listing(
    registry: SkillRegistry | None,
    state: Mapping[str, Any],
    *,
    max_bytes: int,
) -> dict[str, Any] | None:
    if registry is None:
        return None
    sent = _sent_skill_names(state)
    skills = [
        value
        for value in sorted(registry.all(), key=lambda item: item.name)
        if not value.disable_model_invocation and value.name not in sent
    ]
    if not skills:
        return None
    entries: list[str] = []
    for value in skills:
        description = _listing_description(value)
        if len(description) > MAX_LISTING_DESC_CHARS:
            description = description[: MAX_LISTING_DESC_CHARS - 1] + "..."
        entries.append(f"- {value.name}: {description}")
    content = (
        f"<system-reminder>\n{SKILL_LISTING_LEAD}\n\n" + "\n".join(entries) + "\n</system-reminder>"
    )
    actual_bytes = len(content.encode())
    if actual_bytes > max_bytes:
        raise SkillCapacityExceededError(
            "Agent Skill listing exceeds configured limit of "
            f"{max_bytes} UTF-8 bytes (actual: {actual_bytes})."
        )
    return {
        "role": "user",
        "content": content,
        "is_meta": True,
        "attachment_type": "skill_listing",
        "skill_names": tuple(value.name for value in skills),
    }


def _mark_listing_sent(state: dict[str, Any], registry: SkillRegistry | None) -> None:
    if registry is None:
        return
    sent = _sent_skill_names(state)
    sent.update(value.name for value in registry.all() if not value.disable_model_invocation)
    state[SENT_SKILL_NAMES_KEY] = sorted(sent)


def _skill_command_message(skill_name: str, args: str) -> dict[str, Any]:
    content = "\n".join(
        (
            f"<command-message>{skill_name}</command-message>",
            f"<command-name>/{skill_name}</command-name>",
            f"<command-args>{args}</command-args>",
        )
    )
    return {
        "role": "user",
        "content": content,
        "is_meta": True,
        "attachment_type": "skill_command_metadata",
        "skill_name": skill_name,
    }


def _skill_content_message(skill_name: str, content: str) -> dict[str, Any]:
    return {
        "role": "user",
        "content": content,
        "is_meta": True,
        "attachment_type": "skill_content",
        "skill_name": skill_name,
    }


async def _selected_scene_message(
    *,
    invocation_target: InvocationTarget | None,
    scene_catalog: SceneCatalog | None,
    registry: SkillRegistry | None,
) -> dict[str, Any] | None:
    if scene_catalog is None:
        return None
    if invocation_target is None or invocation_target.kind is not InvocationTargetKind.SCENE:
        return None
    scene = await scene_catalog.get_scene(invocation_target.resource_id)
    if scene is None:
        return None
    content = _scene_reminder_content(scene, registry)
    return {
        "role": "user",
        "content": content,
        "is_meta": True,
        "attachment_type": "scene_reminder",
        "skill_name": scene.name,
    }


def _scene_reminder_content(
    scene: SceneDocument,
    registry: SkillRegistry | None,
) -> str:
    scene_path = scene.workspace_path.rstrip("/") or "/"
    lines = [
        "User-selected Scene:",
        f"- Scene name: {scene.name}",
        f"- Scene entry path: {scene_path}",
    ]
    skill_name = (
        registry.skill_name_for_asset_key(scene.required_skill_asset_key)
        if registry is not None and scene.required_skill_asset_key
        else ""
    )
    if skill_name:
        lines.append(f"- Bound skill: {skill_name}")
    return "<system-reminder>\n" + "\n".join(lines) + "\n</system-reminder>"


def _sent_skill_names(state: Mapping[str, Any]) -> set[str]:
    raw = state.get(SENT_SKILL_NAMES_KEY)
    if not isinstance(raw, list):
        return set()
    return {str(value) for value in raw if str(value)}


def _skill_key(skill_document: SkillDocument) -> str:
    return skill_document.asset_key or skill_document.name


def _matches_descriptor_identity(
    loaded: SkillDocument | None,
    descriptor: SkillDocument,
) -> bool:
    """Return whether a loaded body is the exact catalog winner that was indexed."""

    return bool(
        loaded is not None
        and loaded.name == descriptor.name
        and _skill_key(loaded) == _skill_key(descriptor)
    )


def _listing_description(skill_document: SkillDocument) -> str:
    description = skill_document.description.strip()
    when_to_use = (skill_document.when_to_use or "").strip()
    if description and when_to_use:
        return f"{description} - {when_to_use}"
    return when_to_use or description
