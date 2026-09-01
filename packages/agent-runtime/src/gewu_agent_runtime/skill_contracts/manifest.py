"""Parse and validate subscriber-provided ``SKILL.md`` manifests."""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gewu_core.yaml import (
    YamlMappingError,
    YamlMappingIssue,
    dump_yaml_mapping,
    load_yaml_mapping_entries,
)

MAX_SKILL_FILE_BYTES = 64 * 1024
MAX_SKILL_DESCRIPTION_CHARS = 512
SKILL_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class SkillDescriptor(BaseModel):
    """Validated fields declared in ``SKILL.md`` frontmatter."""

    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    when_to_use: str | None = None
    arguments: list[str] = Field(default_factory=list)
    user_invocable: bool = True
    allowed_tools: list[str] | None = None
    argument_hint: str | None = None
    model: str | None = None
    disable_model_invocation: bool = False
    paths: list[str] | None = None
    hooks: dict[str, Any] | None = None
    effort: str | None = None
    context: str | None = None
    agent: str | None = None
    version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SkillFrontmatterSnapshot(BaseModel):
    """Recognized source values without adding runtime defaults."""

    model_config = ConfigDict(frozen=True)

    values: dict[str, Any] = Field(default_factory=dict)
    origin_key_names: dict[str, str] = Field(default_factory=dict)

    def to_descriptor(self) -> SkillDescriptor:
        """Build the descriptor and apply declared defaults."""

        return SkillDescriptor.model_validate(self.values)


class ParsedSkillManifest(BaseModel):
    """Validated frontmatter snapshot, Markdown body, and source hash."""

    model_config = ConfigDict(frozen=True)

    frontmatter: SkillFrontmatterSnapshot
    content: str = ""
    content_hash: str

    @property
    def descriptor(self) -> SkillDescriptor:
        """Return the validated descriptor."""

        return self.frontmatter.to_descriptor()


_OPTIONAL_STRING_FIELDS = {
    "when_to_use",
    "argument_hint",
    "model",
    "effort",
    "context",
    "agent",
    "version",
}
_OPTIONAL_STRING_LIST_FIELDS = {"allowed_tools", "paths"}
_BOOLEAN_FIELDS = {"user_invocable", "disable_model_invocation"}


def _camel_case(value: str) -> str:
    first, *remaining = value.split("_")
    return first + "".join(part[:1].upper() + part[1:] for part in remaining)


def _frontmatter_aliases() -> dict[str, str]:
    aliases: dict[str, str] = {}
    for canonical_name in SkillDescriptor.model_fields:
        for source_name in {
            canonical_name,
            canonical_name.replace("_", "-"),
            _camel_case(canonical_name),
        }:
            aliases[source_name] = canonical_name
    return aliases


FRONTMATTER_FIELD_ALIASES = _frontmatter_aliases()


def parse_skill_manifest(
    content: str,
    *,
    expected_name: str | None = None,
    include_content: bool = True,
) -> ParsedSkillManifest:
    """Parse one complete ``SKILL.md`` using the shared contract."""

    encoded = content.encode("utf-8")
    if len(encoded) > MAX_SKILL_FILE_BYTES:
        raise ValueError(f"Skill file exceeds {MAX_SKILL_FILE_BYTES} bytes")
    if not content.strip():
        raise ValueError("Skill content is empty")
    frontmatter_text, markdown_content = _split_frontmatter(content)
    entries = _load_frontmatter_entries(frontmatter_text)
    values: dict[str, Any] = {}
    origin_key_names: dict[str, str] = {}
    for source_name, value in entries:
        canonical_name = FRONTMATTER_FIELD_ALIASES.get(source_name)
        if canonical_name is None:
            continue
        values[canonical_name] = value
        origin_key_names[canonical_name] = source_name

    _validate_descriptor_values(values, expected_name=expected_name)
    return ParsedSkillManifest(
        frontmatter=SkillFrontmatterSnapshot(
            values=values,
            origin_key_names=origin_key_names,
        ),
        content=markdown_content.strip() if include_content else "",
        content_hash=hashlib.sha256(encoded).hexdigest(),
    )


def rewrite_skill_manifest_identity(content: str, *, name: str, description: str) -> str:
    """Write request-form identity fields while retaining other source fields."""

    frontmatter_text, markdown_content = _split_frontmatter(content)
    entries = _load_frontmatter_entries(frontmatter_text)
    frontmatter: dict[str, Any] = {}
    for key, value in entries:
        frontmatter.pop(key, None)
        frontmatter[key] = value
    frontmatter["name"] = name.strip()
    if description.strip():
        frontmatter["description"] = description.strip()
    dumped = dump_yaml_mapping(frontmatter)
    suffix = markdown_content.lstrip("\r\n")
    rewritten = f"---\n{dumped}\n---"
    return f"{rewritten}\n\n{suffix}" if suffix else rewritten


def _load_frontmatter_entries(frontmatter_text: str) -> list[tuple[str, Any]]:
    try:
        return load_yaml_mapping_entries(frontmatter_text)
    except YamlMappingError as exc:
        if exc.issue is YamlMappingIssue.EMPTY:
            raise ValueError("Skill file has empty YAML frontmatter") from exc
        if exc.issue is YamlMappingIssue.ROOT_NOT_MAPPING:
            raise ValueError("Skill file YAML frontmatter must be a mapping") from exc
        if exc.issue is YamlMappingIssue.NON_STRING_KEY:
            raise ValueError("Skill file YAML frontmatter keys must be strings") from exc
        raise ValueError(f"Invalid YAML in skill frontmatter: {exc.detail}") from exc


def _split_frontmatter(content: str) -> tuple[str, str]:
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("Skill file missing YAML frontmatter marker (---)")
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            return "".join(lines[1:index]).strip(), "".join(lines[index + 1 :])
    raise ValueError("Skill file missing closing YAML frontmatter marker (---)")


def _validate_descriptor_values(values: dict[str, Any], *, expected_name: str | None) -> None:
    name = _required_string(values, "name")
    if not SKILL_NAME_RE.fullmatch(name):
        raise ValueError(
            "Skill 'name' must start with a letter or digit and only contain "
            "letters, digits, '_' or '-'"
        )
    if expected_name is not None and name != expected_name:
        raise ValueError(f"Skill frontmatter name '{name}' must match directory '{expected_name}'")

    description = _required_string(values, "description")
    if len(description) > MAX_SKILL_DESCRIPTION_CHARS:
        raise ValueError(f"Skill 'description' exceeds {MAX_SKILL_DESCRIPTION_CHARS} characters")
    for field_name in _OPTIONAL_STRING_FIELDS:
        if field_name in values:
            _require_optional_string(values[field_name], field_name)
    if "arguments" in values:
        _require_string_list(values["arguments"], "arguments", optional=False)
    for field_name in _OPTIONAL_STRING_LIST_FIELDS:
        if field_name in values:
            _require_string_list(values[field_name], field_name, optional=True)
    for field_name in _BOOLEAN_FIELDS:
        if field_name in values and not isinstance(values[field_name], bool):
            raise ValueError(f"Skill '{field_name}' must be a boolean")
    if "hooks" in values:
        _require_mapping(values["hooks"], "hooks", optional=True)
    if "metadata" in values:
        _require_mapping(values["metadata"], "metadata", optional=False)
    for field_name, value in values.items():
        _validate_json_value(value, path=field_name)
    SkillDescriptor.model_validate(values)


def _required_string(values: dict[str, Any], key: str) -> str:
    if key not in values:
        raise ValueError("Skill file missing required fields: 'name' and 'description'")
    value = values[key]
    if not isinstance(value, str):
        raise ValueError(f"Skill '{key}' must be a string")
    if not value.strip():
        raise ValueError(f"Skill '{key}' field is empty")
    return value


def _require_optional_string(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"Skill '{field_name}' must be a string or null")


def _require_string_list(value: object, field_name: str, *, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"Skill '{field_name}' must be a list of strings")


def _require_mapping(value: object, field_name: str, *, optional: bool) -> None:
    if value is None and optional:
        return
    if not isinstance(value, dict):
        raise ValueError(f"Skill '{field_name}' must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"Skill '{field_name}' keys must be strings")


def _validate_json_value(value: object, *, path: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Skill '{path}' must not contain NaN or infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, path=f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"Skill '{path}' keys must be strings")
            _validate_json_value(item, path=f"{path}.{key}")
        return
    raise ValueError(
        f"Skill '{path}' must use JSON-compatible YAML values, got {type(value).__name__}"
    )
