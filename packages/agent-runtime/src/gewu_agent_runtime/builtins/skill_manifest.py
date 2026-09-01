"""Runtime projection of shared ``SKILL.md`` manifest contracts."""

from __future__ import annotations

from pydantic import BaseModel

from gewu_agent_runtime.builtins.skills import SkillDocument
from gewu_agent_runtime.skill_contracts import (
    MAX_SKILL_DESCRIPTION_CHARS,
    MAX_SKILL_FILE_BYTES,
    ParsedSkillManifest,
    SkillDescriptor,
    SkillFrontmatterSnapshot,
    parse_skill_manifest,
    rewrite_skill_manifest_identity,
)


class SkillLoadResult(BaseModel):
    """Non-throwing result used when an indexed Skill body becomes invalid."""

    skill: SkillDocument | None = None
    error: str | None = None


def build_skill_document(
    descriptor: SkillDescriptor,
    *,
    content: str = "",
    content_hash: str = "",
    asset_key: str = "",
    base_path: str = "",
    source_path: str = "",
) -> SkillDocument:
    """Build a Runtime Skill document from a persisted descriptor."""

    return SkillDocument(
        **descriptor.model_dump(mode="python"),
        content=content,
        content_hash=content_hash,
        asset_key=asset_key,
        base_path=base_path,
        source_path=source_path,
    )


def load_skill_document(
    content: str,
    *,
    expected_name: str | None = None,
    asset_key: str = "",
    base_path: str = "",
    source_path: str = "",
) -> SkillLoadResult:
    """Load a complete document without leaking parse failures into a turn."""

    try:
        parsed = parse_skill_manifest(content, expected_name=expected_name)
        return SkillLoadResult(
            skill=build_skill_document(
                parsed.descriptor,
                content=parsed.content,
                content_hash=parsed.content_hash,
                asset_key=asset_key,
                base_path=base_path,
                source_path=source_path,
            )
        )
    except ValueError as exc:
        return SkillLoadResult(error=str(exc))
    except Exception as exc:
        return SkillLoadResult(error=f"Unexpected error loading skill ({type(exc).__name__}).")


__all__ = [
    "MAX_SKILL_DESCRIPTION_CHARS",
    "MAX_SKILL_FILE_BYTES",
    "ParsedSkillManifest",
    "SkillDescriptor",
    "SkillFrontmatterSnapshot",
    "SkillLoadResult",
    "build_skill_document",
    "load_skill_document",
    "parse_skill_manifest",
    "rewrite_skill_manifest_identity",
]
