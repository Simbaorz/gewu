"""Stable ``SKILL.md`` parsing independent of subscriber authorization."""

from __future__ import annotations

import pytest

import gewu_agent_runtime.builtins.skill_manifest as skill_manifest_module
from gewu_agent_runtime.builtins import (
    load_skill_document,
    parse_skill_manifest,
    rewrite_skill_manifest_identity,
)


def test_manifest_preserves_source_snapshot_and_applies_runtime_defaults() -> None:
    manifest = parse_skill_manifest("""---
name: report
description: Write reports
allowed-tools: [read]
allowed_tools: [write]
allowedTools: [read, write]
userInvocable: false
metadata:
  nested-config:
    child-key: value
unknown: ignored
---
# Report
""")

    assert manifest.frontmatter.values == {
        "name": "report",
        "description": "Write reports",
        "allowed_tools": ["read", "write"],
        "user_invocable": False,
        "metadata": {"nested-config": {"child-key": "value"}},
    }
    assert manifest.frontmatter.origin_key_names["allowed_tools"] == "allowedTools"
    assert manifest.descriptor.arguments == []
    assert manifest.content == "# Report"
    assert len(manifest.content_hash) == 64


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("no frontmatter", "missing YAML frontmatter marker"),
        ("---\nname: report\n---\n", "missing required fields"),
        (
            "---\nname: report\ndescription: Reports\npaths: docs\n---\n",
            "must be a list of strings",
        ),
        (
            "---\nname: report\ndescription: Reports\nuser-invocable: 'false'\n---\n",
            "must be a boolean",
        ),
    ],
)
def test_manifest_rejects_the_same_invalid_contracts(content: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_skill_manifest(content)


def test_manifest_requires_name_to_match_storage_directory() -> None:
    with pytest.raises(ValueError, match="must match directory 'review'"):
        parse_skill_manifest(
            "---\nname: report\ndescription: Reports\n---\n",
            expected_name="review",
        )


def test_manifest_ignores_unknown_yaml_but_rejects_non_json_metadata() -> None:
    manifest = parse_skill_manifest(
        "---\nname: report\ndescription: Reports\nreleased: 2026-07-16\n---\n"
    )
    assert "released" not in manifest.frontmatter.values

    with pytest.raises(ValueError, match="JSON-compatible YAML values"):
        parse_skill_manifest(
            "---\nname: report\ndescription: Reports\n" "metadata:\n  released: 2026-07-16\n---\n"
        )


def test_manifest_enforces_utf8_byte_limit_before_yaml_parsing() -> None:
    with pytest.raises(ValueError, match="exceeds 65536 bytes"):
        parse_skill_manifest("---\nname: report\ndescription: Reports\n---\n" + "x" * 65_536)


def test_non_throwing_loader_returns_full_runtime_document() -> None:
    result = load_skill_document(
        "---\nname: review\ndescription: Review code\n---\n# Body\n",
        expected_name="review",
        asset_key="skill-review",
        base_path="/workspace/private/.skills/review",
        source_path=".skills/review/SKILL.md",
    )

    assert result.error is None
    assert result.skill is not None
    assert result.skill.content == "# Body"
    assert result.skill.asset_key == "skill-review"
    assert result.skill.base_path == "/workspace/private/.skills/review"


def test_non_throwing_loader_hides_unexpected_exception_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_parse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("skill-loader-private-secret")

    monkeypatch.setattr(skill_manifest_module, "parse_skill_manifest", fail_parse)

    result = load_skill_document("content")

    assert result.skill is None
    assert result.error == "Unexpected error loading skill (RuntimeError)."
    assert "skill-loader-private-secret" not in result.error


def test_identity_rewrite_keeps_last_duplicate_source_value() -> None:
    rewritten = rewrite_skill_manifest_identity(
        """---
name: report
description: Reports
custom: first
custom: second
user-invocable: false
user_invocable: true
user-invocable: false
---
""",
        name="report",
        description="Reports",
    )

    assert "custom: second" in rewritten
    assert "custom: first" not in rewritten
    assert parse_skill_manifest(rewritten).descriptor.user_invocable is False
