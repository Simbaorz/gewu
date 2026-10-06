"""Subscriber-neutral prompt assembly tests."""

from __future__ import annotations

from datetime import UTC, datetime

from gewu_agent_runtime.prompts import (
    DYNAMIC_BOUNDARY,
    PromptProfile,
    WorkspacePromptContext,
    build_environment_section,
    build_system_prompt,
    get_dynamic_prompt,
    get_static_prompt,
)


def test_static_prompt_uses_neutral_runtime_default() -> None:
    value = get_static_prompt()

    assert "You are Agent" in value
    assert "Subscriber" not in value
    assert DYNAMIC_BOUNDARY not in value


def test_system_prompt_assembles_profile_and_neutral_workspace_context() -> None:
    workspace = WorkspacePromptContext(
        writable_roots=("/workspace/private",),
        readable_roots=(
            "/workspace/shared/tenant",
            "/workspace/shared/province",
            "/workspace/shared/city",
            "/workspace/shared/teams",
        ),
        relative_path_root="/workspace/private",
        relative_path_description=(
            "Relative paths resolve under `/workspace/private` and never refer to shared files."
        ),
        rules=(
            "Absolute paths must be within one of the allowed roots above.",
            "`/workspace` and `/workspace/shared` are namespaces and cannot be accessed directly.",
            "Shared paths must be absolute.",
            "`/workspace/shared/teams` can be listed or searched across available workspaces.",
            "All shared paths are read-only.",
        ),
    )

    prompt = build_system_prompt(
        profile=PromptProfile(
            sections=(
                "# Subscriber Context\n\n"
                "<AgentPolicy>\nbot instructions\n</AgentPolicy>\n\n"
                "<PrincipalPreferences>\nuser preferences\n</PrincipalPreferences>",
            )
        ),
        workspace=workspace,
    )

    assert "You are Agent" in prompt.full
    assert DYNAMIC_BOUNDARY in prompt.dynamic_sections
    assert "<AgentPolicy>\nbot instructions\n</AgentPolicy>" in prompt.full
    assert "<PrincipalPreferences>\nuser preferences\n</PrincipalPreferences>" in prompt.full
    assert "`/workspace/shared/province`" in prompt.full
    assert "Host filesystem paths are unavailable" in prompt.full
    assert "tenant_id" not in prompt.full
    assert "team_id" not in prompt.full


def test_assistant_name_and_dynamic_sections_are_subscriber_configurable() -> None:
    prompt = build_system_prompt(
        assistant_name="Subscriber Agent",
        extra_dynamic_sections={"catalog": "# Available Resources\n\nresource-a"},
    )

    assert "You are Subscriber Agent" in prompt.full
    assert "You are Agent" not in prompt.full
    assert any("# Available Resources" in section for section in prompt.dynamic_sections)


def test_subscriber_can_replace_static_sections_without_changing_runtime_default() -> None:
    prompt = build_system_prompt(
        static_sections=("You are a specialist.", "# Specialist Rules"),
        extra_dynamic_sections=("# Current Context",),
    )

    assert prompt.static_sections == ("You are a specialist.", "# Specialist Rules")
    assert prompt.full.startswith("You are a specialist.\n\n---\n\n# Specialist Rules")
    assert "# Doing Tasks" not in prompt.full
    assert "# Doing Tasks" in get_static_prompt()


def test_environment_section_uses_explicit_timezone() -> None:
    section = build_environment_section(
        current_time=datetime(2026, 5, 29, 4, 52, tzinfo=UTC),
        timezone_name="Asia/Shanghai",
    )

    assert "2026-05-29 12:52:00 Asia/Shanghai (UTC+08:00)" in section
    assert "2026-05-29T04:52:00Z" in section


def test_empty_dynamic_context_does_not_emit_sections_or_empty_entries() -> None:
    profile = PromptProfile(sections=("", " \n "))
    workspace = WorkspacePromptContext()
    extras = {"empty": " \n "}

    assert (
        get_dynamic_prompt(profile=profile, workspace=workspace, extra_dynamic_sections=extras)
        == ""
    )
    prompt = build_system_prompt(
        profile=profile, workspace=workspace, extra_dynamic_sections=extras
    )
    assert prompt.dynamic_sections == (DYNAMIC_BOUNDARY,)
    for title in (
        "# Workspace",
        "# Memory",
        "# Default Scene",
        "# Environment",
        "# Available Scenes",
    ):
        assert title not in prompt.full


def test_stable_dynamic_context_precedes_memory_and_has_no_clock() -> None:
    workspace = WorkspacePromptContext(readable_roots=("/wiki",), relative_path_root="/wiki")
    profile = PromptProfile(sections=("# Memory\n\nRemember this preference.",))
    extras = ("# Default Scene\n\nEntry path: /wiki/policies",)
    prompt = build_system_prompt(
        workspace=workspace, profile=profile, extra_dynamic_sections=extras
    )

    assert prompt.full.index("# Workspace") < prompt.full.index("# Default Scene")
    assert prompt.full.index("# Default Scene") < prompt.full.index("# Memory")
    assert "Writable:\n- None" in prompt.full
    assert "# Environment" not in prompt.full
    assert get_dynamic_prompt(
        workspace=workspace, profile=profile, extra_dynamic_sections=extras
    ) == "\n\n---\n\n".join(prompt.dynamic_sections[1:])
    assert prompt == build_system_prompt(
        workspace=workspace, profile=profile, extra_dynamic_sections=extras
    )
