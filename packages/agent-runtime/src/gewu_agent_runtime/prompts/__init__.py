"""Provider- and subscriber-neutral system prompt assembly."""

from gewu_agent_runtime.prompts.system import (
    DYNAMIC_BOUNDARY,
    PromptProfile,
    SystemPrompt,
    WorkspacePromptContext,
    build_environment_section,
    build_system_prompt,
    get_dynamic_prompt,
    get_static_prompt,
)

__all__ = [
    "DYNAMIC_BOUNDARY",
    "PromptProfile",
    "SystemPrompt",
    "WorkspacePromptContext",
    "build_environment_section",
    "build_system_prompt",
    "get_dynamic_prompt",
    "get_static_prompt",
]
