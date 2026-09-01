"""Compatibility exports for the separately installable Runtime persistence schema."""

from gewu_agent_runtime.adapters.mysql.schema import (
    AgentRunRow,
    AgentRuntimeBase,
    AttachmentRow,
    ConversationCompactionRow,
    ConversationMessageRow,
    ConversationRow,
    ConversationStateRow,
)

__all__ = [
    "AgentRunRow",
    "AgentRuntimeBase",
    "AttachmentRow",
    "ConversationCompactionRow",
    "ConversationMessageRow",
    "ConversationRow",
    "ConversationStateRow",
]
