"""Rebuild provider-neutral model context from the append-only message log."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from gewu_agent_runtime.domain import ConversationMessage, FileStateCache, MessageKind
from gewu_agent_runtime.llm import ContentPart, ContentPartType, Message, MessageRole, ToolCall
from gewu_agent_runtime.persistence import RuntimeStore

SUMMARY_TAG = "conversation-summary"
COMPACTABLE_TOOL_RESULT_NAMES = frozenset(
    {
        "append",
        "bash",
        "delete",
        "edit",
        "glob",
        "grep",
        "read",
        "write",
    }
)
CLEARED_TOOL_RESULT_CONTENT = "[Old tool result content cleared]"
CLEARED_TOOL_RESULT_PAYLOAD = {
    "cleared": True,
    "message": CLEARED_TOOL_RESULT_CONTENT,
}
DEFAULT_KEEP_RECENT_TOOL_RESULTS = 5
DEFAULT_HISTORY_PAGE_SIZE = 500
LEGACY_TRUNCATED_READ_MARKER = "...(truncated at "


class ConversationContextBuilder:
    """Build model messages from the latest valid compaction boundary."""

    def __init__(
        self,
        store: RuntimeStore,
        *,
        keep_recent_tool_results: int = DEFAULT_KEEP_RECENT_TOOL_RESULTS,
        history_page_size: int = DEFAULT_HISTORY_PAGE_SIZE,
    ) -> None:
        if history_page_size <= 0:
            raise ValueError("history_page_size must be greater than zero")
        self._store = store
        self.keep_recent_tool_results = max(0, keep_recent_tool_results)
        self.history_page_size = history_page_size

    async def build(
        self,
        conversation_id: str,
        *,
        system_prompt: str = "",
        file_state_cache: FileStateCache | None = None,
        skip_input_run_id: str = "",
        current_input_message: Message | None = None,
    ) -> list[Message]:
        compaction = await self._store.get_latest_compaction(conversation_id)
        boundary = compaction.through_sequence if compaction is not None else 0
        persisted = await self._load_history_pages(
            conversation_id,
            after_sequence=boundary,
        )
        compacted = self.compact_history_messages(
            persisted,
            keep_recent_tool_results=self.keep_recent_tool_results,
        )
        self.reconcile_file_state(compacted, file_state_cache)
        return self.build_messages(
            Message.system(system_prompt.strip()) if system_prompt.strip() else None,
            summary=compaction.summary if compaction is not None else "",
            history=compacted,
            skip_input_run_id=skip_input_run_id,
            current_input_message=current_input_message,
        )

    async def _load_history_pages(
        self,
        conversation_id: str,
        *,
        after_sequence: int,
    ) -> tuple[ConversationMessage, ...]:
        """Load complete post-boundary history through bounded ascending pages."""

        messages: list[ConversationMessage] = []
        cursor = after_sequence
        while True:
            page = await self._store.list_message_page(
                conversation_id,
                after_sequence=cursor,
                limit=self.history_page_size,
            )
            if not page:
                break
            next_cursor = page[-1].sequence
            if next_cursor <= cursor:
                raise RuntimeError("Message history page did not advance the sequence cursor.")
            messages.extend(page)
            cursor = next_cursor
            if len(page) < self.history_page_size:
                break
        return tuple(messages)

    @classmethod
    def build_messages(
        cls,
        system_message: Message | None,
        *,
        summary: str,
        history: Sequence[ConversationMessage],
        skip_input_run_id: str = "",
        current_input_message: Message | None = None,
    ) -> list[Message]:
        """Assemble one normalized provider-neutral model request."""

        messages: list[Message] = []
        if system_message is not None:
            messages.append(system_message)
        if summary.strip():
            messages.append(Message.user(f"<{SUMMARY_TAG}>\n{summary.strip()}\n</{SUMMARY_TAG}>"))
        messages.extend(
            cls.convert_messages(
                history,
                skip_input_run_id=skip_input_run_id,
            )
        )
        if current_input_message is not None:
            messages.append(current_input_message)
        return cls.normalize(messages)

    @classmethod
    def reconcile_file_state(
        cls,
        messages: Sequence[ConversationMessage],
        file_state_cache: FileStateCache | None,
    ) -> None:
        """Keep writable state only while its real Read evidence remains in context."""

        if file_state_cache is None:
            return
        retained_reads: set[tuple[str, int, int]] = set()
        for message in messages:
            result = cls._retained_read_result(message)
            if result is None:
                continue
            file_path = result.get("file_path")
            start_line = result.get("start_line")
            num_lines = result.get("num_lines")
            if (
                isinstance(file_path, str)
                and file_path
                and isinstance(start_line, int)
                and isinstance(num_lines, int)
            ):
                retained_reads.add((file_path, start_line, num_lines))
        file_state_cache.retain_read_states(retained_reads)

    @staticmethod
    def _retained_read_result(message: ConversationMessage) -> dict[str, Any] | None:
        if (
            message.kind is not MessageKind.TOOL_RESULT
            or message.payload.get("is_error") is True
            or message.payload.get("tool_name") != "read"
        ):
            return None
        result = message.payload.get("result")
        if not isinstance(result, dict):
            return None
        if (
            result.get("type") != "text"
            or result.get("cleared") is True
            or result.get("unchanged") is True
            or result.get("truncated") is True
        ):
            return None
        content = result.get("content")
        if isinstance(content, str) and LEGACY_TRUNCATED_READ_MARKER in content:
            return None
        return result

    @classmethod
    def compact_history_messages(
        cls,
        messages: Sequence[ConversationMessage],
        *,
        keep_recent_tool_results: int = DEFAULT_KEEP_RECENT_TOOL_RESULTS,
    ) -> Sequence[ConversationMessage]:
        """Replace older low-value Tool results in only the model projection."""

        keep_recent_tool_results = max(0, keep_recent_tool_results)
        eligible_indexes = [
            index
            for index, message in enumerate(messages)
            if cls._is_compactable_tool_result(message)
        ]
        if not eligible_indexes:
            return messages
        keep_indexes = (
            set(eligible_indexes[-keep_recent_tool_results:]) if keep_recent_tool_results else set()
        )
        compacted: list[ConversationMessage] = []
        for index, message in enumerate(messages):
            if index not in eligible_indexes or index in keep_indexes:
                compacted.append(message)
                continue
            compacted.append(
                message.model_copy(
                    update={
                        "payload": {
                            **message.payload,
                            "result": CLEARED_TOOL_RESULT_PAYLOAD.copy(),
                        }
                    }
                )
            )
        return compacted

    @classmethod
    def compactable_tool_result_count(
        cls,
        messages: Sequence[ConversationMessage],
    ) -> int:
        """Count successful low-value Tool results eligible for Micro Compact."""

        return sum(cls._is_compactable_tool_result(message) for message in messages)

    @staticmethod
    def _is_compactable_tool_result(message: ConversationMessage) -> bool:
        if message.kind is not MessageKind.TOOL_RESULT:
            return False
        if message.payload.get("is_error") is True:
            return False
        tool_name = message.payload.get("tool_name")
        return isinstance(tool_name, str) and tool_name in COMPACTABLE_TOOL_RESULT_NAMES

    @classmethod
    def convert_messages(
        cls,
        values: Sequence[ConversationMessage],
        *,
        skip_input_run_id: str = "",
    ) -> list[Message]:
        """Convert persisted messages while preserving tool-call structure."""

        converted: list[Message] = []
        for value in values:
            if (
                skip_input_run_id
                and value.kind is MessageKind.INPUT
                and value.run_id == skip_input_run_id
            ):
                continue
            message = cls.convert_message(value)
            if message is not None:
                converted.append(message)
        return converted

    @classmethod
    def convert_message(cls, value: ConversationMessage) -> Message | None:
        """Convert one persisted message or ignore transport-only records."""

        if value.payload.get("llm_ignore") is True:
            return None
        if value.kind is MessageKind.INPUT:
            return Message.user(cls._input_content_with_attachment_placeholders(value))
        if value.kind is MessageKind.META:
            if value.role is MessageRole.SYSTEM:
                return Message.system(value.content)
            if value.role is MessageRole.ASSISTANT:
                return Message.assistant(value.content)
            if value.role is MessageRole.TOOL:
                return Message.tool(
                    str(value.payload.get("tool_call_id") or ""),
                    value.content,
                )
            return Message.user(value.content)
        if value.kind is MessageKind.SYSTEM:
            return Message.system(value.content)
        if value.kind is MessageKind.ASSISTANT:
            return Message.assistant(value.content)
        if value.kind is MessageKind.TOOL_USE:
            call = ToolCall(
                tool_call_id=str(value.payload.get("tool_call_id") or ""),
                name=str(value.payload.get("tool_name") or ""),
                arguments=_mapping(value.payload.get("arguments")),
            )
            return Message.assistant(value.content, (call,))
        if value.kind is MessageKind.TOOL_RESULT:
            return Message.tool(
                str(value.payload.get("tool_call_id") or ""),
                json.dumps(value.payload.get("result", {}), ensure_ascii=False),
                trace_result=value.payload.get("trace_result") is not False,
            )
        return None

    @staticmethod
    def normalize(messages: Sequence[Message]) -> list[Message]:
        """Merge adjacent user messages without losing structured image parts."""

        normalized: list[Message] = []
        for message in messages:
            if (
                message.role is MessageRole.USER
                and normalized
                and normalized[-1].role is MessageRole.USER
            ):
                if normalized[-1].content_parts or message.content_parts:
                    normalized[-1] = Message.user_parts(
                        (
                            *ConversationContextBuilder._message_content_parts(normalized[-1]),
                            *ConversationContextBuilder._message_content_parts(message),
                        )
                    )
                    continue
                normalized[-1] = Message.user(
                    "\n\n".join(item for item in (normalized[-1].content, message.content) if item)
                )
                continue
            normalized.append(message)
        return normalized

    @classmethod
    def _input_content_with_attachment_placeholders(
        cls,
        message: ConversationMessage,
    ) -> str:
        """Represent historical images as public references, never retained bytes."""

        placeholders = cls._attachment_placeholders(message.payload.get("attachments"))
        return "\n\n".join(part for part in (message.content, *placeholders) if part)

    @staticmethod
    def _attachment_placeholders(value: Any) -> list[str]:
        if not isinstance(value, (list, tuple)):
            return []
        placeholders: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            attachment_id = str(item.get("attachment_id") or "")
            if not attachment_id:
                continue
            original_name = str(item.get("original_name") or attachment_id)
            mime_type = str(item.get("mime_type") or "image")
            placeholders.append(
                f"[Image: {original_name}, {mime_type}, attachment_id={attachment_id}]"
            )
        return placeholders

    @staticmethod
    def _message_content_parts(message: Message) -> tuple[ContentPart, ...]:
        if message.content_parts:
            return message.content_parts
        if message.content:
            return (ContentPart(part_type=ContentPartType.TEXT, text=message.content),)
        return ()


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
