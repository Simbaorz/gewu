"""Conservative local token estimates for provider-bound Agent requests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import tiktoken
from pydantic import BaseModel, ConfigDict, SkipValidation
from tiktoken.model import encoding_name_for_model

from gewu_agent_runtime.llm import ContentPartType, Message
from gewu_agent_runtime.tools import Tool
from gewu_core.blocking import run_cpu_task
from gewu_core.file_tasks import FileTaskLane, run_file_task

MESSAGE_OVERHEAD_TOKENS = 4
TOOL_OVERHEAD_TOKENS = 12
IMAGE_ESTIMATE_TOKENS = 1_600
ESTIMATE_SAFETY_RATIO = 0.15
_ENCODING_SOURCES = {
    "cl100k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken",
        "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7",
    ),
    "o200k_base": (
        "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken",
        "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d",
    ),
}


class _EncodingRegistry(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    encodings: SkipValidation[dict[str, tiktoken.Encoding]]


_REGISTRY_LOCK = threading.Lock()
_REGISTRY: _EncodingRegistry | None = None


async def initialize_context_token_encodings(*, require_complete_cache: bool) -> None:
    """Validate tokenizer assets and construct the process registry before serving."""

    cache_dir = _tiktoken_cache_dir()
    available_names = await run_file_task(
        _valid_cached_encoding_names,
        cache_dir,
        lane=FileTaskLane.INTERACTIVE,
    )
    missing_names = tuple(name for name in _ENCODING_SOURCES if name not in available_names)
    if require_complete_cache and missing_names:
        missing = ", ".join(missing_names)
        raise RuntimeError(
            f"Required tiktoken cache is incomplete at {cache_dir}: missing {missing}."
        )
    registry = await run_cpu_task(_build_encoding_registry, available_names)
    _install_encoding_registry(registry)


def _registry_for_request() -> _EncodingRegistry:
    registry = _REGISTRY
    if registry is not None:
        return registry
    cache_dir = _tiktoken_cache_dir()
    available_names = _valid_cached_encoding_names(cache_dir)
    registry = _build_encoding_registry(available_names)
    _install_encoding_registry(registry)
    return registry


def _install_encoding_registry(registry: _EncodingRegistry) -> None:
    global _REGISTRY
    with _REGISTRY_LOCK:
        _REGISTRY = registry


def _build_encoding_registry(names: Sequence[str]) -> _EncodingRegistry:
    encodings: dict[str, tiktoken.Encoding] = {}
    for name in names:
        try:
            encodings[name] = tiktoken.get_encoding(name)
        except Exception:  # noqa: BLE001
            continue
    return _EncodingRegistry(encodings=encodings)


def _valid_cached_encoding_names(cache_dir: Path) -> tuple[str, ...]:
    valid: list[str] = []
    for name, (source_url, expected_hash) in _ENCODING_SOURCES.items():
        cache_path = cache_dir / hashlib.sha1(source_url.encode()).hexdigest()
        try:
            contents = cache_path.read_bytes()
        except OSError:
            continue
        if hashlib.sha256(contents).hexdigest() == expected_hash:
            valid.append(name)
    return tuple(valid)


def _tiktoken_cache_dir() -> Path:
    configured = os.getenv("TIKTOKEN_CACHE_DIR")
    if configured is None:
        configured = os.getenv("DATA_GYM_CACHE_DIR")
    if configured is None:
        configured = str(Path(tempfile.gettempdir()) / "data-gym-cache")
    return Path(configured).expanduser()


def _select_encodings(
    model_name: str,
) -> tuple[tiktoken.Encoding | None, tuple[tiktoken.Encoding, ...]]:
    registry = _registry_for_request().encodings
    model_encoding: tiktoken.Encoding | None = None
    try:
        model_encoding = registry.get(encoding_name_for_model(model_name))
    except KeyError:
        pass
    fallback = [
        encoding
        for name in ("cl100k_base", "o200k_base")
        if (encoding := registry.get(name)) is not None and encoding is not model_encoding
    ]
    return model_encoding, tuple(fallback)


class ContextTokenEstimator:
    """Estimate one request without claiming provider billing precision."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model_encoding, self._fallback_encodings = _select_encodings(model_name)

    async def estimate_async(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
    ) -> int:
        return await run_cpu_task(self.estimate, messages, tools)

    async def estimate_raw_async(
        self,
        messages: Sequence[Message],
        tools: Sequence[Tool] = (),
    ) -> int:
        return await run_cpu_task(self.estimate_raw, messages, tools)

    def estimate(self, messages: Sequence[Message], tools: Sequence[Tool] = ()) -> int:
        return self.apply_safety(self.estimate_raw(messages, tools))

    def estimate_raw(self, messages: Sequence[Message], tools: Sequence[Tool] = ()) -> int:
        raw = 3
        for message in messages:
            raw += self.estimate_message_raw(message)
        for tool in tools:
            raw += TOOL_OVERHEAD_TOKENS
            raw += self.count_text(tool.name)
            raw += self.count_text(tool.description)
            raw += self.count_json(tool.input_schema)
        return raw

    def estimate_message_raw(self, message: Message) -> int:
        raw = MESSAGE_OVERHEAD_TOKENS
        raw += self.count_text(message.role.value)
        raw += self.count_text(message.tool_call_id)
        if message.content_parts:
            for part in message.content_parts:
                raw += (
                    IMAGE_ESTIMATE_TOKENS
                    if part.part_type is ContentPartType.IMAGE
                    else self.count_text(part.text)
                )
        else:
            raw += self.count_text(message.content)
        for tool_call in message.tool_calls:
            raw += self.count_text(tool_call.tool_call_id)
            raw += self.count_text(tool_call.name)
            raw += self.count_json(tool_call.arguments)
        return raw

    @classmethod
    def apply_safety(cls, raw_tokens: int) -> int:
        return math.ceil(raw_tokens * (1 + ESTIMATE_SAFETY_RATIO))

    def count_text(self, value: str) -> int:
        if not value:
            return 0
        if self._model_encoding is not None:
            return len(self._model_encoding.encode(value, disallowed_special=()))
        if self._fallback_encodings:
            return max(
                len(encoding.encode(value, disallowed_special=()))
                for encoding in self._fallback_encodings
            )
        return max(math.ceil(len(value.encode("utf-8")) / 2), 1)

    def count_json(self, value: Any) -> int:
        return self.count_text(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
