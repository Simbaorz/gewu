"""Turn-scoped image loading, encoding and process-memory admission."""

from __future__ import annotations

import asyncio
import base64
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.llm import ContentPart, Message
from gewu_core.blocking import run_cpu_task
from gewu_core.concurrency import (
    WeightedCapacityExceededError,
    WeightedCapacityLimiter,
    WeightedCapacityReservation,
)

_PROVIDER_IMAGE_BLOCK_OVERHEAD_BYTES = 128


class AttachmentRef(BaseModel):
    """Authorized image metadata plus an opaque turn-only loader key."""

    model_config = ConfigDict(frozen=True)

    attachment_id: str = Field(min_length=1, max_length=128)
    resource_key: str = Field(min_length=1, exclude=True)
    original_name: str = ""
    mime_type: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)

    def to_message_payload(self) -> dict[str, object]:
        """Return the public reference safe for conversation persistence."""

        return {
            "attachment_id": self.attachment_id,
            "original_name": self.original_name,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
        }


class AttachmentLoader(Protocol):
    """Read bytes for one already-authorized opaque resource key."""

    async def read(self, resource_key: str) -> bytes:
        """Return exact attachment bytes."""


class ImageCapacityExceededError(RuntimeError):
    """Raised when one image turn cannot enter process memory capacity."""


class ImagePayloadManager:
    """Own process-local raw, encoded and provider-payload image budgets."""

    def __init__(
        self,
        *,
        raw_capacity_bytes: int = 128 * 1024 * 1024,
        encoded_capacity_bytes: int = 256 * 1024 * 1024,
        provider_payload_capacity_bytes: int = 256 * 1024 * 1024,
        admission_timeout_seconds: float = 2.0,
    ) -> None:
        self._raw = WeightedCapacityLimiter(
            name="Agent image raw bytes",
            capacity=raw_capacity_bytes,
            admission_timeout_seconds=admission_timeout_seconds,
        )
        self._encoded = WeightedCapacityLimiter(
            name="Agent image encoded bytes",
            capacity=encoded_capacity_bytes,
            admission_timeout_seconds=admission_timeout_seconds,
        )
        self._provider_payload = WeightedCapacityLimiter(
            name="Agent image provider payload bytes",
            capacity=provider_payload_capacity_bytes,
            admission_timeout_seconds=admission_timeout_seconds,
        )

    @asynccontextmanager
    async def prepare(
        self,
        content: str,
        attachments: Sequence[AttachmentRef],
        loader: AttachmentLoader,
    ) -> AsyncIterator[Message]:
        """Yield one pre-encoded message and release all budgets afterward."""

        if not attachments:
            yield Message.user(content)
            return
        encoded_bytes = sum(_base64_size(value.size_bytes) for value in attachments)
        provider_payload_bytes = encoded_bytes + sum(
            len(value.mime_type) + _PROVIDER_IMAGE_BLOCK_OVERHEAD_BYTES for value in attachments
        )
        reservations: list[WeightedCapacityReservation] = []
        try:
            reservations.append(await self._encoded.acquire(encoded_bytes))
            reservations.append(await self._provider_payload.acquire(provider_payload_bytes))
            parts: list[ContentPart] = []
            if content:
                parts.append(ContentPart.text_part(content))
            for attachment in attachments:
                raw_reservation = await self._raw.acquire(attachment.size_bytes)
                try:
                    data = await loader.read(attachment.resource_key)
                    if len(data) != attachment.size_bytes:
                        raise RuntimeError("Chat attachment bytes do not match persisted metadata.")
                    encoded = await run_cpu_task(_encode_image, data)
                finally:
                    await asyncio.shield(raw_reservation.release())
                data = b""
                parts.append(
                    ContentPart.encoded_image(
                        mime_type=attachment.mime_type,
                        base64_data=encoded,
                        resource_id=attachment.attachment_id,
                        name=attachment.original_name,
                    )
                )
            yield Message.user_parts(parts)
        except WeightedCapacityExceededError as exc:
            raise ImageCapacityExceededError(
                "Agent image processing capacity is exhausted."
            ) from exc
        finally:
            for reservation in reversed(reservations):
                await asyncio.shield(reservation.release())

    @property
    def raw_bytes_used(self) -> int:
        return self._raw.used

    @property
    def encoded_bytes_used(self) -> int:
        return self._encoded.used

    @property
    def provider_payload_bytes_used(self) -> int:
        return self._provider_payload.used


def _base64_size(raw_bytes: int) -> int:
    return 4 * ((raw_bytes + 2) // 3)


def _encode_image(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
