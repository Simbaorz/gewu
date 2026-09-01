"""Turn-scoped image payload and capacity behavior."""

from __future__ import annotations

import asyncio
import base64
import time

import pytest

from gewu_agent_runtime import AttachmentRef, ImageCapacityExceededError, ImagePayloadManager
from gewu_agent_runtime import media as media_module
from gewu_agent_runtime.llm import ContentPartType


class RecordingAttachmentLoader:
    """In-memory authorized attachment loader with controllable reads."""

    def __init__(self, data: bytes, *, read_gate: asyncio.Event | None = None) -> None:
        self.data = data
        self.read_gate = read_gate
        self.read_started = asyncio.Event()
        self.read_keys: list[str] = []

    async def read(self, resource_key: str) -> bytes:
        self.read_keys.append(resource_key)
        self.read_started.set()
        if self.read_gate is not None:
            await self.read_gate.wait()
        return self.data


def _attachment(*, size_bytes: int = 3) -> AttachmentRef:
    return AttachmentRef(
        attachment_id="image-1",
        resource_key="opaque/image-1",
        original_name="image.png",
        mime_type="image/png",
        size_bytes=size_bytes,
    )


def _manager(
    *,
    raw_capacity_bytes: int = 1024,
    encoded_capacity_bytes: int = 1024,
    provider_payload_capacity_bytes: int = 1024,
) -> ImagePayloadManager:
    return ImagePayloadManager(
        raw_capacity_bytes=raw_capacity_bytes,
        encoded_capacity_bytes=encoded_capacity_bytes,
        provider_payload_capacity_bytes=provider_payload_capacity_bytes,
        admission_timeout_seconds=0.05,
    )


def _assert_released(manager: ImagePayloadManager) -> None:
    assert manager.raw_bytes_used == 0
    assert manager.encoded_bytes_used == 0
    assert manager.provider_payload_bytes_used == 0


async def test_image_payload_is_encoded_once_and_retained_for_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bytes] = []

    def record_encode(data: bytes) -> str:
        calls.append(data)
        return base64.b64encode(data).decode("ascii")

    monkeypatch.setattr(media_module, "_encode_image", record_encode)
    manager = _manager()
    loader = RecordingAttachmentLoader(b"abc")

    async with manager.prepare("inspect", (_attachment(),), loader) as message:
        image = message.content_parts[1]
        assert image.part_type is ContentPartType.IMAGE
        assert image.data == b""
        assert image.base64_data == "YWJj"
        assert image.resource_id == "image-1"
        assert calls == [b"abc"]
        assert loader.read_keys == ["opaque/image-1"]
        assert manager.raw_bytes_used == 0
        assert manager.encoded_bytes_used == 4
        assert manager.provider_payload_bytes_used == 141

    _assert_released(manager)


@pytest.mark.parametrize(
    ("encoded_capacity", "provider_capacity"),
    ((3, 1024), (1024, 140)),
)
async def test_retained_capacity_rejects_before_attachment_read(
    encoded_capacity: int,
    provider_capacity: int,
) -> None:
    manager = _manager(
        encoded_capacity_bytes=encoded_capacity,
        provider_payload_capacity_bytes=provider_capacity,
    )
    loader = RecordingAttachmentLoader(b"abc")

    with pytest.raises(ImageCapacityExceededError):
        async with manager.prepare("inspect", (_attachment(),), loader):
            pass

    assert loader.read_keys == []
    _assert_released(manager)


async def test_raw_capacity_rejects_before_attachment_read() -> None:
    manager = _manager(raw_capacity_bytes=2)
    loader = RecordingAttachmentLoader(b"abc")

    with pytest.raises(ImageCapacityExceededError):
        async with manager.prepare("inspect", (_attachment(),), loader):
            pass

    assert loader.read_keys == []
    _assert_released(manager)


async def test_attachment_size_mismatch_releases_all_capacity() -> None:
    manager = _manager()
    loader = RecordingAttachmentLoader(b"abcd")

    with pytest.raises(RuntimeError, match="do not match persisted metadata"):
        async with manager.prepare("inspect", (_attachment(size_bytes=3),), loader):
            pass

    _assert_released(manager)


async def test_downstream_failure_releases_retained_capacity() -> None:
    manager = _manager()
    loader = RecordingAttachmentLoader(b"abc")

    with pytest.raises(RuntimeError, match="downstream failed"):
        async with manager.prepare("inspect", (_attachment(),), loader):
            raise RuntimeError("downstream failed")

    _assert_released(manager)


async def test_attachment_read_cancellation_releases_all_capacity() -> None:
    manager = _manager()
    loader = RecordingAttachmentLoader(b"abc", read_gate=asyncio.Event())

    async def consume() -> None:
        async with manager.prepare("inspect", (_attachment(),), loader):
            pass

    task = asyncio.create_task(consume())
    await loader.read_started.wait()
    assert manager.raw_bytes_used == 3
    assert manager.encoded_bytes_used == 4
    assert manager.provider_payload_bytes_used == 141

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    _assert_released(manager)


async def test_slow_base64_encoding_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def slow_encode(data: bytes) -> str:
        time.sleep(0.05)
        return base64.b64encode(data).decode("ascii")

    monkeypatch.setattr(media_module, "_encode_image", slow_encode)
    manager = _manager()
    loader = RecordingAttachmentLoader(b"abc")
    finished = asyncio.Event()

    async def prepare() -> None:
        async with manager.prepare("inspect", (_attachment(),), loader):
            pass
        finished.set()

    task = asyncio.create_task(prepare())
    heartbeats = 0
    while not finished.is_set():
        await asyncio.sleep(0.005)
        heartbeats += 1
    await task

    assert heartbeats >= 3
    _assert_released(manager)
