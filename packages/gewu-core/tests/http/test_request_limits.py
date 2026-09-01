"""Bounded ASGI request body regression tests."""

from __future__ import annotations

import asyncio
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from io import BytesIO
from pathlib import Path

import anyio
import pytest
from fastapi import HTTPException, Request, UploadFile

from gewu_core.http.request_limits import (
    read_limited_request_body,
    read_limited_upload_file,
    stream_limited_request_file,
)
from gewu_core.runtime_temp import set_runtime_temp_root_provider


@pytest.fixture(autouse=True)
def _runtime_temp_root(tmp_path: Path) -> Iterator[None]:
    previous = set_runtime_temp_root_provider(lambda: tmp_path)
    try:
        yield
    finally:
        set_runtime_temp_root_provider(previous)


def _request(
    messages: list[dict[str, object]],
    *,
    content_length: int | None = None,
) -> tuple[Request, Callable[[], Awaitable[dict[str, object]]]]:
    pending = iter(messages)

    async def receive() -> dict[str, object]:
        return next(pending)

    headers = [] if content_length is None else [(b"content-length", str(content_length).encode())]
    request = Request(
        {
            "type": "http",
            "method": "PUT",
            "path": "/upload",
            "headers": headers,
        },
        receive,  # type: ignore[arg-type]
    )
    return request, receive


async def test_content_length_rejects_oversize_body_before_receive() -> None:
    request, _ = _request([], content_length=11)

    with pytest.raises(HTTPException) as error:
        await read_limited_request_body(request, 10)

    assert error.value.status_code == 413


async def test_chunked_body_aborts_when_accumulated_limit_is_exceeded() -> None:
    request, _ = _request(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )

    with pytest.raises(HTTPException) as error:
        await read_limited_request_body(request, 5)

    assert error.value.status_code == 413


async def test_multipart_upload_aborts_when_accumulated_limit_is_exceeded() -> None:
    upload = UploadFile(filename="image.png", file=BytesIO(b"abcdef"))

    with pytest.raises(HTTPException) as error:
        await read_limited_upload_file(upload, 5)

    assert error.value.status_code == 413


async def test_package_body_streams_to_file_and_cleans_up() -> None:
    request, _ = _request(
        [
            {"type": "http.request", "body": b"abc", "more_body": True},
            {"type": "http.request", "body": b"def", "more_body": False},
        ]
    )

    async with stream_limited_request_file(
        request,
        6,
        prefix="subscriber-test-upload-",
    ) as path:
        assert await anyio.Path(path).read_bytes() == b"abcdef"
        temporary_path = path

    assert not await anyio.Path(temporary_path).exists()


async def test_package_temp_file_creation_does_not_block_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, _ = _request([{"type": "http.request", "body": b"body", "more_body": False}])
    original_mkstemp = tempfile.mkstemp
    entered = threading.Event()
    release = threading.Event()

    def slow_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        entered.set()
        release.wait(timeout=1)
        return original_mkstemp(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(tempfile, "mkstemp", slow_mkstemp)
    started = time.perf_counter()

    async def consume_upload() -> bytes:
        async with stream_limited_request_file(
            request,
            4,
            prefix="subscriber-test-upload-",
        ) as path:
            return await anyio.Path(path).read_bytes()

    task = asyncio.create_task(consume_upload())
    assert await asyncio.to_thread(entered.wait, 1)
    await asyncio.sleep(0.02)
    heartbeat_elapsed = time.perf_counter() - started
    release.set()

    assert await task == b"body"
    assert heartbeat_elapsed < 0.2
