"""Bounded ASGI request-body readers for upload endpoints."""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import BinaryIO

from fastapi import HTTPException, Request, UploadFile

from gewu_core import WeightedCapacityExceededError
from gewu_core.file_tasks import FileTaskLane, run_file_task
from gewu_core.http.io_capacity import get_upload_ingress_controller
from gewu_core.runtime_temp import runtime_temp_subdir


def _validate_content_length(request: Request, max_bytes: int) -> None:
    value = request.headers.get("content-length")
    if value is None:
        return
    try:
        content_length = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header.") from exc
    if content_length < 0:
        raise HTTPException(status_code=400, detail="Invalid Content-Length header.")
    if content_length > max_bytes:
        raise HTTPException(status_code=413, detail=f"Request body exceeds {max_bytes} bytes.")


async def read_limited_request_body(request: Request, max_bytes: int) -> bytes:
    _validate_content_length(request, max_bytes)
    body = bytearray()
    async for chunk in request.stream():
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Request body exceeds {max_bytes} bytes.")
        body.extend(chunk)
    return bytes(body)


async def read_limited_upload_file(upload: UploadFile, max_bytes: int) -> bytes:
    body = bytearray()
    while chunk := await upload.read(64 * 1024):
        if len(body) + len(chunk) > max_bytes:
            raise HTTPException(status_code=413, detail=f"Upload exceeds {max_bytes} bytes.")
        body.extend(chunk)
    return bytes(body)


@asynccontextmanager
async def buffered_limited_request_body(
    request: Request,
    max_bytes: int,
) -> AsyncIterator[bytes]:
    controller = get_upload_ingress_controller(request.scope)
    try:
        reservation = await controller.buffer_limiter.acquire(max_bytes)
    except WeightedCapacityExceededError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload buffering is busy. Please retry later.",
            headers={"Retry-After": "1"},
        ) from exc
    async with reservation:
        yield await read_limited_request_body(request, max_bytes)


@asynccontextmanager
async def buffered_limited_upload_file(
    request: Request,
    upload: UploadFile,
    max_bytes: int,
) -> AsyncIterator[bytes]:
    controller = get_upload_ingress_controller(request.scope)
    try:
        reservation = await controller.buffer_limiter.acquire(max_bytes)
    except WeightedCapacityExceededError as exc:
        raise HTTPException(
            status_code=503,
            detail="Upload buffering is busy. Please retry later.",
            headers={"Retry-After": "1"},
        ) from exc
    async with reservation:
        yield await read_limited_upload_file(upload, max_bytes)


@asynccontextmanager
async def stream_limited_request_file(
    request: Request,
    max_bytes: int,
    *,
    prefix: str,
) -> AsyncIterator[Path]:
    _validate_content_length(request, max_bytes)
    path, target = await run_file_task(
        _open_upload_temp_file,
        prefix,
        lane=FileTaskLane.BULK,
        cancel_result_cleanup=_cleanup_open_upload_temp_file,
        wait_on_cancel=True,
    )
    total = 0
    try:
        try:
            async for chunk in request.stream():
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Request body exceeds {max_bytes} bytes.",
                    )
                await run_file_task(
                    target.write,
                    chunk,
                    lane=FileTaskLane.BULK,
                    wait_on_cancel=True,
                )
        finally:
            await run_file_task(target.close, lane=FileTaskLane.BULK, wait_on_cancel=True)
        yield path
    finally:
        await run_file_task(
            path.unlink,
            missing_ok=True,
            lane=FileTaskLane.BULK,
            wait_on_cancel=True,
        )


def _open_upload_temp_file(prefix: str) -> tuple[Path, BinaryIO]:
    file_descriptor, file_name = tempfile.mkstemp(
        prefix=prefix,
        suffix=".zip",
        dir=runtime_temp_subdir("uploads"),
    )
    return Path(file_name), os.fdopen(file_descriptor, "wb")


def _cleanup_open_upload_temp_file(result: tuple[Path, BinaryIO]) -> None:
    path, target = result
    target.close()
    path.unlink(missing_ok=True)
