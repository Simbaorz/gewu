"""Shared bounded file and directory download responses."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Awaitable, Callable, Mapping
from pathlib import Path
from secrets import token_hex
from typing import BinaryIO

from fastapi import HTTPException
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse, Response
from starlette.types import Receive, Scope, Send

from gewu_core import AsyncAdmissionCapacityExceededError, WeightedCapacityExceededError
from gewu_core.archive import (
    DirectoryArchiveTooLargeError,
    create_directory_archive,
    delete_temp_file_async,
)
from gewu_core.file_tasks import FileTaskLane, run_file_task
from gewu_core.http.io_capacity import get_download_egress_controller


async def build_path_download_response(
    target_path: Path,
    *,
    fallback_root_name: str,
    max_archive_bytes: int,
    archive_too_large_detail: str,
) -> FileResponse:
    is_directory, stat_result = await run_file_task(
        _path_info,
        target_path,
        lane=FileTaskLane.INTERACTIVE,
    )
    if not is_directory:
        return BoundedFileResponse(
            target_path,
            filename=target_path.name,
            media_type="application/octet-stream",
            stat_result=stat_result,
        )
    try:
        archive_path = await create_directory_archive(
            target_path,
            fallback_root_name=fallback_root_name,
            max_bytes=max_archive_bytes,
        )
    except DirectoryArchiveTooLargeError:
        raise HTTPException(status_code=413, detail=archive_too_large_detail) from None
    archive_stat = await run_file_task(archive_path.stat, lane=FileTaskLane.INTERACTIVE)
    return BoundedFileResponse(
        archive_path,
        filename=f"{target_path.name}.zip",
        media_type="application/zip",
        stat_result=archive_stat,
        background=BackgroundTask(delete_temp_file_async, archive_path),
    )


class BoundedFileResponse(FileResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        background = self.background
        self.background = None
        try:
            if scope["type"] == "http" and str(scope.get("method") or "").upper() == "HEAD":
                await super().__call__(scope, receive, send)
                return
            controller = get_download_egress_controller(scope)
            try:
                lease = await controller.acquire_open_file()
            except AsyncAdmissionCapacityExceededError:
                response = JSONResponse(
                    status_code=503,
                    content={"detail": "Download file capacity is busy. Please retry later."},
                    headers={"Retry-After": "1"},
                )
                await response(scope, receive, send)
                return
            async with lease:
                await super().__call__(scope, receive, send)
        finally:
            if background is not None:
                await background()

    async def _handle_simple(
        self,
        send: Send,
        send_header_only: bool,
        send_pathsend: bool,
    ) -> None:
        del send_pathsend
        await send(
            {"type": "http.response.start", "status": self.status_code, "headers": self.raw_headers}
        )
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        file = await self._open_file()
        try:
            more_body = True
            while more_body:
                chunk = await self._read_file(file, self.chunk_size)
                more_body = len(chunk) == self.chunk_size
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
        finally:
            await self._close_file(file)

    async def _handle_single_range(
        self,
        send: Send,
        start: int,
        end: int,
        file_size: int,
        send_header_only: bool,
    ) -> None:
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-range"] = f"bytes {start}-{end - 1}/{file_size}"
        headers["content-length"] = str(end - start)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        file = await self._open_file()
        try:
            await self._seek_file(file, start)
            more_body = True
            while more_body:
                chunk = await self._read_file(file, min(self.chunk_size, end - start))
                start += len(chunk)
                more_body = len(chunk) == self.chunk_size and start < end
                await send({"type": "http.response.body", "body": chunk, "more_body": more_body})
        finally:
            await self._close_file(file)

    async def _handle_multiple_ranges(
        self,
        send: Send,
        ranges: list[tuple[int, int]],
        file_size: int,
        send_header_only: bool,
    ) -> None:
        boundary = token_hex(13)
        content_length, header_generator = self.generate_multipart(
            ranges,
            boundary,
            file_size,
            self.headers["content-type"],
        )
        headers = MutableHeaders(raw=list(self.raw_headers))
        headers["content-type"] = f"multipart/byteranges; boundary={boundary}"
        headers["content-length"] = str(content_length)
        await send({"type": "http.response.start", "status": 206, "headers": headers.raw})
        if send_header_only:
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return
        file = await self._open_file()
        try:
            for start, end in ranges:
                await send(
                    {
                        "type": "http.response.body",
                        "body": header_generator(start, end),
                        "more_body": True,
                    }
                )
                await self._seek_file(file, start)
                while start < end:
                    chunk = await self._read_file(file, min(self.chunk_size, end - start))
                    start += len(chunk)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
                await send({"type": "http.response.body", "body": b"\r\n", "more_body": True})
            await send(
                {
                    "type": "http.response.body",
                    "body": f"--{boundary}--".encode("latin-1"),
                    "more_body": False,
                }
            )
        finally:
            await self._close_file(file)

    async def _open_file(self) -> BinaryIO:
        return await run_file_task(
            _open_binary_file,
            Path(self.path),
            lane=FileTaskLane.INTERACTIVE,
            wait_on_cancel=True,
        )

    async def _read_file(self, file: BinaryIO, size: int) -> bytes:
        return await run_file_task(
            file.read,
            size,
            lane=FileTaskLane.INTERACTIVE,
            wait_on_cancel=True,
        )

    async def _seek_file(self, file: BinaryIO, offset: int) -> None:
        await run_file_task(
            file.seek,
            offset,
            lane=FileTaskLane.INTERACTIVE,
            wait_on_cancel=True,
        )

    async def _close_file(self, file: BinaryIO) -> None:
        await run_file_task(
            file.close,
            lane=FileTaskLane.INTERACTIVE,
            wait_on_cancel=True,
        )


class BoundedMediaResponse(Response):
    """Load media only after reserving bytes and hold them through socket send."""

    def __init__(
        self,
        *,
        loader: Callable[[], Awaitable[bytes]],
        reservation_size_bytes: int,
        media_type: str,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(content=b"", media_type=media_type, headers=headers)
        self._loader = loader
        self._reservation_size_bytes = reservation_size_bytes
        self._response_headers = dict(headers or {})

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        controller = get_download_egress_controller(scope)
        try:
            reservation = await controller.media_bytes_limiter.acquire(self._reservation_size_bytes)
        except WeightedCapacityExceededError:
            capacity_response = JSONResponse(
                status_code=503,
                content={"detail": "Download processing is busy. Please retry later."},
                headers={"Retry-After": "1"},
            )
            await capacity_response(scope, receive, send)
            return
        try:
            data = await self._loader()
            response = Response(
                content=data,
                media_type=self.media_type,
                headers=self._response_headers,
            )
            await response(scope, receive, send)
        finally:
            await asyncio.shield(reservation.release())


def _path_info(path: Path) -> tuple[bool, os.stat_result]:
    stat_result = path.stat()
    return stat.S_ISDIR(stat_result.st_mode), stat_result


def _open_binary_file(path: Path) -> BinaryIO:
    return path.open("rb")
