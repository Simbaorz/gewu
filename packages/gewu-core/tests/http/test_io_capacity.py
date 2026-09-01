"""Upload and download lifecycle capacity tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException, Request
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gewu_core.http import downloads as download_module
from gewu_core.http.downloads import BoundedFileResponse, BoundedMediaResponse
from gewu_core.http.io_capacity import (
    DownloadEgressController,
    DownloadEgressMiddleware,
    UploadIngressMiddleware,
    get_upload_ingress_controller,
)
from gewu_core.http.request_limits import buffered_limited_request_body
from gewu_core.http.settings import (
    DownloadEgressSettings,
    HttpBlockingIoSettings,
    HttpInfrastructureSettings,
    UploadIngressSettings,
)


async def test_upload_capacity_is_bounded_by_process_total() -> None:
    first_receive_started = asyncio.Event()
    release_first_body = asyncio.Event()

    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        if _header(scope, b"x-request-id") == b"first":
            first_receive_started.set()
            await release_first_body.wait()
        await receive()
        await JSONResponse({"ok": True})(scope, receive, send)

    app = _app(upload=_upload_settings(max_active=1, queue_capacity=0))
    middleware = UploadIngressMiddleware(endpoint, _select_request)
    first = asyncio.create_task(
        _call_asgi(
            middleware,
            _scope(app, method="PUT", path="/api/workspace/upload", request_id="first"),
            body=b"data",
        )
    )
    await first_receive_started.wait()

    second = await _call_asgi(
        middleware,
        _scope(app, method="PUT", path="/api/workspace/upload"),
        body=b"data",
    )

    assert _status(second) == 503
    release_first_body.set()
    assert _status(await first) == 200


async def test_upload_ingress_releases_when_body_finishes_before_service_work() -> None:
    first_body_consumed = asyncio.Event()
    release_first_service = asyncio.Event()

    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        await receive()
        if _header(scope, b"x-request-id") == b"first":
            first_body_consumed.set()
            await release_first_service.wait()
        await JSONResponse({"ok": True})(scope, receive, send)

    app = _app(upload=_upload_settings(max_active=1, queue_capacity=0))
    middleware = UploadIngressMiddleware(endpoint, _select_request)
    first = asyncio.create_task(
        _call_asgi(
            middleware,
            _scope(app, method="PUT", path="/api/workspace/upload", request_id="first"),
            body=b"data",
        )
    )
    await first_body_consumed.wait()

    second = await _call_asgi(
        middleware,
        _scope(app, method="PUT", path="/api/workspace/upload", request_id="second"),
        body=b"data",
    )

    assert _status(second) == 200
    assert not first.done()
    release_first_service.set()
    assert _status(await first) == 200


async def test_upload_body_timeout_returns_408_and_releases_capacity() -> None:
    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        await receive()
        await JSONResponse({"ok": True})(scope, receive, send)

    app = _app(
        upload=_upload_settings(
            max_active=1,
            queue_capacity=0,
            body_timeout_seconds=0.01,
        )
    )
    middleware = UploadIngressMiddleware(endpoint, _select_request)

    timed_out = await _call_asgi(
        middleware,
        _scope(app, method="PUT", path="/api/workspace/upload"),
        body=b"data",
        receive_delay_seconds=1,
    )
    completed = await _call_asgi(
        middleware,
        _scope(app, method="PUT", path="/api/workspace/upload"),
        body=b"data",
    )

    assert _status(timed_out) == 408
    assert _status(completed) == 200
    stats = await get_upload_ingress_controller(
        _scope(app, method="PUT", path="/api/workspace/upload")
    ).receive_limiter.snapshot()
    assert stats.active == 0
    assert stats.queued == 0


async def test_buffered_upload_bytes_stay_reserved_until_service_finishes() -> None:
    app = _app(
        upload=_upload_settings(
            max_active=1,
            queue_capacity=0,
            max_buffered_file_bytes=6,
            admission_timeout_seconds=0.01,
        )
    )
    first = _request_with_body(app, b"abc")
    second = _request_with_body(app, b"def")

    async with buffered_limited_request_body(first, 4) as body:
        assert body == b"abc"
        controller = get_upload_ingress_controller(first.scope)
        assert controller.buffer_limiter.used == 4
        with pytest.raises(HTTPException) as error:
            async with buffered_limited_request_body(second, 4):
                raise AssertionError("capacity rejection should happen before entering")
        assert error.value.status_code == 503
        assert controller.buffer_limiter.used == 4

    assert controller.buffer_limiter.used == 0


async def test_egress_capacity_is_held_until_final_body_send_completes() -> None:
    final_send_started = asyncio.Event()
    release_final_send = asyncio.Event()

    async def endpoint(scope: Scope, receive: Receive, send: Send) -> None:
        await Response(content=b"data")(scope, receive, send)

    app = _app(download=_download_settings(max_egress=1, max_open_files=1, queue_capacity=0))
    middleware = DownloadEgressMiddleware(endpoint, _select_request)
    first = asyncio.create_task(
        _call_asgi(
            middleware,
            _scope(app, path="/api/workspace/download"),
            block_final_send=(final_send_started, release_final_send),
        )
    )
    await final_send_started.wait()

    second = await _call_asgi(middleware, _scope(app, path="/api/workspace/download"))

    assert _status(second) == 503
    release_final_send.set()
    assert _status(await first) == 200


async def test_file_send_does_not_hold_interactive_execution_and_fd_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "download.bin"
    path.write_bytes(b"x" * (BoundedFileResponse.chunk_size + 1))
    app = _app(download=_download_settings(max_egress=2, max_open_files=1, queue_capacity=0))
    controller = DownloadEgressController(
        _download_settings(max_egress=2, max_open_files=1, queue_capacity=0)
    )
    app.state.download_egress_controller = controller
    active_file_tasks = 0
    original_run_file_task = download_module.run_file_task

    async def tracked_run_file_task(*args, **kwargs):
        nonlocal active_file_tasks
        active_file_tasks += 1
        try:
            return await original_run_file_task(*args, **kwargs)
        finally:
            active_file_tasks -= 1

    monkeypatch.setattr(download_module, "run_file_task", tracked_run_file_task)
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    first_response = BoundedFileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        stat_result=path.stat(),
    )
    first = asyncio.create_task(
        _call_asgi(
            first_response,
            _scope(app, path="/api/workspace/download"),
            block_nonempty_body_send=(send_started, release_send),
        )
    )
    await send_started.wait()

    assert active_file_tasks == 0
    assert (await controller.open_file_limiter.snapshot()).active == 1
    second_response = BoundedFileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        stat_result=path.stat(),
    )
    second = await _call_asgi(second_response, _scope(app, path="/api/workspace/download"))
    assert _status(second) == 503

    release_send.set()
    assert _status(await first) == 200
    assert (await controller.open_file_limiter.snapshot()).active == 0


async def test_bounded_file_response_preserves_single_range_protocol(tmp_path: Path) -> None:
    path = tmp_path / "range.bin"
    path.write_bytes(b"0123456789")
    app = _app(download=_download_settings(max_egress=1, max_open_files=1, queue_capacity=0))
    response = BoundedFileResponse(
        path,
        filename=path.name,
        media_type="application/octet-stream",
        stat_result=path.stat(),
    )

    messages = await _call_asgi(
        response,
        _scope(
            app,
            path="/api/workspace/download",
            headers=[(b"range", b"bytes=2-5")],
        ),
    )

    assert _status(messages) == 206
    assert _response_header(messages, b"content-range") == b"bytes 2-5/10"
    assert _response_body(messages) == b"2345"


async def test_temporary_download_cleanup_runs_when_client_send_fails(tmp_path: Path) -> None:
    path = tmp_path / "archive.zip"
    path.write_bytes(b"archive")
    app = _app(download=_download_settings(max_egress=1, max_open_files=1, queue_capacity=0))
    response = BoundedFileResponse(
        path,
        filename=path.name,
        media_type="application/zip",
        stat_result=path.stat(),
        background=BackgroundTask(path.unlink),
    )

    async def receive() -> Message:
        return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.body":
            raise ConnectionError("client disconnected")

    with pytest.raises(ConnectionError, match="client disconnected"):
        await response(_scope(app, path="/api/workspace/download"), receive, send)

    assert not path.exists()


async def test_media_bytes_remain_reserved_until_client_send_finishes() -> None:
    app = _app(
        download=_download_settings(
            max_egress=2,
            max_open_files=2,
            queue_capacity=0,
            max_media_bytes=6,
            admission_timeout_seconds=0.01,
        )
    )
    controller = DownloadEgressController(
        _download_settings(
            max_egress=2,
            max_open_files=2,
            queue_capacity=0,
            max_media_bytes=6,
            admission_timeout_seconds=0.01,
        )
    )
    app.state.download_egress_controller = controller
    send_started = asyncio.Event()
    release_send = asyncio.Event()
    first_response = BoundedMediaResponse(
        loader=lambda: _loaded_bytes(b"data"),
        reservation_size_bytes=4,
        media_type="image/png",
    )
    first = asyncio.create_task(
        _call_asgi(
            first_response,
            _scope(app, path="/api/chat/attachments/A1"),
            block_final_send=(send_started, release_send),
        )
    )
    await send_started.wait()

    assert controller.media_bytes_limiter.used == 4
    second_loader_called = False

    async def second_loader() -> bytes:
        nonlocal second_loader_called
        second_loader_called = True
        return b"more"

    second = await _call_asgi(
        BoundedMediaResponse(
            loader=second_loader,
            reservation_size_bytes=4,
            media_type="image/png",
        ),
        _scope(app, path="/api/chat/attachments/A2"),
    )

    assert _status(second) == 503
    assert not second_loader_called
    assert controller.media_bytes_limiter.used == 4
    release_send.set()
    assert _status(await first) == 200
    assert controller.media_bytes_limiter.used == 0


def _app(
    *,
    upload: UploadIngressSettings | None = None,
    download: DownloadEgressSettings | None = None,
) -> FastAPI:
    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        settings=HttpInfrastructureSettings(
            blocking_io=HttpBlockingIoSettings(
                upload=upload or UploadIngressSettings(),
                download=download or DownloadEgressSettings(),
            )
        )
    )
    return app


def _upload_settings(
    *,
    max_active: int,
    queue_capacity: int,
    body_timeout_seconds: float = 1,
    max_buffered_file_bytes: int = 1024,
    admission_timeout_seconds: float = 0.05,
) -> UploadIngressSettings:
    return UploadIngressSettings(
        max_concurrent_ingress=max_active,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout_seconds,
        body_timeout_seconds=body_timeout_seconds,
        max_buffered_file_bytes=max_buffered_file_bytes,
    )


def _download_settings(
    *,
    max_egress: int,
    max_open_files: int,
    queue_capacity: int,
    max_media_bytes: int = 1024,
    admission_timeout_seconds: float = 0.05,
) -> DownloadEgressSettings:
    return DownloadEgressSettings(
        max_concurrent_egress=max_egress,
        max_open_files=max_open_files,
        max_media_bytes_in_flight=max_media_bytes,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout_seconds,
    )


async def _loaded_bytes(data: bytes) -> bytes:
    return data


def _select_request(_scope: Scope) -> bool:
    return True


async def _call_asgi(
    app: ASGIApp,
    scope: Scope,
    *,
    body: bytes = b"",
    receive_delay_seconds: float = 0,
    block_final_send: tuple[asyncio.Event, asyncio.Event] | None = None,
    block_nonempty_body_send: tuple[asyncio.Event, asyncio.Event] | None = None,
) -> list[Message]:
    sent: list[Message] = []
    consumed = False
    blocked_final = False
    blocked_body = False

    async def receive() -> Message:
        nonlocal consumed
        if receive_delay_seconds:
            await asyncio.sleep(receive_delay_seconds)
        if consumed:
            return {"type": "http.disconnect"}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message: Message) -> None:
        nonlocal blocked_body, blocked_final
        sent.append(message)
        if (
            block_nonempty_body_send is not None
            and not blocked_body
            and message["type"] == "http.response.body"
            and message.get("body", b"")
        ):
            blocked_body = True
            block_nonempty_body_send[0].set()
            await block_nonempty_body_send[1].wait()
        if (
            block_final_send is not None
            and not blocked_final
            and message["type"] == "http.response.body"
            and not message.get("more_body", False)
        ):
            blocked_final = True
            block_final_send[0].set()
            await block_final_send[1].wait()

    await app(scope, receive, send)
    return sent


def _scope(
    app: FastAPI,
    *,
    path: str,
    method: str = "GET",
    request_id: str = "request",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> Scope:
    return {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers or [(b"x-request-id", request_id.encode())],
        "client": ("127.0.0.1", 1),
        "server": ("testserver", 80),
        "app": app,
    }


def _request_with_body(app: FastAPI, body: bytes) -> Request:
    consumed = False

    async def receive() -> Message:
        nonlocal consumed
        if consumed:
            return {"type": "http.disconnect"}
        consumed = True
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(_scope(app, method="PUT", path="/api/workspace/upload"), receive)


def _status(messages: list[Message]) -> int:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return int(start["status"])


def _response_header(messages: list[Message], name: bytes) -> bytes:
    start = next(message for message in messages if message["type"] == "http.response.start")
    return bytes(dict(start["headers"])[name])


def _response_body(messages: list[Message]) -> bytes:
    return b"".join(
        bytes(message.get("body", b""))
        for message in messages
        if message["type"] == "http.response.body"
    )


def _header(scope: Scope, name: bytes) -> bytes:
    return bytes(dict(scope.get("headers", ()))[name])
