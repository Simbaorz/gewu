"""Bound complete HTTP upload and download lifecycles."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from starlette.datastructures import State
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gewu_core import (
    AsyncAdmissionCapacityExceededError,
    FairAsyncCapacityLimiter,
    WeightedCapacityLimiter,
)
from gewu_core.http.settings import (
    DownloadEgressSettings,
    HttpInfrastructureSettings,
    UploadIngressSettings,
)

UPLOAD_CONTROLLER_STATE = "upload_ingress_controller"
DOWNLOAD_CONTROLLER_STATE = "download_egress_controller"
RequestSelector = Callable[[Scope], bool]


class UploadIngressController:
    def __init__(self, settings: UploadIngressSettings) -> None:
        self.settings = settings
        self.receive_limiter = FairAsyncCapacityLimiter(
            capacity_name="Upload ingress",
            max_active=settings.max_concurrent_ingress,
            queue_capacity=settings.queue_capacity,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )
        self.buffer_limiter = WeightedCapacityLimiter(
            name="Upload buffered bytes",
            capacity=settings.max_buffered_file_bytes,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )

    async def acquire_receive(self) -> AbstractAsyncContextManager[None]:
        return await self.receive_limiter.acquire()


class DownloadEgressController:
    def __init__(self, settings: DownloadEgressSettings) -> None:
        self.settings = settings
        self.egress_limiter = FairAsyncCapacityLimiter(
            capacity_name="Download egress",
            max_active=settings.max_concurrent_egress,
            queue_capacity=settings.queue_capacity,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )
        self.open_file_limiter = FairAsyncCapacityLimiter(
            capacity_name="Download open files",
            max_active=settings.max_open_files,
            queue_capacity=settings.queue_capacity,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )
        self.media_bytes_limiter = WeightedCapacityLimiter(
            name="Download media bytes",
            capacity=settings.max_media_bytes_in_flight,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )

    async def acquire_egress(self) -> AbstractAsyncContextManager[None]:
        return await self.egress_limiter.acquire()

    async def acquire_open_file(self) -> AbstractAsyncContextManager[None]:
        return await self.open_file_limiter.acquire()


def get_upload_ingress_controller(scope: Scope) -> UploadIngressController:
    state = _app_state(scope)
    controller = getattr(state, UPLOAD_CONTROLLER_STATE, None)
    if isinstance(controller, UploadIngressController):
        return controller
    controller = UploadIngressController(_upload_settings(scope))
    if state is not None:
        setattr(state, UPLOAD_CONTROLLER_STATE, controller)
    return controller


def get_download_egress_controller(scope: Scope) -> DownloadEgressController:
    state = _app_state(scope)
    controller = getattr(state, DOWNLOAD_CONTROLLER_STATE, None)
    if isinstance(controller, DownloadEgressController):
        return controller
    controller = DownloadEgressController(_download_settings(scope))
    if state is not None:
        setattr(state, DOWNLOAD_CONTROLLER_STATE, controller)
    return controller


class UploadIngressMiddleware:
    def __init__(self, app: ASGIApp, request_selector: RequestSelector) -> None:
        self.app = app
        self._request_selector = request_selector

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._request_selector(scope):
            await self.app(scope, receive, send)
            return
        controller = get_upload_ingress_controller(scope)
        try:
            lease = await controller.acquire_receive()
        except AsyncAdmissionCapacityExceededError:
            await _send_json_error(
                scope,
                receive,
                send,
                status_code=503,
                detail="Upload processing is busy. Please retry later.",
                headers={"Retry-After": "1"},
            )
            return
        await self._call_with_deadline(
            scope,
            receive,
            send,
            lease,
            controller.settings.body_timeout_seconds,
        )

    async def _call_with_deadline(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        lease: AbstractAsyncContextManager[None],
        timeout_seconds: float,
    ) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_seconds
        body_complete = False
        timed_out = False
        released = False
        pending_messages: list[Message] = []

        async def release() -> None:
            nonlocal released
            if not released:
                released = True
                await lease.__aexit__(None, None, None)

        async def limited_receive() -> Message:
            nonlocal body_complete, timed_out
            remaining = deadline - loop.time()
            if remaining <= 0:
                timed_out = True
                body_complete = True
                await release()
                return {"type": "http.disconnect"}
            try:
                async with asyncio.timeout(remaining):
                    message = await receive()
            except TimeoutError:
                timed_out = True
                body_complete = True
                await release()
                return {"type": "http.disconnect"}
            if message["type"] == "http.disconnect":
                body_complete = True
                await release()
            elif message["type"] == "http.request" and not message.get("more_body", False):
                body_complete = True
                await release()
                for pending in pending_messages:
                    await send(pending)
                pending_messages.clear()
            return message

        async def limited_send(message: Message) -> None:
            if timed_out:
                return
            if body_complete:
                await send(message)
            else:
                pending_messages.append(message)

        await lease.__aenter__()
        try:
            try:
                await self.app(scope, limited_receive, limited_send)
            except Exception:
                if not timed_out:
                    raise
            if timed_out:
                await _send_json_error(
                    scope,
                    receive,
                    send,
                    status_code=408,
                    detail="Upload request body timed out.",
                )
            else:
                for pending in pending_messages:
                    await send(pending)
        finally:
            await release()


class DownloadEgressMiddleware:
    def __init__(self, app: ASGIApp, request_selector: RequestSelector) -> None:
        self.app = app
        self._request_selector = request_selector

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if not self._request_selector(scope):
            await self.app(scope, receive, send)
            return
        controller = get_download_egress_controller(scope)
        try:
            lease = await controller.acquire_egress()
        except AsyncAdmissionCapacityExceededError:
            await _send_download_capacity_error(scope, receive, send)
            return
        released = False

        async def release() -> None:
            nonlocal released
            if not released:
                released = True
                await lease.__aexit__(None, None, None)

        async def limited_send(message: Message) -> None:
            await send(message)
            if message["type"] == "http.response.pathsend" or (
                message["type"] == "http.response.body" and not message.get("more_body", False)
            ):
                await release()

        await lease.__aenter__()
        try:
            await self.app(scope, receive, limited_send)
        finally:
            await release()


async def _send_download_capacity_error(
    scope: Scope,
    receive: Receive,
    send: Send,
) -> None:
    await _send_json_error(
        scope,
        receive,
        send,
        status_code=503,
        detail="Download processing is busy. Please retry later.",
        headers={"Retry-After": "1"},
    )


async def _send_json_error(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    status_code: int,
    detail: str,
    headers: dict[str, str] | None = None,
) -> None:
    response = JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)
    await response(scope, receive, send)


def _app_state(scope: Scope) -> State | None:
    state = getattr(scope.get("app"), "state", None)
    return state if isinstance(state, State) else None


def _runtime_settings(scope: Scope) -> HttpInfrastructureSettings | None:
    state = _app_state(scope)
    runtime = getattr(state, "runtime", None)
    settings = getattr(runtime, "settings", None)
    return settings if isinstance(settings, HttpInfrastructureSettings) else None


def _upload_settings(scope: Scope) -> UploadIngressSettings:
    blocking = getattr(_runtime_settings(scope), "blocking_io", None)
    settings = getattr(blocking, "upload", None)
    return settings if isinstance(settings, UploadIngressSettings) else UploadIngressSettings()


def _download_settings(scope: Scope) -> DownloadEgressSettings:
    blocking = getattr(_runtime_settings(scope), "blocking_io", None)
    settings = getattr(blocking, "download", None)
    return settings if isinstance(settings, DownloadEgressSettings) else DownloadEgressSettings()
