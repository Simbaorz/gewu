"""ASGI receive limits applied before FastAPI parses request bodies."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from starlette.datastructures import State
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gewu_core import AsyncAdmissionCapacityExceededError, FairAsyncCapacityLimiter
from gewu_core.http.settings import HttpInfrastructureSettings, HttpIngressSettings

_BODY_METHODS = frozenset({"POST", "PUT", "PATCH"})
BodyLimitResolver = Callable[[Scope], int | None]


class HttpRequestBodyLimitMiddleware:
    """Bound configured request bodies before framework parsing."""

    def __init__(self, app: ASGIApp, body_limit_resolver: BodyLimitResolver) -> None:
        self.app = app
        self._body_limit_resolver = body_limit_resolver
        self._limiter: FairAsyncCapacityLimiter | None = None

    def _get_limiter(self, scope: Scope) -> FairAsyncCapacityLimiter:
        if self._limiter is not None:
            return self._limiter
        settings = _http_ingress_settings(scope)
        self._limiter = FairAsyncCapacityLimiter(
            capacity_name="HTTP request body",
            max_active=settings.max_concurrent_bodies,
            queue_capacity=settings.queue_capacity,
            admission_timeout_seconds=settings.admission_timeout_seconds,
        )
        return self._limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        method = str(scope.get("method") or "").upper()
        if method not in _BODY_METHODS:
            await self.app(scope, receive, send)
            return
        limit = self._body_limit_resolver(scope)
        if limit is None:
            await self.app(scope, receive, send)
            return
        try:
            content_length = _content_length(scope)
        except _InvalidContentLength:
            await _reject(scope, receive, send, 400, "Invalid Content-Length header.")
            return
        if content_length is not None and content_length > limit:
            await _reject(scope, receive, send, 413, "Request body is too large.")
            return
        try:
            lease = await self._get_limiter(scope).acquire()
        except AsyncAdmissionCapacityExceededError:
            await _reject(
                scope,
                receive,
                send,
                503,
                "Request body processing is busy. Please retry later.",
                headers={"Retry-After": "1"},
            )
            return
        await self._call_with_limit(scope, receive, send, limit, lease)

    async def _call_with_limit(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
        limit: int,
        lease: AbstractAsyncContextManager[None],
    ) -> None:
        consumed = 0
        released = False
        body_complete = False
        body_too_large = False
        pending_messages: list[Message] = []

        async def release() -> None:
            nonlocal released
            if not released:
                released = True
                await lease.__aexit__(None, None, None)

        async def limited_receive() -> Message:
            nonlocal body_complete, body_too_large, consumed
            message = await receive()
            if message["type"] == "http.request":
                consumed += len(message.get("body", b""))
                if consumed > limit:
                    body_too_large = True
                    body_complete = True
                    pending_messages.clear()
                    await release()
                    return {"type": "http.disconnect"}
                if not message.get("more_body", False):
                    body_complete = True
                    await release()
                    for pending in pending_messages:
                        await send(pending)
                    pending_messages.clear()
            return message

        async def limited_send(message: Message) -> None:
            if body_too_large:
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
                if not body_too_large:
                    raise
            if body_too_large:
                await _reject(scope, receive, send, 413, "Request body is too large.")
            else:
                for pending in pending_messages:
                    await send(pending)
        finally:
            await release()


class _InvalidContentLength(ValueError):
    """Internal signal for malformed Content-Length headers."""


def _content_length(scope: Scope) -> int | None:
    value = _header(scope, b"content-length")
    if not value:
        return None
    try:
        content_length = int(value)
    except ValueError as exc:
        raise _InvalidContentLength from exc
    if content_length < 0:
        raise _InvalidContentLength
    return content_length


def _header(scope: Scope, target: bytes) -> bytes:
    for name, value in scope.get("headers", ()):
        if name.lower() == target:
            return bytes(value)
    return b""


def _runtime_settings(scope: Scope) -> HttpInfrastructureSettings:
    state = _app_state(scope)
    runtime = getattr(state, "runtime", None)
    settings = getattr(runtime, "settings", None)
    return (
        settings
        if isinstance(settings, HttpInfrastructureSettings)
        else HttpInfrastructureSettings()
    )


def _http_ingress_settings(scope: Scope) -> HttpIngressSettings:
    return _runtime_settings(scope).http_ingress


def _app_state(scope: Scope) -> State | None:
    state = getattr(scope.get("app"), "state", None)
    return state if isinstance(state, State) else None


async def _reject(
    scope: Scope,
    receive: Receive,
    send: Send,
    status_code: int,
    detail: str,
    *,
    headers: dict[str, str] | None = None,
) -> None:
    response = JSONResponse(
        status_code=status_code,
        content={"detail": detail},
        headers=headers,
    )
    await response(scope, receive, send)
