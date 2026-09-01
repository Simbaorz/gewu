"""HTTP request-body middleware parity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.types import Message, Scope

from gewu_core.http import (
    HttpInfrastructureSettings,
    HttpIngressSettings,
    HttpRequestBodyLimitMiddleware,
)


def _app() -> FastAPI:
    app = FastAPI()
    app.state.runtime = SimpleNamespace(
        settings=HttpInfrastructureSettings(
            http_ingress=HttpIngressSettings(
                max_concurrent_bodies=32,
            )
        )
    )
    app.add_middleware(
        HttpRequestBodyLimitMiddleware,
        body_limit_resolver=_body_limit,
    )

    @app.post("/api/chat/stream")
    @app.post("/api/items")
    @app.post("/api/admin/items")
    async def json_endpoint(payload: dict[str, object]) -> dict[str, object]:
        return payload

    @app.post("/api/chat/attachments")
    async def multipart_endpoint(request: Request) -> dict[str, int]:
        form = await request.form()
        return {"fields": len(form)}

    @app.put("/api/workspace/upload")
    async def raw_file_endpoint(request: Request) -> dict[str, int]:
        return {"bytes": len(await request.body())}

    return app


def _body_limit(scope: Scope) -> int | None:
    path = str(scope.get("path") or "")
    if path == "/api/admin/items" or path == "/api/chat/attachments":
        return 2048
    if path == "/api/workspace/upload":
        return None
    return 1024 if path.startswith("/api/") else None


def test_chat_body_limit_rejects_declared_size_before_json_parsing() -> None:
    response = TestClient(_app()).post(
        "/api/chat/stream",
        json={"content": "x" * 2048},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "Request body is too large."}


def test_admin_and_user_json_use_separate_limits() -> None:
    body = {"content": "x" * 1500}

    assert TestClient(_app()).post("/api/items", json=body).status_code == 413
    assert TestClient(_app()).post("/api/admin/items", json=body).status_code == 200


@pytest.mark.parametrize("content_length", [None, b"1"])
async def test_actual_json_bytes_reject_missing_or_forged_content_length(
    content_length: bytes | None,
) -> None:
    body = b'{"content":"' + (b"x" * 2048) + b'"}'
    headers = [(b"content-type", b"application/json")]
    if content_length is not None:
        headers.append((b"content-length", content_length))

    response = await _call_asgi(_app(), "/api/items", body, headers=headers)

    assert response["status"] == 413


@pytest.mark.parametrize(
    "path",
    ["/api/chat/stream", "/api/items", "/api/admin/items"],
)
@pytest.mark.parametrize("content_type", [None, b"text/plain"])
async def test_api_body_limit_does_not_trust_content_type(
    path: str,
    content_type: bytes | None,
) -> None:
    body = b'{"content":"' + (b"x" * 4096) + b'"}'
    headers = [] if content_type is None else [(b"content-type", content_type)]

    response = await _call_asgi(_app(), path, body, headers=headers)

    assert response["status"] == 413


async def test_chat_multipart_rejects_before_framework_form_parsing() -> None:
    boundary = b"subscriber-boundary"
    body = (
        b"--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="file"; filename="a.png"\r\n'
        + b"Content-Type: image/png\r\n\r\n"
        + (b"x" * 4096)
        + b"\r\n--"
        + boundary
        + b"--\r\n"
    )

    response = await _call_asgi(
        _app(),
        "/api/chat/attachments",
        body,
        headers=[
            (b"content-type", b"multipart/form-data; boundary=" + boundary),
            (b"content-length", b"1"),
        ],
    )

    assert response["status"] == 413


async def test_raw_upload_stream_is_not_limited_by_small_json_budget() -> None:
    response = await _call_asgi(
        _app(),
        "/api/workspace/upload",
        b"x" * 4096,
        method="PUT",
        headers=[(b"content-type", b"application/json")],
    )

    assert response["status"] == 200


async def test_invalid_content_length_is_rejected_before_receive() -> None:
    response = await _call_asgi(
        _app(),
        "/api/items",
        b"{}",
        headers=[
            (b"content-type", b"application/json"),
            (b"content-length", b"invalid"),
        ],
    )

    assert response["status"] == 400


async def _call_asgi(
    app: FastAPI,
    path: str,
    body: bytes,
    *,
    method: str = "POST",
    headers: list[tuple[bytes, bytes]],
) -> dict[str, object]:
    midpoint = max(1, len(body) // 2)
    messages = iter(
        (
            {"type": "http.request", "body": body[:midpoint], "more_body": True},
            {"type": "http.request", "body": body[midpoint:], "more_body": False},
        )
    )
    sent: list[Message] = []

    async def receive() -> Message:
        try:
            return next(messages)  # type: ignore[return-value]
        except StopIteration:
            return {"type": "http.disconnect"}

    async def send(message: Message) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": headers,
            "client": ("127.0.0.1", 1),
            "server": ("testserver", 80),
            "app": app,
        },
        receive,
        send,
    )
    start = next(message for message in sent if message["type"] == "http.response.start")
    return {"status": start["status"], "messages": sent}
