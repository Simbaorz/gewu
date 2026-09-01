"""Shared HTTP contract behavior."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gewu_core.blocking import BlockingTaskCapacityExceededError
from gewu_core.errors import ApplicationError, ApplicationErrorKind
from gewu_core.file_tasks import FileTaskCapacityExceededError
from gewu_core.http.app_factory import create_base_http_app
from gewu_core.http.application_errors import application_error_status_code
from gewu_core.redis import RedisCapacityExceededError
from gewu_core.runtime_health import RuntimeReadinessSnapshot


@asynccontextmanager
async def _empty_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


class _ReadyRuntime:
    def readiness_snapshot(self) -> RuntimeReadinessSnapshot:
        return RuntimeReadinessSnapshot(ready=True)


def test_base_app_preserves_public_metadata_and_hidden_health_routes() -> None:
    app = create_base_http_app(_empty_lifespan)

    assert app.title == "Gewu Service"
    assert app.description == "Gewu HTTP service"
    assert app.version == "0.1.0"
    assert app.openapi()["paths"] == {}


def test_health_and_readiness_do_not_report_unstarted_process_ready() -> None:
    app = create_base_http_app(_empty_lifespan)
    with TestClient(app) as client:
        health = client.get("/healthz")
        unstarted = client.get("/readyz")
        app.state.runtime = _ReadyRuntime()
        ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert unstarted.status_code == 503
    assert unstarted.json() == {
        "status": "not_ready",
        "reasons": ["process_not_started"],
    }
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "reasons": []}


@pytest.mark.parametrize(
    ("error_type", "detail"),
    [
        (FileTaskCapacityExceededError, "File processing is busy. Please retry later."),
        (BlockingTaskCapacityExceededError, "Server processing is busy. Please retry later."),
        (RedisCapacityExceededError, "Server state service is busy. Please retry later."),
    ],
)
def test_capacity_errors_map_to_retryable_service_unavailable(
    error_type: type[Exception],
    detail: str,
) -> None:
    app = create_base_http_app(_empty_lifespan)

    @app.get("/capacity")
    async def capacity() -> None:
        raise error_type("capacity")

    with TestClient(app) as client:
        response = client.get("/capacity")

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert response.json() == {"detail": detail}


@pytest.mark.parametrize(
    ("kind", "status_code"),
    [
        (ApplicationErrorKind.BAD_REQUEST, 400),
        (ApplicationErrorKind.UNAUTHENTICATED, 401),
        (ApplicationErrorKind.FORBIDDEN, 403),
        (ApplicationErrorKind.NOT_FOUND, 404),
        (ApplicationErrorKind.CONFLICT, 409),
        (ApplicationErrorKind.PAYLOAD_TOO_LARGE, 413),
        (ApplicationErrorKind.INVALID_INPUT, 422),
        (ApplicationErrorKind.INTERNAL, 500),
        (ApplicationErrorKind.UNAVAILABLE, 503),
        (ApplicationErrorKind.TIMEOUT, 504),
    ],
)
def test_application_error_status_mapping(
    kind: ApplicationErrorKind,
    status_code: int,
) -> None:
    error = ApplicationError(kind, "expected failure")
    assert application_error_status_code(error) == status_code


def test_application_error_handler_preserves_public_detail() -> None:
    app = create_base_http_app(_empty_lifespan)

    @app.get("/application-error")
    async def application_error() -> None:
        raise ApplicationError(ApplicationErrorKind.UNAVAILABLE, "authorization is too large")

    with TestClient(app) as client:
        response = client.get("/application-error")

    assert response.status_code == 503
    assert response.json() == {"detail": "authorization is too large"}


def test_api_datetime_fields_are_returned_as_local_wall_clock_strings() -> None:
    app = create_base_http_app(_empty_lifespan, timezone_name="Asia/Shanghai")

    @app.get("/datetime")
    async def datetime_payload() -> dict[str, object]:
        return {
            "create_time": "2026-07-31T01:02:03Z",
            "nested": [{"updated_at": "2026-07-31T01:02:03+00:00"}],
            "unchanged": "2026-07-31T01:02:03Z",
        }

    with TestClient(app) as client:
        response = client.get("/datetime")

    assert response.json() == {
        "create_time": "2026-07-31 09:02:03",
        "nested": [{"updated_at": "2026-07-31 09:02:03"}],
        "unchanged": "2026-07-31T01:02:03Z",
    }
