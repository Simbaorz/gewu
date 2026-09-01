"""Shared FastAPI construction without selecting application routes."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractAsyncContextManager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from gewu_core.blocking import BlockingTaskCapacityExceededError
from gewu_core.errors import ApplicationError
from gewu_core.file_tasks import FileTaskCapacityExceededError
from gewu_core.http.application_errors import application_error_status_code
from gewu_core.http.health import router as health_router
from gewu_core.http.serialization import response_class_for_timezone
from gewu_core.redis import RedisCapacityExceededError


async def _file_task_capacity_handler(
    _request: Request,
    _error: Exception,
) -> JSONResponse:
    """Return a retryable response when filesystem work capacity is exhausted."""
    return JSONResponse(
        status_code=503,
        content={"detail": "File processing is busy. Please retry later."},
        headers={"Retry-After": "1"},
    )


async def _blocking_task_capacity_handler(
    _request: Request,
    _error: Exception,
) -> JSONResponse:
    """Return a retryable response when blocking-work capacity is exhausted."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Server processing is busy. Please retry later."},
        headers={"Retry-After": "1"},
    )


async def _redis_capacity_handler(
    _request: Request,
    _error: Exception,
) -> JSONResponse:
    """Return a retryable response when the Redis client pool is exhausted."""
    return JSONResponse(
        status_code=503,
        content={"detail": "Server state service is busy. Please retry later."},
        headers={"Retry-After": "1"},
    )


async def _application_error_handler(
    _request: Request,
    error: Exception,
) -> JSONResponse:
    """Map expected application failures into their public response."""
    if not isinstance(error, ApplicationError):
        raise error
    return JSONResponse(
        status_code=application_error_status_code(error),
        content={"detail": error.detail},
    )


def create_base_http_app(
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]],
    *,
    timezone_name: str = "Asia/Shanghai",
    title: str = "Gewu Service",
    description: str = "Gewu HTTP service",
    version: str = "0.1.0",
) -> FastAPI:
    """Create a route-neutral HTTP application shell."""
    app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        default_response_class=response_class_for_timezone(timezone_name),
    )
    app.add_exception_handler(
        BlockingTaskCapacityExceededError,
        _blocking_task_capacity_handler,
    )
    app.add_exception_handler(FileTaskCapacityExceededError, _file_task_capacity_handler)
    app.add_exception_handler(RedisCapacityExceededError, _redis_capacity_handler)
    app.add_exception_handler(ApplicationError, _application_error_handler)
    app.include_router(health_router)
    return app
