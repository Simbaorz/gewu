"""FastAPI lifespan integration for one explicit process runtime."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from gewu_core.logging import shutdown_logging
from gewu_core.runtime_health import RuntimeReadinessSnapshot


class HttpProcessRuntime(Protocol):
    async def startup(self) -> None: ...

    async def shutdown(self) -> None: ...

    def readiness_snapshot(self) -> RuntimeReadinessSnapshot: ...


def create_lifespan(
    runtime: HttpProcessRuntime,
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """Create lifecycle management for one explicitly constructed runtime."""

    @asynccontextmanager
    async def runtime_lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            await runtime.startup()
            app.state.runtime = runtime
            yield
        finally:
            try:
                await runtime.shutdown()
            finally:
                shutdown_logging()

    return runtime_lifespan
