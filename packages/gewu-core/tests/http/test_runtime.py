"""Shared HTTP process infrastructure behavior."""

from __future__ import annotations

from typing import Any, cast

import pytest
from fastapi import FastAPI

from gewu_core.config import BootstrapSettings
from gewu_core.http.lifecycle import HttpProcessRuntime, create_lifespan
from gewu_core.http.runtime import HttpInfrastructureRuntime
from gewu_core.http.settings import HttpInfrastructureSettings
from gewu_core.logging import shutdown_logging
from gewu_core.runtime_health import RuntimeReadinessSnapshot


async def test_runtime_transitions_readiness_and_event_loop_monitor() -> None:
    runtime = HttpInfrastructureRuntime(
        BootstrapSettings(),
        settings=HttpInfrastructureSettings(),
    )

    assert runtime.readiness_snapshot().reasons == ("process_not_started",)
    try:
        await runtime.startup()
        assert runtime.readiness_snapshot().ready is True
        assert runtime.event_loop_lag_snapshot().running is True
    finally:
        await runtime.shutdown()
        shutdown_logging()

    assert runtime.readiness_snapshot().reasons == ("process_not_started",)
    assert runtime.event_loop_lag_snapshot().running is False


class _FailingStartupRuntime:
    def __init__(self) -> None:
        self.shutdown_called = False

    async def startup(self) -> None:
        raise RuntimeError("startup failed")

    async def shutdown(self) -> None:
        self.shutdown_called = True

    def readiness_snapshot(self) -> RuntimeReadinessSnapshot:
        return RuntimeReadinessSnapshot(ready=False, reasons=("process_not_started",))


async def test_lifespan_cleans_partially_started_runtime_and_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = _FailingStartupRuntime()
    logging_shutdowns: list[None] = []
    monkeypatch.setattr(
        "gewu_core.http.lifecycle.shutdown_logging",
        lambda: logging_shutdowns.append(None),
    )
    lifespan = create_lifespan(cast(HttpProcessRuntime, cast(Any, runtime)))

    with pytest.raises(RuntimeError, match="startup failed"):
        async with lifespan(FastAPI()):
            pass

    assert runtime.shutdown_called is True
    assert logging_shutdowns == [None]
