"""Process liveness and sustained-capacity readiness probes."""

from __future__ import annotations

from typing import Protocol, cast

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gewu_core.runtime_health import RuntimeReadinessSnapshot


class _ReadinessProvider(Protocol):
    def readiness_snapshot(self) -> RuntimeReadinessSnapshot: ...


router = APIRouter(include_in_schema=False)


@router.get("/healthz")
async def health() -> dict[str, str]:
    """Report that the HTTP process and event loop can answer requests."""
    return {"status": "ok"}


@router.get("/readyz")
async def readiness(request: Request) -> JSONResponse:
    """Reject traffic before startup or after sustained file capacity loss."""
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None or not hasattr(runtime, "readiness_snapshot"):
        snapshot = RuntimeReadinessSnapshot(
            ready=False,
            reasons=("process_not_started",),
        )
    else:
        snapshot = cast(_ReadinessProvider, runtime).readiness_snapshot()
    return JSONResponse(
        status_code=200 if snapshot.ready else 503,
        content={
            "status": "ready" if snapshot.ready else "not_ready",
            "reasons": list(snapshot.reasons),
        },
    )
