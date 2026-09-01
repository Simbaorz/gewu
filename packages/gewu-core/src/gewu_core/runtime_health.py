"""Event-loop health sampling and process readiness evaluation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict

from gewu_core.file_tasks import file_task_stats


class EventLoopLagStats(BaseModel):
    """Immutable event-loop heartbeat snapshot."""

    model_config = ConfigDict(frozen=True)

    running: bool
    sample_count: int
    current_seconds: float
    max_seconds: float


class RuntimeReadinessSnapshot(BaseModel):
    """Minimal readiness result safe to expose through a process probe."""

    model_config = ConfigDict(frozen=True)

    ready: bool
    reasons: tuple[str, ...] = ()


class EventLoopLagMonitor:
    """Measure scheduling delay without adding work to request paths."""

    def __init__(
        self,
        interval_seconds: float = 1.0,
        *,
        record_lag: Callable[[float], None] | None = None,
    ) -> None:
        """Initialize a process-local heartbeat interval and optional metric sink."""
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self.interval_seconds = interval_seconds
        self._record_lag = record_lag
        self._task: asyncio.Task[None] | None = None
        self._sample_count = 0
        self._current_seconds = 0.0
        self._max_seconds = 0.0

    def start(self) -> None:
        """Start sampling on the current event loop once."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop sampling and consume monitor cancellation."""
        task = self._task
        self._task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    def snapshot(self) -> EventLoopLagStats:
        """Return the latest and process-maximum observed lag."""
        return EventLoopLagStats(
            running=self._task is not None and not self._task.done(),
            sample_count=self._sample_count,
            current_seconds=self._current_seconds,
            max_seconds=self._max_seconds,
        )

    async def _run(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            deadline = loop.time() + self.interval_seconds
            await asyncio.sleep(self.interval_seconds)
            lag_seconds = max(0.0, loop.time() - deadline)
            self._sample_count += 1
            self._current_seconds = lag_seconds
            self._max_seconds = max(self._max_seconds, lag_seconds)
            if self._record_lag is not None:
                self._record_lag(lag_seconds)


def process_readiness(
    *,
    started: bool,
    filesystem_saturation_unready_seconds: float,
) -> RuntimeReadinessSnapshot:
    """Evaluate startup state and sustained filesystem capacity loss."""
    reasons: list[str] = []
    if not started:
        reasons.append("process_not_started")
    for lane, stats in file_task_stats().items():
        if stats.saturated_seconds >= filesystem_saturation_unready_seconds:
            reasons.append(f"filesystem_{lane.value}_saturated")
    return RuntimeReadinessSnapshot(ready=not reasons, reasons=tuple(reasons))
