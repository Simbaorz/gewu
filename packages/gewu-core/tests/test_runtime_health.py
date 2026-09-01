"""Runtime lag monitoring and readiness behavior."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

from gewu_core.file_tasks import FileTaskLane
from gewu_core.runtime_health import EventLoopLagMonitor, process_readiness


async def test_event_loop_monitor_records_a_blocking_delay() -> None:
    """Expose a synchronous event-loop stall as current and maximum lag."""
    recorded: list[float] = []
    monitor = EventLoopLagMonitor(interval_seconds=0.01, record_lag=recorded.append)
    monitor.start()
    try:
        await asyncio.sleep(0)
        asyncio.get_running_loop().call_soon(time.sleep, 0.04)
        await asyncio.sleep(0.06)

        stats = monitor.snapshot()

        assert stats.running is True
        assert stats.sample_count >= 1
        assert stats.max_seconds >= 0.02
        assert recorded
        assert max(recorded) == stats.max_seconds
    finally:
        await monitor.stop()
    assert monitor.snapshot().running is False


def test_process_readiness_requires_startup_and_sustained_file_capacity(
    monkeypatch,
) -> None:
    """Keep transient load ready while marking a persistently full lane unready."""
    monkeypatch.setattr(
        "gewu_core.runtime_health.file_task_stats",
        lambda: {
            FileTaskLane.INTERACTIVE: SimpleNamespace(saturated_seconds=59.9),
            FileTaskLane.BULK: SimpleNamespace(saturated_seconds=60.0),
        },
    )

    not_started = process_readiness(
        started=False,
        filesystem_saturation_unready_seconds=60.0,
    )
    saturated = process_readiness(
        started=True,
        filesystem_saturation_unready_seconds=60.0,
    )

    assert not_started.ready is False
    assert not_started.reasons == (
        "process_not_started",
        "filesystem_bulk_saturated",
    )
    assert saturated.ready is False
    assert saturated.reasons == ("filesystem_bulk_saturated",)
