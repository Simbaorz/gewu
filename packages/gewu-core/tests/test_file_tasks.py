"""Isolated bounded filesystem execution lanes."""

from __future__ import annotations

import asyncio
import threading

import pytest

from gewu_core.file_tasks import (
    FileTaskLane,
    FileTaskLaneSettings,
    FileTaskSettings,
    configure_file_task_runners,
    run_file_mutation,
    run_file_task,
)


def _constrained() -> FileTaskSettings:
    lane = FileTaskLaneSettings(
        max_workers=1,
        queue_capacity=0,
        admission_timeout_seconds=0.05,
    )
    return FileTaskSettings(interactive=lane, bulk=lane, execution_warn_seconds=10)


def test_default_filesystem_lanes_match_subscriber_capacity() -> None:
    settings = FileTaskSettings()

    assert settings.interactive.model_dump() == {
        "max_workers": 8,
        "queue_capacity": 32,
        "admission_timeout_seconds": 2.0,
    }
    assert settings.bulk.model_dump() == {
        "max_workers": 4,
        "queue_capacity": 16,
        "admission_timeout_seconds": 2.0,
    }


async def test_bulk_saturation_does_not_consume_interactive_capacity() -> None:
    configure_file_task_runners(_constrained())
    release = threading.Event()
    started = threading.Event()

    def stalled_bulk_operation() -> None:
        started.set()
        release.wait(timeout=1)

    try:
        bulk = asyncio.create_task(run_file_task(stalled_bulk_operation))
        assert await asyncio.to_thread(started.wait, 0.5)
        assert (
            await run_file_task(lambda: "responsive", lane=FileTaskLane.INTERACTIVE) == "responsive"
        )
        release.set()
        await bulk
    finally:
        release.set()
        configure_file_task_runners(FileTaskSettings())


async def test_file_mutation_cancellation_waits_for_physical_completion() -> None:
    configure_file_task_runners(_constrained())
    release = threading.Event()
    started = threading.Event()

    def stalled_mutation() -> None:
        started.set()
        release.wait(timeout=1)

    try:
        task = asyncio.create_task(
            run_file_mutation(stalled_mutation, lane=FileTaskLane.INTERACTIVE)
        )
        assert await asyncio.to_thread(started.wait, 0.5)
        task.cancel()
        await asyncio.sleep(0)
        assert task.done() is False
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        release.set()
        configure_file_task_runners(FileTaskSettings())
