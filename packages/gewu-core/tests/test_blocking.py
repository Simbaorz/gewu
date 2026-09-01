"""Bounded blocking execution and cancellation behavior."""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import pytest

from gewu_core.blocking import (
    BlockingExecutorSettings,
    BlockingTaskCapacityExceededError,
    BlockingTaskRunner,
    BlockingTaskSettings,
    CpuTaskCapacityExceededError,
    blocking_task_stats,
    configure_blocking_task_runners,
    run_cpu_task,
    run_external_task,
)


def _runner(
    *,
    max_workers: int = 1,
    queue_capacity: int = 1,
    admission_timeout_seconds: float = 0.5,
) -> BlockingTaskRunner:
    return BlockingTaskRunner(
        name="test",
        max_workers=max_workers,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout_seconds,
        execution_warn_seconds=10,
    )


async def test_cancelled_queued_task_never_reaches_the_executor() -> None:
    runner = _runner()
    release = threading.Event()
    started = threading.Event()

    def blocking_operation() -> None:
        started.set()
        release.wait(timeout=1)

    blocker = asyncio.create_task(runner.run(blocking_operation))
    assert await asyncio.to_thread(started.wait, 0.5)
    queued_executed = threading.Event()
    queued = asyncio.create_task(runner.run(queued_executed.set))
    await asyncio.sleep(0)

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    release.set()
    await blocker
    assert not queued_executed.is_set()
    assert runner.snapshot().queued == 0
    runner.shutdown()


async def test_cancelled_running_task_retains_capacity_and_cleans_result() -> None:
    runner = _runner(queue_capacity=0)
    release = threading.Event()
    started = threading.Event()
    cleaned: list[str] = []

    def blocking_operation() -> str:
        started.set()
        release.wait(timeout=1)
        return "artifact"

    task = asyncio.create_task(runner.run(blocking_operation, cancel_result_cleanup=cleaned.append))
    assert await asyncio.to_thread(started.wait, 0.5)
    before = time.perf_counter()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert time.perf_counter() - before < 0.2
    assert runner.snapshot().caller_cancelled_but_running == 1
    with pytest.raises(BlockingTaskCapacityExceededError):
        await runner.run(lambda: None)

    release.set()
    await _wait_until_idle(runner)
    assert cleaned == ["artifact"]
    runner.shutdown()


async def test_cancelled_result_cleanup_failure_hides_exception_body(
    caplog: pytest.LogCaptureFixture,
) -> None:
    runner = _runner(queue_capacity=0)
    release = threading.Event()
    started = threading.Event()

    def blocking_operation() -> str:
        started.set()
        release.wait(timeout=1)
        return "artifact"

    def fail_cleanup(_result: str) -> None:
        raise RuntimeError("blocking-cleanup-private-secret")

    task = asyncio.create_task(runner.run(blocking_operation, cancel_result_cleanup=fail_cleanup))
    assert await asyncio.to_thread(started.wait, 0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    with caplog.at_level(logging.ERROR, logger="gewu_core.blocking"):
        release.set()
        await _wait_until_idle(runner)

    assert "Unable to clean a cancelled blocking-task result" in caplog.text
    assert "exception_type=RuntimeError" in caplog.text
    assert "blocking-cleanup-private-secret" not in caplog.text
    runner.shutdown()


async def test_wait_queue_is_bounded_and_times_out_without_leaking() -> None:
    runner = _runner(queue_capacity=1, admission_timeout_seconds=0.02)
    release = threading.Event()
    started = threading.Event()

    def blocking_operation() -> None:
        started.set()
        release.wait(timeout=1)

    blocker = asyncio.create_task(runner.run(blocking_operation))
    assert await asyncio.to_thread(started.wait, 0.5)
    with pytest.raises(BlockingTaskCapacityExceededError, match="timed out"):
        await runner.run(lambda: None)
    assert runner.snapshot().queued == 0
    assert runner.snapshot().admission_timeout_total == 1
    release.set()
    await blocker
    runner.shutdown()


async def test_wait_on_cancel_keeps_caller_attached_until_mutation_finishes() -> None:
    runner = _runner(queue_capacity=0)
    release = threading.Event()
    started = threading.Event()

    def blocking_mutation() -> None:
        started.set()
        release.wait(timeout=1)

    task = asyncio.create_task(runner.run(blocking_mutation, wait_on_cancel=True))
    assert await asyncio.to_thread(started.wait, 0.5)
    task.cancel()
    await asyncio.sleep(0)
    assert task.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert runner.snapshot().running == 0
    runner.shutdown()


async def test_cpu_and_external_global_lanes_have_independent_capacity() -> None:
    constrained = BlockingTaskSettings(
        cpu=BlockingExecutorSettings(
            max_workers=1,
            queue_capacity=0,
            admission_timeout_seconds=0.02,
            execution_warn_seconds=10,
        ),
        external=BlockingExecutorSettings(
            max_workers=1,
            queue_capacity=0,
            admission_timeout_seconds=0.02,
            execution_warn_seconds=10,
        ),
    )
    configure_blocking_task_runners(constrained)
    release = threading.Event()
    started = threading.Event()

    def stalled_cpu_operation() -> None:
        started.set()
        release.wait(timeout=1)

    try:
        cpu_task = asyncio.create_task(run_cpu_task(stalled_cpu_operation))
        assert await asyncio.to_thread(started.wait, 0.5)
        with pytest.raises(CpuTaskCapacityExceededError):
            await run_cpu_task(lambda: None)
        assert await run_external_task(lambda: "responsive") == "responsive"
        assert blocking_task_stats()["cpu"].queue_full_rejected_total == 1
        release.set()
        await cpu_task
    finally:
        release.set()
        configure_blocking_task_runners(BlockingTaskSettings())


async def _wait_until_idle(runner: BlockingTaskRunner) -> None:
    for _ in range(100):
        if runner.snapshot().running == 0:
            return
        await asyncio.sleep(0.01)
    raise AssertionError("blocking task runner did not become idle")
