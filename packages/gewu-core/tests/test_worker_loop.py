"""Process-owned async loop behavior for synchronous workers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from gewu_core import WorkerAsyncLoop
from gewu_core import worker_loop as worker_loop_module


def test_worker_loop_reuses_one_event_loop_across_jobs() -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop")

    async def identify() -> tuple[int, int]:
        return id(asyncio.get_running_loop()), threading.get_ident()

    try:
        first = worker_loop.run(identify())
        second = worker_loop.run(identify())
    finally:
        worker_loop.shutdown()

    assert first == second
    assert first[1] != threading.get_ident()


def test_worker_loop_propagates_job_failure_and_remains_reusable() -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop-failure")

    async def fail() -> None:
        raise RuntimeError("injected failure")

    async def succeed() -> str:
        return "ok"

    try:
        try:
            worker_loop.run(fail())
        except RuntimeError as exc:
            assert str(exc) == "injected failure"
        else:  # pragma: no cover
            raise AssertionError("Worker failure was not propagated.")
        assert worker_loop.run(succeed()) == "ok"
    finally:
        worker_loop.shutdown()


def test_worker_loop_shutdown_is_idempotent_and_can_restart() -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop-shutdown")

    async def current_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = worker_loop.run(current_loop())
    worker_loop.shutdown()
    worker_loop.shutdown()
    second = worker_loop.run(current_loop())
    worker_loop.shutdown()

    assert first is not second


def test_worker_loop_cancels_coroutine_when_sync_caller_is_interrupted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop-cancel")
    started = threading.Event()
    cancelled = threading.Event()
    original_result = worker_loop_module._WorkerSubmission.result
    result_calls = 0

    def interrupted_result(submission: Any) -> Any:
        nonlocal result_calls
        result_calls += 1
        if result_calls == 1:
            assert started.wait(timeout=1)
            raise KeyboardInterrupt
        return original_result(submission)

    monkeypatch.setattr(worker_loop_module._WorkerSubmission, "result", interrupted_result)

    async def wait_forever() -> None:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    try:
        with pytest.raises(KeyboardInterrupt):
            worker_loop.run(wait_forever())
    finally:
        worker_loop.shutdown()

    assert cancelled.is_set()
    assert result_calls == 2


def test_worker_loop_shutdown_cancels_and_drains_background_tasks() -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop-background")
    cancelled = threading.Event()

    async def background() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    async def spawn_background() -> None:
        asyncio.create_task(background())
        await asyncio.sleep(0)

    worker_loop.run(spawn_background())
    worker_loop.shutdown()

    assert cancelled.is_set()


def test_worker_loop_reset_after_fork_discards_inherited_state() -> None:
    worker_loop = WorkerAsyncLoop(thread_name="test-worker-loop-reset")

    async def current_loop() -> asyncio.AbstractEventLoop:
        return asyncio.get_running_loop()

    first = worker_loop.run(current_loop())
    worker_loop.shutdown()
    worker_loop.reset_after_fork()
    try:
        second = worker_loop.run(current_loop())
    finally:
        worker_loop.shutdown()

    assert first is not second
