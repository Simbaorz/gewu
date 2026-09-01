"""Task-reentrant cross-process filesystem locks."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from gewu_core import file_locks
from gewu_core.file_locks import TaskReentrantFileLock
from gewu_core.runtime_temp import set_runtime_temp_root_provider


async def test_file_lock_serializes_independent_coordinators(tmp_path: Path) -> None:
    lock_path = tmp_path / "shared.lock"
    first = TaskReentrantFileLock("first")
    second = TaskReentrantFileLock("second")
    acquired = asyncio.Event()

    async with first.async_lock(lock_path):
        waiter = asyncio.create_task(_acquire_and_signal(second, lock_path, acquired))
        await asyncio.sleep(0.05)
        assert acquired.is_set() is False

    await asyncio.wait_for(waiter, timeout=0.5)
    assert acquired.is_set() is True


async def test_file_lock_is_reentrant_for_owning_task(tmp_path: Path) -> None:
    coordinator = TaskReentrantFileLock("reentrant")
    lock_path = tmp_path / "shared.lock"

    async with coordinator.async_lock(lock_path):
        async with coordinator.async_lock(lock_path):
            pass


async def test_filesystem_mutation_lock_does_not_use_default_executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop = asyncio.get_running_loop()
    original_run_in_executor = loop.run_in_executor

    def reject_default_executor(executor, operation, *args):
        if executor is None:
            raise AssertionError("filesystem lock used asyncio's default executor")
        return original_run_in_executor(executor, operation, *args)

    monkeypatch.setattr(loop, "run_in_executor", reject_default_executor)
    previous = set_runtime_temp_root_provider(lambda: tmp_path / ".runtime-temp")
    try:
        async with file_locks.filesystem_mutation_lock((tmp_path / "asset",)):
            pass
    finally:
        set_runtime_temp_root_provider(previous)


async def test_bounded_file_lock_releases_late_acquire_after_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    acquire_started = threading.Event()
    finish_acquire = threading.Event()
    released = threading.Event()

    def delayed_acquire(_lock) -> bool:
        acquire_started.set()
        finish_acquire.wait(timeout=2)
        return True

    def record_release(_lock, acquired: bool) -> None:
        if acquired:
            released.set()

    monkeypatch.setattr(file_locks, "_try_acquire_file_lock", delayed_acquire)
    monkeypatch.setattr(file_locks, "_release_file_lock_if_acquired", record_release)

    async def hold_lock() -> None:
        async with file_locks._bounded_file_lock(tmp_path / "asset.lock"):
            pytest.fail("cancelled lock waiter entered its critical section")

    task = asyncio.create_task(hold_lock())
    assert await asyncio.to_thread(acquire_started.wait, 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    finish_acquire.set()

    assert await asyncio.to_thread(released.wait, 2)


async def _acquire_and_signal(
    coordinator: TaskReentrantFileLock,
    lock_path: Path,
    acquired: asyncio.Event,
) -> None:
    async with coordinator.async_lock(lock_path):
        acquired.set()
