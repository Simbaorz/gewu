"""Async-friendly cross-process filesystem mutation locks."""

from __future__ import annotations

import asyncio
import hashlib
from asyncio import current_task
from collections.abc import AsyncIterator, Awaitable, Sequence
from contextlib import (
    AbstractAsyncContextManager,
    AbstractContextManager,
    AsyncExitStack,
    asynccontextmanager,
    nullcontext,
)
from contextvars import ContextVar
from pathlib import Path

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from gewu_core.file_tasks import FileTaskLane, run_file_mutation, run_file_task
from gewu_core.runtime_temp import runtime_temp_subdir

_FILE_LOCK_POLL_INTERVAL_SECONDS = 0.05


class TaskReentrantFileLock:
    """Coordinate file locks while allowing nested use by the owning asyncio task."""

    def __init__(self, name: str) -> None:
        self._held_locks: ContextVar[frozenset[tuple[str, object]]] = ContextVar(
            f"{name}_held_file_locks_{id(self)}",
            default=frozenset(),
        )

    def async_lock(self, path: Path) -> AbstractAsyncContextManager[object]:
        """Return an async lock that is reentrant for its owning task."""

        return self._async_lock(path)

    def sync_lock(self, path: Path) -> AbstractContextManager[object]:
        """Return a synchronous lock, or a no-op when this task owns it."""

        owner_key = _current_lock_owner_key(path)
        if owner_key is not None and owner_key in self._held_locks.get():
            return nullcontext()
        return FileLock(path)

    @asynccontextmanager
    async def _async_lock(self, path: Path) -> AsyncIterator[None]:
        owner_key = _current_lock_owner_key(path)
        held_locks = self._held_locks.get()
        if owner_key is not None and owner_key in held_locks:
            yield
            return

        token = None
        if owner_key is not None:
            token = self._held_locks.set(held_locks | {owner_key})
        try:
            async with _bounded_file_lock(path):
                yield
        finally:
            if token is not None:
                self._held_locks.reset(token)


@asynccontextmanager
async def filesystem_mutation_lock(paths: Sequence[Path]) -> AsyncIterator[None]:
    """Serialize mutations to the same physical paths across processes."""

    lock_root = await run_file_task(
        runtime_temp_subdir,
        "file-mutation-locks",
        lane=FileTaskLane.INTERACTIVE,
    )
    lock_paths = await run_file_task(
        _mutation_lock_paths,
        tuple(paths),
        lock_root,
        lane=FileTaskLane.INTERACTIVE,
    )
    async with AsyncExitStack() as stack:  # noqa
        for lock_path in lock_paths:
            await stack.enter_async_context(_bounded_file_lock(lock_path))
        yield


@asynccontextmanager
async def _bounded_file_lock(path: Path) -> AsyncIterator[None]:
    lock = FileLock(path, thread_local=False)
    acquired = False
    try:
        while not acquired:
            acquired = await run_file_task(
                _try_acquire_file_lock,
                lock,
                lane=FileTaskLane.INTERACTIVE,
                cancel_result_cleanup=lambda result: _release_file_lock_if_acquired(lock, result),
            )
            if not acquired:
                await asyncio.sleep(_FILE_LOCK_POLL_INTERVAL_SECONDS)
        yield
    finally:
        if acquired:
            await _finish_critical(
                run_file_mutation(
                    lock.release,
                    force=True,
                    lane=FileTaskLane.INTERACTIVE,
                )
            )


async def _finish_critical[ResultT](awaitable: Awaitable[ResultT]) -> ResultT:
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _try_acquire_file_lock(lock: FileLock) -> bool:
    try:
        lock.acquire(timeout=0)
    except FileLockTimeout:
        return False
    return True


def _release_file_lock_if_acquired(lock: FileLock, acquired: bool) -> None:
    if acquired:
        lock.release(force=True)


def _mutation_lock_paths(paths: tuple[Path, ...], lock_root: Path) -> tuple[Path, ...]:
    lock_names = {
        hashlib.sha256(str(path.resolve(strict=False)).encode("utf-8")).hexdigest()
        for path in paths
    }
    return tuple(lock_root / f"{name}.lock" for name in sorted(lock_names))


def _current_lock_owner_key(lock_path: Path) -> tuple[str, object] | None:
    try:
        task = current_task()
    except RuntimeError:
        task = None
    if task is None:
        return None
    return str(lock_path), task
