"""Compensation guards for filesystem mutations paired with durable metadata."""

from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path
from typing import Any

from gewu_core.errors import CommitOutcomeUnknownError
from gewu_core.file_locks import filesystem_mutation_lock
from gewu_core.file_tasks import run_file_mutation, run_file_task
from gewu_core.filesystem import remove_path
from gewu_core.ids import new_entity_id
from gewu_core.runtime_temp import runtime_temp_subdir

logger = logging.getLogger(__name__)


@asynccontextmanager
async def restore_directory_on_error(
    target: Path,
    *,
    cleanup_paths: Sequence[Path] = (),
    mutation_lock: AbstractAsyncContextManager[object] | None = None,
) -> AsyncIterator[None]:
    """Restore a directory snapshot when a later metadata mutation fails.

    An unknown database COMMIT keeps the new filesystem state because rolling it
    back could remove content already referenced by durable metadata.
    """

    async with (
        filesystem_mutation_lock((target, *cleanup_paths)),
        _optional_mutation_lock(mutation_lock),
    ):
        rollback_parent = await run_file_task(runtime_temp_subdir, "file-rollback")
        rollback_root = Path(
            await run_file_mutation(
                tempfile.mkdtemp,
                prefix="gewu-file-rollback-",
                dir=rollback_parent,
            )
        )
        backup = rollback_root / "directory"
        try:
            existed_before, is_directory, is_symlink = await run_file_task(
                _path_state,
                target,
            )
            cleanup_created: list[Path] = []
            for path in cleanup_paths:
                if path != target and not await run_file_task(_path_exists, path):
                    cleanup_created.append(path)
            if existed_before:
                if not is_directory or is_symlink:
                    raise NotADirectoryError(target)
                await run_file_mutation(shutil.copytree, target, backup, symlinks=True)
            try:
                yield
            except CommitOutcomeUnknownError:
                raise
            except BaseException:
                try:
                    await _run_file_task_critical(
                        _restore_directory,
                        target,
                        backup,
                        existed_before,
                        tuple(cleanup_created),
                    )
                except Exception as exc:
                    logger.error(
                        "Unable to restore filesystem directory: %s exception_type=%s",
                        target,
                        type(exc).__name__,
                    )
                raise
        finally:
            try:
                await _run_file_task_critical(remove_path, rollback_root)
            except Exception as exc:
                logger.error(
                    "Unable to remove filesystem rollback directory: %s exception_type=%s",
                    rollback_root,
                    type(exc).__name__,
                )


@asynccontextmanager
async def remove_directory_created_on_error(
    target: Path,
    *,
    mutation_lock: AbstractAsyncContextManager[object] | None = None,
) -> AsyncIterator[None]:
    """Remove a newly created directory when its metadata mutation fails."""

    async with (
        filesystem_mutation_lock((target,)),
        _optional_mutation_lock(mutation_lock),
    ):
        existed_before, is_directory, is_symlink = await run_file_task(_path_state, target)
        if existed_before and (not is_directory or is_symlink):
            raise NotADirectoryError(target)
        try:
            yield
        except CommitOutcomeUnknownError:
            raise
        except BaseException:
            if not existed_before and await run_file_task(_path_exists, target):
                try:
                    await _run_file_task_critical(remove_path, target)
                except Exception as exc:
                    logger.error(
                        "Unable to remove failed filesystem directory: %s exception_type=%s",
                        target,
                        type(exc).__name__,
                    )
            raise


@asynccontextmanager
async def quarantine_path_until_success(
    target: Path | None,
    *,
    mutation_lock: AbstractAsyncContextManager[object] | None = None,
) -> AsyncIterator[None]:
    """Hide a path until metadata deletion succeeds, restoring it on failure."""

    if target is None:
        yield
        return
    async with (
        filesystem_mutation_lock((target,)),
        _optional_mutation_lock(mutation_lock),
    ):
        if not await run_file_task(_path_exists, target):
            yield
            return
        quarantine = target.with_name(f".{target.name}.delete-{new_entity_id()}")
        try:
            await run_file_task(target.replace, quarantine, wait_on_cancel=True)
        except asyncio.CancelledError:
            try:
                if await _run_file_task_critical(_path_exists, quarantine):
                    await _run_file_task_critical(
                        _restore_quarantined_path,
                        target,
                        quarantine,
                    )
            except Exception as exc:
                logger.error(
                    "Unable to restore filesystem path after quarantine cancellation: %s "
                    "exception_type=%s",
                    target,
                    type(exc).__name__,
                )
            raise
        try:
            yield
        except CommitOutcomeUnknownError:
            raise
        except BaseException:
            try:
                await _run_file_task_critical(
                    _restore_quarantined_path,
                    target,
                    quarantine,
                )
            except Exception as exc:
                logger.error(
                    "Unable to restore quarantined path: %s exception_type=%s",
                    target,
                    type(exc).__name__,
                )
            raise
        else:
            try:
                await _run_file_task_critical(remove_path, quarantine)
            except Exception as exc:
                logger.warning(
                    "Unable to remove committed filesystem quarantine: %s exception_type=%s",
                    quarantine,
                    type(exc).__name__,
                )


@asynccontextmanager
async def _optional_mutation_lock(
    mutation_lock: AbstractAsyncContextManager[object] | None,
) -> AsyncIterator[None]:
    if mutation_lock is None:
        yield
        return
    async with mutation_lock:
        yield


async def _run_file_task_critical[ResultT](
    operation: Callable[..., ResultT],
    /,
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    """Finish one short compensation task despite repeated cancellation."""

    return await _finish_critical(
        run_file_task(
            operation,
            *args,
            wait_on_cancel=True,
            **kwargs,
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


def _restore_directory(
    target: Path,
    backup: Path,
    existed_before: bool,
    cleanup_paths: tuple[Path, ...],
) -> None:
    for path in cleanup_paths:
        remove_path(path)
    remove_path(target)
    if existed_before:
        shutil.copytree(backup, target, symlinks=True)


def _restore_quarantined_path(target: Path, quarantine: Path) -> None:
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite concurrently created path: {target}")
    quarantine.replace(target)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _path_state(path: Path) -> tuple[bool, bool, bool]:
    is_symlink = path.is_symlink()
    return path.exists() or is_symlink, path.is_dir(), is_symlink
