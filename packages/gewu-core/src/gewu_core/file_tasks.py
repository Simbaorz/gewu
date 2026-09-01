"""Bounded execution lanes for blocking filesystem workloads."""

from __future__ import annotations

import threading
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol

from pydantic import Field

from gewu_core.blocking import (
    BlockingTaskCapacityExceededError,
    BlockingTaskRunner,
    BlockingTaskRunnerStats,
)
from gewu_core.config import SettingsModel

MAX_CONCURRENT_FILE_TASKS = 4
MAX_QUEUED_FILE_TASKS = 16
DEFAULT_INTERACTIVE_FILE_TASKS = 8
DEFAULT_INTERACTIVE_QUEUED_TASKS = 32
DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS = 2.0
DEFAULT_FILE_TASK_WARN_SECONDS = 2.0


class FileTaskLane(StrEnum):
    """Filesystem workload classes with isolated process capacity."""

    INTERACTIVE = "interactive"
    BULK = "bulk"


class FileTaskCapacityExceededError(BlockingTaskCapacityExceededError):
    """Raised when a bounded filesystem lane cannot admit more work."""


class FileTaskLaneSettings(SettingsModel):
    """Capacity and admission settings for one filesystem lane."""

    max_workers: int = Field(ge=1)
    queue_capacity: int = Field(ge=0)
    admission_timeout_seconds: float = Field(gt=0)


class FileTaskSettings(SettingsModel):
    """Independent interactive and bulk filesystem execution settings."""

    interactive: FileTaskLaneSettings = Field(
        default_factory=lambda: FileTaskLaneSettings(
            max_workers=DEFAULT_INTERACTIVE_FILE_TASKS,
            queue_capacity=DEFAULT_INTERACTIVE_QUEUED_TASKS,
            admission_timeout_seconds=DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
        )
    )
    bulk: FileTaskLaneSettings = Field(
        default_factory=lambda: FileTaskLaneSettings(
            max_workers=MAX_CONCURRENT_FILE_TASKS,
            queue_capacity=MAX_QUEUED_FILE_TASKS,
            admission_timeout_seconds=DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
        )
    )
    execution_warn_seconds: float = Field(default=DEFAULT_FILE_TASK_WARN_SECONDS, gt=0)
    saturation_unready_seconds: float = Field(default=60.0, gt=0)


class FileTaskLaneConf(Protocol):
    @property
    def max_workers(self) -> int: ...

    @property
    def queue_capacity(self) -> int: ...

    @property
    def admission_timeout_seconds(self) -> float: ...


class FilesystemTaskConf(Protocol):
    @property
    def interactive(self) -> FileTaskLaneConf: ...

    @property
    def bulk(self) -> FileTaskLaneConf: ...

    @property
    def execution_warn_seconds(self) -> float: ...


def _runner(
    *,
    lane: FileTaskLane,
    max_workers: int,
    queue_capacity: int,
    admission_timeout_seconds: float,
    execution_warn_seconds: float,
) -> BlockingTaskRunner:
    return BlockingTaskRunner(
        name=f"file-{lane.value}",
        max_workers=max_workers,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout_seconds,
        execution_warn_seconds=execution_warn_seconds,
        capacity_error_type=FileTaskCapacityExceededError,
        capacity_name=f"Filesystem file-{lane.value}",
    )


_RUNNERS_LOCK = threading.Lock()
_RUNNER_CONFIG: tuple[object, ...] = (
    DEFAULT_INTERACTIVE_FILE_TASKS,
    DEFAULT_INTERACTIVE_QUEUED_TASKS,
    DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
    MAX_CONCURRENT_FILE_TASKS,
    MAX_QUEUED_FILE_TASKS,
    DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
    DEFAULT_FILE_TASK_WARN_SECONDS,
)
_FILE_TASK_RUNNERS = {
    FileTaskLane.INTERACTIVE: _runner(
        lane=FileTaskLane.INTERACTIVE,
        max_workers=DEFAULT_INTERACTIVE_FILE_TASKS,
        queue_capacity=DEFAULT_INTERACTIVE_QUEUED_TASKS,
        admission_timeout_seconds=DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
        execution_warn_seconds=DEFAULT_FILE_TASK_WARN_SECONDS,
    ),
    FileTaskLane.BULK: _runner(
        lane=FileTaskLane.BULK,
        max_workers=MAX_CONCURRENT_FILE_TASKS,
        queue_capacity=MAX_QUEUED_FILE_TASKS,
        admission_timeout_seconds=DEFAULT_FILE_TASK_ADMISSION_TIMEOUT_SECONDS,
        execution_warn_seconds=DEFAULT_FILE_TASK_WARN_SECONDS,
    ),
}


def configure_file_task_runners(conf: FilesystemTaskConf) -> None:
    """Apply startup lane sizing without replacing busy executors."""

    global _FILE_TASK_RUNNERS, _RUNNER_CONFIG
    config = (
        conf.interactive.max_workers,
        conf.interactive.queue_capacity,
        conf.interactive.admission_timeout_seconds,
        conf.bulk.max_workers,
        conf.bulk.queue_capacity,
        conf.bulk.admission_timeout_seconds,
        conf.execution_warn_seconds,
    )
    with _RUNNERS_LOCK:
        if config == _RUNNER_CONFIG:
            return
        replacements = {
            FileTaskLane.INTERACTIVE: _runner(
                lane=FileTaskLane.INTERACTIVE,
                max_workers=conf.interactive.max_workers,
                queue_capacity=conf.interactive.queue_capacity,
                admission_timeout_seconds=conf.interactive.admission_timeout_seconds,
                execution_warn_seconds=conf.execution_warn_seconds,
            ),
            FileTaskLane.BULK: _runner(
                lane=FileTaskLane.BULK,
                max_workers=conf.bulk.max_workers,
                queue_capacity=conf.bulk.queue_capacity,
                admission_timeout_seconds=conf.bulk.admission_timeout_seconds,
                execution_warn_seconds=conf.execution_warn_seconds,
            ),
        }
        try:
            for runner in _FILE_TASK_RUNNERS.values():
                runner.shutdown()
        except BaseException:
            for runner in replacements.values():
                runner.shutdown()
            raise
        _FILE_TASK_RUNNERS = replacements
        _RUNNER_CONFIG = config


async def run_file_task[T](
    operation: Callable[..., T],
    /,
    *args: Any,
    lane: FileTaskLane = FileTaskLane.BULK,
    cancel_result_cleanup: Callable[[T], None] | None = None,
    wait_on_cancel: bool = False,
    **kwargs: Any,
) -> T:
    """Run one blocking filesystem operation in the selected bounded lane."""

    return await _FILE_TASK_RUNNERS[lane].run(
        operation,
        *args,
        cancel_result_cleanup=cancel_result_cleanup,
        wait_on_cancel=wait_on_cancel,
        **kwargs,
    )


async def run_file_mutation[T](
    operation: Callable[..., T],
    /,
    *args: Any,
    lane: FileTaskLane = FileTaskLane.BULK,
    **kwargs: Any,
) -> T:
    """Keep the caller attached until a physical mutation completes."""

    return await run_file_task(
        operation,
        *args,
        lane=lane,
        wait_on_cancel=True,
        **kwargs,
    )


def file_task_stats() -> dict[FileTaskLane, BlockingTaskRunnerStats]:
    """Return process-local snapshots for both filesystem lanes."""

    return {lane: runner.snapshot() for lane, runner in _FILE_TASK_RUNNERS.items()}
