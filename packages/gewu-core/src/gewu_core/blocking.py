"""Bounded execution lanes for synchronous infrastructure operations."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from gewu_core.config import SettingsModel

logger = logging.getLogger(__name__)

DEFAULT_CPU_TASK_CONF = (4, 16, 1.0, 1.0)
DEFAULT_EXTERNAL_TASK_CONF = (8, 32, 2.0, 2.0)


class BlockingTaskCapacityExceededError(RuntimeError):
    """Raised when a bounded blocking-work runner cannot admit more work."""


class CpuTaskCapacityExceededError(BlockingTaskCapacityExceededError):
    """Raised when process CPU-task capacity is exhausted."""


class ExternalTaskCapacityExceededError(BlockingTaskCapacityExceededError):
    """Raised when synchronous external-client capacity is exhausted."""


class BlockingExecutorSettings(SettingsModel):
    """Capacity and latency settings for one blocking execution lane."""

    max_workers: int = Field(default=4, ge=1)
    queue_capacity: int = Field(default=16, ge=0)
    admission_timeout_seconds: float = Field(default=1.0, gt=0)
    execution_warn_seconds: float = Field(default=1.0, gt=0)


class BlockingTaskSettings(SettingsModel):
    """Independent CPU and synchronous external-client lanes."""

    cpu: BlockingExecutorSettings = Field(default_factory=BlockingExecutorSettings)
    external: BlockingExecutorSettings = Field(
        default_factory=lambda: BlockingExecutorSettings(
            max_workers=8,
            queue_capacity=32,
            admission_timeout_seconds=2.0,
            execution_warn_seconds=2.0,
        )
    )


class BlockingExecutorTaskConf(Protocol):
    """Values required to configure one blocking executor."""

    @property
    def max_workers(self) -> int: ...

    @property
    def queue_capacity(self) -> int: ...

    @property
    def admission_timeout_seconds(self) -> float: ...

    @property
    def execution_warn_seconds(self) -> float: ...


class BlockingInfrastructureConf(Protocol):
    """CPU and external-client executor configuration."""

    @property
    def cpu(self) -> BlockingExecutorTaskConf: ...

    @property
    def external(self) -> BlockingExecutorTaskConf: ...


class BlockingTaskRunnerStats(BaseModel):
    """Immutable process-local snapshot for one blocking task runner."""

    model_config = ConfigDict(frozen=True)

    capacity: int
    queue_capacity: int
    running: int
    queued: int
    submitted_total: int
    completed_total: int
    failed_total: int
    rejected_total: int
    queue_full_rejected_total: int
    admission_timeout_total: int
    caller_cancelled_total: int
    caller_cancelled_but_running: int
    max_queue_wait_seconds: float
    max_execution_seconds: float
    oldest_running_seconds: float
    oldest_running_operation: str
    saturated_seconds: float


class _AdmissionWaiter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    loop: asyncio.AbstractEventLoop
    future: asyncio.Future[None]
    granted: bool = False


class _CancelledResultCleanup:
    def __init__(self, cleanup: Callable[[Any], None] | None) -> None:
        self._cleanup = cleanup
        self._lock = threading.Lock()
        self._caller_cancelled = False
        self._cleaned = False

    def mark_caller_cancelled(self, future: ConcurrentFuture[Any]) -> None:
        with self._lock:
            self._caller_cancelled = True
        if future.done():
            self.clean_completed_result(future)

    def clean_completed_result(self, future: ConcurrentFuture[Any]) -> None:
        with self._lock:
            if not self._caller_cancelled or self._cleaned or self._cleanup is None:
                return
            if future.cancelled() or future.exception() is not None:
                self._cleaned = True
                return
            self._cleaned = True
            cleanup = self._cleanup
        try:
            cleanup(future.result())
        except Exception as exc:
            logger.error(
                "Unable to clean a cancelled blocking-task result exception_type=%s",
                type(exc).__name__,
            )


class BlockingTaskRunner:
    """Run blocking callables with a bounded, loop-independent wait queue."""

    def __init__(
        self,
        *,
        name: str,
        max_workers: int,
        queue_capacity: int,
        admission_timeout_seconds: float,
        execution_warn_seconds: float,
        capacity_error_type: type[BlockingTaskCapacityExceededError] = (
            BlockingTaskCapacityExceededError
        ),
        capacity_name: str | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must not be negative.")
        if admission_timeout_seconds <= 0:
            raise ValueError("admission_timeout_seconds must be positive.")
        if execution_warn_seconds <= 0:
            raise ValueError("execution_warn_seconds must be positive.")
        self.name = name
        self.max_workers = max_workers
        self.queue_capacity = queue_capacity
        self.admission_timeout_seconds = admission_timeout_seconds
        self.execution_warn_seconds = execution_warn_seconds
        self._capacity_error_type = capacity_error_type
        self._capacity_name = capacity_name or name
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=f"gewu-{name}",
        )
        self._lock = threading.Lock()
        self._running = 0
        self._waiters: deque[_AdmissionWaiter] = deque()
        self._closed = False
        self._submitted_total = 0
        self._completed_total = 0
        self._failed_total = 0
        self._rejected_total = 0
        self._queue_full_rejected_total = 0
        self._admission_timeout_total = 0
        self._caller_cancelled_total = 0
        self._cancelled_running_futures: set[ConcurrentFuture[Any]] = set()
        self._running_futures: dict[ConcurrentFuture[Any], tuple[float, str]] = {}
        self._max_queue_wait_seconds = 0.0
        self._max_execution_seconds = 0.0
        self._saturated_since: float | None = None

    async def run[T](
        self,
        operation: Callable[..., T],
        /,
        *args: Any,
        cancel_result_cleanup: Callable[[T], None] | None = None,
        wait_on_cancel: bool = False,
        **kwargs: Any,
    ) -> T:
        """Admit and execute one callable without blocking its event loop."""

        queue_started = time.perf_counter()
        await self._acquire()
        queue_seconds = time.perf_counter() - queue_started
        execution_started = time.perf_counter()
        operation_name = self._operation_name(operation)
        cleanup_state = _CancelledResultCleanup(cancel_result_cleanup)
        try:
            concurrent_future = self._executor.submit(partial(operation, *args, **kwargs))
        except BaseException:
            self._release()
            raise
        with self._lock:
            self._submitted_total += 1
            self._running_futures[concurrent_future] = (execution_started, operation_name)
            self._max_queue_wait_seconds = max(self._max_queue_wait_seconds, queue_seconds)
        concurrent_future.add_done_callback(
            lambda completed: self._on_completed(
                completed,
                execution_started,
                queue_seconds,
                operation_name,
                cleanup_state,
            )
        )
        future = asyncio.wrap_future(concurrent_future)
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            with self._lock:
                self._caller_cancelled_total += 1
            if wait_on_cancel:
                await self._wait_for_completion_after_cancel(future)
                cleanup_state.mark_caller_cancelled(concurrent_future)
                raise
            cleanup_state.mark_caller_cancelled(concurrent_future)
            with self._lock:
                if not concurrent_future.done():
                    self._cancelled_running_futures.add(concurrent_future)
            concurrent_future.cancel()
            raise

    @classmethod
    async def _wait_for_completion_after_cancel(cls, future: asyncio.Future[Any]) -> None:
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError:
                continue
            except BaseException:
                return

    def snapshot(self) -> BlockingTaskRunnerStats:
        """Return current capacity and cumulative outcome counters."""

        with self._lock:
            now = time.perf_counter()
            oldest_operation = ""
            oldest_running_seconds = 0.0
            if self._running_futures:
                oldest_started, oldest_operation = min(
                    self._running_futures.values(), key=lambda item: item[0]
                )
                oldest_running_seconds = max(0.0, now - oldest_started)
            return BlockingTaskRunnerStats(
                capacity=self.max_workers,
                queue_capacity=self.queue_capacity,
                running=self._running,
                queued=len(self._waiters),
                submitted_total=self._submitted_total,
                completed_total=self._completed_total,
                failed_total=self._failed_total,
                rejected_total=self._rejected_total,
                queue_full_rejected_total=self._queue_full_rejected_total,
                admission_timeout_total=self._admission_timeout_total,
                caller_cancelled_total=self._caller_cancelled_total,
                caller_cancelled_but_running=len(self._cancelled_running_futures),
                max_queue_wait_seconds=self._max_queue_wait_seconds,
                max_execution_seconds=self._max_execution_seconds,
                oldest_running_seconds=oldest_running_seconds,
                oldest_running_operation=oldest_operation,
                saturated_seconds=(
                    max(0.0, now - self._saturated_since)
                    if self._saturated_since is not None
                    else 0.0
                ),
            )

    def shutdown(self) -> None:
        """Close an idle runner during process reconfiguration or shutdown."""

        with self._lock:
            if self._running or self._waiters:
                raise RuntimeError(f"Cannot shut down busy blocking task runner: {self.name}")
            self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _acquire(self) -> None:
        loop = asyncio.get_running_loop()
        with self._lock:
            if self._closed:
                raise RuntimeError(f"Blocking task runner is closed: {self.name}")
            self._discard_cancelled_waiters()
            self._dispatch_locked()
            if self._running < self.max_workers:
                self._grant_locked()
                return
            if len(self._waiters) >= self.queue_capacity:
                self._rejected_total += 1
                self._queue_full_rejected_total += 1
                raise self._capacity_error_locked()
            waiter = _AdmissionWaiter(loop=loop, future=loop.create_future())
            self._waiters.append(waiter)
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter.future), timeout=self.admission_timeout_seconds
            )
        except TimeoutError as exc:
            self._abort_waiter(waiter)
            with self._lock:
                self._rejected_total += 1
                self._admission_timeout_total += 1
                error = self._capacity_error_locked(admission_timeout=True)
            raise error from exc
        except asyncio.CancelledError:
            self._abort_waiter(waiter)
            raise

    def _abort_waiter(self, waiter: _AdmissionWaiter) -> None:
        release_grant = False
        with self._lock:
            if waiter.granted:
                waiter.granted = False
                release_grant = True
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
        if release_grant:
            self._release()

    def _release(self) -> None:
        with self._lock:
            self._release_counts_locked()
            self._discard_cancelled_waiters()
            self._dispatch_locked()

    def _on_completed(
        self,
        future: ConcurrentFuture[Any],
        execution_started: float,
        queue_seconds: float,
        operation_name: str,
        cleanup_state: _CancelledResultCleanup,
    ) -> None:
        execution_seconds = time.perf_counter() - execution_started
        cancelled = future.cancelled()
        failed = not cancelled and future.exception() is not None
        cleanup_state.clean_completed_result(future)
        with self._lock:
            self._completed_total += 1
            if failed:
                self._failed_total += 1
            self._running_futures.pop(future, None)
            self._max_execution_seconds = max(self._max_execution_seconds, execution_seconds)
            self._cancelled_running_futures.discard(future)
        self._release()
        if execution_seconds >= self.execution_warn_seconds:
            logger.warning(
                (
                    "Slow blocking task runner=%s operation=%s "
                    "queue_seconds=%.3f execution_seconds=%.3f"
                ),
                self.name,
                operation_name,
                queue_seconds,
                execution_seconds,
            )

    def _dispatch_locked(self) -> None:
        while self._running < self.max_workers and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.future.done():
                continue
            waiter.granted = True
            self._grant_locked()
            waiter.loop.call_soon_threadsafe(self._finish_waiter, waiter.future)

    def _capacity_error_locked(
        self, *, admission_timeout: bool = False
    ) -> BlockingTaskCapacityExceededError:
        detail = "admission timed out" if admission_timeout else "capacity is exhausted"
        return self._capacity_error_type(f"{self._capacity_name} task {detail}.")

    def _grant_locked(self) -> None:
        self._running += 1
        if self._running >= self.max_workers and self._saturated_since is None:
            self._saturated_since = time.perf_counter()

    def _release_counts_locked(self) -> None:
        self._running -= 1
        if self._running < 0:
            raise RuntimeError("Blocking task admission counter became negative.")
        if self._running < self.max_workers:
            self._saturated_since = None

    @classmethod
    def _operation_name(cls, operation: Callable[..., Any]) -> str:
        return str(
            getattr(operation, "__qualname__", None)
            or getattr(operation, "__name__", None)
            or operation.__class__.__name__
        )

    def _discard_cancelled_waiters(self) -> None:
        self._waiters = deque(waiter for waiter in self._waiters if not waiter.future.done())

    @classmethod
    def _finish_waiter(cls, future: asyncio.Future[None]) -> None:
        if not future.done():
            future.set_result(None)


def _build_runner(
    name: str,
    conf: tuple[int, int, float, float],
    capacity_error_type: type[BlockingTaskCapacityExceededError],
) -> BlockingTaskRunner:
    max_workers, queue_capacity, admission_timeout, execution_warn = conf
    return BlockingTaskRunner(
        name=name,
        max_workers=max_workers,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout,
        execution_warn_seconds=execution_warn,
        capacity_error_type=capacity_error_type,
        capacity_name=name.upper(),
    )


_RUNNERS_LOCK = threading.Lock()
_RUNNER_CONFIG = (DEFAULT_CPU_TASK_CONF, DEFAULT_EXTERNAL_TASK_CONF)
_CPU_TASK_RUNNER = _build_runner("cpu", DEFAULT_CPU_TASK_CONF, CpuTaskCapacityExceededError)
_EXTERNAL_TASK_RUNNER = _build_runner(
    "external", DEFAULT_EXTERNAL_TASK_CONF, ExternalTaskCapacityExceededError
)


def configure_blocking_task_runners(conf: BlockingInfrastructureConf) -> None:
    """Apply startup CPU/external sizing without replacing busy executors."""

    global _CPU_TASK_RUNNER, _EXTERNAL_TASK_RUNNER, _RUNNER_CONFIG
    cpu_conf = (
        conf.cpu.max_workers,
        conf.cpu.queue_capacity,
        conf.cpu.admission_timeout_seconds,
        conf.cpu.execution_warn_seconds,
    )
    external_conf = (
        conf.external.max_workers,
        conf.external.queue_capacity,
        conf.external.admission_timeout_seconds,
        conf.external.execution_warn_seconds,
    )
    config = (cpu_conf, external_conf)
    with _RUNNERS_LOCK:
        if config == _RUNNER_CONFIG:
            return
        cpu_runner = _build_runner("cpu", cpu_conf, CpuTaskCapacityExceededError)
        external_runner = _build_runner(
            "external", external_conf, ExternalTaskCapacityExceededError
        )
        try:
            _CPU_TASK_RUNNER.shutdown()
            _EXTERNAL_TASK_RUNNER.shutdown()
        except BaseException:
            cpu_runner.shutdown()
            external_runner.shutdown()
            raise
        _CPU_TASK_RUNNER = cpu_runner
        _EXTERNAL_TASK_RUNNER = external_runner
        _RUNNER_CONFIG = config


async def run_cpu_task[T](operation: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run predictable CPU work outside the event-loop thread."""

    return await _CPU_TASK_RUNNER.run(operation, *args, **kwargs)


async def run_external_task[T](
    operation: Callable[..., T],
    /,
    *args: Any,
    cancel_result_cleanup: Callable[[T], None] | None = None,
    wait_on_cancel: bool = False,
    **kwargs: Any,
) -> T:
    """Run one synchronous external-client operation in its isolated lane."""

    return await _EXTERNAL_TASK_RUNNER.run(
        operation,
        *args,
        cancel_result_cleanup=cancel_result_cleanup,
        wait_on_cancel=wait_on_cancel,
        **kwargs,
    )


def blocking_task_stats() -> dict[str, BlockingTaskRunnerStats]:
    """Return process-local CPU and external runner snapshots."""

    return {
        "cpu": _CPU_TASK_RUNNER.snapshot(),
        "external": _EXTERNAL_TASK_RUNNER.snapshot(),
    }
