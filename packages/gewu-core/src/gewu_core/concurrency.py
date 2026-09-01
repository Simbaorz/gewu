"""Cancellation-safe process-local weighted capacity controls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from contextlib import AbstractAsyncContextManager
from types import TracebackType

from pydantic import BaseModel, ConfigDict


class AsyncAdmissionCapacityExceededError(RuntimeError):
    """Raised when an asynchronous workload cannot enter bounded capacity."""


class AsyncAdmissionStats(BaseModel):
    """Immutable process-local asynchronous admission snapshot."""

    model_config = ConfigDict(frozen=True)

    active: int
    queued: int
    admitted_total: int
    rejected_total: int


class _AdmissionWaiter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    future: asyncio.Future[None]
    queued_at: float
    granted: bool = False


class _AdmissionLease(AbstractAsyncContextManager[None]):
    def __init__(self, limiter: FairAsyncCapacityLimiter) -> None:
        self._limiter = limiter
        self._released = False

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        if not self._released:
            self._released = True
            await self._limiter._release()


class FairAsyncCapacityLimiter:
    """Bound active async work and dispatch queued callers in FIFO order."""

    def __init__(
        self,
        *,
        capacity_name: str,
        max_active: int,
        queue_capacity: int,
        admission_timeout_seconds: float,
    ) -> None:
        if max_active < 1:
            raise ValueError("max_active must be at least 1.")
        if queue_capacity < 0:
            raise ValueError("queue_capacity must not be negative.")
        if admission_timeout_seconds <= 0:
            raise ValueError("admission_timeout_seconds must be positive.")
        self.capacity_name = capacity_name
        self.max_active = max_active
        self.queue_capacity = queue_capacity
        self.admission_timeout_seconds = admission_timeout_seconds
        self._lock = asyncio.Lock()
        self._active = 0
        self._waiters: deque[_AdmissionWaiter] = deque()
        self._queued = 0
        self._admitted_total = 0
        self._rejected_total = 0

    async def acquire(self) -> AbstractAsyncContextManager[None]:
        """Wait for process capacity and return an idempotent async lease."""

        loop = asyncio.get_running_loop()
        async with self._lock:
            if not self._waiters and self._active < self.max_active:
                self._grant()
                return _AdmissionLease(self)
            if self._queued >= self.queue_capacity:
                self._rejected_total += 1
                raise AsyncAdmissionCapacityExceededError(
                    f"{self.capacity_name} capacity is exhausted."
                )
            waiter = _AdmissionWaiter(
                future=loop.create_future(),
                queued_at=time.perf_counter(),
            )
            self._waiters.append(waiter)
            self._queued += 1
            self._dispatch_locked()
        try:
            await asyncio.wait_for(
                asyncio.shield(waiter.future),
                timeout=self.admission_timeout_seconds,
            )
        except TimeoutError as exc:
            await self._abort_waiter(waiter)
            async with self._lock:
                self._rejected_total += 1
            raise AsyncAdmissionCapacityExceededError(
                f"{self.capacity_name} admission timed out."
            ) from exc
        except asyncio.CancelledError:
            await self._abort_waiter(waiter)
            raise
        return _AdmissionLease(self)

    async def snapshot(self) -> AsyncAdmissionStats:
        async with self._lock:
            return AsyncAdmissionStats(
                active=self._active,
                queued=self._queued,
                admitted_total=self._admitted_total,
                rejected_total=self._rejected_total,
            )

    async def _abort_waiter(self, waiter: _AdmissionWaiter) -> None:
        async with self._lock:
            if waiter.granted:
                self._release_count()
            else:
                try:
                    self._waiters.remove(waiter)
                except ValueError:
                    pass
                else:
                    self._queued -= 1
            if not waiter.future.done():
                waiter.future.cancel()
            self._dispatch_locked()

    async def _release(self) -> None:
        async with self._lock:
            self._release_count()
            self._dispatch_locked()

    def _dispatch_locked(self) -> None:
        while self._active < self.max_active and self._waiters:
            waiter = self._waiters.popleft()
            if waiter.future.done():
                continue
            self._queued -= 1
            waiter.granted = True
            self._grant()
            waiter.future.set_result(None)

    def _grant(self) -> None:
        self._active += 1
        self._admitted_total += 1

    def _release_count(self) -> None:
        if self._active <= 0:
            raise RuntimeError("Asynchronous admission lease was released twice.")
        self._active -= 1


class WeightedCapacityExceededError(RuntimeError):
    """Raised when capacity cannot be admitted before its deadline."""


class WeightedCapacityLimiter:
    """Bound aggregate in-flight units for variable-size operations."""

    def __init__(
        self,
        *,
        name: str,
        capacity: int,
        admission_timeout_seconds: float,
    ) -> None:
        if capacity < 1:
            raise ValueError("Weighted capacity must be at least 1.")
        if admission_timeout_seconds <= 0:
            raise ValueError("Weighted admission timeout must be positive.")
        self.name = name
        self.capacity = capacity
        self.admission_timeout_seconds = admission_timeout_seconds
        self._condition = asyncio.Condition()
        self._used = 0

    @property
    def used(self) -> int:
        """Return currently reserved units."""

        return self._used

    async def acquire(self, units: int) -> WeightedCapacityReservation:
        """Reserve units or fail without changing capacity state."""

        if units < 0:
            raise ValueError("Weighted admission units cannot be negative.")
        if units == 0:
            return WeightedCapacityReservation(self, 0)
        if units > self.capacity:
            raise WeightedCapacityExceededError(f"{self.name} request exceeds process capacity.")
        try:
            async with asyncio.timeout(self.admission_timeout_seconds):
                async with self._condition:
                    await self._condition.wait_for(lambda: self._used + units <= self.capacity)
                    self._used += units
        except TimeoutError as exc:
            raise WeightedCapacityExceededError(f"{self.name} admission timed out.") from exc
        return WeightedCapacityReservation(self, units)

    async def _release(self, units: int) -> None:
        if units == 0:
            return
        async with self._condition:
            self._used -= units
            if self._used < 0:
                self._used = 0
                raise RuntimeError(f"{self.name} capacity was released more than once.")
            self._condition.notify_all()


class WeightedCapacityReservation:
    """Idempotent ownership token returned by a weighted limiter."""

    def __init__(self, limiter: WeightedCapacityLimiter, units: int) -> None:
        self._limiter = limiter
        self.units = units
        self._released = False

    async def release(self) -> None:
        """Return owned capacity exactly once."""

        if self._released:
            return
        self._released = True
        await self._limiter._release(self.units)  # noqa

    async def __aenter__(self) -> WeightedCapacityReservation:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object | None,
    ) -> None:
        await asyncio.shield(self.release())
