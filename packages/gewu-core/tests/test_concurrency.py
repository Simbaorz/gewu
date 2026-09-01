"""Weighted capacity behavior shared by process-local Gewu components."""

from __future__ import annotations

import asyncio

import pytest

from gewu_core.concurrency import (
    AsyncAdmissionCapacityExceededError,
    FairAsyncCapacityLimiter,
    WeightedCapacityExceededError,
    WeightedCapacityLimiter,
)


def fair_limiter(
    *,
    max_active: int = 1,
    queue_capacity: int = 4,
    admission_timeout_seconds: float = 1.0,
) -> FairAsyncCapacityLimiter:
    return FairAsyncCapacityLimiter(
        capacity_name="test",
        max_active=max_active,
        queue_capacity=queue_capacity,
        admission_timeout_seconds=admission_timeout_seconds,
    )


async def test_async_admission_enforces_process_capacity() -> None:
    limiter = fair_limiter(max_active=2, queue_capacity=0)
    first = await limiter.acquire()
    second = await limiter.acquire()
    await first.__aenter__()
    await second.__aenter__()
    try:
        with pytest.raises(AsyncAdmissionCapacityExceededError):
            await limiter.acquire()
    finally:
        await second.__aexit__(None, None, None)
        await first.__aexit__(None, None, None)


async def test_async_admission_dispatches_waiters_fifo() -> None:
    limiter = fair_limiter()
    first = await limiter.acquire()
    await first.__aenter__()
    order: list[str] = []

    async def run(name: str) -> None:
        async with await limiter.acquire():
            order.append(name)
            await asyncio.sleep(0)

    second = asyncio.create_task(run("second"))
    third = asyncio.create_task(run("third"))
    await asyncio.sleep(0)
    await first.__aexit__(None, None, None)
    await asyncio.gather(second, third)

    assert order == ["second", "third"]
    stats = await limiter.snapshot()
    assert stats.active == 0
    assert stats.queued == 0
    assert stats.admitted_total == 3


async def test_async_admission_enforces_queue_capacity() -> None:
    limiter = fair_limiter(queue_capacity=1)
    first = await limiter.acquire()
    await first.__aenter__()
    queued = asyncio.create_task(limiter.acquire())
    await asyncio.sleep(0)

    with pytest.raises(AsyncAdmissionCapacityExceededError, match="capacity is exhausted"):
        await limiter.acquire()

    queued.cancel()
    with pytest.raises(asyncio.CancelledError):
        await queued
    await first.__aexit__(None, None, None)
    stats = await limiter.snapshot()
    assert stats.active == 0
    assert stats.queued == 0
    assert stats.rejected_total == 1


async def test_async_admission_timeout_does_not_leak_capacity() -> None:
    limiter = fair_limiter(admission_timeout_seconds=0.01)
    async with await limiter.acquire():
        with pytest.raises(AsyncAdmissionCapacityExceededError, match="admission timed out"):
            await limiter.acquire()

    stats = await limiter.snapshot()
    assert stats.active == 0
    assert stats.queued == 0
    assert stats.rejected_total == 1


async def test_weighted_capacity_waits_and_releases_idempotently() -> None:
    limiter = WeightedCapacityLimiter(
        name="bytes",
        capacity=3,
        admission_timeout_seconds=0.1,
    )
    first = await limiter.acquire(3)
    waiting = asyncio.create_task(limiter.acquire(1))
    await asyncio.sleep(0)
    assert waiting.done() is False

    await first.release()
    second = await waiting
    assert limiter.used == 1
    await second.release()
    await second.release()
    assert limiter.used == 0


async def test_weighted_capacity_rejects_impossible_and_timed_out_requests() -> None:
    limiter = WeightedCapacityLimiter(
        name="bytes",
        capacity=2,
        admission_timeout_seconds=0.01,
    )
    with pytest.raises(WeightedCapacityExceededError, match="exceeds process capacity"):
        await limiter.acquire(3)

    reservation = await limiter.acquire(2)
    with pytest.raises(WeightedCapacityExceededError, match="admission timed out"):
        await limiter.acquire(1)
    await reservation.release()
    assert limiter.used == 0
