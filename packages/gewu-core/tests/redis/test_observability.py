"""Redis observation boundary behavior."""

from __future__ import annotations

import asyncio

import pytest
from redis.exceptions import MaxConnectionsError

from gewu_core.redis import RedisCapacityExceededError, observe_redis


async def test_observer_counts_failure_without_changing_exception() -> None:
    operations: list[str] = []
    original = TimeoutError("redis timeout")

    async def fail() -> None:
        raise original

    with pytest.raises(TimeoutError) as error:
        await observe_redis("agent_run.renew", fail(), record_error=operations.append)

    assert error.value is original
    assert operations == ["agent_run.renew"]


async def test_observer_does_not_report_caller_cancellation() -> None:
    operations: list[str] = []

    async def wait_forever() -> None:
        await asyncio.Future()

    task = asyncio.create_task(
        observe_redis("publish", wait_forever(), record_error=operations.append)
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert operations == []


async def test_observer_maps_pool_exhaustion_to_capacity_error() -> None:
    operations: list[str] = []
    original = MaxConnectionsError("Too many connections")

    async def fail() -> None:
        raise original

    with pytest.raises(RedisCapacityExceededError) as error:
        await observe_redis(
            "chat.concurrency.acquire",
            fail(),
            record_error=operations.append,
        )

    assert error.value.__cause__ is original
    assert operations == ["chat.concurrency.acquire"]
