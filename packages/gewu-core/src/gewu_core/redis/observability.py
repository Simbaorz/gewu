"""Redis await boundary with consistent capacity and cancellation semantics."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from redis.exceptions import MaxConnectionsError


class RedisCapacityExceededError(RuntimeError):
    """Raised when a process-local Redis connection pool is exhausted."""


async def observe_redis[T](
    operation: str,
    awaitable: Awaitable[T],
    *,
    record_error: Callable[[str], None] | None = None,
) -> T:
    """Await Redis work while preserving cancellation and exception semantics."""
    try:
        return await awaitable
    except asyncio.CancelledError:
        raise
    except MaxConnectionsError as exc:
        if record_error is not None:
            record_error(operation)
        raise RedisCapacityExceededError("Redis connection capacity is exhausted.") from exc
    except BaseException:
        if record_error is not None:
            record_error(operation)
        raise
