"""Cancellation-safe transaction boundaries for Runtime SQL adapters."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession

from gewu_core.errors import CommitOutcomeUnknownError

logger = logging.getLogger(__name__)
SessionFactory = Callable[[], AsyncSession]


@asynccontextmanager
async def committed_session(
    session_factory: SessionFactory,
    *,
    operation: str,
) -> AsyncIterator[AsyncSession]:
    """Commit one mutation while preserving an explicit unknown-outcome state."""

    session = session_factory()
    try:
        try:
            yield session
        except BaseException:
            await _rollback_best_effort(session, operation)
            raise
        try:
            await session.flush()
        except BaseException:
            await _rollback_best_effort(session, operation)
            raise
        try:
            await _finish_critical(session.commit())
        except BaseException as exc:
            await _rollback_best_effort(session, operation)
            raise CommitOutcomeUnknownError(
                f"Database COMMIT outcome is unknown for {operation}."
            ) from exc
    finally:
        try:
            await _finish_critical(session.close())
        except BaseException as exc:
            logger.error(
                "Unable to close database session after %s exception_type=%s",
                operation,
                type(exc).__name__,
            )


async def _rollback_best_effort(session: AsyncSession, operation: str) -> None:
    try:
        await _finish_critical(session.rollback())
    except BaseException as exc:
        logger.error(
            "Unable to roll back database session after %s exception_type=%s",
            operation,
            type(exc).__name__,
        )


async def _finish_critical[ResultT](awaitable: Awaitable[ResultT]) -> ResultT:
    task = asyncio.ensure_future(awaitable)
    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()
