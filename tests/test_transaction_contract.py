"""Shared transaction semantics for Runtime and Subscriber-owned repositories."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from typing import Protocol, cast
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gewu_agent_runtime.adapters.mysql.transactions import (
    committed_session as runtime_committed_session,
)
from gewu_core import CommitOutcomeUnknownError
from gewu_core.database import committed_session as subscriber_committed_session


class CommittedSession(Protocol):
    def __call__(
        self,
        session_factory: Callable[[], AsyncSession],
        *,
        operation: str,
    ) -> AbstractAsyncContextManager[AsyncSession]: ...


BOUNDARIES: tuple[CommittedSession, ...] = (
    subscriber_committed_session,
    runtime_committed_session,
)


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=("subscriber", "runtime"))
async def test_committed_session_commits_and_closes(boundary: CommittedSession) -> None:
    session = AsyncMock(spec=AsyncSession)

    async with boundary(
        lambda: cast(AsyncSession, session),
        operation="test operation",
    ) as current:
        assert current is session

    session.flush.assert_awaited_once()
    session.commit.assert_awaited_once()
    session.rollback.assert_not_awaited()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=("subscriber", "runtime"))
async def test_committed_session_rolls_back_body_failure(boundary: CommittedSession) -> None:
    session = AsyncMock(spec=AsyncSession)

    with pytest.raises(RuntimeError, match="body failed"):
        async with boundary(
            lambda: cast(AsyncSession, session),
            operation="test operation",
        ):
            raise RuntimeError("body failed")

    session.flush.assert_not_awaited()
    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=("subscriber", "runtime"))
async def test_committed_session_propagates_flush_error_as_known_failure(
    boundary: CommittedSession,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.flush.side_effect = RuntimeError("flush failed")

    with pytest.raises(RuntimeError, match="flush failed"):
        async with boundary(
            lambda: cast(AsyncSession, session),
            operation="test operation",
        ):
            pass

    session.commit.assert_not_awaited()
    session.rollback.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=("subscriber", "runtime"))
async def test_committed_session_reports_commit_error_as_unknown(
    boundary: CommittedSession,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    session.commit.side_effect = RuntimeError("commit response lost")

    with pytest.raises(CommitOutcomeUnknownError, match="outcome is unknown"):
        async with boundary(
            lambda: cast(AsyncSession, session),
            operation="test operation",
        ):
            pass

    session.rollback.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.parametrize("boundary", BOUNDARIES, ids=("subscriber", "runtime"))
async def test_committed_session_finishes_commit_despite_request_cancellation(
    boundary: CommittedSession,
) -> None:
    session = AsyncMock(spec=AsyncSession)
    commit_started = asyncio.Event()
    release_commit = asyncio.Event()

    async def commit() -> None:
        commit_started.set()
        await release_commit.wait()

    session.commit.side_effect = commit

    async def persist() -> str:
        async with boundary(
            lambda: cast(AsyncSession, session),
            operation="test operation",
        ):
            pass
        return "saved"

    task = asyncio.create_task(persist())
    await commit_started.wait()
    task.cancel()
    release_commit.set()

    assert await task == "saved"
    session.rollback.assert_not_awaited()
    session.close.assert_awaited_once()
