"""Database cleanup logs do not retain infrastructure exception details."""

from __future__ import annotations

import logging
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from gewu_agent_runtime.adapters.mysql.transactions import (
    committed_session as runtime_committed_session,
)
from gewu_core.database.commit import committed_session as subscriber_committed_session


class _FailingCleanupSession:
    async def rollback(self) -> None:
        raise RuntimeError("database-rollback-private-secret")

    async def close(self) -> None:
        raise OSError("database-close-private-secret")


async def test_subscriber_transaction_cleanup_hides_exception_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = cast(AsyncSession, _FailingCleanupSession())

    with caplog.at_level(logging.ERROR, logger="gewu_core.database.commit"):
        with pytest.raises(ValueError, match="business failure"):
            async with subscriber_committed_session(lambda: session, operation="test mutation"):
                raise ValueError("business failure")

    _assert_sanitized_cleanup_logs(caplog.text)


async def test_runtime_transaction_cleanup_hides_exception_bodies(
    caplog: pytest.LogCaptureFixture,
) -> None:
    session = cast(AsyncSession, _FailingCleanupSession())

    with caplog.at_level(
        logging.ERROR,
        logger="gewu_agent_runtime.adapters.mysql.transactions",
    ):
        with pytest.raises(ValueError, match="business failure"):
            async with runtime_committed_session(lambda: session, operation="test mutation"):
                raise ValueError("business failure")

    _assert_sanitized_cleanup_logs(caplog.text)


def _assert_sanitized_cleanup_logs(log_text: str) -> None:
    assert "Unable to roll back database session" in log_text
    assert "exception_type=RuntimeError" in log_text
    assert "Unable to close database session" in log_text
    assert "exception_type=OSError" in log_text
    assert "database-rollback-private-secret" not in log_text
    assert "database-close-private-secret" not in log_text
