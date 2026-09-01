"""Exclusive run ownership contract."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol


class RunLeaseStatus(StrEnum):
    """Shared status visible to a run owner."""

    ACTIVE = "active"
    CANCEL_REQUESTED = "cancel_requested"
    LOST = "lost"


class RunLease(Protocol):
    """Coordinate at most one active run for each conversation."""

    @property
    def monitor_interval_seconds(self) -> float | None:
        """Return the cancellation and ownership polling interval."""

    @property
    def renewal_interval_seconds(self) -> float | None:
        """Return how often a renewable lease must be refreshed."""

    async def acquire(self, conversation_id: str, run_id: str) -> bool:
        """Acquire exclusive ownership."""

    async def renew(self, conversation_id: str, run_id: str) -> bool:
        """Refresh or verify current ownership."""

    async def release(self, conversation_id: str, run_id: str) -> None:
        """Release ownership only when held by this run."""

    async def request_cancel(
        self,
        conversation_id: str,
        *,
        expected_run_id: str | None = None,
    ) -> str | None:
        """Mark the current owner for cancellation and return its run ID."""

    async def status(self, conversation_id: str, run_id: str) -> RunLeaseStatus:
        """Return the current shared ownership state."""
