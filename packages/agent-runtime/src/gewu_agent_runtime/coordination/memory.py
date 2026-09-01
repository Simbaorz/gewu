"""Process-local run ownership."""

from __future__ import annotations

import asyncio

from gewu_agent_runtime.coordination.contracts import RunLeaseStatus


class InMemoryRunLease:
    """Coordinate conversation runs within one Python process."""

    def __init__(self) -> None:
        self._owners: dict[str, tuple[str, bool]] = {}
        self._lock = asyncio.Lock()

    @property
    def monitor_interval_seconds(self) -> float | None:
        return None

    @property
    def renewal_interval_seconds(self) -> float | None:
        return None

    async def acquire(self, conversation_id: str, run_id: str) -> bool:
        async with self._lock:
            if conversation_id in self._owners:
                return False
            self._owners[conversation_id] = (run_id, False)
            return True

    async def renew(self, conversation_id: str, run_id: str) -> bool:
        async with self._lock:
            owner = self._owners.get(conversation_id)
            return owner is not None and owner[0] == run_id

    async def release(self, conversation_id: str, run_id: str) -> None:
        async with self._lock:
            owner = self._owners.get(conversation_id)
            if owner is not None and owner[0] == run_id:
                self._owners.pop(conversation_id, None)

    async def request_cancel(
        self,
        conversation_id: str,
        *,
        expected_run_id: str | None = None,
    ) -> str | None:
        async with self._lock:
            owner = self._owners.get(conversation_id)
            if owner is None or (expected_run_id is not None and owner[0] != expected_run_id):
                return None
            self._owners[conversation_id] = (owner[0], True)
            return owner[0]

    async def status(self, conversation_id: str, run_id: str) -> RunLeaseStatus:
        async with self._lock:
            owner = self._owners.get(conversation_id)
            if owner is None or owner[0] != run_id:
                return RunLeaseStatus.LOST
            return RunLeaseStatus.CANCEL_REQUESTED if owner[1] else RunLeaseStatus.ACTIVE
