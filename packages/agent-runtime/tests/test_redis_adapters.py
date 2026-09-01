"""Redis lease and state cache tests using fakeredis."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fakeredis.aioredis import FakeRedis

from gewu_agent_runtime.adapters.redis import RedisRunLease, RedisStateCache
from gewu_agent_runtime.coordination import RunLeaseStatus
from gewu_agent_runtime.domain import ConversationState
from gewu_core.time import utc_now


async def test_redis_run_lease_ownership_and_cancel() -> None:
    client = FakeLeaseRedis()
    lease = RedisRunLease(client, ttl_seconds=30)

    assert lease.monitor_interval_seconds == 1.0
    assert await lease.acquire("conversation", "run-1") is True
    assert await lease.acquire("conversation", "run-2") is False
    assert await lease.renew("conversation", "run-1") is True
    assert await lease.renew("conversation", "run-2") is False
    assert await lease.request_cancel("conversation", expected_run_id="run-1") == "run-1"
    assert await lease.status("conversation", "run-1") is RunLeaseStatus.CANCEL_REQUESTED
    await lease.release("conversation", "run-1")
    assert await lease.status("conversation", "run-1") is RunLeaseStatus.LOST
    assert client.keys == {"gewu:agent-runtime:conversation:{conversation}:run"}


class FakeLeaseRedis:
    """Apply the operation encoded by each Redis lease script marker."""

    def __init__(self) -> None:
        self.records: dict[str, tuple[str, bool]] = {}
        self.keys: set[str] = set()

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: object,
    ) -> Any:
        assert numkeys == 1
        key = str(keys_and_args[0])
        self.keys.add(key)
        operation = script.splitlines()[0]
        if operation == "-- agent-run-acquire":
            if key in self.records:
                return 0
            self.records[key] = (str(keys_and_args[1]), False)
            return 1
        if operation == "-- agent-run-renew":
            record = self.records.get(key)
            return int(record is not None and record[0] == str(keys_and_args[1]))
        if operation == "-- agent-run-release":
            record = self.records.get(key)
            if record is None or record[0] != str(keys_and_args[1]):
                return 0
            self.records.pop(key)
            return 1
        if operation == "-- agent-run-cancel":
            record = self.records.get(key)
            if record is None:
                return ""
            expected = str(keys_and_args[1])
            if expected and expected != record[0]:
                return ""
            self.records[key] = (record[0], True)
            return record[0]
        if operation == "-- agent-run-status":
            record = self.records.get(key)
            if record is None or record[0] != str(keys_and_args[1]):
                return 0
            return 2 if record[1] else 1
        raise AssertionError(f"Unexpected script: {operation}")


async def test_redis_state_cache_round_trip() -> None:
    client = FakeRedis()
    cache = RedisStateCache(client)
    state = ConversationState(
        conversation_id="conversation",
        kind="pending_ask",
        revision=1,
        payload={"ask_id": "ask"},
        expires_at=utc_now() + timedelta(minutes=1),
    )

    await cache.set(state)
    assert await cache.get(state.conversation_id, state.kind) == state
    await cache.delete(state.conversation_id, state.kind)
    assert await cache.get(state.conversation_id, state.kind) is None
    await client.aclose()
