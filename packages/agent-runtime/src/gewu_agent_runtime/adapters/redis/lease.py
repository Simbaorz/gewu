"""TTL-backed distributed conversation run lease."""

from __future__ import annotations

from typing import Protocol

from gewu_agent_runtime.coordination import RunLeaseStatus

DEFAULT_LEASE_TTL_SECONDS = 300
DEFAULT_RENEWAL_INTERVAL_SECONDS = 30.0
DEFAULT_MONITOR_INTERVAL_SECONDS = 1.0

_ACQUIRE_SCRIPT = """-- agent-run-acquire
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1], 'run_id', ARGV[1], 'cancel_requested', '0')
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

_RENEW_SCRIPT = """-- agent-run-renew
if redis.call('HGET', KEYS[1], 'run_id') ~= ARGV[1] then return 0 end
redis.call('PEXPIRE', KEYS[1], ARGV[2])
return 1
"""

_RELEASE_SCRIPT = """-- agent-run-release
if redis.call('HGET', KEYS[1], 'run_id') ~= ARGV[1] then return 0 end
return redis.call('DEL', KEYS[1])
"""

_CANCEL_SCRIPT = """-- agent-run-cancel
local run_id = redis.call('HGET', KEYS[1], 'run_id')
if not run_id then return '' end
if ARGV[1] ~= '' and run_id ~= ARGV[1] then return '' end
redis.call('HSET', KEYS[1], 'cancel_requested', '1')
return run_id
"""

_STATUS_SCRIPT = """-- agent-run-status
if redis.call('HGET', KEYS[1], 'run_id') ~= ARGV[1] then return 0 end
if redis.call('HGET', KEYS[1], 'cancel_requested') == '1' then return 2 end
return 1
"""


class RedisEvalClient(Protocol):
    """Redis operation shared by standalone, Sentinel, and Cluster clients."""

    async def eval(
        self,
        script: str,
        numkeys: int,
        *keys_and_args: object,
    ) -> object: ...


class RedisRunLease:
    """Coordinate one active run per conversation through one cluster-safe key."""

    def __init__(
        self,
        client: RedisEvalClient,
        *,
        key_prefix: str = "gewu:agent-runtime",
        ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS,
        renewal_interval_seconds: float | None = None,
    ) -> None:
        if ttl_seconds < 3:
            raise ValueError("ttl_seconds must be at least 3.")
        resolved_renewal_interval = (
            min(DEFAULT_RENEWAL_INTERVAL_SECONDS, ttl_seconds / 3)
            if renewal_interval_seconds is None
            else renewal_interval_seconds
        )
        if resolved_renewal_interval <= 0 or resolved_renewal_interval >= ttl_seconds:
            raise ValueError("renewal_interval_seconds must be positive and less than ttl_seconds.")
        self._client = client
        self._prefix = key_prefix.rstrip(":")
        self._ttl_ms = ttl_seconds * 1_000
        self._renewal_interval_seconds = resolved_renewal_interval

    @property
    def monitor_interval_seconds(self) -> float:
        return DEFAULT_MONITOR_INTERVAL_SECONDS

    @property
    def renewal_interval_seconds(self) -> float:
        return self._renewal_interval_seconds

    async def acquire(self, conversation_id: str, run_id: str) -> bool:
        result = await self._client.eval(
            _ACQUIRE_SCRIPT,
            1,
            self._key(conversation_id),
            run_id,
            self._ttl_ms,
        )
        return _int_value(result) == 1

    async def renew(self, conversation_id: str, run_id: str) -> bool:
        result = await self._client.eval(
            _RENEW_SCRIPT,
            1,
            self._key(conversation_id),
            run_id,
            self._ttl_ms,
        )
        return _int_value(result) == 1

    async def release(self, conversation_id: str, run_id: str) -> None:
        await self._client.eval(
            _RELEASE_SCRIPT,
            1,
            self._key(conversation_id),
            run_id,
        )

    async def request_cancel(
        self,
        conversation_id: str,
        *,
        expected_run_id: str | None = None,
    ) -> str | None:
        result = await self._client.eval(
            _CANCEL_SCRIPT,
            1,
            self._key(conversation_id),
            expected_run_id or "",
        )
        run_id = _text(result)
        return run_id or None

    async def status(self, conversation_id: str, run_id: str) -> RunLeaseStatus:
        result = _int_value(
            await self._client.eval(
                _STATUS_SCRIPT,
                1,
                self._key(conversation_id),
                run_id,
            )
        )
        if result == 2:
            return RunLeaseStatus.CANCEL_REQUESTED
        if result == 1:
            return RunLeaseStatus.ACTIVE
        return RunLeaseStatus.LOST

    def _key(self, conversation_id: str) -> str:
        return f"{self._prefix}:conversation:{{{conversation_id}}}:run"


def _int_value(value: object) -> int:
    if not isinstance(value, (str, bytes, bytearray, int)):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return ""
