"""Redis cache for disposable versioned conversation state."""

from __future__ import annotations

import math
from typing import Protocol

from gewu_agent_runtime.domain import ConversationState
from gewu_core.time import utc_now


class RedisStateClient(Protocol):
    """Redis commands used by the disposable state cache."""

    async def get(self, key: str) -> object: ...

    async def set(self, key: str, value: str, *, ex: int) -> object: ...

    async def delete(self, *keys: str) -> object: ...


class RedisStateCache:
    """Cache state JSON while leaving the RuntimeStore as the fact source."""

    def __init__(
        self,
        client: RedisStateClient,
        *,
        key_prefix: str = "gewu:agent-runtime:state",
        default_ttl_seconds: int = 3_600,
    ) -> None:
        if default_ttl_seconds < 1:
            raise ValueError("default_ttl_seconds must be positive.")
        self._client = client
        self._prefix = key_prefix.rstrip(":")
        self._default_ttl = default_ttl_seconds

    async def get(self, conversation_id: str, kind: str) -> ConversationState | None:
        value = await self._client.get(self._key(conversation_id, kind))
        if value is None:
            return None
        if not isinstance(value, (str, bytes, bytearray)):
            await self.delete(conversation_id, kind)
            return None
        try:
            state = ConversationState.model_validate_json(value)
        except ValueError:
            await self.delete(conversation_id, kind)
            return None
        if state.expires_at is not None and state.expires_at <= utc_now():
            await self.delete(conversation_id, kind)
            return None
        return state

    async def set(self, state: ConversationState) -> None:
        ttl = self._default_ttl
        if state.expires_at is not None:
            ttl = max(1, math.ceil((state.expires_at - utc_now()).total_seconds()))
        await self._client.set(
            self._key(state.conversation_id, state.kind),
            state.model_dump_json(),
            ex=ttl,
        )

    async def delete(self, conversation_id: str, kind: str) -> None:
        await self._client.delete(self._key(conversation_id, kind))

    def _key(self, conversation_id: str, kind: str) -> str:
        return f"{self._prefix}:{conversation_id}:{kind}"
