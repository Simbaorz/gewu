"""Disposable process-local Runtime state cache behavior."""

from __future__ import annotations

from gewu_agent_runtime.domain import ConversationState
from gewu_agent_runtime.persistence import InMemoryStateCache


async def test_memory_state_cache_evicts_the_least_recently_used_entry() -> None:
    cache = InMemoryStateCache(max_entries=2, ttl_seconds=60)
    first = ConversationState(conversation_id="c1", kind="file", revision=1)
    second = ConversationState(conversation_id="c2", kind="file", revision=1)
    third = ConversationState(conversation_id="c3", kind="file", revision=1)

    await cache.set(first)
    await cache.set(second)
    assert await cache.get("c1", "file") == first
    await cache.set(third)

    assert await cache.get("c1", "file") == first
    assert await cache.get("c2", "file") is None
    assert await cache.get("c3", "file") == third


async def test_memory_state_cache_expires_entries_by_cache_ttl(monkeypatch) -> None:
    now = 100.0
    monkeypatch.setattr(
        "gewu_agent_runtime.persistence.memory.time.monotonic",
        lambda: now,
    )
    cache = InMemoryStateCache(max_entries=2, ttl_seconds=1)
    state = ConversationState(conversation_id="c1", kind="file", revision=1)
    await cache.set(state)

    assert await cache.get("c1", "file") == state
    now = 101.0

    assert await cache.get("c1", "file") is None
