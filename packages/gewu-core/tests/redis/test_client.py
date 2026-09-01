"""Redis topology settings and process-owned client lifecycle."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import gewu_core.redis.client as client_module
from gewu_core.redis import (
    RedisClient,
    RedisConnectionSettings,
    RedisDatabasesSettings,
    RedisMode,
    RedisSettings,
)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, int] = {}
        self.ping_count = 0
        self.closed = False

    async def ping(self) -> bool:
        self.ping_count += 1
        return True

    async def aclose(self) -> None:
        self.closed = True

    async def get(self, key: str) -> int | None:
        return self.values.get(key)

    async def incr(self, key: str) -> int:
        self.values[key] = self.values.get(key, 0) + 1
        return self.values[key]

    async def expire(self, key: str, seconds: int) -> bool:
        del seconds
        return key in self.values

    async def persist(self, key: str) -> bool:
        return key in self.values

    async def ttl(self, key: str) -> int:
        return 30 if key in self.values else -2

    async def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)


def test_redis_settings_fail_closed_for_missing_or_invalid_topology() -> None:
    with pytest.raises(ValidationError, match="redis.connection is required"):
        RedisSettings(enabled=True)

    with pytest.raises(ValidationError, match="redis.connection.nodes is required"):
        RedisConnectionSettings(mode=RedisMode.SENTINEL, master_name="primary")

    with pytest.raises(ValidationError, match="redis.databases.app must be 0"):
        RedisSettings(
            enabled=True,
            connection=RedisConnectionSettings(
                mode=RedisMode.CLUSTER,
                nodes=["redis-1:6379"],
            ),
            databases=RedisDatabasesSettings(app=1),
        )


@pytest.mark.parametrize("mode", [RedisMode.STANDALONE, RedisMode.SENTINEL, RedisMode.CLUSTER])
async def test_redis_client_initializes_each_supported_topology(
    monkeypatch: pytest.MonkeyPatch,
    mode: RedisMode,
) -> None:
    fake = FakeRedis()
    captured: dict[str, object] = {}

    def standalone_factory(**kwargs: object) -> FakeRedis:
        captured.update(kwargs)
        return fake

    def cluster_factory(**kwargs: object) -> FakeRedis:
        captured.update(kwargs)
        return fake

    class FakeSentinel:
        def __init__(self, nodes: list[tuple[str, int]], **kwargs: object) -> None:
            captured["sentinel_nodes"] = nodes
            captured.update(kwargs)

        def master_for(self, master_name: str, **kwargs: object) -> FakeRedis:
            captured["master_name"] = master_name
            captured.update(kwargs)
            return fake

    monkeypatch.setattr(client_module, "Redis", standalone_factory)
    monkeypatch.setattr(client_module, "RedisCluster", cluster_factory)
    monkeypatch.setattr(client_module, "Sentinel", FakeSentinel)
    settings = RedisSettings(
        enabled=True,
        connection=_connection(mode),
        databases=RedisDatabasesSettings(app=0 if mode is RedisMode.CLUSTER else 1),
    )
    client = RedisClient(settings)

    await client.initialize()
    assert client.initialized
    assert fake.ping_count == 1
    if mode is RedisMode.STANDALONE:
        assert captured["host"] == "redis.internal"
        assert captured["db"] == 1
    elif mode is RedisMode.SENTINEL:
        assert captured["sentinel_nodes"] == [("sentinel.internal", 26379)]
        assert captured["master_name"] == "primary"
    else:
        nodes = captured["startup_nodes"]
        assert isinstance(nodes, list)
        assert [(node.host, node.port) for node in nodes] == [("cluster.internal", 6379)]

    assert await client.incr("login") == 1
    assert await client.get("login") == 1
    assert await client.ttl("login") == 30
    assert await client.expire("login", 60)
    assert await client.persist("login")
    assert await client.delete("login") == 1
    await client.close()
    assert fake.closed
    assert not client.initialized


def _connection(mode: RedisMode) -> RedisConnectionSettings:
    if mode is RedisMode.STANDALONE:
        return RedisConnectionSettings(mode=mode, host="redis.internal")
    if mode is RedisMode.SENTINEL:
        return RedisConnectionSettings(
            mode=mode,
            nodes="sentinel.internal:26379",
            master_name="primary",
        )
    return RedisConnectionSettings(mode=mode, nodes=["cluster.internal:6379"])
