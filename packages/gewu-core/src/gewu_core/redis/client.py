"""Lifecycle-owned Redis client for standalone, Sentinel, and Cluster deployments."""

from __future__ import annotations

from redis.asyncio import Redis, RedisCluster
from redis.asyncio.cluster import ClusterNode, LoadBalancingStrategy
from redis.asyncio.sentinel import Sentinel

from gewu_core.redis.observability import observe_redis
from gewu_core.redis.settings import RedisMode, RedisSettings

RedisConnection = Redis | RedisCluster


class RedisClient:
    """Own one bounded redis-py client and expose observed basic commands."""

    def __init__(self, settings: RedisSettings) -> None:
        self.settings = settings
        self._client: RedisConnection | None = None

    @property
    def initialized(self) -> bool:
        return self._client is not None

    @property
    def connection(self) -> RedisConnection:
        """Expose the initialized redis-py client to technology-specific adapters."""

        return self._require_client()

    async def initialize(self) -> None:
        """Construct and verify the configured Redis client once."""

        if self._client is not None or not self.settings.enabled:
            return
        connection = self.settings.connection
        if connection is None:
            raise RuntimeError("redis.connection is required when Redis is enabled.")
        options = self.settings.client_options()
        if connection.mode is RedisMode.CLUSTER:
            client: RedisConnection = RedisCluster(
                startup_nodes=[self._cluster_node(node) for node in connection.nodes],
                password=connection.password or None,
                decode_responses=True,
                **options,
                load_balancing_strategy=(
                    LoadBalancingStrategy.ROUND_ROBIN if connection.read_from_replicas else None
                ),
                require_full_coverage=connection.require_full_coverage,
                dynamic_startup_nodes=connection.dynamic_startup_nodes,
                reinitialize_steps=5,
            )
        elif connection.mode is RedisMode.SENTINEL:
            sentinel = Sentinel(
                [self._parse_node(node) for node in connection.nodes],
                sentinel_kwargs={
                    "password": connection.sentinel_password or None,
                    **options,
                    "retry_on_timeout": False,
                },
                min_other_sentinels=connection.min_other_sentinels,
            )
            client = sentinel.master_for(
                connection.master_name,
                password=connection.redis_password or None,
                db=self.settings.databases.app,
                decode_responses=True,
                **options,
                retry_on_timeout=False,
            )
        else:
            client = Redis(
                host=connection.host,
                port=connection.port,
                password=connection.password or None,
                db=self.settings.databases.app,
                decode_responses=True,
                **options,
                retry_on_timeout=False,
            )
        try:
            await observe_redis("initialize.ping", client.ping())
        except BaseException:
            await client.aclose()
            raise
        self._client = client

    async def close(self) -> None:
        """Close the current pool and make future commands fail closed."""

        client = self._client
        self._client = None
        if client is not None:
            await observe_redis("close", client.aclose())

    async def get(self, key: str) -> object:
        return await observe_redis("get", self._require_client().get(key))

    async def incr(self, key: str) -> int:
        value = await observe_redis("incr", self._require_client().incr(key))
        return int(value)

    async def expire(self, key: str, seconds: int) -> bool:
        value = await observe_redis("expire", self._require_client().expire(key, seconds))
        return bool(value)

    async def persist(self, key: str) -> bool:
        value = await observe_redis("persist", self._require_client().persist(key))
        return bool(value)

    async def ttl(self, key: str) -> int:
        value = await observe_redis("ttl", self._require_client().ttl(key))
        return int(value)

    async def delete(self, key: str) -> int:
        value = await observe_redis("delete", self._require_client().delete(key))
        return int(value)

    def _require_client(self) -> RedisConnection:
        if self._client is None:
            raise RuntimeError("Redis is not initialized.")
        return self._client

    @staticmethod
    def _cluster_node(value: str) -> ClusterNode:
        host, port = RedisClient._parse_node(value)
        return ClusterNode(host=host, port=port)

    @staticmethod
    def _parse_node(value: str) -> tuple[str, int]:
        try:
            host, raw_port = value.strip().rsplit(":", maxsplit=1)
            port = int(raw_port)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Redis node: {value}") from exc
        if not host or not 1 <= port <= 65535:
            raise ValueError(f"Invalid Redis node: {value}")
        return host, port
