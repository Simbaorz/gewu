"""Typed Redis topology and connection-pool settings."""

from __future__ import annotations

from enum import StrEnum
from typing import TypedDict

from pydantic import Field, field_validator, model_validator

from gewu_core.config import SettingsModel


class RedisMode(StrEnum):
    """Supported Redis deployment topologies."""

    STANDALONE = "standalone"
    SENTINEL = "sentinel"
    CLUSTER = "cluster"


class RedisConnectionSettings(SettingsModel):
    """Connection fields interpreted according to the selected topology."""

    mode: RedisMode
    host: str = ""
    port: int = Field(default=6379, ge=1, le=65535)
    password: str = Field(default="", exclude=True, repr=False)
    nodes: list[str] = Field(default_factory=list)
    master_name: str = ""
    redis_password: str = Field(default="", exclude=True, repr=False)
    sentinel_password: str = Field(default="", exclude=True, repr=False)
    min_other_sentinels: int = Field(default=0, ge=0)
    read_from_replicas: bool = False
    require_full_coverage: bool = True
    dynamic_startup_nodes: bool = True

    @field_validator("nodes", mode="before")
    @classmethod
    def normalize_nodes(cls, value: object) -> object:
        """Accept a YAML list or a comma-separated remote override."""

        if isinstance(value, str):
            return [node.strip() for node in value.split(",") if node.strip()]
        return value

    @model_validator(mode="after")
    def validate_topology(self) -> RedisConnectionSettings:
        """Require the addressing fields used by the selected topology."""

        if self.mode is RedisMode.STANDALONE and not self.host.strip():
            raise ValueError("redis.connection.host is required in standalone mode.")
        if self.mode in {RedisMode.SENTINEL, RedisMode.CLUSTER} and not self.nodes:
            raise ValueError(f"redis.connection.nodes is required in {self.mode.value} mode.")
        if self.mode is RedisMode.SENTINEL and not self.master_name.strip():
            raise ValueError("redis.connection.master_name is required in sentinel mode.")
        return self


class RedisClientSettings(SettingsModel):
    """Redis connection-pool and socket settings."""

    max_connections: int | None = Field(default=None, ge=1)
    socket_timeout_seconds: float = Field(default=3.0, gt=0)
    socket_connect_timeout_seconds: float = Field(default=3.0, gt=0)
    socket_keepalive: bool = True
    health_check_interval_seconds: int = Field(default=30, ge=0)


class RedisClientOptions(TypedDict):
    """Common keyword arguments accepted by redis-py clients."""

    max_connections: int
    socket_timeout: float
    socket_connect_timeout: float
    socket_keepalive: bool
    health_check_interval: int


class RedisDatabasesSettings(SettingsModel):
    """Logical databases assigned to application consumers."""

    app: int = Field(default=1, ge=0)
    celery: int = Field(default=0, ge=0)


class RedisSettings(SettingsModel):
    """Complete Redis configuration shared by application processes."""

    enabled: bool = False
    connection: RedisConnectionSettings | None = None
    client: RedisClientSettings = Field(default_factory=RedisClientSettings)
    databases: RedisDatabasesSettings = Field(default_factory=RedisDatabasesSettings)

    @model_validator(mode="after")
    def validate_connection(self) -> RedisSettings:
        """Fail closed for missing connections and invalid Cluster databases."""

        if self.enabled and self.connection is None:
            raise ValueError("redis.connection is required when Redis is enabled.")
        if (
            self.connection is not None
            and self.connection.mode is RedisMode.CLUSTER
            and self.databases.app != 0
        ):
            raise ValueError("redis.databases.app must be 0 in Redis Cluster mode.")
        return self

    def resolved_max_connections(self) -> int:
        """Return the explicit or topology-specific connection limit."""

        if self.client.max_connections is not None:
            return self.client.max_connections
        if self.connection is not None and self.connection.mode is RedisMode.CLUSTER:
            return 32
        return 100

    def client_options(self) -> RedisClientOptions:
        """Return common redis-py connection options."""

        return {
            "max_connections": self.resolved_max_connections(),
            "socket_timeout": self.client.socket_timeout_seconds,
            "socket_connect_timeout": self.client.socket_connect_timeout_seconds,
            "socket_keepalive": self.client.socket_keepalive,
            "health_check_interval": self.client.health_check_interval_seconds,
        }
