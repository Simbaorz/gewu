"""Shared Redis adapter boundaries for Gewu applications."""

from gewu_core.redis.celery import celery_broker_url, celery_transport_options
from gewu_core.redis.client import RedisClient
from gewu_core.redis.observability import RedisCapacityExceededError, observe_redis
from gewu_core.redis.settings import (
    RedisClientSettings,
    RedisConnectionSettings,
    RedisDatabasesSettings,
    RedisMode,
    RedisSettings,
)

__all__ = [
    "RedisCapacityExceededError",
    "RedisClient",
    "RedisClientSettings",
    "RedisConnectionSettings",
    "RedisDatabasesSettings",
    "RedisMode",
    "RedisSettings",
    "celery_broker_url",
    "celery_transport_options",
    "observe_redis",
]
