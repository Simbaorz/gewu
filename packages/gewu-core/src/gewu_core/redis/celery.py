"""Celery broker addressing over the shared Redis topology settings."""

from __future__ import annotations

from urllib.parse import quote

from gewu_core.redis.settings import RedisMode, RedisSettings


def celery_broker_url(settings: RedisSettings) -> str:
    """Build a Celery broker URL or reject an unsupported Redis topology."""

    if not settings.enabled:
        raise RuntimeError("Redis must be enabled for the Celery broker.")
    connection = settings.connection
    if connection is None:
        raise RuntimeError("redis.connection is required for the Celery broker.")
    if connection.mode is RedisMode.CLUSTER:
        raise RuntimeError("Celery does not support the configured Redis Cluster mode.")
    if connection.mode is RedisMode.SENTINEL:
        if settings.databases.celery != 0:
            raise RuntimeError("Celery Sentinel broker currently requires Redis database 0.")
        credentials = _password_credentials(connection.redis_password)
        return ";".join(f"sentinel://{credentials}{node.strip()}" for node in connection.nodes)
    credentials = _password_credentials(connection.password)
    return f"redis://{credentials}{connection.host}:{connection.port}/{settings.databases.celery}"


def celery_transport_options(
    settings: RedisSettings,
    *,
    project_name: str = "gewu",
    mode: str = "dev",
) -> dict[str, object]:
    """Build bounded, namespaced Kombu Redis transport options."""

    options: dict[str, object] = {
        "global_keyprefix": f"{project_name}:{mode}:celery:",
        **settings.client_options(),
        "retry_on_timeout": False,
    }
    connection = settings.connection
    if connection is None or connection.mode is not RedisMode.SENTINEL:
        return options
    options["master_name"] = connection.master_name
    if connection.min_other_sentinels:
        options["min_other_sentinels"] = connection.min_other_sentinels
    sentinel_options: dict[str, object] = {
        **settings.client_options(),
        "retry_on_timeout": False,
    }
    if connection.sentinel_password:
        sentinel_options["password"] = connection.sentinel_password
    options["sentinel_kwargs"] = sentinel_options
    return options


def _password_credentials(password: str) -> str:
    return f":{quote(password, safe='')}@" if password else ""
