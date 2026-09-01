"""Apollo-backed typed configuration loading and change monitoring."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import json
import logging
import os
import random
import re
import tempfile
import time
from collections.abc import Awaitable, Callable, Collection, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import yaml
from pydantic import BaseModel, ConfigDict

from gewu_core.config import (
    ApolloBootstrapSettings,
    ApolloStartupPolicy,
    ConfigurationSource,
    flatten_yaml_settings,
    load_settings,
    load_settings_from_values,
)

logger = logging.getLogger(__name__)


class ApolloConfigurationError(RuntimeError):
    """Apollo could not provide a valid configuration snapshot."""


class ApolloNamespaceSnapshot(BaseModel):
    """One namespace release decoded into flattened settings paths."""

    model_config = ConfigDict(frozen=True)

    namespace: str
    values: Mapping[str, object]
    release_key: str


class ApolloConfigSnapshot(BaseModel):
    """Merged configuration from all ordered Apollo namespaces."""

    model_config = ConfigDict(frozen=True)

    values: Mapping[str, object]
    revision: str


SettingsChangeHandler = Callable[[Any, Any], Awaitable[None] | None]


class ApolloClient:
    """Small async client for Apollo Config Service's read and notification APIs."""

    def __init__(
        self,
        bootstrap: ApolloBootstrapSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._bootstrap = bootstrap
        self._base_url = bootstrap.apollo_base_url.rstrip("/")
        self._owns_client = http_client is None
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(bootstrap.apollo_long_poll_timeout_seconds)
        )

    async def fetch_namespace(
        self,
        namespace: str,
        *,
        release_key: str = "",
    ) -> ApolloNamespaceSnapshot | None:
        """Fetch one complete namespace, returning ``None`` for HTTP 304."""

        encoded_app_id = quote(self._bootstrap.apollo_app_id, safe="")
        encoded_cluster = quote(self._bootstrap.apollo_cluster, safe="")
        encoded_namespace = quote(namespace, safe=".")
        url = f"{self._base_url}/configs/{encoded_app_id}/{encoded_cluster}/{encoded_namespace}"
        params: dict[str, str] = {}
        if release_key:
            params["releaseKey"] = release_key
        if self._bootstrap.apollo_client_ip.strip():
            params["ip"] = self._bootstrap.apollo_client_ip.strip()
        if self._bootstrap.apollo_label.strip():
            params["label"] = self._bootstrap.apollo_label.strip()
        response = await self._get(url, params=params, long_poll=False)
        if response.status_code == 304:
            return None
        self._raise_for_status(response, operation=f"fetch namespace {namespace}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApolloConfigurationError(
                f"Apollo namespace {namespace} returned invalid JSON"
            ) from exc
        if not isinstance(payload, dict):
            raise ApolloConfigurationError(
                f"Apollo namespace {namespace} response must be a JSON object"
            )
        configurations = payload.get("configurations")
        if not isinstance(configurations, dict):
            raise ApolloConfigurationError(
                f"Apollo namespace {namespace} response is missing configurations"
            )
        values = _decode_namespace_values(namespace, configurations)
        release = payload.get("releaseKey", "")
        return ApolloNamespaceSnapshot(
            namespace=namespace,
            values=values,
            release_key=release if isinstance(release, str) else str(release),
        )

    async def poll_notifications(
        self,
        notification_ids: Mapping[str, int],
    ) -> dict[str, int]:
        """Long-poll namespace release notifications."""

        notifications = [
            {"namespaceName": namespace, "notificationId": notification_ids[namespace]}
            for namespace in notification_ids
        ]
        response = await self._get(
            f"{self._base_url}/notifications/v2",
            params={
                "appId": self._bootstrap.apollo_app_id,
                "cluster": self._bootstrap.apollo_cluster,
                "notifications": json.dumps(
                    notifications,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
            long_poll=True,
        )
        if response.status_code == 304:
            return {}
        self._raise_for_status(response, operation="poll notifications")
        try:
            payload = response.json()
        except ValueError as exc:
            raise ApolloConfigurationError("Apollo notifications returned invalid JSON") from exc
        if not isinstance(payload, list):
            raise ApolloConfigurationError("Apollo notifications response must be a JSON array")
        changed: dict[str, int] = {}
        for entry in payload:
            if not isinstance(entry, dict):
                raise ApolloConfigurationError("Apollo notification entry must be an object")
            namespace = entry.get("namespaceName")
            notification_id = entry.get("notificationId")
            if not isinstance(namespace, str) or not isinstance(notification_id, int):
                raise ApolloConfigurationError("Apollo notification entry is malformed")
            if namespace in notification_ids:
                changed[namespace] = notification_id
        return changed

    async def close(self) -> None:
        """Close the owned HTTP client."""

        if self._owns_client:
            await self._client.aclose()

    async def _get(
        self,
        url: str,
        *,
        params: Mapping[str, str],
        long_poll: bool,
    ) -> httpx.Response:
        request = self._client.build_request("GET", url, params=params)
        request.headers.update(self._authorization_headers(request.url))
        timeout = (
            self._bootstrap.apollo_long_poll_timeout_seconds + 5.0
            if long_poll
            else self._bootstrap.apollo_startup_timeout_seconds
        )
        try:
            async with asyncio.timeout(timeout):
                return await self._client.send(request)
        except TimeoutError as exc:
            operation = "long poll" if long_poll else "configuration fetch"
            raise ApolloConfigurationError(f"Apollo {operation} timed out") from exc
        except httpx.TransportError as exc:
            raise ApolloConfigurationError(f"Apollo request failed: {type(exc).__name__}") from exc

    def _authorization_headers(self, url: httpx.URL) -> dict[str, str]:
        secret = self._bootstrap.apollo_access_key_secret
        if not secret:
            return {}
        timestamp = str(int(time.time() * 1000))
        path_with_query = url.raw_path.decode("ascii")
        string_to_sign = f"{timestamp}\n{path_with_query}".encode()
        digest = hmac.new(secret.encode(), string_to_sign, hashlib.sha1).digest()
        signature = base64.b64encode(digest).decode("ascii")
        return {
            "Authorization": f"Apollo {self._bootstrap.apollo_app_id}:{signature}",
            "Timestamp": timestamp,
        }

    @staticmethod
    def _raise_for_status(response: httpx.Response, *, operation: str) -> None:
        if response.status_code < 400:
            return
        raise ApolloConfigurationError(
            f"Apollo {operation} failed with HTTP {response.status_code}"
        )


class ApolloConfigSource:
    """Ordered Apollo namespaces with last-known-good disk caching."""

    def __init__(
        self,
        bootstrap: ApolloBootstrapSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.bootstrap = bootstrap
        self.client = ApolloClient(bootstrap, http_client=http_client)
        self._namespace_snapshots: dict[str, ApolloNamespaceSnapshot] = {}
        self._notification_ids = {namespace: -1 for namespace in bootstrap.apollo_namespaces}

    async def fetch_all(self) -> ApolloConfigSnapshot:
        """Fetch and merge every configured namespace in declaration order."""

        fetched = await asyncio.gather(
            *(
                self.client.fetch_namespace(namespace)
                for namespace in self.bootstrap.apollo_namespaces
            )
        )
        for namespace, snapshot in zip(self.bootstrap.apollo_namespaces, fetched, strict=True):
            if snapshot is None:
                raise ApolloConfigurationError(
                    f"Apollo returned not-modified without a cached namespace: {namespace}"
                )
            self._namespace_snapshots[namespace] = snapshot
        return self._merged_snapshot()

    async def watch(self, handler: Callable[[ApolloConfigSnapshot], Awaitable[None]]) -> None:
        """Continuously long-poll Apollo and deliver changed merged snapshots."""

        last_refresh = time.monotonic()
        retry_seconds = 1.0
        while True:
            try:
                changed = await self.client.poll_notifications(self._notification_ids)
                for namespace, notification_id in changed.items():
                    current = self._namespace_snapshots.get(namespace)
                    snapshot = await self.client.fetch_namespace(
                        namespace,
                        release_key=current.release_key if current is not None else "",
                    )
                    self._notification_ids[namespace] = notification_id
                    if snapshot is not None:
                        self._namespace_snapshots[namespace] = snapshot
                now = time.monotonic()
                refresh_due = now - last_refresh >= self.bootstrap.apollo_refresh_interval_seconds
                if refresh_due:
                    await self._refresh_all_namespaces()
                    last_refresh = now
                if changed or refresh_due:
                    await handler(self._merged_snapshot())
                retry_seconds = 1.0
                # Real Config Service long polls always suspend, but a mock or proxy
                # may answer immediately. Yield so cancellation and peer tasks stay fair.
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Apollo configuration watch failed; retaining current settings")
                delay = retry_seconds + random.uniform(0, min(1.0, retry_seconds / 4))
                await asyncio.sleep(delay)
                retry_seconds = min(retry_seconds * 2, 30.0)

    def load_cache(self) -> ApolloConfigSnapshot:
        """Load the last validated merged snapshot from disk."""

        path = self._cache_file
        if not path.exists():
            raise ApolloConfigurationError(f"Apollo cache does not exist: {path}")
        try:
            values = flatten_yaml_settings(path.read_text(encoding="utf-8"), source=str(path))
        except OSError as exc:
            raise ApolloConfigurationError(f"Apollo cache cannot be read: {path}") from exc
        return _snapshot(values)

    def save_cache(self, snapshot: ApolloConfigSnapshot) -> None:
        """Atomically save one already validated snapshot with private permissions."""

        path = self._cache_file
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        content = yaml.safe_dump(
            dict(snapshot.values),
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=True,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        temporary_path = Path(temporary_name)
        try:
            os.chmod(temporary_path, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    async def close(self) -> None:
        """Close network resources."""

        await self.client.close()

    async def _refresh_all_namespaces(self) -> None:
        for namespace in self.bootstrap.apollo_namespaces:
            current = self._namespace_snapshots.get(namespace)
            snapshot = await self.client.fetch_namespace(
                namespace,
                release_key=current.release_key if current is not None else "",
            )
            if snapshot is not None:
                self._namespace_snapshots[namespace] = snapshot

    def _merged_snapshot(self) -> ApolloConfigSnapshot:
        values: dict[str, object] = {}
        for namespace in self.bootstrap.apollo_namespaces:
            snapshot = self._namespace_snapshots.get(namespace)
            if snapshot is None:
                raise ApolloConfigurationError(f"Apollo namespace has not been loaded: {namespace}")
            values.update(snapshot.values)
        return _snapshot(values)

    @property
    def _cache_file(self) -> Path:
        identity = "\n".join(
            (
                self.bootstrap.apollo_base_url.rstrip("/"),
                self.bootstrap.apollo_app_id,
                self.bootstrap.apollo_cluster,
                *self.bootstrap.apollo_namespaces,
            )
        )
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        safe_app_id = re.sub(r"[^A-Za-z0-9_.-]", "_", self.bootstrap.apollo_app_id)[:64]
        return self.bootstrap.apollo_cache_path / f"{safe_app_id or 'apollo'}-{digest}.yml"


class SettingsRuntime[SettingsT: BaseModel]:
    """Resolve typed settings and optionally monitor Apollo for valid changes."""

    def __init__(
        self,
        settings_type: type[SettingsT],
        bootstrap: ApolloBootstrapSettings,
        *,
        required_paths: Collection[str] = (),
        environ: Mapping[str, str] | None = None,
        change_handler: SettingsChangeHandler | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings_type = settings_type
        self._bootstrap = bootstrap
        self._required_paths = tuple(required_paths)
        self._environ = environ
        self._change_handler = change_handler
        self._http_client = http_client
        self._source: ApolloConfigSource | None = None
        self._settings: SettingsT | None = None
        self._watch_task: asyncio.Task[None] | None = None

    @property
    def settings(self) -> SettingsT:
        """Return the active immutable settings snapshot."""

        if self._settings is None:
            raise RuntimeError("SettingsRuntime has not started")
        return self._settings

    async def startup(self, *, watch: bool = False) -> SettingsT:
        """Resolve initial settings and optionally start Apollo monitoring."""

        if self._settings is not None:
            return self._settings
        if self._bootstrap.config_source is ConfigurationSource.LOCAL:
            self._settings = load_settings(
                self._settings_type,
                self._bootstrap,
                environ=self._environ,
                required_paths=self._required_paths,
            )
            return self._settings
        self._source = ApolloConfigSource(self._bootstrap, http_client=self._http_client)
        snapshot, cache_snapshot = await self._load_initial_snapshot()
        self._settings = self._validate(snapshot)
        if cache_snapshot:
            self._save_cache(snapshot)
        if watch:
            if self._change_handler is None:
                raise ValueError("change_handler is required when Apollo watch is enabled")
            self._watch_task = asyncio.create_task(
                self._source.watch(self._handle_snapshot),
                name=f"apollo-config-watch:{self._bootstrap.apollo_app_id}",
            )
        return self._settings

    async def shutdown(self) -> None:
        """Stop monitoring and close Apollo resources."""

        task = self._watch_task
        self._watch_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        source = self._source
        self._source = None
        if source is not None:
            await source.close()
        self._settings = None

    async def _load_initial_snapshot(self) -> tuple[ApolloConfigSnapshot, bool]:
        assert self._source is not None
        try:
            remote = await self._source.fetch_all()
            self._validate(remote)
            return remote, True
        except Exception:
            policy = self._bootstrap.apollo_startup_policy
            if policy is ApolloStartupPolicy.REMOTE_ONLY:
                raise
            if policy is ApolloStartupPolicy.LOCAL_FALLBACK:
                logger.warning("Apollo startup failed; using explicitly configured local fallback")
                path = self._bootstrap.config_path
                if not path.exists():
                    return _snapshot({}), False
                return (
                    _snapshot(
                        flatten_yaml_settings(path.read_text(encoding="utf-8"), source=str(path))
                    ),
                    False,
                )
            try:
                cached = self._source.load_cache()
                self._validate(cached)
            except Exception as cache_exc:
                raise ApolloConfigurationError(
                    "Apollo startup failed and no valid last-known-good cache is available"
                ) from cache_exc
            logger.warning("Apollo startup failed; using last-known-good cached configuration")
            return cached, False

    def _validate(self, snapshot: ApolloConfigSnapshot) -> SettingsT:
        return load_settings_from_values(
            self._settings_type,
            snapshot.values,
            environ=self._environ,
            required_paths=self._required_paths,
            source=(
                f"Apollo app={self._bootstrap.apollo_app_id} "
                f"cluster={self._bootstrap.apollo_cluster}"
            ),
        )

    async def _handle_snapshot(self, snapshot: ApolloConfigSnapshot) -> None:
        assert self._source is not None
        current = self.settings
        try:
            candidate = self._validate(snapshot)
        except Exception as exc:
            logger.warning(
                "Rejected invalid Apollo configuration update (%s)",
                type(exc).__name__,
            )
            return
        self._save_cache(snapshot)
        if candidate == current:
            return
        self._settings = candidate
        assert self._change_handler is not None
        result = self._change_handler(current, candidate)
        if inspect.isawaitable(result):
            await result

    def _save_cache(self, snapshot: ApolloConfigSnapshot) -> None:
        assert self._source is not None
        try:
            self._source.save_cache(snapshot)
        except OSError as exc:
            logger.warning(
                "Could not persist Apollo last-known-good cache (%s)",
                type(exc).__name__,
            )


async def load_settings_once[SettingsT: BaseModel](
    settings_type: type[SettingsT],
    bootstrap: ApolloBootstrapSettings,
    *,
    required_paths: Collection[str] = (),
    environ: Mapping[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> SettingsT:
    """Load one local or Apollo snapshot without starting a listener."""

    runtime = SettingsRuntime(
        settings_type,
        bootstrap,
        required_paths=required_paths,
        environ=environ,
        http_client=http_client,
    )
    try:
        return await runtime.startup(watch=False)
    finally:
        await runtime.shutdown()


def _decode_namespace_values(
    namespace: str,
    configurations: Mapping[object, object],
) -> dict[str, object]:
    if namespace.lower().endswith((".yml", ".yaml")):
        content = configurations.get("content")
        if not isinstance(content, str):
            raise ApolloConfigurationError(
                f"Apollo YAML namespace {namespace} must expose string key 'content'"
            )
        return flatten_yaml_settings(content, source=f"Apollo namespace {namespace}")
    output: dict[str, object] = {}
    for key, value in configurations.items():
        if not isinstance(key, str):
            raise ApolloConfigurationError(
                f"Apollo namespace {namespace} configuration keys must be strings"
            )
        output[key] = value
    return output


def _snapshot(values: Mapping[str, object]) -> ApolloConfigSnapshot:
    normalized = dict(values)
    canonical = yaml.safe_dump(
        normalized,
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=True,
    )
    return ApolloConfigSnapshot(
        values=normalized,
        revision=hashlib.sha256(canonical.encode()).hexdigest(),
    )
