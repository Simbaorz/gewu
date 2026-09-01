"""Apollo configuration source and typed runtime behavior."""

from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest
from pydantic import Field

from gewu_core.apollo_config import ApolloClient, SettingsRuntime
from gewu_core.config import ApolloBootstrapSettings, SettingsModel


class DatabaseSettings(SettingsModel):
    pool_size: int = Field(default=5, ge=1)


class ApplicationSettings(SettingsModel):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, alias="db")
    feature_enabled: bool = False


def _bootstrap(tmp_path: Path, **overrides: object) -> ApolloBootstrapSettings:
    values: dict[str, object] = {
        "PROJECT_HOME": tmp_path,
        "CONFIG_SOURCE": "apollo",
        "APOLLO_BASE_URL": "http://apollo.test",
        "APOLLO_APP_ID": "expert-api",
        "APOLLO_NAMESPACES": "common.yml,application.yml",
    }
    values.update(overrides)
    return ApolloBootstrapSettings.model_validate(values)


@pytest.mark.asyncio
async def test_runtime_merges_yaml_namespaces_then_environment(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        namespace = request.url.path.rsplit("/", 1)[-1]
        content = (
            "db:\n  pool_size: 3\nfeature_enabled: false\n"
            if namespace == "common.yml"
            else "db:\n  pool_size: 7\n"
        )
        return httpx.Response(
            200,
            request=request,
            json={"configurations": {"content": content}, "releaseKey": namespace},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = SettingsRuntime(
        ApplicationSettings,
        _bootstrap(tmp_path),
        environ={"FEATURE_ENABLED": "true"},
        http_client=client,
    )
    try:
        settings = await runtime.startup()
    finally:
        await runtime.shutdown()
        await client.aclose()

    assert settings.database.pool_size == 7
    assert settings.feature_enabled is True
    assert list((tmp_path / "var" / "apollo").glob("expert-api-*.yml"))


@pytest.mark.asyncio
async def test_runtime_uses_last_known_good_cache_when_apollo_is_unavailable(
    tmp_path: Path,
) -> None:
    bootstrap = _bootstrap(tmp_path, APOLLO_NAMESPACES="application.yml")

    def available(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "configurations": {"content": "db:\n  pool_size: 9\n"},
                "releaseKey": "release-1",
            },
        )

    first_client = httpx.AsyncClient(transport=httpx.MockTransport(available))
    first = SettingsRuntime(
        ApplicationSettings,
        bootstrap,
        http_client=first_client,
    )
    await first.startup()
    await first.shutdown()
    await first_client.aclose()

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    second_client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    second = SettingsRuntime(
        ApplicationSettings,
        bootstrap,
        http_client=second_client,
    )
    try:
        settings = await second.startup()
    finally:
        await second.shutdown()
        await second_client.aclose()

    assert settings.database.pool_size == 9


@pytest.mark.asyncio
async def test_local_fallback_is_not_written_as_an_apollo_cache(tmp_path: Path) -> None:
    config_path = tmp_path / "fallback.yml"
    config_path.write_text("db:\n  pool_size: 13\n", encoding="utf-8")
    bootstrap = _bootstrap(
        tmp_path,
        APOLLO_NAMESPACES="application.yml",
        APOLLO_STARTUP_POLICY="local_fallback",
        CONFIG_FILE=config_path,
    )

    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("offline", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(unavailable))
    runtime = SettingsRuntime(ApplicationSettings, bootstrap, http_client=client)
    try:
        settings = await runtime.startup()
    finally:
        await runtime.shutdown()
        await client.aclose()

    assert settings.database.pool_size == 13
    assert list((tmp_path / "var" / "apollo").glob("*.yml")) == []


@pytest.mark.asyncio
async def test_runtime_can_start_again_after_shutdown(tmp_path: Path) -> None:
    calls = 0

    def available(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            request=request,
            json={
                "configurations": {"content": f"db:\n  pool_size: {calls + 1}\n"},
                "releaseKey": f"release-{calls}",
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(available))
    runtime = SettingsRuntime(
        ApplicationSettings,
        _bootstrap(tmp_path, APOLLO_NAMESPACES="application.yml"),
        http_client=client,
    )
    try:
        first = await runtime.startup()
        await runtime.shutdown()
        second = await runtime.startup()
    finally:
        await runtime.shutdown()
        await client.aclose()

    assert first.database.pool_size == 2
    assert second.database.pool_size == 3
    assert calls == 2


@pytest.mark.asyncio
async def test_runtime_notifies_only_after_valid_changed_snapshot(tmp_path: Path) -> None:
    notification_calls = 0
    config_calls = 0
    changed = asyncio.Event()
    observed: list[tuple[int, int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal notification_calls, config_calls
        if request.url.path == "/notifications/v2":
            notification_calls += 1
            if notification_calls == 1:
                return httpx.Response(
                    200,
                    request=request,
                    json=[{"namespaceName": "application.yml", "notificationId": 1}],
                )
            return httpx.Response(304, request=request)
        config_calls += 1
        pool_size = 3 if config_calls == 1 else 11
        return httpx.Response(
            200,
            request=request,
            json={
                "configurations": {"content": f"db:\n  pool_size: {pool_size}\n"},
                "releaseKey": f"release-{config_calls}",
            },
        )

    async def on_change(old: ApplicationSettings, new: ApplicationSettings) -> None:
        observed.append((old.database.pool_size, new.database.pool_size))
        changed.set()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    runtime = SettingsRuntime(
        ApplicationSettings,
        _bootstrap(tmp_path, APOLLO_NAMESPACES="application.yml"),
        change_handler=on_change,
        http_client=client,
    )
    try:
        initial = await runtime.startup(watch=True)
        await asyncio.wait_for(changed.wait(), timeout=1)
    finally:
        await runtime.shutdown()
        await client.aclose()

    assert initial.database.pool_size == 3
    assert observed == [(3, 11)]


@pytest.mark.asyncio
async def test_client_signs_access_key_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(
            200,
            request=request,
            json={"configurations": {"content": "feature_enabled: true\n"}},
        )

    monkeypatch.setattr("gewu_core.apollo_config.time.time", lambda: 1_700_000_000.0)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = ApolloClient(
        _bootstrap(
            tmp_path,
            APOLLO_NAMESPACES="application.yml",
            APOLLO_ACCESS_KEY_SECRET="access-secret",
        ),
        http_client=http_client,
    )
    try:
        await client.fetch_namespace("application.yml")
    finally:
        await client.close()
        await http_client.aclose()

    assert observed[0].headers["Timestamp"] == "1700000000000"
    assert observed[0].headers["Authorization"].startswith("Apollo expert-api:")
