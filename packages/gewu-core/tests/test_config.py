"""Tests for two-stage typed configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from gewu_core.config import (
    ApolloBootstrapSettings,
    BootstrapSettings,
    DeploymentMode,
    SettingsModel,
    load_bootstrap_settings,
    load_bootstrap_settings_as,
    load_settings,
)
from gewu_core.logging import LoggingSettings


class DatabaseSettings(SettingsModel):
    enabled: bool = True
    pool_size: int = Field(default=5, ge=1)


class ApplicationSettings(SettingsModel):
    database: DatabaseSettings = Field(default_factory=DatabaseSettings, alias="db")
    log: LoggingSettings = Field(default_factory=LoggingSettings)
    upload_max_bytes: int = 1024
    api_key: str = ""


class SubscriberStyleSettings(SettingsModel):
    pool_size: int = Field(default=5, alias="db.pool_size")
    max_file_bytes: int = Field(default=1024, alias="workspace.max_file_bytes")


class ProcessBootstrapSettings(BootstrapSettings):
    secure_cookie: bool = Field(default=False, alias="SECURE_COOKIE")


def test_bootstrap_uses_dotenv_local_then_process_environment(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "PROJECT_NAME=base\nMODE=dev\nTIMEZONE=UTC\n",
        encoding="utf-8",
    )
    (tmp_path / ".env.local").write_text(
        "PROJECT_NAME=local\nMODE=test\n",
        encoding="utf-8",
    )

    settings = load_bootstrap_settings(
        environ={"PROJECT_HOME": str(tmp_path), "PROJECT_NAME": "process"}
    )

    assert settings.project_name == "process"
    assert settings.project_home == tmp_path
    assert settings.mode is DeploymentMode.TEST
    assert settings.timezone == "UTC"
    assert settings.config_path == tmp_path / "conf" / "conf.yml"


def test_explicit_dotenv_does_not_mutate_or_search_process_environment(tmp_path: Path) -> None:
    env_file = tmp_path / "service.env"
    env_file.write_text(
        "PROJECT_NAME=web\nPROJECT_HOME=/srv/gewu\nMODE=prod\nCONFIG_FILE=web.yml\n",
        encoding="utf-8",
    )

    settings = load_bootstrap_settings(env_file, environ={})

    assert settings.project_name == "web"
    assert settings.project_home == Path("/srv/gewu")
    assert settings.mode is DeploymentMode.PROD
    assert settings.config_path == Path("/srv/gewu/web.yml")


def test_invalid_bootstrap_mode_fails_at_startup(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODE=staging\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_bootstrap_settings(env_file, environ={})


def test_apollo_bootstrap_parses_namespaces_and_requires_identity() -> None:
    settings = load_bootstrap_settings_as(
        ApolloBootstrapSettings,
        environ={
            "CONFIG_SOURCE": "apollo",
            "APOLLO_BASE_URL": "http://apollo.test/",
            "APOLLO_APP_ID": "expert-api",
            "APOLLO_NAMESPACES": "common.yml, application.yml",
        },
    )

    assert settings.apollo_namespaces == ("common.yml", "application.yml")

    with pytest.raises(ValidationError, match="APOLLO_APP_ID"):
        load_bootstrap_settings_as(
            ApolloBootstrapSettings,
            environ={
                "CONFIG_SOURCE": "apollo",
                "APOLLO_BASE_URL": "http://apollo.test",
            },
        )


def test_process_specific_bootstrap_subtype_reads_only_declared_values() -> None:
    settings = load_bootstrap_settings_as(
        ProcessBootstrapSettings,
        environ={
            "PROJECT_NAME": "subscriber",
            "SECURE_COOKIE": "true",
            "UNRELATED_SECRET": "ignored",
        },
    )

    assert settings.project_name == "subscriber"
    assert settings.secure_cookie is True


def test_yaml_then_environment_are_loaded_into_application_model(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "conf.yml").write_text(
        "\n".join(
            [
                "db:",
                "  enabled: true",
                "  pool_size: 7",
                "log:",
                "  level: WARNING",
                "  queue_capacity: 64",
                "upload_max_bytes: 20 MB",
                "api_key: yaml-value",
            ]
        ),
        encoding="utf-8",
    )
    bootstrap = BootstrapSettings(PROJECT_HOME=tmp_path)

    settings = load_settings(
        ApplicationSettings,
        bootstrap,
        environ={
            "DB_POOL_SIZE": "11",
            "LOG_LEVEL": "DEBUG",
            "API_KEY": "environment-value",
        },
    )

    assert settings.database.enabled is True
    assert settings.database.pool_size == 11
    assert settings.log.level == "DEBUG"
    assert settings.log.queue_capacity == 64
    assert settings.upload_max_bytes == 20 * 1024 * 1024
    assert settings.api_key == "environment-value"


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    config_path = conf_dir / "conf.yml"
    config_path.write_text("db:\n  pool_szie: 7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="db.pool_szie"):
        load_settings(
            ApplicationSettings,
            BootstrapSettings(PROJECT_HOME=tmp_path),
            environ={},
        )


def test_required_configuration_path_must_be_explicitly_configured(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Missing required configuration paths.*db"):
        load_settings(
            ApplicationSettings,
            BootstrapSettings(PROJECT_HOME=tmp_path),
            environ={},
            required_paths=("db",),
        )


def test_required_configuration_path_can_be_supplied_by_environment(tmp_path: Path) -> None:
    settings = load_settings(
        ApplicationSettings,
        BootstrapSettings(PROJECT_HOME=tmp_path),
        environ={"DB_POOL_SIZE": "9"},
        required_paths=("db",),
    )

    assert settings.database.pool_size == 9


def test_subscriber_style_dotted_field_aliases_remain_supported(tmp_path: Path) -> None:
    conf_dir = tmp_path / "conf"
    conf_dir.mkdir()
    (conf_dir / "conf.yml").write_text(
        "db:\n  pool_size: 7\nworkspace:\n  max_file_bytes: 5MB\n",
        encoding="utf-8",
    )

    settings = load_settings(
        SubscriberStyleSettings,
        BootstrapSettings(PROJECT_HOME=tmp_path),
        environ={"DB_POOL_SIZE": "9"},
    )

    assert settings.pool_size == 9
    assert settings.max_file_bytes == 5 * 1024 * 1024


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yml"
    config_path.write_text("- invalid\n- root\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a mapping"):
        load_settings(
            ApplicationSettings,
            BootstrapSettings(PROJECT_HOME=tmp_path, CONFIG_FILE=config_path),
            environ={},
        )
