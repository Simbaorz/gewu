"""Two-stage bootstrap and typed application configuration loading."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from enum import StrEnum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gewu_core.size import parse_size_bytes

DEFAULT_CONFIG_FILE = Path("conf") / "conf.yml"


class DeploymentMode(StrEnum):
    """Supported application deployment modes."""

    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class ConfigurationSource(StrEnum):
    """Supported sources for typed application configuration."""

    LOCAL = "local"
    APOLLO = "apollo"


class ApolloStartupPolicy(StrEnum):
    """Fallback behavior when Apollo cannot serve startup configuration."""

    CACHE_OR_FAIL = "cache_or_fail"
    REMOTE_ONLY = "remote_only"
    LOCAL_FALLBACK = "local_fallback"


class BootstrapSettings(BaseModel):
    """Local settings needed before the typed YAML configuration is available."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)

    project_name: str = Field(default="gewu", alias="PROJECT_NAME")
    project_home: Path = Field(default_factory=Path.cwd, alias="PROJECT_HOME")
    mode: DeploymentMode = Field(default=DeploymentMode.DEV, alias="MODE")
    timezone: str = Field(default="Asia/Shanghai", alias="TIMEZONE")
    config_file: Path = Field(default=DEFAULT_CONFIG_FILE, alias="CONFIG_FILE")

    @property
    def config_path(self) -> Path:
        """Return the absolute YAML configuration path for this application."""

        if self.config_file.is_absolute():
            return self.config_file
        return self.project_home / self.config_file


class ApolloBootstrapSettings(BootstrapSettings):
    """Bootstrap material needed before local or Apollo configuration can load."""

    config_source: ConfigurationSource = Field(
        default=ConfigurationSource.LOCAL,
        alias="CONFIG_SOURCE",
    )
    apollo_base_url: str = Field(default="", alias="APOLLO_BASE_URL")
    apollo_app_id: str = Field(default="", alias="APOLLO_APP_ID")
    apollo_cluster: str = Field(default="default", alias="APOLLO_CLUSTER")
    apollo_namespaces: tuple[str, ...] = Field(
        default=("application.yml",),
        alias="APOLLO_NAMESPACES",
    )
    apollo_access_key_secret: str = Field(
        default="",
        alias="APOLLO_ACCESS_KEY_SECRET",
        repr=False,
    )
    apollo_client_ip: str = Field(default="", alias="APOLLO_CLIENT_IP")
    apollo_label: str = Field(default="", alias="APOLLO_LABEL")
    apollo_startup_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        alias="APOLLO_STARTUP_TIMEOUT_SECONDS",
    )
    apollo_long_poll_timeout_seconds: float = Field(
        default=70.0,
        gt=60,
        alias="APOLLO_LONG_POLL_TIMEOUT_SECONDS",
    )
    apollo_refresh_interval_seconds: float = Field(
        default=300.0,
        gt=0,
        alias="APOLLO_REFRESH_INTERVAL_SECONDS",
    )
    apollo_cache_dir: Path = Field(default=Path("var/apollo"), alias="APOLLO_CACHE_DIR")
    apollo_startup_policy: ApolloStartupPolicy = Field(
        default=ApolloStartupPolicy.CACHE_OR_FAIL,
        alias="APOLLO_STARTUP_POLICY",
    )

    @field_validator("apollo_namespaces", mode="before")
    @classmethod
    def parse_apollo_namespaces(cls, value: object) -> object:
        """Accept comma-separated environment values as an ordered namespace list."""

        if isinstance(value, str):
            return tuple(item.strip() for item in value.split(",") if item.strip())
        return value

    @model_validator(mode="after")
    def validate_apollo_bootstrap(self) -> ApolloBootstrapSettings:
        """Require connection identity only when Apollo is selected."""

        if self.config_source is not ConfigurationSource.APOLLO:
            return self
        if not self.apollo_base_url.strip():
            raise ValueError("APOLLO_BASE_URL is required when CONFIG_SOURCE=apollo")
        if not self.apollo_app_id.strip():
            raise ValueError("APOLLO_APP_ID is required when CONFIG_SOURCE=apollo")
        if not self.apollo_cluster.strip():
            raise ValueError("APOLLO_CLUSTER cannot be empty")
        if not self.apollo_namespaces:
            raise ValueError("APOLLO_NAMESPACES must contain at least one namespace")
        return self

    @property
    def apollo_cache_path(self) -> Path:
        """Return the absolute directory for last-known-good Apollo snapshots."""

        if self.apollo_cache_dir.is_absolute():
            return self.apollo_cache_dir
        return self.project_home / self.apollo_cache_dir


class SettingsModel(BaseModel):
    """Strict base class for application-owned typed settings models."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid", frozen=True)


_BOOTSTRAP_ENV_NAMES = {
    "CONFIG_FILE",
    "MODE",
    "PROJECT_HOME",
    "PROJECT_NAME",
    "TIMEZONE",
}


def load_bootstrap_settings(
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BootstrapSettings:
    """Load `.env`, `.env.local`, and process environment bootstrap values.

    Process environment has the highest priority. When ``env_file`` is supplied,
    only that dotenv file is read before the process environment.
    """

    return load_bootstrap_settings_as(
        BootstrapSettings,
        env_file,
        environ=environ,
    )


def load_bootstrap_settings_as[BootstrapT: BootstrapSettings](
    settings_type: type[BootstrapT],
    env_file: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> BootstrapT:
    """Load a process-specific BootstrapSettings subtype from local sources."""

    process_env = os.environ if environ is None else environ
    raw_env: dict[str, str] = {}
    if env_file is not None:
        raw_env.update(_read_dotenv(Path(env_file).expanduser()))
    else:
        project_dir = Path(process_env.get("PROJECT_HOME", Path.cwd())).expanduser()
        raw_env.update(_read_dotenv(project_dir / ".env"))
        raw_env.update(_read_dotenv(project_dir / ".env.local"))
    raw_env.update(process_env)

    accepted_names = {
        str(field.alias or name) for name, field in settings_type.model_fields.items()
    }
    values = {key: value for key, value in raw_env.items() if key in accepted_names}
    values.setdefault("PROJECT_NAME", raw_env.get("UV_PROJECT_NAME", "gewu"))
    values.setdefault("PROJECT_HOME", raw_env.get("PROJECT_HOME", str(Path.cwd())))
    values.setdefault("MODE", raw_env.get("ENV", DeploymentMode.DEV.value))
    return settings_type.model_validate(values)


def load_settings[SettingsT: BaseModel](
    settings_type: type[SettingsT],
    bootstrap: BootstrapSettings | None = None,
    *,
    env_file: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
    required_paths: Collection[str] = (),
) -> SettingsT:
    """Load YAML and environment overrides into one application settings model.

    The settings model defines the complete accepted configuration schema. YAML
    keys outside that schema are rejected so configuration typos fail at startup.
    Environment names are derived from field paths, for example ``db.pool_size``
    becomes ``DB_POOL_SIZE``.
    """

    resolved_bootstrap = bootstrap or load_bootstrap_settings(env_file, environ=environ)
    flat_values = _read_flat_yaml(resolved_bootstrap.config_path)
    return load_settings_from_values(
        settings_type,
        flat_values,
        environ=environ,
        required_paths=required_paths,
        source=str(resolved_bootstrap.config_path),
    )


def load_settings_from_values[SettingsT: BaseModel](
    settings_type: type[SettingsT],
    values: Mapping[str, object],
    *,
    environ: Mapping[str, str] | None = None,
    required_paths: Collection[str] = (),
    source: str = "configuration",
) -> SettingsT:
    """Validate flattened configuration values using the standard override rules."""

    flat_values = dict(values)
    model_paths = _model_paths(settings_type)
    unknown = sorted(set(flat_values) - set(model_paths))
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown configuration keys in {source}: {joined}")

    process_env = os.environ if environ is None else environ
    env_paths = {_to_env_name(path): path for path in model_paths}
    for env_name, path in env_paths.items():
        if env_name in process_env:
            flat_values[path] = process_env[env_name]

    missing = sorted(
        path
        for path in required_paths
        if not any(
            configured == path or configured.startswith(f"{path}.") for configured in flat_values
        )
    )
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required configuration paths in {source}: {joined}")

    parsed = {
        path: _parse_config_value(path, value)
        for path, value in flat_values.items()
        if path in model_paths
    }
    return settings_type.model_validate(_materialize_model_values(settings_type, parsed))


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return {key: value for key, value in dotenv_values(path).items() if value is not None}


def _read_flat_yaml(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return flatten_yaml_settings(path.read_text(encoding="utf-8"), source=str(path))


def flatten_yaml_settings(content: str, *, source: str) -> dict[str, object]:
    """Decode one YAML mapping into the flattened paths used by settings models."""

    try:
        loaded = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid config YAML: {source}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config YAML must be a mapping: {source}")
    output: dict[str, object] = {}
    _flatten_mapping(loaded, "", output, source)
    return output


def _flatten_mapping(
    mapping: dict[object, object],
    prefix: str,
    output: dict[str, object],
    source: str,
) -> None:
    for key, value in mapping.items():
        if not isinstance(key, str):
            raise ValueError(f"Config YAML keys must be strings: {source}")
        normalized = key.strip()
        if not normalized:
            raise ValueError(f"Config YAML keys cannot be empty: {source}")
        full_key = f"{prefix}.{normalized}" if prefix else normalized
        if isinstance(value, dict):
            _flatten_mapping(value, full_key, output, source)
        else:
            output[full_key] = value


def _model_paths(model_type: type[BaseModel], prefix: str = "") -> set[str]:
    paths: set[str] = set()
    for name, field in model_type.model_fields.items():
        alias = str(field.alias or name)
        full_path = f"{prefix}.{alias}" if prefix else alias
        nested_type = _nested_model_type(field.annotation)
        if nested_type is None:
            paths.add(full_path)
        else:
            paths.update(_model_paths(nested_type, full_path))
    return paths


def _nested_model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    origin = get_origin(annotation)
    if origin not in (UnionType, Union):
        return None
    nested = [
        value
        for value in get_args(annotation)
        if isinstance(value, type) and issubclass(value, BaseModel)
    ]
    return nested[0] if len(nested) == 1 else None


def _materialize_model_values(
    model_type: type[BaseModel],
    values: Mapping[str, object],
    prefix: str = "",
) -> dict[str, object]:
    """Project flat paths into the aliases expected by each nested model."""

    output: dict[str, object] = {}
    for name, field in model_type.model_fields.items():
        alias = str(field.alias or name)
        full_path = f"{prefix}.{alias}" if prefix else alias
        nested_type = _nested_model_type(field.annotation)
        if nested_type is None:
            if full_path in values:
                output[alias] = values[full_path]
            continue
        nested_values = _materialize_model_values(nested_type, values, full_path)
        if nested_values:
            output[alias] = nested_values
    return output


def _to_env_name(path: str) -> str:
    return path.upper().replace(".", "_").replace("-", "_")


def _parse_config_value(path: str, value: object) -> object:
    if path.endswith("_bytes") or path.endswith(".bytes"):
        return parse_size_bytes(value)
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return float(stripped) if "." in stripped else int(stripped)
    except ValueError:
        return stripped
