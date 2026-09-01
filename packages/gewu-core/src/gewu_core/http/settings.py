"""Typed settings for reusable HTTP process infrastructure."""

from __future__ import annotations

from pydantic import Field, field_validator

from gewu_core.blocking import BlockingTaskSettings
from gewu_core.config import SettingsModel
from gewu_core.file_tasks import FileTaskSettings
from gewu_core.logging import LoggingSettings
from gewu_core.size import parse_size_bytes


class UploadIngressSettings(SettingsModel):
    """Receive concurrency, timeout, and buffered-byte limits for uploads."""

    max_concurrent_ingress: int = Field(default=16, ge=1)
    queue_capacity: int = Field(default=32, ge=0)
    admission_timeout_seconds: float = Field(default=2.0, gt=0)
    body_timeout_seconds: float = Field(default=60.0, gt=0)
    max_buffered_file_bytes: int = Field(default=128 * 1024 * 1024, ge=1)

    @field_validator("max_buffered_file_bytes", mode="before")
    @classmethod
    def parse_buffered_file_bytes(cls, value: object) -> int:
        return parse_size_bytes(value)


class DownloadEgressSettings(SettingsModel):
    """Concurrency, open-file, and media-memory limits for downloads."""

    max_concurrent_egress: int = Field(default=32, ge=1)
    max_open_files: int = Field(default=64, ge=1)
    max_media_bytes_in_flight: int = Field(default=64 * 1024 * 1024, ge=1)
    queue_capacity: int = Field(default=64, ge=0)
    admission_timeout_seconds: float = Field(default=2.0, gt=0)

    @field_validator("max_media_bytes_in_flight", mode="before")
    @classmethod
    def parse_media_bytes_in_flight(cls, value: object) -> int:
        return parse_size_bytes(value)


class HttpBlockingIoSettings(BlockingTaskSettings):
    """Blocking execution settings including filesystem-specific lanes."""

    filesystem: FileTaskSettings = Field(default_factory=FileTaskSettings)
    upload: UploadIngressSettings = Field(default_factory=UploadIngressSettings)
    download: DownloadEgressSettings = Field(default_factory=DownloadEgressSettings)


class PasswordTransportSettings(SettingsModel):
    """RSA key-ring settings for browser password transport."""

    private_key_path: str = ""
    previous_private_key_paths_json: str = "[]"


class RuntimeFilesystemSettings(SettingsModel):
    """Process-local temporary filesystem settings."""

    temp_dir: str = "temp"


class HttpIngressSettings(SettingsModel):
    """Request receive admission limits shared by HTTP processes."""

    max_concurrent_bodies: int = Field(default=32, ge=1)
    queue_capacity: int = Field(default=64, ge=0)
    admission_timeout_seconds: float = Field(default=2.0, gt=0)


class HttpInfrastructureSettings(SettingsModel):
    """Configuration consumed by reusable HTTP process infrastructure."""

    blocking_io: HttpBlockingIoSettings = Field(default_factory=HttpBlockingIoSettings)
    log: LoggingSettings = Field(default_factory=LoggingSettings)
    runtime: RuntimeFilesystemSettings = Field(default_factory=RuntimeFilesystemSettings)
    http_ingress: HttpIngressSettings = Field(default_factory=HttpIngressSettings)
