"""Shared process-resource lifecycle for HTTP applications."""

from __future__ import annotations

from collections.abc import Callable

from gewu_core.blocking import configure_blocking_task_runners
from gewu_core.config import BootstrapSettings
from gewu_core.file_tasks import configure_file_task_runners
from gewu_core.http.password_transport import RsaPasswordTransport
from gewu_core.http.settings import HttpInfrastructureSettings, PasswordTransportSettings
from gewu_core.logging import configure_logging
from gewu_core.runtime_health import (
    EventLoopLagMonitor,
    EventLoopLagStats,
    RuntimeReadinessSnapshot,
    process_readiness,
)
from gewu_core.runtime_temp import (
    prepare_runtime_temp_subdirs,
    resolve_runtime_temp_root,
    set_runtime_temp_root_provider,
)


class HttpInfrastructureRuntime:
    """Own shared process configuration and bounded execution resources."""

    def __init__(
        self,
        bootstrap: BootstrapSettings,
        *,
        settings: HttpInfrastructureSettings,
        password_transport_settings: PasswordTransportSettings | None = None,
        require_password_transport: bool = False,
        record_event_loop_lag: Callable[[float], None] | None = None,
    ) -> None:
        self.bootstrap = bootstrap
        self.settings = settings
        self._password_transport_settings = password_transport_settings
        self._require_password_transport = require_password_transport
        self._event_loop_lag_monitor = EventLoopLagMonitor(record_lag=record_event_loop_lag)
        self.password_transport: RsaPasswordTransport | None = None
        self._started = False

    async def startup(self) -> None:
        """Apply validated settings and start process-local resources."""
        if self._started:
            return
        settings = self.settings
        configure_logging(settings.log)
        configure_blocking_task_runners(settings.blocking_io)
        configure_file_task_runners(settings.blocking_io.filesystem)
        temp_root = resolve_runtime_temp_root(
            settings.runtime.temp_dir,
            self.bootstrap.project_home,
        )
        set_runtime_temp_root_provider(lambda: temp_root)
        prepare_runtime_temp_subdirs(("bash", "downloads", "extracts", "file-rollback", "uploads"))
        if self._password_transport_settings is not None:
            self.password_transport = await RsaPasswordTransport.load(
                self._password_transport_settings,
                self.bootstrap.project_home,
                required=self._require_password_transport,
            )
        self._event_loop_lag_monitor.start()
        self._started = True

    async def shutdown(self) -> None:
        """Mark the process unready before stopping its heartbeat."""
        self._started = False
        await self._event_loop_lag_monitor.stop()
        self.password_transport = None
        set_runtime_temp_root_provider(None)

    def readiness_snapshot(self) -> RuntimeReadinessSnapshot:
        """Return readiness without performing network or filesystem I/O."""
        return process_readiness(
            started=self._started,
            filesystem_saturation_unready_seconds=(
                self.settings.blocking_io.filesystem.saturation_unready_seconds
            ),
        )

    def event_loop_lag_snapshot(self) -> EventLoopLagStats:
        """Return the latest event-loop heartbeat measurements."""
        return self._event_loop_lag_monitor.snapshot()
