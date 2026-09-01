"""Bounded non-blocking process logging initialization."""

from __future__ import annotations

import atexit
import copy
import logging
import os
import queue
import threading
from collections.abc import Mapping
from logging.config import dictConfig
from logging.handlers import QueueHandler, QueueListener
from typing import Any, Protocol, cast

from pydantic import BaseModel, ConfigDict, Field

from gewu_core.config import SettingsModel


class LogLevelConfig(Protocol):
    """Configuration values required by bounded process logging."""

    @property
    def level(self) -> str: ...

    @property
    def queue_capacity(self) -> int: ...


class LoggingSettings(SettingsModel):
    """Application logging settings."""

    level: str = "INFO"
    queue_capacity: int = Field(default=8192, ge=1)


class LoggingQueueStats(BaseModel):
    """Process-local bounded logging queue snapshot."""

    model_config = ConfigDict(frozen=True)

    capacity: int
    queued: int
    dropped_total: int
    dropped_pending_notice: int
    listener_alive: bool


class _DeferredQueueHandler(QueueHandler):
    def __init__(
        self,
        log_queue: queue.Queue[logging.LogRecord],
        sinks: tuple[logging.Handler, ...],
    ) -> None:
        super().__init__(log_queue)
        self.sinks = sinks
        self._drop_lock = threading.Lock()
        self._dropped_total = 0
        self._dropped_pending_notice = 0

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        prepared = _sanitize_log_record(copy.copy(record))
        prepared._gewu_log_sinks = self.sinks  # type: ignore[attr-defined]
        return prepared

    def enqueue(self, record: logging.LogRecord) -> None:
        self._enqueue_pending_drop_notice(record)
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            with self._drop_lock:
                self._dropped_total += 1
                self._dropped_pending_notice += 1

    def drop_counts(self) -> tuple[int, int]:
        with self._drop_lock:
            return self._dropped_total, self._dropped_pending_notice

    def _enqueue_pending_drop_notice(self, source: logging.LogRecord) -> None:
        with self._drop_lock:
            dropped = self._dropped_pending_notice
            if not dropped:
                return
            notice = logging.LogRecord(
                name="gewu_core.logging",
                level=logging.WARNING,
                pathname=__file__,
                lineno=0,
                msg="Dropped %d log records because the bounded logging queue was full.",
                args=(dropped,),
                exc_info=None,
            )
            notice.created = source.created
            notice._gewu_log_sinks = self.sinks  # type: ignore[attr-defined]
            try:
                self.queue.put_nowait(notice)
            except queue.Full:
                return
            self._dropped_pending_notice = 0


class _DispatchQueueListener(QueueListener):
    def handle(self, record: logging.LogRecord) -> None:
        sinks: tuple[logging.Handler, ...] = getattr(record, "_gewu_log_sinks", ())
        for sink in sinks:
            if record.levelno < sink.level:
                continue
            try:
                sink.handle(record)
            except Exception:  # noqa: BLE001
                sink.handleError(record)

    def enqueue_sentinel(self) -> None:
        cast(queue.Queue[object], self.queue).put(None)


class _LoggerRegistration:
    def __init__(
        self,
        logger: logging.Logger,
        queue_handler: _DeferredQueueHandler,
        sinks: tuple[logging.Handler, ...],
    ) -> None:
        self.logger = logger
        self.queue_handler = queue_handler
        self.sinks = sinks


class _LoggingRuntime:
    def __init__(
        self,
        *,
        capacity: int,
        log_queue: queue.Queue[logging.LogRecord],
        listener: _DispatchQueueListener,
        registrations: tuple[_LoggerRegistration, ...],
    ) -> None:
        self.pid = os.getpid()
        self.capacity = capacity
        self.log_queue = log_queue
        self.listener = listener
        self.registrations = registrations


_RUNTIME_LOCK = threading.RLock()
_RUNTIME: _LoggingRuntime | None = None
_UPSTREAM_LOG_RECORD_FACTORY = logging.getLogRecordFactory()


def init_logging(level: str = "INFO") -> None:
    """Initialize synchronous logging for the bootstrap phase."""

    _install_safe_log_record_factory()
    shutdown_logging()
    normalized_level = _normalize_level(level)
    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s %(levelname)s [%(name)s] %(message)s",
                    "datefmt": "%Y-%m-%d %H:%M:%S",
                }
            },
            "handlers": {
                "default": {
                    "class": "logging.StreamHandler",
                    "formatter": "default",
                    "level": normalized_level,
                }
            },
            "root": {"handlers": ["default"], "level": normalized_level},
        }
    )


def configure_logging(conf: LogLevelConfig) -> None:
    """Move configured process logging onto one bounded listener thread."""

    _install_safe_log_record_factory()
    normalized_level = _normalize_level(conf.level)
    capacity = max(1, conf.queue_capacity)
    with _RUNTIME_LOCK:
        _shutdown_logging_locked()
        root_logger = logging.getLogger()
        root_logger.setLevel(normalized_level)
        if not root_logger.handlers:
            root_logger.addHandler(_default_stream_handler(normalized_level))
        for handler in root_logger.handlers:
            handler.setLevel(normalized_level)

        log_queue: queue.Queue[logging.LogRecord] = queue.Queue(maxsize=capacity)
        registrations: list[_LoggerRegistration] = []
        for logger in _loggers_with_handlers():
            sinks = tuple(logger.handlers)
            if not sinks:
                continue
            queue_handler = _DeferredQueueHandler(log_queue, sinks)
            logger.handlers[:] = [queue_handler]
            registrations.append(_LoggerRegistration(logger, queue_handler, sinks))

        listener = _DispatchQueueListener(log_queue)
        listener.start()
        global _RUNTIME
        _RUNTIME = _LoggingRuntime(
            capacity=capacity,
            log_queue=log_queue,
            listener=listener,
            registrations=tuple(registrations),
        )


def shutdown_logging() -> None:
    """Drain, stop, and detach the process logging listener."""

    with _RUNTIME_LOCK:
        _shutdown_logging_locked()


def logging_queue_stats() -> LoggingQueueStats:
    """Return bounded logging queue capacity and drop counters."""

    with _RUNTIME_LOCK:
        runtime = _RUNTIME
        if runtime is None or runtime.pid != os.getpid():
            return LoggingQueueStats(
                capacity=0,
                queued=0,
                dropped_total=0,
                dropped_pending_notice=0,
                listener_alive=False,
            )
        dropped_total = 0
        dropped_pending = 0
        for registration in runtime.registrations:
            total, pending = registration.queue_handler.drop_counts()
            dropped_total += total
            dropped_pending += pending
        listener_thread = runtime.listener._thread
        return LoggingQueueStats(
            capacity=runtime.capacity,
            queued=runtime.log_queue.qsize(),
            dropped_total=dropped_total,
            dropped_pending_notice=dropped_pending,
            listener_alive=bool(listener_thread and listener_thread.is_alive()),
        )


def _shutdown_logging_locked() -> None:
    global _RUNTIME
    runtime = _RUNTIME
    if runtime is None:
        return
    _RUNTIME = None
    for registration in runtime.registrations:
        if registration.queue_handler in registration.logger.handlers:
            registration.logger.handlers[:] = list(registration.sinks)
    if runtime.pid == os.getpid():
        runtime.listener.stop()


def _loggers_with_handlers() -> tuple[logging.Logger, ...]:
    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.Logger.manager.loggerDict.values()
        if isinstance(logger, logging.Logger) and logger.handlers
    )
    return tuple(loggers)


def _default_stream_handler(level: str) -> logging.StreamHandler:
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    return handler


def _sanitize_log_arguments(
    values: tuple[object, ...] | Mapping[str, object] | None,
) -> tuple[object, ...] | dict[str, object] | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        return {key: _sanitize_log_value(value) for key, value in values.items()}
    return tuple(_sanitize_log_value(value) for value in values)


def _sanitize_log_value(value: object) -> object:
    if isinstance(value, BaseException):
        return f"<{type(value).__name__}>"
    return value


def _sanitize_log_record(record: logging.LogRecord) -> logging.LogRecord:
    record.msg = _sanitize_log_value(record.msg)
    record.args = _sanitize_log_arguments(record.args)
    record.exc_info = None
    record.exc_text = None
    record.stack_info = None
    return record


def _safe_log_record_factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
    return _sanitize_log_record(_UPSTREAM_LOG_RECORD_FACTORY(*args, **kwargs))


def _install_safe_log_record_factory() -> None:
    global _UPSTREAM_LOG_RECORD_FACTORY
    current = logging.getLogRecordFactory()
    if current is _safe_log_record_factory:
        return
    _UPSTREAM_LOG_RECORD_FACTORY = current
    logging.setLogRecordFactory(_safe_log_record_factory)


def _normalize_level(level: str) -> str:
    value = level.strip().upper()
    return value if value in logging.getLevelNamesMapping() else "INFO"


atexit.register(shutdown_logging)
