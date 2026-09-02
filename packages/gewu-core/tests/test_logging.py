"""Tests for bounded process logging isolation."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Iterator

import pytest
from pydantic import ValidationError

from gewu_core.logging import (
    LoggingSettings,
    configure_logging,
    init_logging,
    logging_queue_stats,
    shutdown_logging,
)


def test_logging_settings_only_accept_runtime_consumed_values() -> None:
    assert LoggingSettings(level="DEBUG", queue_capacity=128).model_dump() == {
        "level": "DEBUG",
        "queue_capacity": 128,
    }
    with pytest.raises(ValidationError, match="json_format"):
        LoggingSettings.model_validate({"json_format": True})


class _BlockingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.started = threading.Event()
        self.unblock = threading.Event()
        self.thread_ids: list[int] = []
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.thread_ids.append(threading.get_ident())
        self.started.set()
        self.unblock.wait(timeout=5)
        self.messages.append(self.format(record))


class _RecordingHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(self.format(record))


@pytest.fixture
def isolated_root_logger() -> Iterator[logging.Logger]:
    shutdown_logging()
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    try:
        root.handlers[:] = []
        root.setLevel(logging.INFO)
        yield root
    finally:
        shutdown_logging()
        root.handlers[:] = old_handlers
        root.setLevel(old_level)


async def test_slow_sink_never_runs_on_event_loop_thread(
    isolated_root_logger: logging.Logger,
) -> None:
    sink = _BlockingHandler()
    isolated_root_logger.addHandler(sink)
    configure_logging(LoggingSettings(queue_capacity=8))
    event_loop_thread = threading.get_ident()

    logging.getLogger("test.slow").info("message %s", "value")

    assert sink.started.wait(timeout=1)
    assert sink.thread_ids[0] != event_loop_thread
    await asyncio.sleep(0)
    sink.unblock.set()
    shutdown_logging()
    assert sink.messages == ["message value"]


def test_full_queue_drops_without_blocking_request_thread(
    isolated_root_logger: logging.Logger,
) -> None:
    sink = _BlockingHandler()
    isolated_root_logger.addHandler(sink)
    configure_logging(LoggingSettings(queue_capacity=1))
    logger = logging.getLogger("test.full")
    logger.info("in-flight")
    assert sink.started.wait(timeout=1)
    logger.info("queued")

    started = time.perf_counter()
    logger.info("must-drop")
    elapsed = time.perf_counter() - started

    stats = logging_queue_stats()
    assert elapsed < 0.1
    assert stats.capacity == 1
    assert stats.queued == 1
    assert stats.dropped_total == 1
    assert stats.dropped_pending_notice == 1
    sink.unblock.set()
    shutdown_logging()


def test_shutdown_restores_handlers_and_stops_listener(
    isolated_root_logger: logging.Logger,
) -> None:
    sink = logging.NullHandler()
    isolated_root_logger.addHandler(sink)
    configure_logging(LoggingSettings())

    assert logging_queue_stats().listener_alive is True
    assert isolated_root_logger.handlers != [sink]

    shutdown_logging()

    assert sink in isolated_root_logger.handlers
    assert all(
        handler.__class__.__name__ != "_DeferredQueueHandler"
        for handler in isolated_root_logger.handlers
    )
    assert logging_queue_stats().listener_alive is False


def test_queue_boundary_preserves_third_party_exception_details(
    isolated_root_logger: logging.Logger,
) -> None:
    sink = _BlockingHandler()
    sink.unblock.set()
    isolated_root_logger.addHandler(sink)
    configure_logging(LoggingSettings(queue_capacity=8))
    logger = logging.getLogger("third.party")

    try:
        raise RuntimeError("external adapter failed")
    except RuntimeError as exc:
        logger.error(
            "Third-party request failed error=%s",
            exc,
            exc_info=True,
            stack_info=True,
        )

    assert sink.started.wait(timeout=1)
    shutdown_logging()
    rendered = "\n".join(sink.messages)
    assert rendered.startswith("Third-party request failed error=<RuntimeError>")
    assert "Traceback" in rendered
    assert "RuntimeError: external adapter failed" in rendered
    assert "Stack (most recent call last)" in rendered


def test_bootstrap_record_factory_preserves_late_third_party_tracebacks(
    isolated_root_logger: logging.Logger,
) -> None:
    init_logging()
    sink = _RecordingHandler()
    logger = logging.getLogger("late.third.party")
    old_handlers = list(logger.handlers)
    old_propagate = logger.propagate
    logger.handlers[:] = [sink]
    logger.propagate = False
    try:
        try:
            raise OSError("bootstrap failed")
        except OSError as exc:
            logger.error("Startup failed: %s", exc, exc_info=True)

        assert len(sink.messages) == 1
        assert sink.messages[0].startswith("Startup failed: <OSError>")
        assert "Traceback" in sink.messages[0]
        assert "OSError: bootstrap failed" in sink.messages[0]
    finally:
        logger.handlers[:] = old_handlers
        logger.propagate = old_propagate
