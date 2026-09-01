"""Process-local observation hooks shared by infrastructure primitives."""

from __future__ import annotations

import logging
from collections.abc import Callable

FilesystemScanRecorder = Callable[[str, int, int, str], None]

_LOGGER = logging.getLogger(__name__)


def _ignore_filesystem_scan(
    _operation: str,
    _scanned_entries: int,
    _scanned_bytes: int,
    _outcome: str,
) -> None:
    return None


_filesystem_scan_recorder: FilesystemScanRecorder = _ignore_filesystem_scan


def configure_filesystem_scan_recorder(
    recorder: FilesystemScanRecorder | None,
) -> FilesystemScanRecorder:
    """Install a process recorder and return the previously configured recorder."""

    global _filesystem_scan_recorder
    previous = _filesystem_scan_recorder
    _filesystem_scan_recorder = recorder or _ignore_filesystem_scan
    return previous


def record_filesystem_scan(
    operation: str,
    scanned_entries: int,
    scanned_bytes: int,
    outcome: str,
) -> None:
    """Record one bounded scan without allowing observation to alter its result."""

    try:
        _filesystem_scan_recorder(operation, scanned_entries, scanned_bytes, outcome)
    except Exception as exc:
        _LOGGER.error(
            "Filesystem scan observation failed. operation=%s outcome=%s exception_type=%s",
            operation,
            outcome,
            type(exc).__name__,
        )
