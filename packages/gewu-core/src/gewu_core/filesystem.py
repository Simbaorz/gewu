"""Low-level filesystem metadata and atomic mutation helpers."""

from __future__ import annotations

import os
import shutil
import time
from os import stat_result
from pathlib import Path

from gewu_core.ids import new_id
from gewu_core.observability import record_filesystem_scan


def atomic_write_bytes(path: Path, content: bytes) -> None:
    """Atomically replace one file with bytes staged in the same directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{new_id()}.tmp")
    try:
        staging.write_bytes(content)
        os.replace(staging, path)
    finally:
        try:
            staging.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """Atomically replace one file with encoded text."""

    atomic_write_bytes(path, content.encode(encoding))


def ensure_monotonic_mtime(path: Path, previous_version: int) -> None:
    """Ensure a file's microsecond mtime version advances after replacement."""

    stat = path.stat()
    target_mtime_ns = max(
        stat.st_mtime_ns,
        time.time_ns() // 1_000 * 1_000,
        (previous_version + 1) * 1_000,
    )
    if target_mtime_ns != stat.st_mtime_ns:
        os.utime(path, ns=(stat.st_atime_ns, target_mtime_ns))


def file_version(stat: stat_result) -> int:
    """Return the microsecond optimistic-locking token for filesystem metadata."""

    return stat.st_mtime_ns // 1_000


def regular_file_size_sum(root: Path) -> int:
    """Return total bytes for regular files beneath a path."""

    scanned_entries = 0
    scanned_bytes = 0
    outcome = "success"
    try:
        if not root.exists():
            return 0
        if root.is_file():
            scanned_entries = 1
            scanned_bytes = root.stat().st_size
            return scanned_bytes
        for path in root.rglob("*"):
            scanned_entries += 1
            if path.is_file():
                scanned_bytes += path.stat().st_size
        return scanned_bytes
    except BaseException:
        outcome = "failure"
        raise
    finally:
        record_filesystem_scan(
            "regular_file_size_sum",
            scanned_entries,
            scanned_bytes,
            outcome,
        )


def replace_directory_atomically(source_root: Path, target: Path) -> None:
    """Stage and atomically swap a directory, restoring the old tree on failure."""

    operation_id = new_id()
    staging = target.with_name(f".{target.name}.staging-{operation_id}")
    backup = target.with_name(f".{target.name}.backup-{operation_id}")
    shutil.copytree(source_root, staging)
    old_tree_moved = False
    try:
        if target.exists() or target.is_symlink():
            target.replace(backup)
            old_tree_moved = True
        staging.replace(target)
    except Exception:
        if old_tree_moved and backup.exists():
            remove_path(target)
            backup.replace(target)
        remove_path(staging)
        raise
    else:
        remove_path(backup)


def replace_directory_with_staging(staging: Path, target: Path) -> Path | None:
    """Install a prepared sibling directory and return the previous tree."""

    backup = target.with_name(f".{target.name}.backup-{new_id()}")
    old_tree_moved = False
    try:
        if target.exists() or target.is_symlink():
            target.replace(backup)
            old_tree_moved = True
        staging.replace(target)
    except Exception:
        if old_tree_moved and backup.exists():
            remove_path(target)
            backup.replace(target)
        raise
    return backup if old_tree_moved else None


def remove_path(path: Path) -> None:
    """Remove a file, symlink, or directory tree when present."""

    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
