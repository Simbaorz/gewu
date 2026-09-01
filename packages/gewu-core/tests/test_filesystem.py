"""Low-level filesystem mutation primitives."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pytest

from gewu_core.filesystem import (
    atomic_write_bytes,
    ensure_monotonic_mtime,
    file_version,
    regular_file_size_sum,
    replace_directory_atomically,
)
from gewu_core.observability import configure_filesystem_scan_recorder


def test_atomic_write_replaces_file_without_leaving_staging_files(tmp_path: Path) -> None:
    target = tmp_path / "nested/value.txt"
    target.parent.mkdir()
    target.write_bytes(b"old")

    atomic_write_bytes(target, b"new")

    assert target.read_bytes() == b"new"
    assert list(target.parent.iterdir()) == [target]


def test_atomic_directory_replace_restores_old_tree_when_swap_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "new.txt").write_text("new", encoding="utf-8")
    target = tmp_path / "target"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")
    original_replace = Path.replace

    def fail_staging_swap(path: Path, destination: Path) -> Path:
        if path.name.startswith(".target.staging-"):
            raise OSError("injected swap failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_staging_swap)

    with pytest.raises(OSError, match="injected swap failure"):
        replace_directory_atomically(source, target)

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()
    assert not list(tmp_path.glob(".target.*-*"))


def test_ensure_monotonic_mtime_advances_microsecond_version(tmp_path: Path) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"value")
    previous_version = file_version(target.stat())

    os.utime(target, ns=(target.stat().st_atime_ns, previous_version * 1_000))
    ensure_monotonic_mtime(target, previous_version)

    assert file_version(target.stat()) > previous_version


def test_regular_file_size_sum_counts_nested_regular_files(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"abc")
    (root / "nested/b.txt").write_bytes(b"12345")
    scans: list[tuple[str, int, int, str]] = []
    previous = configure_filesystem_scan_recorder(
        lambda operation, entries, size_bytes, outcome: scans.append(
            (operation, entries, size_bytes, outcome)
        )
    )

    try:
        assert regular_file_size_sum(root) == 8
        assert regular_file_size_sum(tmp_path / "missing") == 0
    finally:
        configure_filesystem_scan_recorder(previous)

    assert scans == [
        ("regular_file_size_sum", 3, 8, "success"),
        ("regular_file_size_sum", 0, 0, "success"),
    ]


def test_regular_file_size_sum_preserves_failure_when_observation_fails(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "value.txt"
    target.write_bytes(b"value")

    def fail_observation(
        _operation: str,
        _entries: int,
        _size_bytes: int,
        _outcome: str,
    ) -> None:
        raise RuntimeError("metric-private-secret")

    previous = configure_filesystem_scan_recorder(fail_observation)

    try:
        with caplog.at_level(logging.ERROR):
            assert regular_file_size_sum(target) == 5
    finally:
        configure_filesystem_scan_recorder(previous)

    assert "exception_type=RuntimeError" in caplog.text
    assert "metric-private-secret" not in caplog.text
