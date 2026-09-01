"""Bounded archive creation parity tests."""

from __future__ import annotations

import asyncio
import threading
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from gewu_core import archive
from gewu_core.archive import (
    ArchiveLimitExceededError,
    DirectoryArchiveTooLargeError,
    create_directory_archive,
    extract_zip_package,
    package_content_size_async,
)
from gewu_core.runtime_temp import runtime_temp_subdir, set_runtime_temp_root_provider


@pytest.fixture(autouse=True)
def configured_runtime_temp(tmp_path: Path) -> Iterator[None]:
    previous = set_runtime_temp_root_provider(lambda: tmp_path / ".runtime-temp")
    try:
        yield
    finally:
        set_runtime_temp_root_provider(previous)


async def test_package_path_size_does_not_block_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "package.zip"
    package.write_bytes(b"zip")
    started = threading.Event()
    release = threading.Event()
    original_size = archive.package_content_size

    def delayed_size(content: archive.PackageContent) -> int:
        started.set()
        release.wait(timeout=1)
        return original_size(content)

    monkeypatch.setattr(archive, "package_content_size", delayed_size)
    timer = threading.Timer(0.1, release.set)
    timer.start()
    try:
        task = asyncio.create_task(package_content_size_async(package))
        await asyncio.sleep(0.02)

        assert started.is_set()
        assert not task.done()
        release.set()
        assert await task == 3
    finally:
        release.set()
        timer.cancel()


async def test_create_directory_archive_uses_runtime_temp_and_cleans_failed_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a", encoding="utf-8")
    (source / "b.txt").write_text("b", encoding="utf-8")
    monkeypatch.setattr(archive, "MAX_ARCHIVE_ENTRIES", 1)

    with pytest.raises(DirectoryArchiveTooLargeError):
        await create_directory_archive(source, fallback_root_name="source", max_bytes=1024)

    assert list(runtime_temp_subdir("downloads").iterdir()) == []


async def test_create_directory_archive_rejects_oversized_source_before_writing(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_bytes(b"abcdef")

    with pytest.raises(DirectoryArchiveTooLargeError):
        await create_directory_archive(source, fallback_root_name="source", max_bytes=5)

    assert list(runtime_temp_subdir("downloads").iterdir()) == []


def test_extract_zip_package_recovers_legacy_utf8_member_names(tmp_path: Path) -> None:
    archive_path = tmp_path / "package.zip"
    expected_path = "package/source/kb/\u4e2d\u56fd\u8054\u901a\u4e1a\u52a1\u9700\u6c42\u5206\u6790\u6587\u6863.doc"
    _write_legacy_encoded_member(archive_path, expected_path, "doc")
    destination = tmp_path / "extracted"
    destination.mkdir()

    with zipfile.ZipFile(archive_path) as package:
        extract_zip_package(package, destination, 1024 * 1024)

    assert (destination / expected_path).read_text(encoding="utf-8") == "doc"


def test_extract_zip_package_rejects_excessive_entry_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    archive_path = tmp_path / "package.zip"
    with zipfile.ZipFile(archive_path, "w") as package:
        package.writestr("one.txt", "1")
        package.writestr("two.txt", "2")
        package.writestr("three.txt", "3")
    monkeypatch.setattr(archive, "MAX_ZIP_ENTRIES", 2)
    destination = tmp_path / "extracted"
    destination.mkdir()

    with (
        zipfile.ZipFile(archive_path) as package,
        pytest.raises(
            ArchiveLimitExceededError,
            match="too many entries",
        ),
    ):
        extract_zip_package(package, destination, 1024 * 1024)


def _write_legacy_encoded_member(
    archive_path: Path,
    member_name: str,
    content: str,
) -> None:
    with zipfile.ZipFile(archive_path, "w") as package:
        package.writestr(member_name, content)
    payload = bytearray(archive_path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = 0
        while True:
            index = payload.find(signature, start)
            if index < 0:
                break
            flag_index = index + flag_offset
            flag_bits = int.from_bytes(payload[flag_index : flag_index + 2], "little")
            payload[flag_index : flag_index + 2] = (flag_bits & ~0x800).to_bytes(2, "little")
            start = index + 4
    archive_path.write_bytes(payload)
