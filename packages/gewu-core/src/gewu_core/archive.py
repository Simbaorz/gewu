"""Safe bounded ZIP package inspection, extraction, and creation."""

from __future__ import annotations

import shutil
import stat
import tempfile
import zipfile
from collections.abc import Iterator
from contextlib import contextmanager
from errno import ENAMETOOLONG
from pathlib import Path, PurePosixPath

from pydantic import BaseModel, ConfigDict

from gewu_core.file_tasks import FileTaskLane, run_file_task
from gewu_core.filesystem import regular_file_size_sum
from gewu_core.runtime_temp import runtime_temp_subdir

MAX_ARCHIVE_ENTRIES = 4096
MAX_ARCHIVE_MEMBER_PATH_BYTES = 1024
MAX_ARCHIVE_TOTAL_PATH_BYTES = 4 * 1024 * 1024
MAX_ZIP_ENTRIES = 4096
MAX_ZIP_MEMBER_PATH_BYTES = 1024
MAX_ZIP_TOTAL_PATH_BYTES = 4 * 1024 * 1024
MAX_ZIP_COMPRESSION_RATIO = 200
type PackageContent = bytes | Path


class ArchiveValidationError(Exception):
    """Base error for an invalid archive package."""


class ArchiveLimitExceededError(ArchiveValidationError):
    """Raised when an archive exceeds a resource budget."""


class DirectoryArchiveTooLargeError(Exception):
    """Raised when a generated directory archive exceeds its budget."""


class ExtractedPackage(BaseModel):
    """Temporary extracted package details valid inside its context."""

    model_config = ConfigDict(frozen=True)

    source_root: Path
    extract_root: Path
    size_bytes: int


def package_content_size(content: PackageContent) -> int:
    """Return package bytes without loading a temporary file into memory."""

    return len(content) if isinstance(content, bytes) else content.stat().st_size


async def package_content_size_async(content: PackageContent) -> int:
    """Return package bytes outside the event-loop thread."""

    if isinstance(content, bytes):
        return len(content)
    return await run_file_task(package_content_size, content, lane=FileTaskLane.INTERACTIVE)


async def create_directory_archive(
    directory: Path,
    *,
    fallback_root_name: str,
    max_bytes: int,
) -> Path:
    """Build a bounded temporary ZIP outside the event-loop thread."""

    return await run_file_task(
        _create_directory_archive,
        directory,
        fallback_root_name=fallback_root_name,
        max_bytes=max_bytes,
        cancel_result_cleanup=delete_temp_file,
    )


def zip_directory(directory: Path, *, fallback_root_name: str) -> Path:
    """Create a bounded temporary ZIP for a directory tree."""

    archive_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".zip",
        dir=runtime_temp_subdir("downloads"),
    )
    archive_path = Path(archive_file.name)
    archive_file.close()
    root_name = directory.name or fallback_root_name
    try:
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            has_items = False
            entry_count = 0
            total_path_bytes = 0
            for item in sorted(directory.rglob("*")):
                if item.is_symlink():
                    continue
                relative_name = item.relative_to(directory).as_posix()
                archive_name = f"{root_name}/{relative_name}"
                entry_count += 1
                path_bytes = len(archive_name.encode("utf-8"))
                total_path_bytes += path_bytes
                if (
                    entry_count > MAX_ARCHIVE_ENTRIES
                    or path_bytes > MAX_ARCHIVE_MEMBER_PATH_BYTES
                    or total_path_bytes > MAX_ARCHIVE_TOTAL_PATH_BYTES
                ):
                    raise DirectoryArchiveTooLargeError
                if item.is_dir():
                    if not any(item.iterdir()):
                        archive.writestr(f"{archive_name}/", "")
                        has_items = True
                    continue
                if item.is_file():
                    archive.write(item, archive_name)
                    has_items = True
            if not has_items:
                archive.writestr(f"{root_name}/", "")
    except Exception:
        archive_path.unlink(missing_ok=True)
        raise
    return archive_path


def delete_temp_file(path: Path) -> None:
    """Remove a temporary archive if present."""

    path.unlink(missing_ok=True)


async def delete_temp_file_async(path: Path) -> None:
    """Remove a temporary archive outside the event-loop thread."""

    await run_file_task(
        delete_temp_file,
        path,
        lane=FileTaskLane.INTERACTIVE,
        wait_on_cancel=True,
    )


def _create_directory_archive(
    directory: Path,
    *,
    fallback_root_name: str,
    max_bytes: int,
) -> Path:
    if regular_file_size_sum(directory) > max_bytes:
        raise DirectoryArchiveTooLargeError
    archive_path = zip_directory(directory, fallback_root_name=fallback_root_name)
    if archive_path.stat().st_size <= max_bytes:
        return archive_path
    archive_path.unlink(missing_ok=True)
    raise DirectoryArchiveTooLargeError


@contextmanager
def extracted_package(
    content: PackageContent,
    max_package_bytes: int,
    *,
    temp_prefix: str,
) -> Iterator[ExtractedPackage]:
    """Safely extract a package into a managed temporary directory."""

    with tempfile.TemporaryDirectory(
        prefix=temp_prefix,
        dir=runtime_temp_subdir("extracts"),
    ) as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = content if isinstance(content, Path) else temp_path / "package.zip"
        extract_root = temp_path / "extracted"
        if isinstance(content, bytes):
            archive_path.write_bytes(content)
        extract_root.mkdir()
        try:
            with zipfile.ZipFile(archive_path) as archive:
                extract_zip_package(archive, extract_root, max_package_bytes)
        except zipfile.BadZipFile as exc:
            raise ArchiveValidationError("must be a valid zip file.") from exc
        source_root = package_source_root(extract_root)
        size_bytes = regular_file_size_sum(source_root)
        if size_bytes > max_package_bytes:
            raise ArchiveLimitExceededError(f"content exceeds {max_package_bytes} bytes limit.")
        yield ExtractedPackage(
            source_root=source_root,
            extract_root=extract_root,
            size_bytes=size_bytes,
        )


def extract_zip_package(
    archive: zipfile.ZipFile,
    destination: Path,
    max_package_bytes: int,
) -> None:
    """Safely extract a ZIP package into destination."""

    destination_root = destination.resolve()
    members = archive.infolist()
    if len(members) > MAX_ZIP_ENTRIES:
        raise ArchiveLimitExceededError("contains too many entries.")
    total_path_bytes = sum(
        len(decoded_zip_member_name(member).encode("utf-8")) for member in members
    )
    if total_path_bytes > MAX_ZIP_TOTAL_PATH_BYTES:
        raise ArchiveLimitExceededError("paths exceed the configured limit.")
    total_size = 0
    for member in members:
        relative_path = safe_zip_member_path(member)
        if relative_path is None:
            continue
        target_path = (destination_root / relative_path).resolve()
        if target_path != destination_root and destination_root not in target_path.parents:
            raise ArchiveValidationError("contains unsafe paths.")
        if member.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        total_size += member.file_size
        if total_size > max_package_bytes:
            raise ArchiveLimitExceededError(f"content exceeds {max_package_bytes} bytes limit.")
        if (
            member.file_size > 1024 * 1024
            and member.compress_size > 0
            and member.file_size / member.compress_size > MAX_ZIP_COMPRESSION_RATIO
        ):
            raise ArchiveLimitExceededError("contains an excessive compression ratio.")
        try:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target_path.open("wb") as target:
                shutil.copyfileobj(source, target)
        except OSError as exc:
            if exc.errno == ENAMETOOLONG:
                raise ArchiveValidationError("contains a file name that is too long.") from exc
            raise


def safe_zip_member_path(member: zipfile.ZipInfo) -> Path | None:
    """Return a safe relative filesystem path for one ZIP member."""

    member_name = decoded_zip_member_name(member).replace("\\", "/")
    if len(member_name.encode("utf-8")) > MAX_ZIP_MEMBER_PATH_BYTES:
        raise ArchiveValidationError("contains a path that is too long.")
    parts = [part for part in PurePosixPath(member_name).parts if part not in {"", "."}]
    if not parts or parts[0] == "__MACOSX":
        return None
    if PurePosixPath(member_name).is_absolute() or any(part == ".." for part in parts):
        raise ArchiveValidationError("contains unsafe paths.")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ArchiveValidationError("cannot contain symbolic links.")
    return Path(*parts)


def decoded_zip_member_name(member: zipfile.ZipInfo) -> str:
    """Recover common UTF-8 or GB18030 names lacking the ZIP UTF-8 flag."""

    member_name = member.filename
    if member.flag_bits & 0x800:
        return member_name
    try:
        raw_name = member_name.encode("cp437")
    except UnicodeEncodeError:
        return member_name
    for encoding in ("utf-8", "gb18030"):
        try:
            return raw_name.decode(encoding)
        except UnicodeDecodeError:
            continue
    return member_name


def package_source_root(extract_root: Path) -> Path:
    """Return the directory whose contents should replace the target directory."""

    children = [child for child in extract_root.iterdir() if child.name != "__MACOSX"]
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return extract_root
