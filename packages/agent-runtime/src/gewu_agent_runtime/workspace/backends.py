"""Built-in memory and local-file workspace backends."""

from __future__ import annotations

import asyncio
import codecs
import os
import shutil
from collections.abc import AsyncIterator, Callable, Iterable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from pathlib import Path

from gewu_agent_runtime.workspace.contracts import (
    EntryType,
    WorkspaceEntry,
    WorkspaceLineVisitor,
    WorkspaceTextRange,
)
from gewu_agent_runtime.workspace.errors import (
    WorkspaceCapacityError,
    WorkspaceConflictError,
    WorkspaceNotFoundError,
    WorkspacePathError,
    WorkspacePermissionError,
    WorkspaceScanLimitError,
)
from gewu_agent_runtime.workspace.paths import normalize_backend_path
from gewu_core.file_locks import TaskReentrantFileLock
from gewu_core.file_tasks import FileTaskLane, run_file_mutation, run_file_task
from gewu_core.filesystem import (
    atomic_write_bytes,
    ensure_monotonic_mtime,
    file_version,
    regular_file_size_sum,
)


class InMemoryWorkspaceBackend:
    """Concurrency-safe in-memory backend for tests and ephemeral workspaces."""

    def __init__(self) -> None:
        """Initialize an empty backend with one root directory."""

        self._files: dict[str, bytes] = {}
        self._versions: dict[str, int] = {}
        self._directories: set[str] = {""}
        self._lock = asyncio.Lock()
        self._next_version = 0

    async def stat(self, path: str) -> WorkspaceEntry:
        normalized = normalize_backend_path(path)
        async with self._lock:
            return self._stat(normalized)

    async def list(
        self,
        path: str,
        *,
        max_entries: int | None = None,
    ) -> tuple[WorkspaceEntry, ...]:
        normalized = normalize_backend_path(path)
        async with self._lock:
            if normalized not in self._directories:
                raise WorkspaceNotFoundError(f"Directory does not exist: {path}")
            prefix = f"{normalized}/" if normalized else ""
            children: dict[str, WorkspaceEntry] = {}
            for directory in self._directories:
                if not directory.startswith(prefix) or directory == normalized:
                    continue
                remainder = directory[len(prefix) :]
                if "/" not in remainder:
                    children[remainder] = self._stat(directory)
            for file_path in self._files:
                if not file_path.startswith(prefix):
                    continue
                remainder = file_path[len(prefix) :]
                if "/" not in remainder:
                    children[remainder] = self._stat(file_path)
            _require_scan_count(len(children), max_entries)
            return tuple(children[name] for name in sorted(children))

    async def read_bytes(self, path: str) -> bytes:
        _, data = await self.read_bytes_with_metadata(path)
        return data

    async def read_bytes_with_metadata(self, path: str) -> tuple[WorkspaceEntry, bytes]:
        normalized = normalize_backend_path(path)
        async with self._lock:
            try:
                data = self._files[normalized]
            except KeyError as exc:
                raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
            return self._stat(normalized), data

    async def read_text_range(
        self,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> WorkspaceTextRange:
        _require_text_range(offset, limit)
        normalized = normalize_backend_path(path)
        async with self._lock:
            try:
                data = self._files[normalized]
            except KeyError as exc:
                raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
            lines = data.decode("utf-8").splitlines(keepends=True)
            selected = lines[offset : offset + limit]
            return WorkspaceTextRange(
                entry=self._stat(normalized),
                content="".join(selected),
                total_lines=len(lines),
                start_line=offset,
                num_lines=len(selected),
            )

    async def visit_text_lines(
        self,
        path: str,
        visitor: WorkspaceLineVisitor,
        *,
        max_bytes: int | None = None,
    ) -> tuple[int, bool]:
        normalized = normalize_backend_path(path)
        async with self._lock:
            try:
                data = self._files[normalized]
            except KeyError as exc:
                raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
            _require_content_scan_bytes(len(data), max_bytes)
            entry = self._stat(normalized)
            for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
                if not visitor(entry, line_number, line):
                    return len(data), False
            return len(data), True

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        overwrite: bool = True,
        expected_version: str | None = None,
    ) -> WorkspaceEntry:
        normalized = normalize_backend_path(path)
        if not normalized:
            raise WorkspacePathError("Cannot write the backend root as a file.")
        async with self._lock:
            current = self._stat(normalized) if normalized in self._files else None
            if expected_version is not None and (
                current is None or current.version != expected_version
            ):
                raise WorkspaceConflictError("File has been modified since read.")
            if not overwrite and normalized in self._files:
                raise FileExistsError(path)
            if normalized in self._directories:
                raise IsADirectoryError(path)
            self._ensure_parents(normalized)
            self._files[normalized] = bytes(data)
            self._next_version += 1
            self._versions[normalized] = self._next_version
            self._directories.discard(normalized)
            return self._stat(normalized)

    async def mkdir(self, path: str, *, parents: bool = True) -> WorkspaceEntry:
        normalized = normalize_backend_path(path)
        async with self._lock:
            if normalized in self._files:
                raise FileExistsError(path)
            parent = normalized.rpartition("/")[0]
            if parent and parent not in self._directories and not parents:
                raise WorkspaceNotFoundError(f"Parent directory does not exist: {parent}")
            self._ensure_parents(f"{normalized}/placeholder" if normalized else "placeholder")
            self._directories.add(normalized)
            return self._stat(normalized)

    async def delete(self, path: str, *, recursive: bool = False) -> None:
        normalized = normalize_backend_path(path)
        if not normalized:
            raise WorkspacePathError("Cannot delete the backend root.")
        async with self._lock:
            if normalized in self._files:
                self._files.pop(normalized)
                self._versions.pop(normalized, None)
                return
            if normalized not in self._directories:
                raise WorkspaceNotFoundError(f"Path does not exist: {path}")
            if not recursive:
                raise IsADirectoryError(path)
            prefix = f"{normalized}/"
            for file_path in tuple(self._files):
                if file_path.startswith(prefix):
                    self._files.pop(file_path)
                    self._versions.pop(file_path, None)
            self._directories = {
                directory
                for directory in self._directories
                if directory != normalized and not directory.startswith(prefix)
            }

    async def move(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        source_path = normalize_backend_path(source)
        destination_path = normalize_backend_path(destination)
        async with self._lock:
            if destination_path in self._files or destination_path in self._directories:
                if not overwrite:
                    raise FileExistsError(destination)
                self._remove_path(destination_path)
            if source_path in self._files:
                data = self._files.pop(source_path)
                version = self._versions.pop(source_path)
                self._ensure_parents(destination_path)
                self._files[destination_path] = data
                self._versions[destination_path] = version
                return self._stat(destination_path)
            if source_path not in self._directories:
                raise WorkspaceNotFoundError(f"Path does not exist: {source}")
            self._ensure_parents(f"{destination_path}/placeholder")
            prefix = f"{source_path}/"
            moved_files = {
                f"{destination_path}/{path[len(prefix):]}": data
                for path, data in self._files.items()
                if path.startswith(prefix)
            }
            moved_versions = {
                f"{destination_path}/{path[len(prefix):]}": version
                for path, version in self._versions.items()
                if path.startswith(prefix)
            }
            moved_directories = {
                f"{destination_path}/{path[len(prefix):]}".rstrip("/")
                for path in self._directories
                if path.startswith(prefix)
            }
            self._files = {
                path: data for path, data in self._files.items() if not path.startswith(prefix)
            }
            self._versions = {
                path: version
                for path, version in self._versions.items()
                if not path.startswith(prefix)
            }
            self._directories = {
                path
                for path in self._directories
                if path != source_path and not path.startswith(prefix)
            }
            self._directories.add(destination_path)
            self._directories.update(moved_directories)
            self._files.update(moved_files)
            self._versions.update(moved_versions)
            return self._stat(destination_path)

    def _stat(self, path: str) -> WorkspaceEntry:
        if path in self._files:
            data = self._files[path]
            return WorkspaceEntry(
                path=path,
                entry_type=EntryType.FILE,
                size=len(data),
                version=str(self._versions[path]),
            )
        if path in self._directories:
            return WorkspaceEntry(path=path, entry_type=EntryType.DIRECTORY)
        raise WorkspaceNotFoundError(f"Path does not exist: {path}")

    def _ensure_parents(self, path: str) -> None:
        parent = path.rpartition("/")[0]
        current = parent
        while current:
            if current in self._files:
                raise NotADirectoryError(current)
            current = current.rpartition("/")[0]
        while parent:
            self._directories.add(parent)
            parent = parent.rpartition("/")[0]
        self._directories.add("")

    def _remove_path(self, path: str) -> None:
        self._files.pop(path, None)
        self._versions.pop(path, None)
        prefix = f"{path}/"
        for file_path in tuple(self._files):
            if file_path.startswith(prefix):
                self._files.pop(file_path)
                self._versions.pop(file_path, None)
        self._directories = {
            directory
            for directory in self._directories
            if directory != path and not directory.startswith(prefix)
        }


class LocalWorkspaceBackend:
    """Local filesystem backend confined to an explicit physical root."""

    def __init__(
        self,
        root: Path,
        *,
        create_root: bool = True,
        max_file_bytes: int | None = None,
        max_total_bytes: int | None = None,
        mutation_locks: TaskReentrantFileLock | None = None,
        mutation_lock_path: Path | None = None,
        protected_directory_names: Iterable[str] = (),
        protected_path_error: str = "Workspace path is protected from mutation.",
        total_capacity_error: Callable[[int], str] | None = None,
    ) -> None:
        """Initialize an explicit root, optionally materializing it eagerly."""

        self._root = root.resolve()
        self._virtual_root = not create_root
        if max_file_bytes is not None and max_file_bytes < 1:
            raise ValueError("max_file_bytes must be at least 1.")
        if max_total_bytes is not None and max_total_bytes < 1:
            raise ValueError("max_total_bytes must be at least 1.")
        self.max_file_bytes = max_file_bytes
        self.max_total_bytes = max_total_bytes
        self._total_capacity_error = total_capacity_error or _default_total_capacity_error
        self._mutation_locks = mutation_locks
        self._mutation_lock_path = mutation_lock_path
        if (mutation_locks is None) != (mutation_lock_path is None):
            raise ValueError("mutation_locks and mutation_lock_path must be provided together.")
        if isinstance(protected_directory_names, str):
            raise TypeError("protected_directory_names must be an iterable of directory names.")
        protected_names: set[str] = set()
        for name in protected_directory_names:
            normalized = normalize_backend_path(name)
            if not normalized or "/" in normalized:
                raise ValueError("Protected Workspace directories must be top-level names.")
            protected_names.add(normalized)
        if protected_names and not protected_path_error:
            raise ValueError("protected_path_error is required when directories are protected.")
        self._protected_directory_names = frozenset(protected_names)
        self._protected_path_error = protected_path_error
        if create_root:
            self._root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def stat(self, path: str) -> WorkspaceEntry:
        return await run_file_task(self._stat_sync, path, lane=FileTaskLane.INTERACTIVE)

    async def list(
        self,
        path: str,
        *,
        max_entries: int | None = None,
    ) -> tuple[WorkspaceEntry, ...]:
        return await run_file_task(
            self._list_sync,
            path,
            max_entries,
            lane=FileTaskLane.INTERACTIVE,
        )

    async def read_bytes(self, path: str) -> bytes:
        _, data = await self.read_bytes_with_metadata(path)
        return data

    async def read_bytes_with_metadata(self, path: str) -> tuple[WorkspaceEntry, bytes]:
        async with self._lock:
            return await run_file_task(
                self._read_snapshot_sync,
                path,
                lane=FileTaskLane.INTERACTIVE,
            )

    async def read_text_range(
        self,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> WorkspaceTextRange:
        _require_text_range(offset, limit)
        async with self._lock:
            return await run_file_task(
                self._read_text_range_sync,
                path,
                offset,
                limit,
                lane=FileTaskLane.INTERACTIVE,
            )

    async def visit_text_lines(
        self,
        path: str,
        visitor: WorkspaceLineVisitor,
        *,
        max_bytes: int | None = None,
    ) -> tuple[int, bool]:
        async with self._lock:
            return await run_file_task(
                self._visit_text_lines_sync,
                path,
                visitor,
                max_bytes,
                lane=FileTaskLane.BULK,
            )

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        overwrite: bool = True,
        expected_version: str | None = None,
    ) -> WorkspaceEntry:
        async with self._lock, self._mutation_guard():
            return await run_file_mutation(
                self._write_sync,
                path,
                data,
                overwrite,
                expected_version,
                lane=FileTaskLane.INTERACTIVE,
            )

    async def mkdir(self, path: str, *, parents: bool = True) -> WorkspaceEntry:
        async with self._lock, self._mutation_guard():
            return await run_file_mutation(
                self._mkdir_sync,
                path,
                parents,
                lane=FileTaskLane.INTERACTIVE,
            )

    async def delete(self, path: str, *, recursive: bool = False) -> None:
        async with self._lock, self._mutation_guard():
            await run_file_mutation(
                self._delete_sync,
                path,
                recursive,
                lane=FileTaskLane.INTERACTIVE,
            )

    async def move(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        async with self._lock, self._mutation_guard():
            return await run_file_mutation(
                self._move_sync,
                source,
                destination,
                overwrite,
                lane=FileTaskLane.INTERACTIVE,
            )

    def _physical(self, path: str) -> Path:
        relative = normalize_backend_path(path)
        physical = (self._root / relative).resolve()
        try:
            physical.relative_to(self._root)
        except ValueError as exc:
            raise WorkspacePathError("Physical path escaped the backend root.") from exc
        return physical

    def _stat_sync(self, path: str) -> WorkspaceEntry:
        relative = normalize_backend_path(path)
        physical = self._physical(relative)
        if not relative and self._virtual_root and not physical.exists():
            return WorkspaceEntry(path="", entry_type=EntryType.DIRECTORY)
        if not physical.exists():
            raise WorkspaceNotFoundError(f"Path does not exist: {path}")
        if physical.is_dir():
            return WorkspaceEntry(path=relative, entry_type=EntryType.DIRECTORY)
        stat = physical.stat()
        return WorkspaceEntry(
            path=relative,
            entry_type=EntryType.FILE,
            size=stat.st_size,
            version=str(file_version(stat)),
        )

    def _read_snapshot_sync(self, path: str) -> tuple[WorkspaceEntry, bytes]:
        relative = normalize_backend_path(path)
        physical = self._physical(relative)
        try:
            with physical.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                self._require_file_size(stat.st_size)
                data = handle.read()
        except OSError as exc:
            raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
        return (
            WorkspaceEntry(
                path=relative,
                entry_type=EntryType.FILE,
                size=stat.st_size,
                version=str(file_version(stat)),
            ),
            data,
        )

    def _read_text_range_sync(self, path: str, offset: int, limit: int) -> WorkspaceTextRange:
        relative = normalize_backend_path(path)
        physical = self._physical(relative)
        selected_lines: list[str] = []
        total_lines = 0
        try:
            with physical.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                self._require_file_size(stat.st_size)
                bytes_read = 0
                for raw_line in handle:
                    bytes_read += len(raw_line)
                    self._require_file_size(bytes_read)
                    line = raw_line.decode("utf-8")
                    if offset <= total_lines < offset + limit:
                        selected_lines.append(line)
                    total_lines += 1
        except OSError as exc:
            raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
        return WorkspaceTextRange(
            entry=WorkspaceEntry(
                path=relative,
                entry_type=EntryType.FILE,
                size=stat.st_size,
                version=str(file_version(stat)),
            ),
            content="".join(selected_lines),
            total_lines=total_lines,
            start_line=offset,
            num_lines=len(selected_lines),
        )

    def _visit_text_lines_sync(
        self,
        path: str,
        visitor: WorkspaceLineVisitor,
        max_bytes: int | None,
    ) -> tuple[int, bool]:
        relative = normalize_backend_path(path)
        physical = self._physical(relative)
        try:
            with physical.open("rb") as handle:
                stat = os.fstat(handle.fileno())
                self._require_file_size(stat.st_size)
                _require_content_scan_bytes(stat.st_size, max_bytes)
                decoder = codecs.getincrementaldecoder("utf-8")()
                bytes_read = 0
                while chunk := handle.read(64 * 1024):
                    bytes_read += len(chunk)
                    self._require_file_size(bytes_read)
                    _require_content_scan_bytes(bytes_read, max_bytes)
                    decoder.decode(chunk)
                decoder.decode(b"", final=True)
                handle.seek(0)
                entry = WorkspaceEntry(
                    path=relative,
                    entry_type=EntryType.FILE,
                    size=stat.st_size,
                    version=str(file_version(stat)),
                )
                for line_number, raw_line in enumerate(handle, 1):
                    line = raw_line.decode("utf-8").rstrip("\r\n")
                    if not visitor(entry, line_number, line):
                        return bytes_read, False
        except OSError as exc:
            raise WorkspaceNotFoundError(f"File does not exist: {path}") from exc
        return bytes_read, True

    def _mkdir_sync(self, path: str, parents: bool) -> WorkspaceEntry:
        physical = self._physical(path)
        self._require_mutation_path(physical)
        physical.mkdir(parents=parents, exist_ok=True)
        return self._stat_sync(path)

    def _delete_sync(self, path: str, recursive: bool) -> None:
        physical = self._physical(path)
        self._require_mutation_path(physical)
        if physical == self._root:
            raise WorkspacePathError("Cannot delete the backend root.")
        if physical.is_file():
            physical.unlink()
            return
        if not physical.exists():
            raise WorkspaceNotFoundError(f"Path does not exist: {path}")
        if not recursive:
            raise IsADirectoryError(path)
        shutil.rmtree(physical)

    def _move_sync(
        self,
        source: str,
        destination: str,
        overwrite: bool,
    ) -> WorkspaceEntry:
        source_path = self._physical(source)
        destination_path = self._physical(destination)
        self._require_mutation_path(source_path)
        self._require_mutation_path(destination_path)
        if not source_path.exists():
            raise WorkspaceNotFoundError(f"Path does not exist: {source}")
        if destination_path.exists() and not overwrite:
            raise FileExistsError(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()
        source_path.replace(destination_path)
        return self._stat_sync(destination)

    def _write_sync(
        self,
        path: str,
        data: bytes,
        overwrite: bool,
        expected_version: str | None,
    ) -> WorkspaceEntry:
        physical = self._physical(path)
        self._require_mutation_path(physical)
        self._require_file_size(len(data))
        exists = physical.is_file()
        current_version = file_version(physical.stat()) if exists else 0
        if expected_version is not None:
            if not exists:
                raise WorkspaceConflictError("File has been modified since read.")
            if str(current_version) != expected_version:
                raise WorkspaceConflictError("File has been modified since read.")
        if physical.exists() and not overwrite:
            raise FileExistsError(path)
        self._require_total_size(physical, len(data))
        physical.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(physical, data)
        ensure_monotonic_mtime(physical, current_version)
        return self._stat_sync(path)

    def _list_sync(self, path: str, max_entries: int | None) -> tuple[WorkspaceEntry, ...]:
        physical = self._physical(path)
        if not physical.exists() and self._virtual_root and physical == self._root:
            return ()
        if not physical.is_dir():
            raise WorkspaceNotFoundError(f"Directory does not exist: {path}")
        entries: list[WorkspaceEntry] = []
        with os.scandir(physical) as children:
            for child in children:
                _require_scan_count(len(entries) + 1, max_entries)
                entries.append(self._stat_sync(str(Path(child.path).relative_to(self._root))))
        return tuple(sorted(entries, key=lambda entry: entry.path))

    def _require_file_size(self, size: int) -> None:
        if self.max_file_bytes is not None and size > self.max_file_bytes:
            raise WorkspaceCapacityError(f"Text file exceeds {self.max_file_bytes} bytes limit.")

    def _require_total_size(self, target: Path, new_size: int) -> None:
        if self.max_total_bytes is None:
            return
        current_total = regular_file_size_sum(self._root)
        old_size = target.stat().st_size if target.is_file() else 0
        if current_total - old_size + new_size > self.max_total_bytes:
            raise WorkspaceCapacityError(self._total_capacity_error(self.max_total_bytes))

    def _require_mutation_path(self, physical: Path) -> None:
        for name in self._protected_directory_names:
            protected_root = (self._root / name).resolve()
            if physical == protected_root or protected_root in physical.parents:
                raise WorkspacePermissionError(self._protected_path_error)

    def _mutation_guard(self) -> AbstractAsyncContextManager[object]:
        if self._mutation_locks is None or self._mutation_lock_path is None:
            return _no_op_async_context()
        return self._mutation_locks.async_lock(self._mutation_lock_path)


@asynccontextmanager
async def _no_op_async_context() -> AsyncIterator[None]:
    yield


def _default_total_capacity_error(max_total_bytes: int) -> str:
    return f"Workspace size limit exceeded: {max_total_bytes} bytes."


def _require_scan_count(current: int, maximum: int | None) -> None:
    if maximum is not None and current > maximum:
        raise WorkspaceScanLimitError(
            f"Workspace file scan exceeds the server limit of {maximum} entries. "
            "Narrow the requested path and retry."
        )


def _require_content_scan_bytes(current: int, maximum: int | None) -> None:
    if maximum is not None and current > maximum:
        raise WorkspaceScanLimitError(
            "Workspace content scan exceeds the server byte limit. "
            "Narrow the requested path and retry."
        )


def _require_text_range(offset: int, limit: int) -> None:
    if offset < 0 or limit <= 0:
        raise ValueError("offset must be >= 0 and limit > 0.")
