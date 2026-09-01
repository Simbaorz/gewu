"""Authorized routing across logical workspace mounts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from gewu_agent_runtime.workspace.contracts import (
    AccessMode,
    EntryType,
    WorkspaceEntry,
    WorkspaceLineVisitor,
    WorkspaceMount,
    WorkspaceTextRange,
)
from gewu_agent_runtime.workspace.errors import (
    WorkspaceCapacityError,
    WorkspaceNotFoundError,
    WorkspacePermissionError,
    WorkspaceScanLimitError,
)
from gewu_agent_runtime.workspace.paths import join_logical, normalize_logical_path
from gewu_core.observability import record_filesystem_scan


class WorkspaceSession:
    """One already-authorized view over arbitrary logical mounts."""

    def __init__(
        self,
        mounts: Sequence[WorkspaceMount],
        *,
        default_root: str = "/",
        write_path_resolver: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize deterministic longest-prefix routing."""

        if not mounts:
            raise ValueError("A workspace session requires at least one mount.")
        ids = {mount.mount_id for mount in mounts}
        if len(ids) != len(mounts):
            raise ValueError("Workspace mount IDs must be unique.")
        self._mounts = tuple(
            sorted(mounts, key=lambda item: (len(item.mount_path), item.priority), reverse=True)
        )
        self._default_root = normalize_logical_path(default_root)
        self._write_path_resolver = write_path_resolver
        try:
            self._resolve(self._default_root)
        except WorkspaceNotFoundError as exc:
            prefix = "/" if self._default_root == "/" else f"{self._default_root}/"
            if not any(mount.mount_path.startswith(prefix) for mount in self._mounts):
                raise ValueError(
                    "Workspace default root must contain an authorized mount."
                ) from exc

    @property
    def mounts(self) -> tuple[WorkspaceMount, ...]:
        """Return the authorized mounts without exposing physical paths."""

        return self._mounts

    @property
    def default_root(self) -> str:
        """Return the logical root used for model-provided relative paths."""

        return self._default_root

    def allowed_roots(self) -> tuple[str, ...]:
        """Return model-visible authorized mount roots."""

        return tuple(mount.mount_path for mount in self._mounts)

    def resolve_path(self, path: str) -> str:
        """Normalize an absolute path or resolve a relative path under the default root."""

        raw = path.replace("\\", "/").strip()
        if raw.startswith("/"):
            return normalize_logical_path(raw)
        return join_logical(self._default_root, raw)

    def resolve_write_path(self, path: str) -> str:
        """Return a subscriber-authorized writable target for a model path."""

        candidate = self._write_path_resolver(path) if self._write_path_resolver else path
        logical = self.resolve_path(candidate)
        mount, _ = self._resolve(logical)
        self._require_write(mount)
        return logical

    async def stat(self, path: str) -> WorkspaceEntry:
        logical = self.resolve_path(path)
        mount, relative = self._resolve(logical)
        return self._logical_entry(mount, await mount.backend.stat(relative))

    async def list(
        self,
        path: str,
        *,
        max_entries: int | None = None,
    ) -> tuple[WorkspaceEntry, ...]:
        _validate_scan_limit(max_entries)
        logical = self.resolve_path(path)
        entries: dict[str, WorkspaceEntry] = {}
        try:
            mount, relative = self._resolve(logical)
        except WorkspaceNotFoundError:
            mount = None
        if mount is not None:
            for entry in await mount.backend.list(relative, max_entries=max_entries):  # noqa
                logical_entry = self._logical_entry(mount, entry)
                entries[logical_entry.path] = logical_entry
        prefix = "/" if logical == "/" else f"{logical}/"
        for candidate in self._mounts:
            if not candidate.mount_path.startswith(prefix) or candidate.mount_path == logical:
                continue
            remainder = candidate.mount_path[len(prefix) :]
            child = remainder.split("/", 1)[0]
            child_path = join_logical(logical, child)
            # A nested mount shadows any entry exposed by its parent backend.
            entries[child_path] = WorkspaceEntry(
                path=child_path,
                entry_type=EntryType.DIRECTORY,
            )
            _require_scan_count(len(entries), max_entries)
        if mount is None and not entries:
            raise WorkspaceNotFoundError(f"No workspace mount contains: {logical}")
        return tuple(entries[key] for key in sorted(entries))

    async def read_bytes(self, path: str) -> bytes:
        mount, relative = self._resolve(self.resolve_path(path))
        return await mount.backend.read_bytes(relative)

    async def read_bytes_with_metadata(self, path: str) -> tuple[WorkspaceEntry, bytes]:
        """Read metadata and bytes from one backend snapshot."""

        mount, relative = self._resolve(self.resolve_path(path))
        entry, data = await mount.backend.read_bytes_with_metadata(relative)
        return self._logical_entry(mount, entry), data

    async def read_text(self, path: str, *, encoding: str = "utf-8") -> str:
        return (await self.read_bytes(path)).decode(encoding)

    async def read_text_with_metadata(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
    ) -> tuple[WorkspaceEntry, str]:
        """Read one text file and its version from the same snapshot."""

        entry, data = await self.read_bytes_with_metadata(path)
        return entry, data.decode(encoding)

    async def read_text_range(
        self,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> WorkspaceTextRange:
        """Read a bounded UTF-8 range without requiring a full-file allocation."""

        mount, relative = self._resolve(self.resolve_path(path))
        result = await mount.backend.read_text_range(relative, offset=offset, limit=limit)
        return result.model_copy(update={"entry": self._logical_entry(mount, result.entry)})

    async def visit_text_lines(
        self,
        entries: Sequence[WorkspaceEntry],
        visitor: WorkspaceLineVisitor,
        *,
        max_total_bytes: int,
    ) -> None:
        """Visit authorized UTF-8 files sequentially under a cumulative byte limit."""

        if max_total_bytes < 1:
            raise ValueError("max_total_bytes must be at least 1.")
        scanned_bytes = 0
        scanned_entries = 0
        outcome = "success"
        try:
            for logical_entry in entries:
                scanned_entries += 1
                mount, relative = self._resolve(logical_entry.path)
                remaining_bytes = max_total_bytes - scanned_bytes
                try:
                    file_bytes, completed = await mount.backend.visit_text_lines(
                        relative,
                        _logical_line_visitor(mount, visitor),
                        max_bytes=remaining_bytes,
                    )
                except WorkspaceScanLimitError as exc:
                    outcome = "capacity_exceeded"
                    raise WorkspaceCapacityError(
                        "Workspace content scan exceeds the server byte limit. "
                        "Narrow the requested path and retry."
                    ) from exc
                except (
                    WorkspaceCapacityError,
                    WorkspaceNotFoundError,
                    UnicodeDecodeError,
                ):
                    continue
                scanned_bytes += file_bytes
                if scanned_bytes > max_total_bytes:
                    outcome = "capacity_exceeded"
                    raise WorkspaceCapacityError(
                        "Workspace content scan exceeds the server byte limit. "
                        "Narrow the requested path and retry."
                    )
                if not completed:
                    return
        except BaseException:
            if outcome == "success":
                outcome = "failure"
            raise
        finally:
            record_filesystem_scan(
                "workspace_line_visit",
                scanned_entries,
                scanned_bytes,
                outcome,
            )

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        overwrite: bool = True,
        expected_version: str | None = None,
    ) -> WorkspaceEntry:
        mount, relative = self._resolve(self.resolve_write_path(path))
        self._require_write(mount)
        return self._logical_entry(
            mount,
            await mount.backend.write_bytes(
                relative,
                data,
                overwrite=overwrite,
                expected_version=expected_version,
            ),
        )

    async def write_text(
        self,
        path: str,
        content: str,
        *,
        encoding: str = "utf-8",
        overwrite: bool = True,
        expected_version: str | None = None,
    ) -> WorkspaceEntry:
        return await self.write_bytes(
            path,
            content.encode(encoding),
            overwrite=overwrite,
            expected_version=expected_version,
        )

    async def mkdir(self, path: str, *, parents: bool = True) -> WorkspaceEntry:
        mount, relative = self._resolve(self.resolve_write_path(path))
        self._require_write(mount)
        return self._logical_entry(mount, await mount.backend.mkdir(relative, parents=parents))

    async def delete(self, path: str, *, recursive: bool = False) -> None:
        mount, relative = self._resolve(self.resolve_write_path(path))
        self._require_write(mount)
        await mount.backend.delete(relative, recursive=recursive)

    async def move(
        self,
        source: str,
        destination: str,
        *,
        overwrite: bool = False,
    ) -> WorkspaceEntry:
        source_mount, source_relative = self._resolve(self.resolve_write_path(source))
        destination_mount, destination_relative = self._resolve(
            self.resolve_write_path(destination)
        )
        self._require_write(source_mount)
        self._require_write(destination_mount)
        if source_mount.mount_id != destination_mount.mount_id:
            raise WorkspacePermissionError("Moves across workspace mounts are not supported.")
        return self._logical_entry(
            source_mount,
            await source_mount.backend.move(
                source_relative,
                destination_relative,
                overwrite=overwrite,
            ),
        )

    async def walk(
        self,
        path: str = "/",
        *,
        max_entries: int | None = None,
    ) -> tuple[WorkspaceEntry, ...]:
        """Return a bounded-by-backend recursive listing for generic search tools."""

        _validate_scan_limit(max_entries)
        pending = [self.resolve_path(path)]
        entries: list[WorkspaceEntry] = []
        scanned_entries = 0
        outcome = "success"
        try:
            while pending:
                current = pending.pop()
                remaining = None if max_entries is None else max_entries - len(entries)
                directory_limit = None if remaining is None else max(1, remaining)
                for entry in await self.list(current, max_entries=directory_limit):
                    scanned_entries += 1
                    entries.append(entry)
                    _require_scan_count(len(entries), max_entries)
                    if entry.entry_type is EntryType.DIRECTORY:
                        pending.append(entry.path)
            return tuple(entries)
        except WorkspaceScanLimitError:
            outcome = "capacity_exceeded"
            if max_entries is not None:
                scanned_entries = max(scanned_entries, max_entries + 1)
            raise
        except BaseException:
            outcome = "failure"
            raise
        finally:
            record_filesystem_scan(
                "workspace_recursive_list",
                scanned_entries,
                0,
                outcome,
            )

    def _resolve(self, path: str) -> tuple[WorkspaceMount, str]:
        logical = normalize_logical_path(path)
        for mount in self._mounts:
            if mount.mount_path == "/":
                return mount, logical.removeprefix("/")
            if logical == mount.mount_path:
                return mount, ""
            prefix = f"{mount.mount_path}/"
            if logical.startswith(prefix):
                return mount, logical[len(prefix) :]
        raise WorkspaceNotFoundError(f"No workspace mount contains: {logical}")

    @staticmethod
    def _require_write(mount: WorkspaceMount) -> None:
        if mount.access_mode is AccessMode.READ_ONLY:
            raise WorkspacePermissionError(f"Workspace mount is read-only: {mount.mount_path}")

    @staticmethod
    def _logical_entry(mount: WorkspaceMount, entry: WorkspaceEntry) -> WorkspaceEntry:
        return entry.model_copy(update={"path": join_logical(mount.mount_path, entry.path)})


def _require_scan_count(current: int, maximum: int | None) -> None:
    if maximum is not None and current > maximum:
        raise WorkspaceScanLimitError(
            f"Workspace file scan exceeds the server limit of {maximum} entries. "
            "Narrow the requested path and retry."
        )


def _validate_scan_limit(maximum: int | None) -> None:
    if maximum is not None and maximum < 1:
        raise ValueError("max_entries must be at least 1.")


def _logical_line_visitor(
    mount: WorkspaceMount,
    visitor: WorkspaceLineVisitor,
) -> WorkspaceLineVisitor:
    def visit(entry: WorkspaceEntry, line_number: int, line: str) -> bool:
        return visitor(WorkspaceSession._logical_entry(mount, entry), line_number, line)  # noqa

    return visit
