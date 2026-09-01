"""Logical workspace routing and backend tests."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from gewu_agent_runtime.workspace import (
    AccessMode,
    EntryType,
    InMemoryWorkspaceBackend,
    LocalWorkspaceBackend,
    WorkspaceCapacityError,
    WorkspaceConflictError,
    WorkspaceMount,
    WorkspaceNotFoundError,
    WorkspacePathError,
    WorkspacePermissionError,
    WorkspaceScanLimitError,
    WorkspaceSession,
)
from gewu_core.observability import configure_filesystem_scan_recorder


async def test_longest_mount_prefix_and_read_only_access() -> None:
    root = InMemoryWorkspaceBackend()
    shared = InMemoryWorkspaceBackend()
    await root.write_bytes("shared/root.txt", b"root")
    await shared.write_bytes("item.txt", b"shared")
    session = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=root,
            ),
            WorkspaceMount(
                mount_id="shared",
                mount_path="/shared",
                access_mode=AccessMode.READ_ONLY,
                backend=shared,
            ),
        ]
    )

    assert await session.read_text("/shared/item.txt") == "shared"
    with pytest.raises(WorkspacePermissionError):
        await session.write_text("/shared/new.txt", "blocked")


async def test_nested_mount_shadows_conflicting_parent_backend_entry() -> None:
    company = InMemoryWorkspaceBackend()
    department = InMemoryWorkspaceBackend()
    await company.write_bytes("engineering", b"parent file")
    await department.write_bytes("policy.md", b"department")
    session = WorkspaceSession(
        (
            WorkspaceMount(
                mount_id="company",
                mount_path="/workspace/company",
                access_mode=AccessMode.READ_ONLY,
                backend=company,
            ),
            WorkspaceMount(
                mount_id="engineering",
                mount_path="/workspace/company/engineering",
                access_mode=AccessMode.READ_ONLY,
                backend=department,
            ),
        ),
        default_root="/workspace/company",
    )

    entries = await session.list("/workspace/company")

    assert [(entry.path, entry.entry_type) for entry in entries] == [
        ("/workspace/company/engineering", EntryType.DIRECTORY)
    ]
    assert await session.read_text("/workspace/company/engineering/policy.md") == "department"


async def test_workspace_walk_and_move(workspace: WorkspaceSession) -> None:
    await workspace.write_text("/a/one.txt", "one")
    await workspace.write_text("/a/two.txt", "two")
    moved = await workspace.move("/a/one.txt", "/b/one.txt")

    assert moved.path == "/b/one.txt"
    assert {entry.path for entry in await workspace.walk()} >= {
        "/a",
        "/a/two.txt",
        "/b",
        "/b/one.txt",
    }


async def test_workspace_rejects_non_positive_scan_and_content_limits(
    workspace: WorkspaceSession,
) -> None:
    with pytest.raises(ValueError, match="max_entries must be at least 1"):
        await workspace.list("/", max_entries=0)
    with pytest.raises(ValueError, match="max_entries must be at least 1"):
        await workspace.walk("/", max_entries=0)
    with pytest.raises(ValueError, match="max_total_bytes must be at least 1"):
        await workspace.visit_text_lines((), lambda _entry, _line, _text: True, max_total_bytes=0)


async def test_workspace_walk_applies_one_cumulative_cross_directory_scan_limit() -> None:
    backend = InMemoryWorkspaceBackend()
    await backend.write_bytes("docs/a.md", b"a")
    await backend.write_bytes("references/b.md", b"b")
    workspace = WorkspaceSession(
        (
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            ),
        )
    )

    scans: list[tuple[str, int, int, str]] = []
    previous = configure_filesystem_scan_recorder(
        lambda operation, entries, size_bytes, outcome: scans.append(
            (operation, entries, size_bytes, outcome)
        )
    )
    try:
        with pytest.raises(WorkspaceScanLimitError, match="server limit of 3 entries"):
            await workspace.walk("/", max_entries=3)
    finally:
        configure_filesystem_scan_recorder(previous)

    assert scans == [("workspace_recursive_list", 4, 0, "capacity_exceeded")]


async def test_workspace_content_budget_stops_before_visiting_overflow_file() -> None:
    backend = InMemoryWorkspaceBackend()
    await backend.write_bytes("first.txt", b"1234")
    await backend.write_bytes("second.txt", b"5678")
    workspace = WorkspaceSession(
        (
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            ),
        )
    )
    entries = (
        await workspace.stat("/first.txt"),
        await workspace.stat("/second.txt"),
    )
    visited: list[str] = []
    scans: list[tuple[str, int, int, str]] = []
    previous = configure_filesystem_scan_recorder(
        lambda operation, count, size_bytes, outcome: scans.append(
            (operation, count, size_bytes, outcome)
        )
    )

    try:
        with pytest.raises(WorkspaceCapacityError, match="content scan exceeds"):
            await workspace.visit_text_lines(
                entries,
                lambda entry, _line, _text: not visited.append(entry.path),
                max_total_bytes=6,
            )
    finally:
        configure_filesystem_scan_recorder(previous)

    assert visited == ["/first.txt"]
    assert scans == [("workspace_line_visit", 2, 4, "capacity_exceeded")]


async def test_local_backend_maps_logical_paths_without_exposing_root(tmp_path: Path) -> None:
    physical_root = tmp_path / "physical"
    session = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="local",
                mount_path="/project",
                access_mode=AccessMode.READ_WRITE,
                backend=LocalWorkspaceBackend(physical_root),
            )
        ]
    )

    written = await session.write_text("/project/docs/readme.txt", "hello")
    listed = await session.list("/project/docs")

    assert written.path == "/project/docs/readme.txt"
    assert await session.read_text(written.path) == "hello"
    assert (physical_root / "docs" / "readme.txt").read_text(encoding="utf-8") == "hello"
    assert [entry.path for entry in listed] == ["/project/docs/readme.txt"]
    assert str(physical_root) not in written.model_dump_json()


async def test_local_backend_rejects_traversal_and_symlink_escape(tmp_path: Path) -> None:
    physical_root = tmp_path / "physical"
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    backend = LocalWorkspaceBackend(physical_root)
    (physical_root / "escape.txt").symlink_to(outside)

    with pytest.raises(WorkspacePathError):
        await backend.read_bytes("../outside.txt")
    with pytest.raises(WorkspacePathError):
        await backend.read_bytes("escape.txt")
    with pytest.raises(WorkspacePathError):
        await backend.write_bytes("escape.txt", b"changed")

    assert outside.read_text(encoding="utf-8") == "outside"


async def test_local_backend_uses_microsecond_versions_and_advances_same_content_write(
    tmp_path: Path,
) -> None:
    backend = LocalWorkspaceBackend(tmp_path / "physical")

    first = await backend.write_bytes("same.txt", b"same")
    second = await backend.write_bytes(
        "same.txt",
        b"same",
        expected_version=first.version,
    )

    physical_version = (tmp_path / "physical/same.txt").stat().st_mtime_ns // 1_000
    assert first.version.isdecimal()
    assert int(second.version) > int(first.version)
    assert second.version == str(physical_version)


async def test_local_backend_range_read_uses_one_bounded_file_snapshot(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path / "physical")
    written = await backend.write_bytes("lines.txt", b"first\nsecond\nthird\n")

    selected = await backend.read_text_range("lines.txt", offset=1, limit=1)

    assert selected.content == "second\n"
    assert selected.total_lines == 3
    assert selected.start_line == 1
    assert selected.num_lines == 1
    assert selected.entry.size == 19
    assert selected.entry.version == written.version


async def test_local_backend_slow_read_does_not_block_the_event_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalWorkspaceBackend(tmp_path / "physical")
    await backend.write_bytes("value.txt", b"value")
    original = backend._read_snapshot_sync  # noqa: SLF001
    started = threading.Event()
    release = threading.Event()

    def slow_read(path: str):
        started.set()
        release.wait(timeout=1)
        return original(path)

    monkeypatch.setattr(backend, "_read_snapshot_sync", slow_read)
    read = asyncio.create_task(backend.read_bytes("value.txt"))
    assert await asyncio.to_thread(started.wait, 0.5)
    try:
        await asyncio.sleep(0)
        assert read.done() is False
    finally:
        release.set()

    assert await read == b"value"


async def test_local_backend_cancelled_write_finishes_atomically_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LocalWorkspaceBackend(tmp_path / "physical")
    original = backend._write_sync  # noqa: SLF001
    started = threading.Event()
    release = threading.Event()

    def slow_write(
        path: str,
        data: bytes,
        overwrite: bool,
        expected_version: str | None,
    ):
        started.set()
        release.wait(timeout=1)
        return original(path, data, overwrite, expected_version)

    monkeypatch.setattr(backend, "_write_sync", slow_write)
    write = asyncio.create_task(backend.write_bytes("value.txt", b"first"))
    assert await asyncio.to_thread(started.wait, 0.5)

    write.cancel()
    await asyncio.sleep(0)
    assert write.done() is False
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await write

    assert await backend.read_bytes("value.txt") == b"first"
    monkeypatch.setattr(backend, "_write_sync", original)
    assert (await backend.write_bytes("value.txt", b"second")).size == 6


async def test_local_backend_checks_file_limit_then_cas_and_preserves_old_file(
    tmp_path: Path,
) -> None:
    backend = LocalWorkspaceBackend(
        tmp_path / "physical",
        max_file_bytes=4,
        max_total_bytes=4,
    )
    await backend.write_bytes("value.txt", b"1234")

    with pytest.raises(WorkspaceCapacityError, match="Text file exceeds 4 bytes limit"):
        await backend.write_bytes("value.txt", b"12345", expected_version="stale")
    with pytest.raises(WorkspaceConflictError, match="modified since read"):
        await backend.write_bytes("value.txt", b"1234", expected_version="stale")

    assert await backend.read_bytes("value.txt") == b"1234"


async def test_local_backend_total_quota_counts_replacements(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path / "physical", max_total_bytes=5)
    await backend.write_bytes("existing.txt", b"old")

    with pytest.raises(
        WorkspaceCapacityError,
        match="Workspace size limit exceeded: 5 bytes",
    ):
        await backend.write_bytes("new.txt", b"new")

    replaced = await backend.write_bytes("existing.txt", b"hello")
    assert replaced.size == 5
    assert await backend.read_bytes("existing.txt") == b"hello"


async def test_local_backend_optional_protected_directories_cover_symlink_aliases(
    tmp_path: Path,
) -> None:
    physical_root = tmp_path / "physical"
    protected = physical_root / ".assets"
    protected.mkdir(parents=True)
    (protected / "existing.txt").write_text("original", encoding="utf-8")
    (physical_root / "alias").symlink_to(protected, target_is_directory=True)
    backend = LocalWorkspaceBackend(
        physical_root,
        protected_directory_names={".assets"},
        protected_path_error="Use the asset API.",
    )

    assert await backend.read_bytes("alias/existing.txt") == b"original"
    for operation in (
        backend.write_bytes("alias/new.txt", b"blocked"),
        backend.mkdir("alias/new"),
        backend.delete("alias/existing.txt"),
        backend.move("alias/existing.txt", "moved.txt"),
        backend.move("ordinary.txt", "alias/moved.txt"),
    ):
        with pytest.raises(WorkspacePermissionError, match="Use the asset API"):
            await operation


@pytest.mark.parametrize("backend_kind", ["memory", "local"])
async def test_workspace_backends_preserve_file_directory_exclusivity(
    backend_kind: str,
    tmp_path: Path,
) -> None:
    backend = (
        InMemoryWorkspaceBackend()
        if backend_kind == "memory"
        else LocalWorkspaceBackend(tmp_path / "physical")
    )
    await backend.write_bytes("occupied", b"file")
    await backend.mkdir("tree")
    await backend.write_bytes("tree/child.txt", b"child")

    with pytest.raises(FileExistsError):
        await backend.mkdir("occupied")
    with pytest.raises(IsADirectoryError):
        await backend.write_bytes("tree", b"replacement")

    assert await backend.read_bytes("occupied") == b"file"
    assert await backend.read_bytes("tree/child.txt") == b"child"


@pytest.mark.parametrize("backend_kind", ["memory", "local"])
async def test_workspace_backends_replace_destination_kind_on_overwrite_move(
    backend_kind: str,
    tmp_path: Path,
) -> None:
    backend = (
        InMemoryWorkspaceBackend()
        if backend_kind == "memory"
        else LocalWorkspaceBackend(tmp_path / "physical")
    )
    await backend.write_bytes("source.txt", b"source")
    await backend.write_bytes("target/old.txt", b"old")

    moved_file = await backend.move("source.txt", "target", overwrite=True)

    assert moved_file.entry_type is EntryType.FILE
    assert await backend.read_bytes("target") == b"source"
    with pytest.raises(WorkspaceNotFoundError):
        await backend.stat("target/old.txt")

    await backend.write_bytes("source/child.txt", b"child")
    await backend.write_bytes("destination", b"old destination")

    moved_directory = await backend.move("source", "destination", overwrite=True)

    assert moved_directory.entry_type is EntryType.DIRECTORY
    assert await backend.read_bytes("destination/child.txt") == b"child"


async def test_move_applies_write_path_policy_to_source_and_destination() -> None:
    backend = InMemoryWorkspaceBackend()
    await backend.write_bytes("protected/source.txt", b"source")

    def admit_mutation(path: str) -> str:
        if path.startswith("/protected"):
            raise WorkspacePermissionError("Protected mutation.")
        return path

    session = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        write_path_resolver=admit_mutation,
    )

    with pytest.raises(WorkspacePermissionError, match="Protected mutation"):
        await session.move("/protected/source.txt", "/moved.txt")

    assert await session.read_text("/protected/source.txt") == "source"


async def test_write_resolver_does_not_change_reads_and_applies_to_mutations() -> None:
    backend = InMemoryWorkspaceBackend()
    await backend.write_bytes("source/item.txt", b"source")
    await backend.write_bytes("target/item.txt", b"target")
    session = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="root",
                mount_path="/",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/source",
        write_path_resolver=lambda path: (path if path.startswith("/") else f"/target/{path}"),
    )

    entry, content = await session.read_text_with_metadata("item.txt")
    written = await session.write_text("item.txt", "changed")
    await session.mkdir("new")
    await session.delete("item.txt")

    assert entry.path == "/source/item.txt"
    assert content == "source"
    assert written.path == "/target/item.txt"
    assert await session.read_text("/source/item.txt") == "source"
    with pytest.raises(WorkspaceNotFoundError):
        await session.stat("/target/item.txt")
    assert (await session.stat("/target/new")).path == "/target/new"
