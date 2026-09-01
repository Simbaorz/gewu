"""Subscriber-compatible file tool behavior over neutral workspace mounts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gewu_agent_runtime.builtins import (
    append,
    delete,
    edit,
    glob,
    grep,
    list_directory,
    read,
    write,
)
from gewu_agent_runtime.builtins.files import (
    FILE_UNCHANGED_STUB,
    MAX_OUTPUT_SIZE,
    FileToolDescriptionProfile,
    build_file_tools,
)
from gewu_agent_runtime.tools import Tool, ToolContext
from gewu_agent_runtime.workspace import (
    AccessMode,
    InMemoryWorkspaceBackend,
    LocalWorkspaceBackend,
    WorkspaceMount,
    WorkspaceSession,
)


def _workspace() -> tuple[WorkspaceSession, InMemoryWorkspaceBackend]:
    backend = InMemoryWorkspaceBackend()
    return (
        WorkspaceSession(
            [
                WorkspaceMount(
                    mount_id="private",
                    mount_path="/workspace/private",
                    access_mode=AccessMode.READ_WRITE,
                    backend=backend,
                )
            ],
            default_root="/workspace/private",
        ),
        backend,
    )


def _runtime(workspace: WorkspaceSession, **values: Any) -> ToolContext:
    return ToolContext(
        conversation_id="conversation-1",
        run_id="run-1",
        workspace=workspace,
        **values,
    )


async def _execute(tool_value: Tool, runtime: ToolContext, **arguments: Any) -> dict[str, Any]:
    result = await tool_value.execute(arguments, runtime)
    return {**result.output_payload(), "_is_error": result.is_error}


def test_file_tool_schema_exposes_context_but_hides_runtime() -> None:
    properties = grep.input_schema["properties"]

    assert "context" in properties
    assert "runtime" not in properties
    assert properties["context"]["type"] == "integer"
    assert properties["output_mode"]["enum"] == [
        "content",
        "files_with_matches",
        "count",
    ]
    assert grep.input_schema["additionalProperties"] is False
    assert read.input_schema["required"] == ["file_path"]


def test_file_tool_language_can_be_bound_to_an_arbitrary_subscriber_workspace() -> None:
    profile = FileToolDescriptionProfile(
        relative_root="`/knowledge/personal`",
        writable_root="`/knowledge/personal`",
        read_only_directories_subject="Department directories",
        read_only_files="department files",
        read_only_files_subject="Department files",
        read_only_paths_subject="Department paths",
        default_root_name="personal root",
        glob_absolute_example="/knowledge/company/docs/*.md",
        mounted_path_pattern="/knowledge/...",
        list_discovery_guidance="Use `/knowledge/company` to discover departments.",
    )
    tools = {value.name: value for value in build_file_tools(profile)}

    assert "`/knowledge/personal`" in tools["read"].description
    assert "/knowledge/company" in tools["list"].description
    assert "/knowledge/company/docs/*.md" in tools["glob"].description
    assert "/workspace/private" not in "\n".join(value.description for value in tools.values())


async def test_read_accepts_numeric_strings_and_returns_unchanged_stub() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("notes.txt", "one\ntwo\nthree\n")
    runtime = _runtime(workspace)

    first = await _execute(read, runtime, file_path="notes.txt", offset="1", limit="1")
    second = await _execute(read, runtime, file_path="notes.txt", offset=1, limit=1)

    assert first == {
        "type": "text",
        "content": "2: two",
        "file_path": "/workspace/private/notes.txt",
        "size": 14,
        "total_lines": 3,
        "start_line": 1,
        "num_lines": 1,
        "error": "",
        "unchanged": False,
        "truncated": False,
        "_is_error": False,
    }
    assert second["type"] == "file_unchanged"
    assert second["content"] == FILE_UNCHANGED_STUB
    assert second["unchanged"] is True


async def test_read_retries_path_with_accidental_whitespace_removed() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("reports/daily.txt", "ready")

    result = await _execute(
        read,
        _runtime(workspace),
        file_path="reports / daily.txt",
    )

    assert result["file_path"] == "/workspace/private/reports/daily.txt"
    assert result["content"] == "1: ready"


async def test_read_preserves_real_path_spaces_when_exact_path_exists() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("docs/a b.md", "spaced\n")
    await workspace.write_text("docs/ab.md", "compact\n")

    result = await _execute(read, _runtime(workspace), file_path="docs/a b.md")

    assert result["file_path"] == "/workspace/private/docs/a b.md"
    assert result["content"] == "1: spaced"


async def test_read_treats_binary_and_oversized_files_as_unreadable(tmp_path: Path) -> None:
    backend = LocalWorkspaceBackend(tmp_path, max_file_bytes=4)
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    (tmp_path / "binary.bin").write_bytes(b"\xff\xfe")
    (tmp_path / "large.txt").write_bytes(b"large")

    binary = await _execute(read, _runtime(workspace), file_path="binary.bin")
    oversized = await _execute(
        read,
        _runtime(workspace),
        file_path="/workspace/private/large.txt",
    )

    assert binary == {
        "type": "error",
        "content": "",
        "file_path": "binary.bin",
        "size": 0,
        "total_lines": 0,
        "start_line": 0,
        "num_lines": 0,
        "error": (
            "File does not exist or is not readable. Relative paths resolve only under "
            "/workspace/private; files in other mounted roots require an absolute path."
        ),
        "unchanged": False,
        "truncated": False,
        "_is_error": True,
    }
    assert oversized["file_path"] == "/workspace/private/large.txt"
    assert oversized["error"] == "File does not exist or is not readable."
    assert oversized["_is_error"] is True


async def test_read_offset_beyond_end_omits_file_path_like_subscriber() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("short.txt", "one\ntwo\n")

    result = await _execute(read, _runtime(workspace), file_path="short.txt", offset=2)

    assert result == {
        "type": "error",
        "content": "",
        "file_path": "",
        "size": 8,
        "total_lines": 2,
        "start_line": 2,
        "num_lines": 0,
        "error": "offset (2) exceeds total lines (2)",
        "unchanged": False,
        "truncated": False,
        "_is_error": True,
    }


async def test_read_uses_backend_line_range_instead_of_full_snapshot() -> None:
    backend = _TextAccessTrackingBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    await workspace.write_text("range.txt", "one\ntwo\nthree\n")

    result = await _execute(
        read,
        _runtime(workspace),
        file_path="range.txt",
        offset=1,
        limit=1,
    )

    assert result["content"] == "2: two"
    assert backend.range_reads == 1
    assert backend.snapshot_reads == 0


async def test_truncated_read_is_not_retained_as_writable_state() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("large.txt", "x" * (MAX_OUTPUT_SIZE + 10))
    runtime = _runtime(workspace)

    read_result = await _execute(read, runtime, file_path="large.txt")
    write_result = await _execute(write, runtime, file_path="large.txt", content="changed")

    assert read_result["truncated"] is True
    assert write_result["error"] == "File has not been read yet. Read it first before writing."
    assert write_result["_is_error"] is True


async def test_read_edit_append_and_rewrite_preserve_read_before_write_state() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("plan.txt", "alpha\nbeta\n")
    runtime = _runtime(workspace)

    unread = await _execute(write, runtime, file_path="plan.txt", content="blocked")
    await _execute(read, runtime, file_path="plan.txt")
    edited = await _execute(
        edit,
        runtime,
        file_path="plan.txt",
        old_string="beta",
        new_string="gamma",
    )
    appended = await _execute(append, runtime, file_path="plan.txt", content="delta\n")
    rewritten = await _execute(write, runtime, file_path="plan.txt", content="final\n")

    assert unread["_is_error"] is True
    assert edited["_is_error"] is False
    assert appended["type"] == "update"
    assert rewritten["type"] == "update"
    assert await workspace.read_text("plan.txt") == "final\n"


async def test_external_version_change_invalidates_write_state() -> None:
    workspace, backend = _workspace()
    await workspace.write_text("versioned.txt", "one")
    runtime = _runtime(workspace)
    await _execute(read, runtime, file_path="versioned.txt")

    await backend.write_bytes("versioned.txt", b"external")
    result = await _execute(write, runtime, file_path="versioned.txt", content="two")

    assert result["error"] == "File has been modified since read. Read it again before writing."
    assert await workspace.read_text("versioned.txt") == "external"


@pytest.mark.parametrize(
    ("tool_value", "arguments"),
    (
        (write, {"content": "replacement\n"}),
        (append, {"content": "tail\n"}),
        (edit, {"old_string": "base", "new_string": "edited"}),
    ),
)
async def test_every_write_tool_rejects_changed_authoritative_version(
    tool_value: Tool,
    arguments: dict[str, Any],
) -> None:
    workspace, backend = _workspace()
    await workspace.write_text("stale.md", "base\n")
    runtime = _runtime(workspace)
    await _execute(read, runtime, file_path="stale.md")
    await backend.write_bytes("stale.md", b"base\nexternal\n")

    result = await _execute(tool_value, runtime, file_path="stale.md", **arguments)

    assert result["_is_error"] is True
    assert "modified since read" in result["error"]
    assert await workspace.read_text("stale.md") == "base\nexternal\n"


async def test_concurrent_create_does_not_overwrite_new_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, backend = _workspace()
    runtime = _runtime(workspace)
    original_write = workspace.write_text

    async def create_before_save(*args: object, **kwargs: object):
        await backend.write_bytes("new.txt", b"concurrent")
        return await original_write(*args, **kwargs)

    monkeypatch.setattr(workspace, "write_text", create_before_save)

    result = await _execute(write, runtime, file_path="new.txt", content="agent")

    assert result["_is_error"] is True
    assert result["error"] == "File was created before this write completed."
    assert await workspace.read_text("new.txt") == "concurrent"


@pytest.mark.parametrize(
    ("tool_value", "arguments", "expected_path"),
    (
        (write, {"content": "replacement\n"}, "docs/a.md"),
        (append, {"content": "tail\n"}, "docs/a.md"),
        (
            edit,
            {"old_string": "base", "new_string": "replacement"},
            "/workspace/private/docs/a.md",
        ),
    ),
)
async def test_failed_save_preserves_subscriber_result_shape(
    tool_value: Tool,
    arguments: dict[str, Any],
    expected_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _ = _workspace()
    await workspace.write_text("docs/a.md", "base\n")
    runtime = _runtime(workspace)
    await _execute(read, runtime, file_path="docs/a.md")

    async def fail_save(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise RuntimeError("save failed")

    monkeypatch.setattr(workspace, "write_text", fail_save)

    result = await _execute(tool_value, runtime, file_path="docs/a.md", **arguments)

    assert result["file_path"] == expected_path
    assert result["error"] == "save failed"
    assert result["_is_error"] is True


async def test_write_caches_version_returned_by_locked_save() -> None:
    backend = _RaceAfterWriteBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    await workspace.write_text("race.md", "base\n")
    runtime = _runtime(workspace)
    await _execute(read, runtime, file_path="race.md")
    backend.race_after_write = True

    first = await _execute(write, runtime, file_path="race.md", content="mine\n")
    cached = runtime.file_state_cache.get("/workspace/private/race.md")
    second = await _execute(write, runtime, file_path="race.md", content="overwrite\n")

    assert first["_is_error"] is False
    assert cached is not None
    assert cached.version == backend.raced_from_version
    assert second["_is_error"] is True
    assert "modified since read" in second["error"]
    assert await workspace.read_text("race.md") == "concurrent\n"


async def test_delete_rejects_expansion_and_root_and_clears_cached_tree() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("folder/item.txt", "value")
    runtime = _runtime(workspace)
    await _execute(read, runtime, file_path="folder/item.txt")

    expansion = await _execute(delete, runtime, path="folder/*.txt")
    root = await _execute(delete, runtime, path=".", recursive=True)
    removed = await _execute(delete, runtime, path="folder", recursive=True)

    assert expansion["_is_error"] is True
    assert root["error"] == "Refusing to delete workspace root."
    assert removed["file_path"] == "/workspace/private/folder"
    assert runtime.file_state_cache.has("/workspace/private/folder/item.txt") is False


async def test_delete_requires_recursive_for_even_an_empty_directory() -> None:
    workspace, _ = _workspace()
    await workspace.mkdir("empty")
    runtime = _runtime(workspace)

    rejected = await _execute(delete, runtime, path="empty")
    removed = await _execute(delete, runtime, path="empty", recursive=True)

    assert rejected["type"] == "delete"
    assert rejected["file_path"] == "empty"
    assert rejected["error"] == "empty"
    assert rejected["_is_error"] is True
    assert not rejected["error"].startswith("Tool execution error")
    assert removed["error"] == ""
    assert removed["_is_error"] is False


async def test_delete_returns_unexpected_os_error_details_like_subscriber() -> None:
    backend = _LeakingDeleteBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )

    result = await _execute(delete, _runtime(workspace), path="private.txt")

    assert result["error"] == "permission denied: /srv/subscriber-a/private.txt"
    assert result["_is_error"] is True


async def test_delete_preserves_subscriber_result_shape_for_non_os_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace, _ = _workspace()

    async def fail_delete(path: str, *, recursive: bool = False) -> None:
        del path, recursive
        raise RuntimeError("delete failed")

    monkeypatch.setattr(workspace, "delete", fail_delete)

    result = await _execute(delete, _runtime(workspace), path="docs/a.md")

    assert result == {
        "type": "delete",
        "file_path": "docs/a.md",
        "written_bytes": 0,
        "error": "delete failed",
        "_is_error": True,
    }


async def test_list_returns_only_stable_direct_children() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("a.txt", "a")
    await workspace.write_text("folder/b.txt", "b")

    result = await _execute(list_directory, _runtime(workspace), path=".")

    assert result["entries"] == [
        "/workspace/private/a.txt",
        "/workspace/private/folder/",
    ]
    assert result["num_entries"] == 2


async def test_edit_after_partial_read_preserves_unread_content() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("long.md", "first\nsecond\nthird\n")
    runtime = _runtime(workspace)

    read_result = await _execute(read, runtime, file_path="long.md", offset=1, limit=1)
    state = runtime.file_state_cache.get("/workspace/private/long.md")
    edited = await _execute(
        edit,
        runtime,
        file_path="long.md",
        old_string="second",
        new_string="SECOND",
    )

    assert read_result["_is_error"] is False
    assert state is not None
    assert (state.offset, state.limit) == (1, 1)
    assert edited["_is_error"] is False
    assert await workspace.read_text("long.md") == "first\nSECOND\nthird\n"


async def test_read_dedup_uses_backend_version_as_content_authority() -> None:
    backend = _MutableSameVersionBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    await workspace.write_text("same.md", "old\n")
    runtime = _runtime(workspace)

    first = await _execute(read, runtime, file_path="same.md")
    backend.replace_content_without_version("same.md", b"new\n")
    second = await _execute(read, runtime, file_path="same.md")

    assert first["type"] == "text"
    assert second["type"] == "file_unchanged"
    assert second["unchanged"] is True


async def test_glob_is_recursive_sorted_and_bounded_to_100_results() -> None:
    workspace, _ = _workspace()
    for index in range(105):
        await workspace.write_text(f"docs/{index:03}.md", str(index))
    await workspace.write_text("docs/ignored.txt", "ignored")

    result = await _execute(glob, _runtime(workspace), pattern="docs/**/*.md")

    assert result["num_files"] == 100
    assert result["truncated"] is True
    assert result["filenames"] == sorted(result["filenames"])
    assert result["filenames"][0] == "/workspace/private/docs/000.md"


async def test_grep_requires_path_supports_re2_context_and_case_insensitive() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("docs/a.txt", "before\nERROR one\nafter\n")
    await workspace.write_text("docs/b.md", "error two\n")
    runtime = _runtime(workspace)

    missing = await _execute(grep, runtime, pattern="error")
    invalid = await _execute(grep, runtime, pattern="(?=error)", path="docs")
    content = await _execute(
        grep,
        runtime,
        pattern="error",
        path="docs",
        glob="*.txt",
        output_mode="content",
        context=1,
        ignore_case=True,
    )

    assert missing["error"].startswith("grep requires a path")
    assert invalid["error"].startswith("Invalid RE2 pattern")
    assert content["content"] == "\n".join(
        [
            "/workspace/private/docs/a.txt-1: before",
            "/workspace/private/docs/a.txt:2: ERROR one",
            "/workspace/private/docs/a.txt-3: after",
        ]
    )
    assert content["num_lines"] == 3


async def test_non_multiline_grep_streams_lines_but_multiline_reads_full_content() -> None:
    backend = _TextAccessTrackingBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    await workspace.write_text("docs/a.md", "before\nneedle\nafter\n")

    streamed = await _execute(
        grep,
        _runtime(workspace),
        pattern="needle",
        path="docs",
        output_mode="content",
    )
    multiline_result = await _execute(
        grep,
        _runtime(workspace),
        pattern="needle",
        path="docs",
        output_mode="files_with_matches",
        multiline=True,
    )

    assert streamed["content"] == "/workspace/private/docs/a.md:2: needle"
    assert multiline_result["filenames"] == ["/workspace/private/docs/a.md"]
    assert backend.line_visits == 1
    assert backend.snapshot_reads == 1


async def test_multiline_grep_stops_reading_files_when_result_budget_is_full() -> None:
    backend = _TextAccessTrackingBackend()
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    await workspace.write_text("docs/a.md", "needle\n")
    await workspace.write_text("docs/b.md", "needle\n")
    await workspace.write_text("docs/c.md", "needle\n")

    result = await _execute(
        grep,
        _runtime(workspace, grep_max_result_lines=1),
        pattern="needle",
        path="docs",
        output_mode="content",
        head_limit=0,
        multiline=True,
    )

    assert result["content"] == "/workspace/private/docs/a.md:1: needle"
    assert result["applied_limit"] == 1
    assert backend.snapshot_reads == 2


@pytest.mark.parametrize("multiline", [False, True])
async def test_grep_skips_binary_and_oversized_files(
    tmp_path: Path,
    multiline: bool,
) -> None:
    backend = LocalWorkspaceBackend(tmp_path, max_file_bytes=8)
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )
    (tmp_path / "good.txt").write_bytes(b"needle\n")
    (tmp_path / "binary.txt").write_bytes(b"needle\xff")
    (tmp_path / "large.txt").write_bytes(b"needle-too-large")

    result = await _execute(
        grep,
        _runtime(workspace),
        pattern="needle",
        path=".",
        multiline=multiline,
    )

    assert result["filenames"] == ["/workspace/private/good.txt"]
    assert result["_is_error"] is False


async def test_grep_rejects_candidate_scope_before_reading_contents() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("one.txt", "needle")
    await workspace.write_text("two.txt", "needle")

    result = await _execute(
        grep,
        _runtime(workspace, grep_max_scan_files=1),
        pattern="needle",
        path=".",
    )

    assert "scope is too large" in result["error"]
    assert result["filenames"] == []


async def test_list_and_glob_stop_at_workspace_scan_limit() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("docs/a.md", "a")
    await workspace.write_text("docs/nested/b.md", "b")

    listed = await _execute(
        list_directory,
        _runtime(workspace, list_max_entries=1),
        path="docs",
    )
    matched = await _execute(
        glob,
        _runtime(workspace, file_search_max_entries=1),
        pattern="**/*.md",
        path="docs",
    )

    assert "server limit of 1 entries" in listed["error"]
    assert "server limit of 1 entries" in matched["error"]


async def test_grep_enforces_actual_cumulative_content_bytes() -> None:
    backend = _UnderreportedSizeBackend()
    await backend.write_bytes("docs/a.md", b"123456")
    workspace = WorkspaceSession(
        [
            WorkspaceMount(
                mount_id="private",
                mount_path="/workspace/private",
                access_mode=AccessMode.READ_WRITE,
                backend=backend,
            )
        ],
        default_root="/workspace/private",
    )

    result = await _execute(
        grep,
        _runtime(workspace, grep_max_scan_bytes=5),
        pattern="123",
        path="docs",
    )

    assert result["error"] == (
        "Workspace content scan exceeds the server byte limit. "
        "Narrow the requested path and retry."
    )


async def test_grep_unlimited_head_still_obeys_result_line_limit() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("matches.md", "needle\nneedle\nneedle\n")

    result = await _execute(
        grep,
        _runtime(workspace, grep_max_result_lines=2),
        pattern="needle",
        path="matches.md",
        output_mode="content",
        head_limit=0,
    )

    assert result["_is_error"] is False
    assert result["num_lines"] == 2
    assert result["applied_limit"] == 2
    assert result["content"].count("needle") == 2


async def test_grep_content_obeys_utf8_result_byte_limit() -> None:
    workspace, _ = _workspace()
    await workspace.write_text("matches.md", "needle\nneedle\n")
    first_row = "/workspace/private/matches.md:1: needle"

    result = await _execute(
        grep,
        _runtime(workspace, grep_max_result_bytes=len(first_row.encode("utf-8"))),
        pattern="needle",
        path="matches.md",
        output_mode="content",
        head_limit=0,
    )

    assert result["content"] == first_row
    assert len(result["content"].encode("utf-8")) <= len(first_row.encode("utf-8"))
    assert result["applied_limit"] == 1


class _UnderreportedSizeBackend(InMemoryWorkspaceBackend):
    async def list(self, path: str, *, max_entries: int | None = None):
        entries = await super().list(path, max_entries=max_entries)
        return tuple(entry.model_copy(update={"size": 1}) for entry in entries)


class _TextAccessTrackingBackend(InMemoryWorkspaceBackend):
    def __init__(self) -> None:
        super().__init__()
        self.snapshot_reads = 0
        self.range_reads = 0
        self.line_visits = 0

    async def read_bytes_with_metadata(self, path: str):
        self.snapshot_reads += 1
        return await super().read_bytes_with_metadata(path)

    async def read_text_range(self, path: str, *, offset: int, limit: int):
        self.range_reads += 1
        return await super().read_text_range(path, offset=offset, limit=limit)

    async def visit_text_lines(self, path: str, visitor, *, max_bytes: int | None = None):
        self.line_visits += 1
        return await super().visit_text_lines(path, visitor, max_bytes=max_bytes)


class _MutableSameVersionBackend(InMemoryWorkspaceBackend):
    def replace_content_without_version(self, path: str, content: bytes) -> None:
        self._files[path] = content


class _LeakingDeleteBackend(InMemoryWorkspaceBackend):
    async def delete(self, path: str, *, recursive: bool = False) -> None:
        del path, recursive
        raise PermissionError("permission denied: /srv/subscriber-a/private.txt")


class _RaceAfterWriteBackend(InMemoryWorkspaceBackend):
    def __init__(self) -> None:
        super().__init__()
        self.race_after_write = False
        self.raced_from_version = ""

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        overwrite: bool = True,
        expected_version: str | None = None,
    ):
        written = await super().write_bytes(
            path,
            data,
            overwrite=overwrite,
            expected_version=expected_version,
        )
        if self.race_after_write:
            self.race_after_write = False
            self.raced_from_version = written.version
            await super().write_bytes(path, b"concurrent\n")
        return written
