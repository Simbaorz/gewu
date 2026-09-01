"""Reference file tools over the subscriber-neutral WorkspaceSession."""

from __future__ import annotations

import fnmatch
import re
import time
from collections import deque
from typing import Any, Literal, Protocol

import re2  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from gewu_agent_runtime.domain import FileState
from gewu_agent_runtime.tools import (
    PersistencePolicy,
    Tool,
    ToolContext,
    ToolResult,
    tool,
    with_tool_description,
)
from gewu_agent_runtime.workspace import (
    EntryType,
    WorkspaceCapacityError,
    WorkspaceConflictError,
    WorkspaceError,
    WorkspaceNotFoundError,
)

MAX_OUTPUT_SIZE = 25_000
MAX_LINES_TO_READ = 2_000
DEFAULT_GLOB_MAX_RESULTS = 100
DEFAULT_GREP_HEAD_LIMIT = 250
MAX_GREP_PATTERN_LENGTH = 512
MAX_COLUMNS = 500
MAX_GREP_CONTEXT_LINES = 100
FILE_UNCHANGED_STUB = (
    "File unchanged since last read. The content from the earlier read tool result "
    "in this conversation is still current."
)
FILE_NOT_READABLE_ERROR = "File does not exist or is not readable."
ERROR_NOT_READ = "File has not been read yet. Read it first before writing."
ERROR_MODIFIED = "File has been modified since read. Read it again before writing."
ERROR_SAME_STRING = "No changes to make: old_string and new_string are exactly the same."
ERROR_STRING_NOT_FOUND = "String to replace not found in file."
ERROR_MULTIPLE_MATCHES = "Found {count} matches of the string to replace, but replace_all is false."
SHELL_EXPANSION_CHARS = frozenset("{}*?[]")
ERROR_EXACT_PATH_REQUIRED = (
    "Delete path must be one exact path. Shell glob or brace expansion is not supported."
)
WHITESPACE_PATTERN = re.compile(r"\s+")
UNREADABLE_FILE_ERRORS = (WorkspaceNotFoundError, WorkspaceCapacityError, UnicodeDecodeError)


class FileToolDescriptionProfile(BaseModel):
    """Subscriber-owned logical-path language for model-visible file tools."""

    model_config = ConfigDict(frozen=True)

    relative_root: str = "the session's default workspace root"
    writable_root: str = "the writable workspace roots"
    read_only_directories_subject: str = "Directories in read-only mounts"
    read_only_files: str = "files in other mounted roots"
    read_only_files_subject: str = "Files in other mounted roots"
    read_only_paths_subject: str = "Read-only mounted paths"
    default_root_name: str = "default root"
    glob_absolute_example: str = "/workspace/mounted/docs/*.md"
    mounted_path_pattern: str = "/workspace/..."
    list_discovery_guidance: str = ""


DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE = FileToolDescriptionProfile()


def build_file_tools(
    profile: FileToolDescriptionProfile = DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE,
) -> tuple[Tool, ...]:
    """Bind model-visible path language without changing file-tool behavior."""

    descriptions = {
        "read": _read_description(profile),
        "write": _write_description(profile),
        "append": _append_description(profile),
        "edit": _edit_description(profile),
        "delete": _delete_description(profile),
        "list": _list_description(profile),
        "glob": _glob_description(profile),
    }
    return tuple(
        (
            with_tool_description(value, descriptions[value.name])
            if value.name in descriptions
            else value
        )
        for value in FILE_TOOLS
    )


class ReadOutput(BaseModel):
    type: str
    content: str = ""
    file_path: str = ""
    size: int = 0
    total_lines: int = 0
    start_line: int = 0
    num_lines: int = 0
    error: str = ""
    unchanged: bool = False
    truncated: bool = False


class WriteOutput(BaseModel):
    type: str
    file_path: str
    written_bytes: int = 0
    error: str = ""


class EditOutput(BaseModel):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False
    error: str = ""


class ListOutput(BaseModel):
    entries: list[str] = Field(default_factory=list)
    num_entries: int = 0
    error: str = ""


class GlobOutput(BaseModel):
    filenames: list[str] = Field(default_factory=list)
    num_files: int = 0
    duration_ms: int = 0
    truncated: bool = False
    error: str = ""


class GrepOutput(BaseModel):
    mode: str
    filenames: list[str] = Field(default_factory=list)
    num_files: int = 0
    content: str | None = None
    num_lines: int | None = None
    num_matches: int | None = None
    applied_limit: int | None = None
    applied_offset: int | None = None
    error: str = ""


def _read_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Read a file from the current session's workspace.

    You can access files under the workspace root. {profile.read_only_directories_subject} inside
    the workspace are read-only references.
    If the user provides a path to a file, assume that path is valid. It is okay
    to read a file that does not exist; an error will be returned.

    Usage:
    - Use paths exactly as provided by the user or returned by `glob` and `grep`.
    - Relative paths resolve under {profile.relative_root} and never refer to {profile.read_only_files}.
      {profile.read_only_files_subject} require an absolute path under a read-only root listed in the
      system prompt.
    - By default, this reads up to 2000 lines starting from the beginning of the
      file. Use `offset` and `limit` for larger files or when you only need a
      specific section.
    - When you already know which part of the file you need, only read that
      part. This can be important for larger files.
    - Results are returned with line numbers starting at 1.
    - This tool can only read files, not directories. To find files by name, use
      `glob`; to search file contents, use `grep`.

    Notes:
    - Only text files are supported. Binary files, images, archives, and rich
      document formats will return an error.
    - Line numbers in output start at 1, but the `offset` parameter is
      0-indexed.
    - If the file was read before and has not changed, this returns a stub
      instead of re-reading the same content.

    Args:
        file_path: Path to the file to read.
        offset: 0-indexed line offset to start reading from. Omit unless reading
            a large file in chunks.
        limit: Maximum number of lines to read. Omit to read the default chunk.

    Returns:
        A read result with file content, size, total line count, read range, and
        error details when the file is missing or unreadable.

    Examples:
        # Read entire file (first 2000 lines by default)
        read("docs/example.md")

        # Read lines 1-500 (offset=0, limit=500)
        read("docs/example.md", offset=0, limit=500)

        # Read from line 100 onward (offset=99 means start at line 100)
        read("docs/example.md", offset=99, limit=500)
    """.strip()


def _write_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Write a UTF-8 text file in the workspace root.

    Usage:
    - This tool will overwrite the existing file if there is one at the provided
      path.
    - This tool writes only under {profile.writable_root}. {profile.read_only_paths_subject} are read-only.
    - Relative paths resolve under {profile.relative_root}.
    - If this is an existing file, you MUST use the Read tool first to read the
      file's contents. This tool will fail if you did not read the file first.
    - Prefer the Edit tool for modifying existing files; only use this tool to
      create new files or for complete rewrites.
    - NEVER create documentation files (*.md) or README files unless explicitly
      requested by the user.
    - Only use emojis if the user explicitly requests it. Avoid writing emojis
      to files unless asked.

    Args:
        file_path: Path to the file to write.
        content: The complete UTF-8 text content to write.

    Returns:
        dict with:
            - type: `create` or `update`.
            - file_path: The path to the file that was written.
            - written_bytes: Number of bytes written.
            - error: Error message if failed, empty on success.
    """.strip()


def _append_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Append UTF-8 text to the end of a file in the workspace root.

    Usage:
    - Use this only when appending is semantically correct.
    - This tool writes only under {profile.writable_root}. {profile.read_only_paths_subject} are read-only.
    - Relative paths resolve under {profile.relative_root}.
    - If the file already exists, you MUST use the Read tool first to read the
      file's contents. This tool will fail if you did not read the file first.
    - The content is appended exactly as provided; include any needed newline.

    Args:
        file_path: Path to the file to append to.
        content: Content to append.

    Returns:
        dict with:
            - type: `create` or `update`.
            - file_path: The path to the file.
            - written_bytes: Number of bytes written.
            - error: Error message if failed, empty on success.
    """.strip()


def _edit_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Perform exact string replacements in a workspace root text file.

    Usage:
    - You must use your Read tool at least once in the conversation before
      editing. This tool will error if you attempt an edit without reading the
      file.
    - This tool modifies files only under {profile.writable_root}. {profile.read_only_paths_subject} are
      read-only.
    - Relative paths resolve under {profile.relative_root}.
    - When editing text from Read tool output, preserve the exact indentation
      after the line number prefix. Never include any part of the line number
      prefix in `old_string` or `new_string`.
    - ALWAYS prefer editing existing files. NEVER write new files unless
      explicitly required.
    - Only use emojis if the user explicitly requests it. Avoid adding emojis to
      files unless asked.
    - The edit will FAIL if `old_string` is not unique in the file. Either
      provide a larger string with more surrounding context to make it unique or
      use `replace_all` to change every instance of `old_string`.
    - Use `replace_all` for replacing and renaming strings across the file.

    Args:
        file_path: The path to the file to modify.
        old_string: The text to replace.
        new_string: The text to replace it with. Must be different from
            `old_string`.
        replace_all: Replace all occurrences of `old_string`.

    Returns:
        dict with:
            - file_path: The path to the file that was edited.
            - old_string: The original string that was replaced.
            - new_string: The new string that replaced it.
            - replace_all: Whether all occurrences were replaced.
            - error: Error message if failed, empty on success.
    """.strip()


def _delete_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Delete a path from the writable workspace.

    Usage:
    - Only use this when the user explicitly asks to remove a file or directory.
    - This tool only deletes paths under {profile.writable_root}.
    - Relative paths resolve under {profile.relative_root}.
    - {profile.read_only_paths_subject} and the {profile.default_root_name} itself cannot be deleted.
    - Pass one exact path. Shell glob or brace expansion is not supported.
    - Directories require `recursive=true`.

    Args:
        path: File or directory path to delete.
        recursive: Whether to delete a directory tree.

    Returns:
        dict with:
            - type: `delete`.
            - file_path: The path that was deleted.
            - written_bytes: Always 0.
            - error: Error message if failed, empty on success.
    """.strip()


def _list_description(profile: FileToolDescriptionProfile) -> str:
    discovery = f" {profile.list_discovery_guidance}" if profile.list_discovery_guidance else ""
    return f"""List the direct children of a readable workspace directory.

    Relative paths resolve under {profile.relative_root}. {profile.read_only_directories_subject} require
    an absolute path.{discovery}

    Args:
        path: Directory to list. Omit it to list {profile.relative_root}.

    Returns:
        A stable list of direct child paths and error details when the directory
        cannot be accessed.
    """.strip()


def _glob_description(profile: FileToolDescriptionProfile) -> str:
    return f"""Fast file pattern matching tool for readable workspace files.

    Usage:
    - Supports glob patterns like `**/*.md`, `docs/**/*.txt`, or
      `{profile.glob_absolute_example}`.
    - Returns matching file paths in a stable sorted order.
    - Use this tool when you need to find files by name patterns.
    - Omit `path` when `pattern` already includes the directory or mounted
      workspace path to search.
    - Use `path` to set the search root. When `path` is provided, `pattern` is
      matched relative to that root and against full workspace paths.
    - Do not pass `"undefined"` or `"null"` for `path`; omit it for the default
      behavior.

    Args:
        pattern: The glob pattern to match files against. It may be a basename,
            relative path, or mounted `{profile.mounted_path_pattern}` path pattern.
        path: Directory or path prefix to search in. Relative paths and omitted
            paths search under {profile.relative_root}. {profile.read_only_paths_subject} must be absolute.

    Returns:
        dict with:
            - filenames: Array of file paths that match the pattern.
            - num_files: Total number of files returned.
            - duration_ms: Time taken to execute the search in milliseconds.
            - truncated: Whether results were truncated.
            - error: Error message if failed, empty on success.
    """.strip()


def _read_result(
    type_: str,
    *,
    content: str = "",
    file_path: str = "",
    size: int = 0,
    total_lines: int = 0,
    start_line: int = 0,
    num_lines: int = 0,
    error: str = "",
    unchanged: bool = False,
    truncated: bool = False,
) -> ToolResult:
    output = ReadOutput(
        type=type_,
        content=content,
        file_path=file_path,
        size=size,
        total_lines=total_lines,
        start_line=start_line,
        num_lines=num_lines,
        error=error,
        unchanged=unchanged,
        truncated=truncated,
    )
    persisted = output.model_dump(mode="json", exclude={"content"})
    persisted["content_omitted"] = bool(content)
    return ToolResult(output=output, is_error=bool(error), persistence_payload=persisted)


@tool(
    name="read",
    description=_read_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    allow_parallel=True,
    persistence_policy=PersistencePolicy.REFERENCE,
)
async def read(
    file_path: str,
    offset: int | None = None,
    limit: int | None = None,
    *,
    runtime: ToolContext,
) -> ToolResult:
    """Execute the read operation against the bound workspace."""

    context = runtime
    actual_offset = offset if offset is not None else 0
    actual_limit = limit if limit is not None else MAX_LINES_TO_READ
    if actual_offset < 0 or actual_limit <= 0:
        return _read_result("error", error="offset must be >= 0 and limit > 0.")

    requested_path = file_path
    try:
        selected_range = await context.workspace.read_text_range(
            requested_path,
            offset=actual_offset,
            limit=actual_limit,
        )
    except UNREADABLE_FILE_ERRORS:
        compact = WHITESPACE_PATTERN.sub("", requested_path)
        if compact == requested_path:
            selected_range = None
        else:
            try:
                selected_range = await context.workspace.read_text_range(
                    compact,
                    offset=actual_offset,
                    limit=actual_limit,
                )
            except UNREADABLE_FILE_ERRORS:
                selected_range = None
            except WorkspaceError as exc:
                return _read_result("error", file_path=file_path, error=str(exc))
    except (WorkspaceError, UnicodeDecodeError) as exc:
        return _read_result("error", file_path=file_path, error=str(exc))

    if selected_range is None:
        if file_path.strip().startswith("/"):
            hint = ""
        elif context.relative_file_not_found_hint:
            hint = context.relative_file_not_found_hint
        else:
            hint = (
                f" Relative paths resolve only under {context.workspace.default_root}; "
                "files in other mounted roots require an absolute path."
            )
        return _read_result(
            "error",
            file_path=file_path,
            error=f"{FILE_NOT_READABLE_ERROR}{hint}",
        )

    entry = selected_range.entry
    total_lines = selected_range.total_lines
    if total_lines and actual_offset >= total_lines:
        return _read_result(
            "error",
            size=entry.size,
            total_lines=total_lines,
            start_line=actual_offset,
            error=f"offset ({actual_offset}) exceeds total lines ({total_lines})",
        )
    num_lines = selected_range.num_lines
    state = context.file_state_cache.get(entry.path)
    if (
        state is not None
        and state.offset == actual_offset
        and state.limit == num_lines
        and state.version == entry.version
    ):
        return _read_result(
            "file_unchanged",
            content=FILE_UNCHANGED_STUB,
            file_path=entry.path,
            size=entry.size,
            total_lines=total_lines,
            start_line=actual_offset,
            num_lines=num_lines,
            unchanged=True,
        )

    output = "\n".join(
        f"{number}: {line.rstrip()}"
        for number, line in enumerate(
            selected_range.content.splitlines(keepends=True),
            start=actual_offset + 1,
        )
    )
    truncated = len(output) > MAX_OUTPUT_SIZE
    if truncated:
        end = actual_offset + num_lines
        output = (
            output[:MAX_OUTPUT_SIZE]
            + f"\n...(truncated at {MAX_OUTPUT_SIZE} chars)\n"
            + f"Use offset={end} and limit={actual_limit} to read more."
        )
        context.file_state_cache.delete(entry.path)
    else:
        context.file_state_cache.set(
            entry.path,
            FileState(version=entry.version, offset=actual_offset, limit=num_lines),
        )
    return _read_result(
        "text",
        content=output,
        file_path=entry.path,
        size=entry.size,
        total_lines=total_lines,
        start_line=actual_offset,
        num_lines=num_lines,
        truncated=truncated,
    )


def _write_result(type_: str, path: str, written_bytes: int = 0, error: str = "") -> ToolResult:
    return ToolResult(
        output=WriteOutput(
            type=type_,
            file_path=path,
            written_bytes=written_bytes,
            error=error,
        ),
        is_error=bool(error),
    )


async def _existing_text(context: ToolContext, path: str) -> tuple[Any, str] | None:
    try:
        return await context.workspace.read_text_with_metadata(path)
    except WorkspaceNotFoundError:
        return None


async def _validate_file_state(
    context: ToolContext,
    path: str,
) -> tuple[FileState | None, str]:
    state = context.file_state_cache.get(path)
    if state is None:
        return None, ERROR_NOT_READ
    try:
        entry, _ = await context.workspace.read_text_with_metadata(path)
    except (WorkspaceError, UnicodeDecodeError):
        return None, ERROR_MODIFIED
    if entry.version != state.version:
        context.file_state_cache.delete(path)
        return None, ERROR_MODIFIED
    return state, ""


async def _save(
    context: ToolContext,
    path: str,
    content: str,
    *,
    expected_version: str | None,
) -> tuple[str, int, str]:
    try:
        entry = await context.workspace.write_text(
            path,
            content,
            overwrite=expected_version is not None,
            expected_version=expected_version,
        )
    except FileExistsError as exc:
        raise WorkspaceConflictError("File was created before this write completed.") from exc
    context.file_state_cache.set(entry.path, FileState(version=entry.version))
    return (
        "update" if expected_version is not None else "create",
        len(content.encode()),
        entry.path,
    )


@tool(
    name="write",
    description=_write_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    writes_workspace=True,
)
async def write(file_path: str, content: str, *, runtime: ToolContext) -> ToolResult:
    """Execute the write operation against the bound workspace."""

    context = runtime
    try:
        existing = await _existing_text(context, file_path)
    except (WorkspaceError, UnicodeDecodeError) as exc:
        return _write_result("create", file_path, error=str(exc))
    expected = None
    display_path = context.workspace.resolve_path(file_path)
    if existing is not None:
        entry, _ = existing
        display_path = entry.path
        state, error = await _validate_file_state(context, display_path)
        if state is None:
            return _write_result("update", display_path, error=error)
        expected = state.version
    try:
        type_, size, normalized = await _save(
            context,
            file_path,
            content,
            expected_version=expected,
        )
        return _write_result(type_, normalized, size)
    except Exception as exc:  # noqa: BLE001
        return _write_result("update" if existing else "create", file_path, error=str(exc))


@tool(
    name="append",
    description=_append_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    writes_workspace=True,
)
async def append(file_path: str, content: str, *, runtime: ToolContext) -> ToolResult:
    """Execute the append operation against the bound workspace."""

    context = runtime
    try:
        existing = await _existing_text(context, file_path)
    except (WorkspaceError, UnicodeDecodeError) as exc:
        return _write_result("create", file_path, error=str(exc))
    old_content = ""
    expected = None
    display_path = context.workspace.resolve_path(file_path)
    if existing is not None:
        entry, old_content = existing
        display_path = entry.path
        state, error = await _validate_file_state(context, display_path)
        if state is None:
            return _write_result("update", display_path, error=error)
        expected = state.version
    try:
        type_, size, normalized = await _save(
            context,
            file_path,
            old_content + content,
            expected_version=expected,
        )
        return _write_result(type_, normalized, size)
    except Exception as exc:  # noqa: BLE001
        return _write_result("update" if existing else "create", file_path, error=str(exc))


def _edit_result(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool,
    error: str = "",
) -> ToolResult:
    return ToolResult(
        output=EditOutput(
            file_path=file_path,
            old_string=old_string,
            new_string=new_string,
            replace_all=replace_all,
            error=error,
        ),
        is_error=bool(error),
    )


@tool(
    name="edit",
    description=_edit_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    writes_workspace=True,
)
async def edit(
    file_path: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
    *,
    runtime: ToolContext,
) -> ToolResult:
    """Execute the edit operation against the bound workspace."""

    context = runtime
    if old_string == new_string:
        return _edit_result(file_path, old_string, new_string, replace_all, ERROR_SAME_STRING)
    try:
        existing = await _existing_text(context, file_path)
    except (WorkspaceError, UnicodeDecodeError) as exc:
        return _edit_result(file_path, old_string, new_string, replace_all, str(exc))
    if existing is None:
        return _edit_result(file_path, old_string, new_string, replace_all, "File does not exist.")
    entry, content = existing
    state, error = await _validate_file_state(context, entry.path)
    if state is None:
        return _edit_result(entry.path, old_string, new_string, replace_all, error)
    if old_string not in content:
        return _edit_result(
            entry.path,
            old_string,
            new_string,
            replace_all,
            f"{ERROR_STRING_NOT_FOUND}\nString: {old_string}",
        )
    count = content.count(old_string)
    if count > 1 and not replace_all:
        return _edit_result(
            entry.path,
            old_string,
            new_string,
            replace_all,
            ERROR_MULTIPLE_MATCHES.format(count=count) + f"\nString: {old_string}",
        )
    updated = content.replace(old_string, new_string, -1 if replace_all else 1)
    try:
        await _save(context, entry.path, updated, expected_version=state.version)
    except Exception as exc:  # noqa: BLE001
        return _edit_result(entry.path, old_string, new_string, replace_all, str(exc))
    return _edit_result(entry.path, old_string, new_string, replace_all)


@tool(
    name="delete",
    description=_delete_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    writes_workspace=True,
)
async def delete(path: str, recursive: bool = False, *, runtime: ToolContext) -> ToolResult:
    """Execute the delete operation against the bound workspace."""

    context = runtime
    if any(char in path for char in SHELL_EXPANSION_CHARS):
        return _write_result("delete", path, error=ERROR_EXACT_PATH_REQUIRED)
    try:
        resolved = context.workspace.resolve_path(path)
        if resolved == context.workspace.default_root:
            return _write_result("delete", resolved, error="Refusing to delete workspace root.")
        await context.workspace.delete(resolved, recursive=recursive)
        if recursive:
            context.file_state_cache.delete_prefix(resolved)
        else:
            context.file_state_cache.delete(resolved)
        return _write_result("delete", resolved)
    except Exception as exc:  # noqa: BLE001
        return _write_result("delete", path, error=str(exc))


@tool(
    name="list",
    description=_list_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    allow_parallel=True,
)
async def list_directory(path: str = ".", *, runtime: ToolContext) -> ToolResult:
    """Execute the list operation against the bound workspace."""

    context = runtime
    try:
        entries = await context.workspace.list(path, max_entries=context.list_max_entries)
        values = [
            f"{entry.path}/" if entry.entry_type is EntryType.DIRECTORY else entry.path
            for entry in entries
        ]
        output = ListOutput(entries=values, num_entries=len(values))
        return ToolResult(output=output)
    except (WorkspaceError, ValueError) as exc:
        return ToolResult(output=ListOutput(error=str(exc)), is_error=True)


def _glob_result(
    filenames: list[str],
    *,
    duration_ms: int = 0,
    truncated: bool = False,
    error: str = "",
) -> ToolResult:
    return ToolResult(
        output=GlobOutput(
            filenames=filenames,
            num_files=len(filenames),
            duration_ms=duration_ms,
            truncated=truncated,
            error=error,
        ),
        is_error=bool(error),
    )


@tool(
    name="glob",
    description=_glob_description(DEFAULT_FILE_TOOL_DESCRIPTION_PROFILE),
    category="filesystem",
    allow_parallel=True,
)
async def glob(pattern: str, path: str | None = None, *, runtime: ToolContext) -> ToolResult:
    """Execute the glob operation against the bound workspace."""

    context = runtime
    started = time.time()
    normalized_pattern = pattern.replace("\\", "/").strip()
    static_prefix = _static_glob_prefix(normalized_pattern)
    try:
        if normalized_pattern.startswith("/"):
            context.workspace.resolve_path(static_prefix or normalized_pattern)
        search_path = path or static_prefix or context.workspace.default_root
        base_prefix = context.workspace.resolve_path(search_path)
        if path and normalized_pattern.startswith("/"):
            pattern_prefix = context.workspace.resolve_path(static_prefix or normalized_pattern)
            if not _is_under_prefix(pattern_prefix, base_prefix):
                return _glob_result(
                    [], error="Absolute glob pattern must be within the provided path."
                )
        entries = await _file_entries(context, search_path)
    except (WorkspaceError, ValueError) as exc:
        return _glob_result([], error=str(exc))
    matches = []
    for entry in entries:
        candidates = _candidate_paths(
            entry.path,
            base_prefix,
            context.workspace.default_root,
        )
        if any(_glob_matches(candidate, normalized_pattern) for candidate in candidates):
            matches.append(entry.path)
    matches.sort()
    truncated = len(matches) > DEFAULT_GLOB_MAX_RESULTS
    return _glob_result(
        matches[:DEFAULT_GLOB_MAX_RESULTS],
        duration_ms=int((time.time() - started) * 1000),
        truncated=truncated,
    )


class _SearchRegex(Protocol):
    def search(self, text: str) -> object | None: ...


class _ContentCollector:
    def __init__(self, offset: int, limit: int | None, max_lines: int, max_bytes: int) -> None:
        self.remaining_offset = offset
        self.line_limit = min(limit or max_lines, max_lines)
        self.max_bytes = max_bytes
        self.result_bytes = 0
        self.values: list[str] = []
        self.applied_limit: int | None = None

    def append(self, value: str) -> bool:
        if self.remaining_offset:
            self.remaining_offset -= 1
            return True
        if len(self.values) >= self.line_limit:
            self.applied_limit = self.line_limit
            return False
        value_bytes = len(value.encode()) + (1 if self.values else 0)
        if self.result_bytes + value_bytes > self.max_bytes:
            self.applied_limit = len(self.values)
            return False
        self.values.append(value)
        self.result_bytes += value_bytes
        return True


def _grep_result(mode: str, **values: Any) -> ToolResult:
    output = GrepOutput(mode=mode, **values)
    return ToolResult(output=output, is_error=bool(output.error))


@tool(name="grep", category="filesystem", allow_parallel=True)
async def grep(
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
    output_mode: Literal["content", "files_with_matches", "count"] = "files_with_matches",
    context: int | None = None,
    ignore_case: bool = False,
    type: str | None = None,
    head_limit: int | None = None,
    offset: int = 0,
    multiline: bool = False,
    *,
    runtime: ToolContext,
) -> ToolResult:
    r"""A powerful search tool for readable workspace text files.

    Usage:
    - Always use this tool for content search instead of `grep` or `rg` through
      Bash.
    - This tool requires a `path`. Provide a readable file or directory path
      returned by the user, `glob`, or another tool result.
    - Supports RE2 safe regular expressions, such as `log.*Error` or
      `function\s+\w+`.
    - Backreferences and look-around assertions are not supported.
    - Filter files with the `glob` parameter, such as `*.md` or `**/*.txt`, or
      the `type` parameter, such as `md`, `txt`, or `json`.
    - Output modes: `content` shows matching lines, `files_with_matches` shows
      only file paths, and `count` shows match counts.
    - Pattern syntax uses RE2. Literal braces need escaping, such as
      `interface\{\}` to find `interface{}`.
    - By default patterns match within single lines only. For cross-line
      patterns, use `multiline: true`.

    Args:
        pattern: The RE2 regular expression pattern to search for in file contents.
        path: Required file or directory path prefix to search in.
        glob: Glob pattern to filter files, such as `*.md` or `**/*.txt`.
        output_mode: Output mode. `content` shows matching lines,
            `files_with_matches` shows file paths, and `count` shows match
            counts.
        context: Number of context lines to include before and after each match.
            Applies only when `output_mode` is `content`.
        ignore_case: Whether to run a case-insensitive search.
        type: File type to search, such as `md`, `markdown`, or `txt`.
        head_limit: Limit output to first N rows or entries. Defaults to 250.
            Pass 0 for unlimited.
        offset: Skip first N rows or entries before applying `head_limit`.
        multiline: Enable multiline mode where `.` matches newlines.

    Returns:
        dict with:
            - mode: Output mode used.
            - filenames: Array of file paths for `files_with_matches` mode.
            - content: Matching content for `content` or `count` mode.
            - num_files: Number of files returned.
            - num_lines: Number of content lines returned.
            - num_matches: Total number of matches for `count` mode.
            - applied_limit: Output limit actually applied.
            - applied_offset: Output offset actually applied.
            - error: Error message if failed, empty on success.
    """

    if len(pattern) > MAX_GREP_PATTERN_LENGTH:
        return _grep_result(
            output_mode,
            error=f"grep pattern exceeds the {MAX_GREP_PATTERN_LENGTH}-character limit.",
        )
    if context is not None and context > MAX_GREP_CONTEXT_LINES:
        return _grep_result(
            output_mode,
            error=f"grep context exceeds the {MAX_GREP_CONTEXT_LINES}-line limit.",
        )
    if head_limit is not None and head_limit < 0:
        return _grep_result(output_mode, error="grep head_limit must be at least 0.")
    if path is None or not path.strip():
        return _grep_result(
            output_mode,
            error="grep requires a path. Specify a readable file or directory path.",
        )
    options = re2.Options()
    options.case_sensitive = not ignore_case
    options.dot_nl = multiline
    options.log_errors = False
    try:
        regex = re2.compile(pattern, options=options)
    except re2.error as exc:
        return _grep_result(output_mode, error=f"Invalid RE2 pattern: {exc}")
    try:
        base_prefix = runtime.workspace.resolve_path(path)
        entries = await _file_entries(runtime, path)
    except (WorkspaceError, ValueError) as exc:
        return _grep_result(output_mode, error=str(exc))
    suffixes = _type_suffixes(type)
    candidates = [
        entry
        for entry in entries
        if (not suffixes or any(entry.path.endswith(suffix) for suffix in suffixes))
        and (
            not glob
            or any(
                fnmatch.fnmatch(candidate, glob)
                for candidate in _candidate_paths(
                    entry.path,
                    base_prefix,
                    runtime.workspace.default_root,
                )
            )
        )
    ]
    total_bytes = sum(entry.size for entry in candidates)
    if len(candidates) > runtime.grep_max_scan_files or total_bytes > runtime.grep_max_scan_bytes:
        return _grep_result(
            output_mode,
            error=(
                "grep search scope is too large, so no file content was scanned. "
                f"Current candidates: {len(candidates)} files / {_format_bytes(total_bytes)}; "
                f"limit: {runtime.grep_max_scan_files} files / "
                f"{_format_bytes(runtime.grep_max_scan_bytes)}. Narrow `path` to a more "
                "specific subdirectory, or add `glob`/`type` before retrying."
            ),
        )
    requested_limit: int | None = DEFAULT_GREP_HEAD_LIMIT if head_limit is None else head_limit
    requested_limit = None if requested_limit == 0 else requested_limit
    safe_offset = max(offset, 0)
    applied_offset = safe_offset or None
    collector = _ContentCollector(
        safe_offset,
        requested_limit,
        runtime.grep_max_result_lines,
        runtime.grep_max_result_bytes,
    )
    files: set[str] = set()
    counts: dict[str, int] = {}
    context_size = max(context or 0, 0)
    if multiline:
        scanned_bytes = 0
        collection_stopped = False
        for entry in candidates:
            try:
                _, content = await runtime.workspace.read_text_with_metadata(entry.path)
            except (WorkspaceError, UnicodeDecodeError):
                continue
            scanned_bytes += len(content.encode("utf-8"))
            if scanned_bytes > runtime.grep_max_scan_bytes:
                return _grep_result(
                    output_mode,
                    error=(
                        "Workspace content scan exceeds the server byte limit. "
                        "Narrow the requested path and retry."
                    ),
                )
            lines = content.splitlines()
            emitted: set[int] = set()
            for index, line in enumerate(lines):
                checked = line[:MAX_COLUMNS] + ("..." if len(line) > MAX_COLUMNS else "")
                if regex.search(checked) is None:
                    continue
                if output_mode == "files_with_matches":
                    files.add(entry.path)
                    break
                if output_mode == "count":
                    counts[entry.path] = counts.get(entry.path, 0) + 1
                    continue
                start = max(0, index - context_size)
                end = min(len(lines), index + context_size + 1)
                for current in range(start, end):
                    if current in emitted:
                        continue
                    emitted.add(current)
                    raw_line = lines[current]
                    current_line = raw_line[:MAX_COLUMNS] + (
                        "..." if len(raw_line) > MAX_COLUMNS else ""
                    )
                    marker = ":" if regex.search(current_line) else "-"
                    if not collector.append(f"{entry.path}{marker}{current + 1}: {current_line}"):
                        collection_stopped = True
                        break
                if collection_stopped:
                    break
            if collection_stopped:
                break
    else:
        current_file_path = ""
        previous_lines: deque[tuple[int, str]] = deque(maxlen=context_size)
        emitted_lines: set[int] = set()
        remaining_after_context = 0

        def append_streamed_line(
            file_path: str,
            line_number: int,
            line: str,
            *,
            is_match: bool,
        ) -> bool:
            if line_number in emitted_lines:
                return True
            emitted_lines.add(line_number)
            marker = ":" if is_match else "-"
            return collector.append(f"{file_path}{marker}{line_number}: {line}")

        def visit_line(entry: Any, line_number: int, line: str) -> bool:
            nonlocal current_file_path, previous_lines, emitted_lines
            nonlocal remaining_after_context
            if entry.path != current_file_path:
                current_file_path = entry.path
                previous_lines = deque(maxlen=context_size)
                emitted_lines = set()
                remaining_after_context = 0
            checked = line[:MAX_COLUMNS] + ("..." if len(line) > MAX_COLUMNS else "")
            is_match = regex.search(checked) is not None
            if is_match and output_mode == "files_with_matches":
                files.add(entry.path)
            elif is_match and output_mode == "count":
                counts[entry.path] = counts.get(entry.path, 0) + 1
            elif output_mode == "content":
                if is_match:
                    for previous_number, previous_line in previous_lines:
                        if not append_streamed_line(
                            entry.path,
                            previous_number,
                            previous_line,
                            is_match=regex.search(previous_line) is not None,
                        ):
                            return False
                    if not append_streamed_line(
                        entry.path,
                        line_number,
                        checked,
                        is_match=True,
                    ):
                        return False
                    remaining_after_context = context_size
                elif remaining_after_context:
                    if not append_streamed_line(
                        entry.path,
                        line_number,
                        checked,
                        is_match=False,
                    ):
                        return False
                    remaining_after_context -= 1
            previous_lines.append((line_number, checked))
            return True

        try:
            await runtime.workspace.visit_text_lines(
                candidates,
                visit_line,
                max_total_bytes=runtime.grep_max_scan_bytes,
            )
        except WorkspaceError as exc:
            return _grep_result(output_mode, error=str(exc))
    if output_mode == "files_with_matches":
        values, applied_limit = _slice(sorted(files), safe_offset, requested_limit)
        return _grep_result(
            output_mode,
            filenames=values,
            num_files=len(values),
            applied_limit=applied_limit,
            applied_offset=applied_offset,
        )
    if output_mode == "count":
        values = [f"{path}: {count}" for path, count in sorted(counts.items())]
        total = sum(counts.values())
        values, applied_limit = _slice(values, safe_offset, requested_limit)
        return _grep_result(
            output_mode,
            content="\n".join(values),
            num_files=len(values),
            num_matches=total,
            applied_limit=applied_limit,
            applied_offset=applied_offset,
        )
    return _grep_result(
        output_mode,
        content="\n".join(collector.values),
        num_lines=len(collector.values),
        applied_limit=collector.applied_limit,
        applied_offset=applied_offset,
    )


async def _file_entries(context: ToolContext, path: str) -> list[Any]:
    entry = await context.workspace.stat(path)
    if entry.entry_type is EntryType.FILE:
        return [entry]
    values = [
        item
        for item in await context.workspace.walk(
            path,
            max_entries=context.file_search_max_entries,
        )
        if item.entry_type is EntryType.FILE
    ]
    values.sort(key=lambda item: item.path)
    return values


def _static_glob_prefix(pattern: str) -> str:
    if not pattern or pattern.lower() in {"undefined", "null"}:
        return ""
    normalized = pattern.rstrip("/")
    segments = normalized.split("/")
    prefix: list[str] = []
    for segment in segments:
        if any(char in segment for char in "*?["):
            break
        prefix.append(segment)
    if len(prefix) == len(segments):
        value = normalized.rsplit("/", 1)[0] if "/" in normalized else ""
    else:
        value = "/".join(prefix).rstrip("/")
    return "" if value in {"", "/"} else value


def _candidate_paths(file_path: str, *base_prefixes: str) -> set[str]:
    candidates = {file_path, file_path.lstrip("/"), file_path.rsplit("/", 1)[-1]}
    for base_prefix in base_prefixes:
        prefix = base_prefix.rstrip("/")
        if file_path == prefix or file_path.startswith(f"{prefix}/"):
            relative = file_path[len(prefix) :].lstrip("/")
            if relative:
                candidates.add(relative)
    return candidates


def _is_under_prefix(path: str, prefix: str) -> bool:
    normalized = prefix.rstrip("/")
    return path == normalized or path.startswith(f"{normalized}/")


def _glob_matches(candidate: str, pattern: str) -> bool:
    return re.fullmatch(_glob_to_regex(pattern), candidate) is not None


def _glob_to_regex(pattern: str) -> str:
    absolute = pattern.startswith("/")
    segments = pattern.split("/")
    index = 1 if absolute else 0
    regex = "^/" if absolute else "^"
    while index < len(segments):
        segment = segments[index]
        last = index == len(segments) - 1
        if segment == "**":
            regex += "(?:[^/]+(?:/[^/]+)*)?" if last else "(?:[^/]+/)*"
        else:
            regex += _glob_segment_to_regex(segment)
            if not last:
                regex += "/"
        index += 1
    return f"{regex}$"


def _glob_segment_to_regex(segment: str) -> str:
    result = ""
    index = 0
    while index < len(segment):
        char = segment[index]
        if char == "*":
            result += "[^/]*"
        elif char == "?":
            result += "[^/]"
        elif char == "[":
            end = segment.find("]", index + 1)
            if end < 0:
                result += re.escape(char)
            else:
                content = segment[index + 1 : end]
                if content.startswith("!"):
                    content = f"^{content[1:]}"
                result += f"[{content}]"
                index = end
        else:
            result += re.escape(char)
        index += 1
    return result


def _slice(values: list[str], offset: int, limit: int | None) -> tuple[list[str], int | None]:
    selected = values[offset:] if offset else values
    if limit is not None and len(selected) > limit:
        return selected[:limit], limit
    return selected, None


def _type_suffixes(value: str | None) -> tuple[str, ...]:
    return {
        "md": (".md", ".markdown"),
        "markdown": (".md", ".markdown"),
        "txt": (".txt",),
    }.get(value or "", ())


def _format_bytes(value: int) -> str:
    if value < 1024:
        return f"{value} B"
    if value < 1024 * 1024:
        return f"{value / 1024:.1f} KiB"
    return f"{value / (1024 * 1024):.1f} MiB"


FILE_TOOLS = (read, write, append, edit, delete, glob, grep, list_directory)
