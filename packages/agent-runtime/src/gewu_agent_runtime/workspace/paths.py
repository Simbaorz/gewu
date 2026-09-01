"""Logical POSIX path normalization."""

from pathlib import PurePosixPath

from gewu_agent_runtime.workspace.errors import WorkspacePathError


def normalize_logical_path(value: str) -> str:
    """Normalize one absolute logical path and reject namespace traversal."""

    candidate = value.replace("\\", "/").strip()
    if not candidate:
        return "/"
    if not candidate.startswith("/"):
        candidate = f"/{candidate}"
    parts: list[str] = []
    for part in PurePosixPath(candidate).parts:
        if part in {"/", "", "."}:
            continue
        if part == "..":
            raise WorkspacePathError("Workspace paths cannot contain '..'.")
        if "\x00" in part:
            raise WorkspacePathError("Workspace paths cannot contain NUL bytes.")
        parts.append(part)
    return f"/{'/'.join(parts)}" if parts else "/"


def normalize_backend_path(value: str) -> str:
    """Normalize a path relative to one backend root."""

    logical = normalize_logical_path(value)
    return logical.removeprefix("/")


def join_logical(parent: str, child: str) -> str:
    """Join a normalized logical parent and one relative child path."""

    return normalize_logical_path(f"{parent.rstrip('/')}/{child.lstrip('/')}")
