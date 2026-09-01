"""Workspace mount and backend contracts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, SkipValidation, field_validator

from gewu_agent_runtime.workspace.paths import normalize_logical_path


class AccessMode(StrEnum):
    """Write capability granted by one mount."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


class EntryType(StrEnum):
    """Kinds of logical workspace entries."""

    FILE = "file"
    DIRECTORY = "directory"


class WorkspaceEntry(BaseModel):
    """Metadata for one logical or backend-relative resource."""

    model_config = ConfigDict(frozen=True)

    path: str
    entry_type: EntryType
    size: int = Field(default=0, ge=0)
    version: str = ""


class WorkspaceTextRange(BaseModel):
    """One UTF-8 line range with metadata for the complete file snapshot."""

    model_config = ConfigDict(frozen=True)

    entry: WorkspaceEntry
    content: str = ""
    total_lines: int = Field(ge=0)
    start_line: int = Field(ge=0)
    num_lines: int = Field(ge=0)


WorkspaceLineVisitor = Callable[[WorkspaceEntry, int, str], bool]


class WorkspaceBackend(Protocol):
    """Storage operations scoped to one backend namespace."""

    async def stat(self, path: str) -> WorkspaceEntry:
        """Return metadata for one backend-relative path."""

    async def list(
        self,
        path: str,
        *,
        max_entries: int | None = None,
    ) -> Sequence[WorkspaceEntry]:
        """List direct children without traversing beyond an optional hard limit."""

    async def read_bytes(self, path: str) -> bytes:
        """Read a file."""

    async def read_bytes_with_metadata(self, path: str) -> tuple[WorkspaceEntry, bytes]:
        """Atomically read file metadata and bytes from one snapshot."""

    async def read_text_range(
        self,
        path: str,
        *,
        offset: int,
        limit: int,
    ) -> WorkspaceTextRange:
        """Read a bounded UTF-8 line range and complete-file metadata."""

    async def visit_text_lines(
        self,
        path: str,
        visitor: WorkspaceLineVisitor,
        *,
        max_bytes: int | None = None,
    ) -> tuple[int, bool]:
        """Visit UTF-8 lines within a byte budget and report scan completion."""

    async def write_bytes(
        self,
        path: str,
        data: bytes,
        *,
        overwrite: bool = True,
        expected_version: str | None = None,
    ) -> WorkspaceEntry:
        """Write a file, optionally requiring its current version."""

    async def mkdir(self, path: str, *, parents: bool = True) -> WorkspaceEntry:
        """Create a directory."""

    async def delete(self, path: str, *, recursive: bool = False) -> None:
        """Delete a file or directory."""

    async def move(
        self, source: str, destination: str, *, overwrite: bool = False
    ) -> WorkspaceEntry:
        """Move one resource inside the backend."""


class WorkspaceMount(BaseModel):
    """One already-authorized logical mount."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    mount_id: str = Field(min_length=1, max_length=128)
    mount_path: str
    access_mode: AccessMode
    backend: SkipValidation[WorkspaceBackend]
    priority: int = 0

    @field_validator("mount_path")
    @classmethod
    def normalize_mount_path(cls, value: str) -> str:
        """Store a canonical logical mount path."""

        return normalize_logical_path(value)
