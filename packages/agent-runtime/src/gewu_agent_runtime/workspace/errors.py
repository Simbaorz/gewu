"""Workspace error vocabulary."""


class WorkspaceError(Exception):
    """Base class for workspace failures."""


class WorkspacePathError(WorkspaceError):
    """Raised when a logical path is invalid or escapes its namespace."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when a logical resource does not exist."""


class WorkspacePermissionError(WorkspaceError):
    """Raised when a session does not grant the requested operation."""


class WorkspaceConflictError(WorkspaceError):
    """Raised when compare-and-swap detects a changed resource version."""


class WorkspaceCapacityError(WorkspaceError):
    """Raised when a file or workspace exceeds an admitted byte limit."""


class WorkspaceScanLimitError(WorkspaceCapacityError):
    """Raised before a workspace metadata traversal exceeds its entry limit."""
