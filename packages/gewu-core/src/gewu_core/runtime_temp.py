"""Configured process temporary-directory infrastructure."""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from pathlib import Path

RuntimeTempRootProvider = Callable[[], Path]

_root_provider: RuntimeTempRootProvider | None = None
_subdirectories: dict[str, Path] = {}
_subdirectories_lock = threading.Lock()


def resolve_runtime_temp_root(configured: str, project_home: str | Path) -> Path:
    """Resolve and validate the configured runtime temporary root."""

    configured = configured.strip()
    if not configured:
        raise RuntimeError("runtime.temp_dir must not be empty.")
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(project_home).expanduser() / path
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Unable to create runtime.temp_dir: {path}") from exc
    if not path.is_dir():
        raise RuntimeError(f"runtime.temp_dir must be a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        raise RuntimeError(f"runtime.temp_dir must be writable: {path}")
    return path.resolve()


def set_runtime_temp_root_provider(
    provider: RuntimeTempRootProvider | None,
) -> RuntimeTempRootProvider | None:
    """Replace the process provider and return its previous value."""

    global _root_provider
    previous = _root_provider
    _root_provider = provider
    with _subdirectories_lock:
        _subdirectories.clear()
    return previous


def runtime_temp_root() -> Path:
    """Return the process temporary root supplied by composition."""

    if _root_provider is None:
        raise RuntimeError("Runtime temporary root is not configured.")
    return _root_provider()  # noqa


def runtime_temp_subdir(name: str) -> Path:
    """Return one named direct child of the configured temporary root."""

    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("Runtime temporary subdirectory name must be one path segment.")
    with _subdirectories_lock:
        cached = _subdirectories.get(name)
        if cached is not None:
            return cached
        path = runtime_temp_root() / name
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(f"Unable to create runtime temporary subdirectory: {path}") from exc
        if not path.is_dir():
            raise RuntimeError(f"Runtime temporary subdirectory must be a directory: {path}")
        _subdirectories[name] = path
        return path


def prepare_runtime_temp_subdirs(names: tuple[str, ...]) -> tuple[Path, ...]:
    """Create known temporary subdirectories during process startup."""

    return tuple(runtime_temp_subdir(name) for name in names)
