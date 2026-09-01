"""Cross-process locking and cancellation-safe filesystem compensation."""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from gewu_core import file_transactions
from gewu_core.errors import CommitOutcomeUnknownError
from gewu_core.file_locks import filesystem_mutation_lock
from gewu_core.file_transactions import (
    quarantine_path_until_success,
    remove_directory_created_on_error,
    restore_directory_on_error,
)
from gewu_core.runtime_temp import set_runtime_temp_root_provider


@pytest.fixture(autouse=True)
def configured_runtime_temp(tmp_path: Path):
    previous = set_runtime_temp_root_provider(lambda: tmp_path / ".runtime-temp")
    try:
        yield
    finally:
        set_runtime_temp_root_provider(previous)


async def test_filesystem_mutation_lock_serializes_same_physical_path(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first() -> None:
        async with filesystem_mutation_lock((target,)):
            first_entered.set()
            await release_first.wait()

    async def second() -> None:
        await first_entered.wait()
        async with filesystem_mutation_lock((target,)):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0.06)
    assert second_entered.is_set() is False

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set() is True


async def test_restore_directory_preserves_new_state_when_commit_is_unknown(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("old", encoding="utf-8")

    with pytest.raises(CommitOutcomeUnknownError):
        async with restore_directory_on_error(target):
            (target / "content.txt").write_text("new", encoding="utf-8")
            raise CommitOutcomeUnknownError("unknown")

    assert (target / "content.txt").read_text(encoding="utf-8") == "new"


async def test_restore_directory_on_error_restores_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected failure"):
        async with restore_directory_on_error(target):
            (target / "old.txt").write_text("changed", encoding="utf-8")
            (target / "new.txt").write_text("new", encoding="utf-8")
            raise RuntimeError("injected failure")

    assert (target / "old.txt").read_text(encoding="utf-8") == "old"
    assert not (target / "new.txt").exists()


async def test_restore_directory_on_error_removes_new_directory(tmp_path: Path) -> None:
    target = tmp_path / "asset"

    with pytest.raises(RuntimeError, match="injected failure"):
        async with restore_directory_on_error(target):
            target.mkdir()
            raise RuntimeError("injected failure")

    assert not target.exists()


async def test_restore_directory_rolls_back_cancellation(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("old", encoding="utf-8")

    with pytest.raises(asyncio.CancelledError):
        async with restore_directory_on_error(target):
            (target / "content.txt").write_text("new", encoding="utf-8")
            raise asyncio.CancelledError

    assert (target / "content.txt").read_text(encoding="utf-8") == "old"


async def test_restore_directory_only_removes_new_cleanup_paths(tmp_path: Path) -> None:
    target = tmp_path / "source"
    destination = tmp_path / "destination"
    new_destination = tmp_path / "new-destination"
    target.mkdir()
    destination.mkdir()
    (target / "source.txt").write_text("source", encoding="utf-8")
    (destination / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected failure"):
        async with restore_directory_on_error(
            target,
            cleanup_paths=(destination, new_destination),
        ):
            (target / "source.txt").write_text("changed", encoding="utf-8")
            new_destination.mkdir()
            raise RuntimeError("injected failure")

    assert (target / "source.txt").read_text(encoding="utf-8") == "source"
    assert (destination / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not new_destination.exists()


async def test_restore_directory_on_error_serializes_same_path_mutations(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("original", encoding="utf-8")
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def fail_first_mutation() -> None:
        with pytest.raises(RuntimeError, match="injected failure"):
            async with restore_directory_on_error(target):
                (target / "content.txt").write_text("first", encoding="utf-8")
                first_entered.set()
                await release_first.wait()
                raise RuntimeError("injected failure")

    async def succeed_second_mutation() -> None:
        await first_entered.wait()
        async with restore_directory_on_error(target):
            (target / "content.txt").write_text("second", encoding="utf-8")

    first_task = asyncio.create_task(fail_first_mutation())
    await first_entered.wait()
    second_task = asyncio.create_task(succeed_second_mutation())
    await asyncio.sleep(0.02)
    assert (target / "content.txt").read_text(encoding="utf-8") == "first"

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert (target / "content.txt").read_text(encoding="utf-8") == "second"


async def test_restore_directory_finishes_compensation_after_repeated_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("old", encoding="utf-8")
    restore_started = threading.Event()
    release_restore = threading.Event()
    original_restore = file_transactions._restore_directory

    def delayed_restore(*args: Any) -> None:
        restore_started.set()
        release_restore.wait(timeout=2)
        original_restore(*args)

    monkeypatch.setattr(file_transactions, "_restore_directory", delayed_restore)

    async def mutate() -> None:
        async with restore_directory_on_error(target):
            (target / "content.txt").write_text("new", encoding="utf-8")
            raise RuntimeError("injected failure")

    task = asyncio.create_task(mutate())
    assert await asyncio.to_thread(restore_started.wait, 2)
    task.cancel()
    task.cancel()
    release_restore.set()

    with pytest.raises(RuntimeError, match="injected failure"):
        await task
    assert (target / "content.txt").read_text(encoding="utf-8") == "old"


async def test_quarantine_restores_and_propagates_initial_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("content", encoding="utf-8")
    original_run_file_task = file_transactions.run_file_task
    cancelled = False
    body_entered = False

    async def cancel_after_replace(operation: Any, /, *args: Any, **kwargs: Any):
        nonlocal cancelled
        result = await original_run_file_task(operation, *args, **kwargs)
        if getattr(operation, "__name__", "") == "replace" and not cancelled:
            cancelled = True
            raise asyncio.CancelledError
        return result

    monkeypatch.setattr(file_transactions, "run_file_task", cancel_after_replace)

    with pytest.raises(asyncio.CancelledError):
        async with quarantine_path_until_success(target):
            body_entered = True

    assert body_entered is False
    assert (target / "content.txt").read_text(encoding="utf-8") == "content"
    assert _quarantines(tmp_path) == []


async def test_quarantine_preserves_hidden_path_when_commit_is_unknown(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()

    with pytest.raises(CommitOutcomeUnknownError):
        async with quarantine_path_until_success(target):
            raise CommitOutcomeUnknownError("unknown")

    assert target.exists() is False
    assert len(_quarantines(tmp_path)) == 1


async def test_quarantine_path_until_success_restores_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("content", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected failure"):
        async with quarantine_path_until_success(target):
            assert not target.exists()
            raise RuntimeError("injected failure")

    assert (target / "content.txt").read_text(encoding="utf-8") == "content"
    assert _quarantines(tmp_path) == []


async def test_quarantine_path_until_success_removes_on_success(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()

    async with quarantine_path_until_success(target):
        assert not target.exists()

    assert not target.exists()
    assert _quarantines(tmp_path) == []


async def test_quarantine_restores_path_after_cancellation(tmp_path: Path) -> None:
    target = tmp_path / "asset"
    target.mkdir()

    with pytest.raises(asyncio.CancelledError):
        async with quarantine_path_until_success(target):
            raise asyncio.CancelledError

    assert target.is_dir()
    assert _quarantines(tmp_path) == []


async def test_quarantine_restore_does_not_overwrite_concurrent_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("original", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected failure"):
        async with quarantine_path_until_success(target):
            target.mkdir()
            (target / "content.txt").write_text("concurrent", encoding="utf-8")
            raise RuntimeError("injected failure")

    assert (target / "content.txt").read_text(encoding="utf-8") == "concurrent"
    assert len(_quarantines(tmp_path)) == 1


async def test_quarantine_cleanup_failure_is_best_effort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    original_remove = file_transactions.remove_path

    def fail_quarantine_remove(path: Path) -> None:
        if ".delete-" in path.name:
            raise OSError("file-cleanup-private-secret")
        original_remove(path)

    monkeypatch.setattr(file_transactions, "remove_path", fail_quarantine_remove)

    async with quarantine_path_until_success(target):
        pass

    assert "Unable to remove committed filesystem quarantine" in caplog.text
    assert "exception_type=OSError" in caplog.text
    assert "file-cleanup-private-secret" not in caplog.text
    assert len(_quarantines(tmp_path)) == 1


async def test_remove_created_directory_preserves_it_when_commit_is_unknown(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"

    with pytest.raises(CommitOutcomeUnknownError):
        async with remove_directory_created_on_error(target):
            target.mkdir()
            raise CommitOutcomeUnknownError("unknown")

    assert target.is_dir()


async def test_remove_directory_created_on_error_keeps_existing_directory(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"
    target.mkdir()
    (target / "content.txt").write_text("existing", encoding="utf-8")

    with pytest.raises(RuntimeError, match="injected failure"):
        async with remove_directory_created_on_error(target):
            raise RuntimeError("injected failure")

    assert (target / "content.txt").read_text(encoding="utf-8") == "existing"


async def test_remove_created_directory_removes_new_directory_on_failure(
    tmp_path: Path,
) -> None:
    target = tmp_path / "asset"

    with pytest.raises(RuntimeError, match="injected failure"):
        async with remove_directory_created_on_error(target):
            target.mkdir()
            raise RuntimeError("injected failure")

    assert not target.exists()


def _quarantines(directory: Path) -> list[Path]:
    return list(directory.glob(".asset.delete-*"))
