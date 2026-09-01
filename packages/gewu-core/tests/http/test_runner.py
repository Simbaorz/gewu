"""Subscriber HTTP command-line runner."""

from __future__ import annotations

import sys

import pytest

from gewu_core.http import runner


def test_runner_leaves_worker_default_to_uvicorn(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["subscriber-web-api"])
    monkeypatch.setattr(runner, "init_logging", lambda: None)
    monkeypatch.setattr(runner.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

    runner.run_http_service("example.app:app", "Example")

    assert captured["workers"] is None


def test_runner_forwards_explicit_worker_count(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(sys, "argv", ["subscriber-web-api", "--workers", "3"])
    monkeypatch.setattr(runner, "init_logging", lambda: None)
    monkeypatch.setattr(runner.uvicorn, "run", lambda **kwargs: captured.update(kwargs))

    runner.run_http_service("example.app:app", "Example")

    assert captured["workers"] == 3


@pytest.mark.parametrize("worker_count", ["0", "-1"])
def test_runner_rejects_non_positive_worker_count(monkeypatch, worker_count: str) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["subscriber-web-api", "--workers", worker_count],
    )

    with pytest.raises(SystemExit):
        runner.run_http_service("example.app:app", "Example")
