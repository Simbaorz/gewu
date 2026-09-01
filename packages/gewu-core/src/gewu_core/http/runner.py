"""Command-line runner shared by dedicated HTTP services."""

from __future__ import annotations

import argparse

import uvicorn

from gewu_core.logging import init_logging


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def run_http_service(app_path: str, description: str) -> None:
    """Parse stable server flags and run one ASGI application."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind.")
    parser.add_argument("--port", default=8000, type=int, help="Port to bind.")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")
    parser.add_argument(
        "--workers",
        default=None,
        type=_positive_int,
        help="Number of Uvicorn worker processes.",
    )
    args = parser.parse_args()

    init_logging()
    uvicorn.run(
        app=app_path,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers,
    )
