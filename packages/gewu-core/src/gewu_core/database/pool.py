"""Instrumented SQLAlchemy pool for native asynchronous database checkout."""

from __future__ import annotations

import time
from collections.abc import Callable

from sqlalchemy.exc import TimeoutError as SqlAlchemyTimeoutError
from sqlalchemy.pool import AsyncAdaptedQueuePool
from sqlalchemy.pool.base import ConnectionPoolEntry

DbCheckoutRecorder = Callable[[float, str], None]


def _ignore_checkout(_seconds: float, _outcome: str) -> None:
    return None


_record_db_checkout: DbCheckoutRecorder = _ignore_checkout


def configure_db_checkout_recorder(recorder: DbCheckoutRecorder | None) -> DbCheckoutRecorder:
    """Install a process metric sink and return the previously configured sink."""
    global _record_db_checkout
    previous = _record_db_checkout
    _record_db_checkout = recorder or _ignore_checkout
    return previous


class InstrumentedAsyncAdaptedQueuePool(AsyncAdaptedQueuePool):
    """Measure pool acquisition while preserving SQLAlchemy's async queue."""

    def _do_get(self) -> ConnectionPoolEntry:
        started = time.perf_counter()
        outcome = "success"
        try:
            return super()._do_get()
        except SqlAlchemyTimeoutError:
            outcome = "timeout"
            raise
        except BaseException:
            outcome = "failure"
            raise
        finally:
            _record_db_checkout(time.perf_counter() - started, outcome)
