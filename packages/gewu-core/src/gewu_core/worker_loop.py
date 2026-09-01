"""Process-owned asyncio loop for synchronous background-job entrypoints."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from concurrent.futures import Future
from typing import Any


class _WorkerSubmission[T]:
    """Bridge one coroutine result and cancellation across worker threads."""

    def __init__(self, coroutine: Coroutine[Any, Any, T]) -> None:
        self._coroutine = coroutine
        self._future: Future[T] = Future()
        self._lock = threading.Lock()
        self._task: asyncio.Task[T] | None = None
        self._cancel_requested = False

    @property
    def done(self) -> bool:
        return self._future.done()

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            task = loop.create_task(self._coroutine)
        except BaseException as exc:
            self._coroutine.close()
            self._future.set_exception(exc)
            return
        with self._lock:
            self._task = task
            cancel_requested = self._cancel_requested
        task.add_done_callback(self._complete)
        if cancel_requested:
            task.cancel()

    def result(self) -> T:
        return self._future.result()

    def cancel(self, loop: asyncio.AbstractEventLoop) -> None:
        with self._lock:
            self._cancel_requested = True
            task = self._task
        if task is not None:
            loop.call_soon_threadsafe(task.cancel)

    def _complete(self, task: asyncio.Task[T]) -> None:
        if task.cancelled():
            self._future.cancel()
            return
        exception = task.exception()
        if exception is not None:
            self._future.set_exception(exception)
            return
        self._future.set_result(task.result())


class WorkerAsyncLoop:
    """Run sync-worker coroutines on one process-owned background event loop."""

    def __init__(self, *, thread_name: str = "gewu-worker-asyncio") -> None:
        self._thread_name = thread_name
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._startup_error: BaseException | None = None

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def run[T](self, coroutine: Coroutine[Any, Any, T]) -> T:
        """Run one coroutine on the process-owned loop and return its result."""

        submission = _WorkerSubmission(coroutine)
        try:
            loop = self._ensure_started()
            loop.call_soon_threadsafe(submission.start, loop)
        except BaseException:
            coroutine.close()
            raise
        try:
            return submission.result()
        except BaseException:
            if not submission.done:
                submission.cancel(loop)
                try:
                    submission.result()
                except BaseException:
                    pass
            raise

    def shutdown(self) -> None:
        """Stop the loop after callers close resources owned by it."""

        with self._lock:
            loop = self._loop
            thread = self._thread
            if loop is None or thread is None:
                self._clear_state()
                return
            loop.call_soon_threadsafe(loop.stop)
        thread.join()
        with self._lock:
            self._clear_state()

    def reset_after_fork(self) -> None:
        """Discard thread state inherited by a newly forked worker child."""

        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._loop = None
        self._thread = None
        self._startup_error = None

    def _ensure_started(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is not None and self._thread is not None:
                if self._thread.is_alive():
                    return self._loop
                self._clear_state()
            self._ready.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._run_loop,
                name=self._thread_name,
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()
            if self._startup_error is not None:
                error = self._startup_error
                self._clear_state()
                raise RuntimeError("Unable to start the worker asyncio loop.") from error
            if self._loop is None:
                self._clear_state()
                raise RuntimeError("Worker asyncio loop did not initialize.")
            return self._loop

    def _run_loop(self) -> None:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self._loop = loop
        except BaseException as exc:
            self._startup_error = exc
            self._ready.set()
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            self._cancel_pending_tasks(loop)
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
            loop.close()

    @staticmethod
    def _cancel_pending_tasks(loop: asyncio.AbstractEventLoop) -> None:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    def _clear_state(self) -> None:
        self._loop = None
        self._thread = None
        self._startup_error = None
        self._ready.clear()
