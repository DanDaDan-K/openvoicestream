"""Generic subprocess-worker IO multiplexer.

Extracted from ``app/backends/jetson/trt_edge_llm_tts.py`` (commit `64185fa`
cooperative cancel protocol; spec ``docs/specs/concurrency-capability-framework.md``
Section 5).

This is the framework-layer abstraction that demuxes a single JSON-line
subprocess (one stdin / one stdout) into N in-flight per-request streams,
keyed by ``request_id`` / ``id``. Used today by the TRT-Edge-LLM TTS
backend. Future ASR worker backends (spec P1+) will reuse this.

Public API (P1a, sync — unchanged from the original `_WorkerIO`):

    wio = WorkerIO(proc, concurrency)
    for event in wio.request(payload): ...   # generator, terminates on done/cancelled
    wio.cancel(request_id)                   # best-effort, any thread
    # No explicit close(): the reader thread is daemon and exits on stdout EOF.

P1b will migrate ``request()`` -> ``async send_request()`` per spec Section 5.
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
from typing import Iterator

logger = logging.getLogger(__name__)


class WorkerExitError(RuntimeError):
    """Raised when the worker subprocess dies while a request is in flight."""


class WorkerIO:
    """Per-worker stdin writer + stdout reader thread, multiplexing N in-flight requests.

    Replaces a coarse per-request lock (which would serialize full
    request→response cycles end-to-end) with:

      * a single ``_stdin_lock`` that only protects the single-line JSON write,
      * a daemon reader thread that demuxes stdout events to per-request
        ``queue.Queue`` instances keyed by ``request_id``/``id``,
      * a ``threading.Semaphore`` bounding in-flight requests to ``concurrency``.

    When the worker subprocess EOFs (crash / restart), the reader thread wakes
    every in-flight caller with a sentinel ``{"event": "_worker_exit"}`` so
    they raise ``WorkerExitError`` instead of hanging on ``q.get(timeout=...)``.

    A given ``WorkerIO`` instance is bound to ONE subprocess. To restart the
    worker, discard the old instance and create a new one (handled by the
    owning backend, e.g. ``_ensure_worker`` / ``_restart_worker``).
    """

    # Class-level temporary instrumentation for Part D disconnect-watcher
    # validation (spec docs/specs/tts-n2-throughput.md §3). Counts every
    # cancel() invocation across all WorkerIO instances since process
    # start. Surfaced via the /health endpoint and via debug log. Remove
    # once Part D is validated and stable.
    _cancel_count: int = 0
    _cancel_count_lock = threading.Lock()

    def __init__(self, proc: subprocess.Popen, concurrency: int):
        self._proc = proc
        self._stdin_lock = threading.Lock()
        self._inflight: dict[str, "queue.Queue"] = {}
        self._inflight_lock = threading.Lock()
        self._sem = threading.Semaphore(max(1, int(concurrency)))
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            name="worker-io-stdout",
            daemon=True,
        )
        self._reader_thread.start()

    def request(self, payload: dict) -> Iterator[dict]:
        """Send ``payload`` to the worker and yield response events until ``done``.

        Caller must include a unique ``id`` field in ``payload``. The generator
        terminates when an ``event=="done"`` or ``event=="cancelled"`` is
        received, or raises ``WorkerExitError`` if the worker dies mid-request.
        """
        self._sem.acquire()
        req_id = payload["id"]
        q: "queue.Queue" = queue.Queue()
        # CRITICAL ordering: insert the queue BEFORE writing stdin so the
        # reader thread can never observe an event for ``req_id`` before
        # the queue exists (would otherwise be dropped as "stale").
        with self._inflight_lock:
            self._inflight[req_id] = q
        try:
            assert self._proc.stdin is not None
            try:
                with self._stdin_lock:
                    self._proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    self._proc.stdin.flush()
            except Exception:
                with self._inflight_lock:
                    self._inflight.pop(req_id, None)
                raise
            while True:
                try:
                    event = q.get(timeout=60.0)
                except Exception:
                    raise
                if event.get("event") == "_worker_exit":
                    raise WorkerExitError("worker subprocess died mid-request")
                yield event
                if event.get("event") in ("done", "cancelled"):
                    return
        finally:
            with self._inflight_lock:
                self._inflight.pop(req_id, None)
            self._sem.release()

    def cancel(self, req_id: str) -> None:
        """Best-effort cancel for an in-flight request.

        Writes a cancel JSON to the worker's stdin. The worker will check
        its per-request atomic flag at the next chunk boundary and emit
        a ``{"event":"cancelled", ...}`` terminal event in lieu of ``done``.

        Safe to call from any thread. Safe to call after the request has
        naturally completed (worker silently drops unknown cancels).
        """
        with WorkerIO._cancel_count_lock:
            WorkerIO._cancel_count += 1
            count_snapshot = WorkerIO._cancel_count
        logger.info(
            "WorkerIO.cancel: req_id=%s total_cancel_count=%d",
            req_id,
            count_snapshot,
        )
        try:
            assert self._proc.stdin is not None
            with self._stdin_lock:
                self._proc.stdin.write(
                    json.dumps({"type": "cancel", "id": req_id}) + "\n"
                )
                self._proc.stdin.flush()
        except Exception:
            # Worker may have already exited; the reader-loop sentinel
            # will surface that via WorkerExitError on the next q.get().
            logger.debug(
                "cancel() write failed; worker may be exiting",
                exc_info=True,
            )

    def _reader_loop(self) -> None:
        """Drain worker stdout, dispatching events to per-request queues."""
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except Exception:
                    logger.debug("worker emitted non-JSON line: %r", line[:200])
                    continue
                rid = event.get("request_id") or event.get("id")
                with self._inflight_lock:
                    q = self._inflight.get(rid) if rid else None
                if q is not None:
                    q.put(event)
                # else: stale / unsolicited (e.g. spurious "ready" replays
                # or events after a restart). Drop silently.
        except Exception:
            logger.exception("worker stdout reader crashed")
        finally:
            # Worker died (or stdout closed). Wake every in-flight caller
            # with the sentinel so they raise WorkerExitError instead of
            # hanging on q.get(timeout=60).
            with self._inflight_lock:
                for q in self._inflight.values():
                    q.put({"event": "_worker_exit"})
                self._inflight.clear()


# Backwards-compat alias for callers still referencing the old private name.
# Will be removed once all internal references migrate to ``WorkerIO``.
_WorkerIO = WorkerIO
