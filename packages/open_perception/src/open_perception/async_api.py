"""AsyncOpenPerception: non-blocking wrapper around `OpenPerception`.

The core `OpenPerception.detect` / `track` calls block the caller until
PyTorch finishes on the GPU — 500 ms – 2 s for SAM3 segmentation and
100 – 500 ms for a SAM3 video step. Robotics control loops running at
30–60 Hz can't afford to block per-frame.

`AsyncOpenPerception` owns a `OpenPerception` instance plus a dedicated
worker thread. `submit_detect()` / `submit_track()` return immediately
with a monotonically increasing seq id; the worker processes submissions
and stores the latest finished `PerceptionOutput` in a lock-guarded
slot that `latest()` / `wait_next()` read.

Backpressure is **drop-oldest**: a new submit overwrites any submit the
worker has not yet picked up. If the worker is mid-inference when you
submit, your submission is queued and runs next. The number of dropped
submissions is exposed via `n_dropped`.

Ordering caveat — detect and track submissions share one input slot.
For the standard robotics pattern *detect once, track every frame*,
wait for the detect to finish before starting to submit tracks:

    with AsyncOpenPerception() as perc:
        perc.submit_detect(["mug"], frames[0])
        perc.wait_next()                              # blocks once
        for frame in frames[1:]:
            perc.submit_track(frame)                  # never blocks
            ...                                       # do policy work
            got = perc.latest()
            if got is not None:
                seq, out = got

Worker exceptions are captured and re-raised on the main thread on the
next `latest()` / `wait_next()` call so GPU failures don't silently kill
the worker.
"""

from __future__ import annotations

import itertools
import threading
import time
from typing import Any

import numpy as np
from loguru import logger

from .api import OpenPerception
from .config import PerceptionConfig
from .encoders.reference import ReferenceBank
from .types import CameraInfo, InferenceOptions, PerceptionOutput

_KIND_DETECT = "detect"
_KIND_TRACK = "track"
_KIND_STOP = "__stop__"


class AsyncOpenPerception:
    """Non-blocking wrapper around `OpenPerception`.

    Parameters mirror `OpenPerception.__init__` plus one test hook:

    perception:
        Optional pre-built `OpenPerception` to wrap. When set, `config`,
        `vlm_client`, and `reference` are ignored. Useful for tests that
        inject fakes without loading SAM3.
    """

    def __init__(
        self,
        config: PerceptionConfig | None = None,
        vlm_client: Any | None = None,
        reference: str | dict[str, Any] | ReferenceBank | None = None,
        *,
        perception: OpenPerception | None = None,
    ) -> None:
        self._perc = (
            perception
            if perception is not None
            else OpenPerception(
                config=config, vlm_client=vlm_client, reference=reference
            )
        )

        # Input: a single slot for pending (not-yet-started) work. Guarded
        # by _submit_lock; signalled by _has_work.
        self._pending: tuple[str, int, tuple, dict] | None = None
        self._submit_lock = threading.Lock()
        self._has_work = threading.Event()
        self._seq_gen = itertools.count()
        self._n_dropped = 0

        # Output: most-recent finished (seq, result) + deferred worker
        # exception. Guarded by _result_cond's underlying lock.
        self._result_cond = threading.Condition()
        self._latest_seq = -1
        self._latest: PerceptionOutput | None = None
        self._last_exc: BaseException | None = None

        # Worker lifecycle.
        self._busy = False
        self._n_processed = 0
        self._stop = threading.Event()
        self._closed = False
        self._worker = threading.Thread(
            target=self._run, name="AsyncOpenPerception-worker", daemon=True
        )
        self._worker.start()

    # ------------------------------------------------------------------
    # Public API

    @property
    def perception(self) -> OpenPerception:
        """The wrapped sync `OpenPerception`. Do not call from multiple threads."""
        return self._perc

    @property
    def busy(self) -> bool:
        """True if the worker is currently mid-inference."""
        return self._busy

    @property
    def pending_seq(self) -> int | None:
        """seq id of the queued-but-not-yet-started submission, else None."""
        with self._submit_lock:
            if self._pending is None:
                return None
            return self._pending[1]

    @property
    def n_dropped(self) -> int:
        """How many submissions were overwritten before the worker saw them."""
        return self._n_dropped

    @property
    def n_processed(self) -> int:
        """How many submissions the worker has finished."""
        return self._n_processed

    def submit_detect(
        self,
        queries: str | list[str],
        rgb: np.ndarray | list[np.ndarray],
        *,
        depth: np.ndarray | list[np.ndarray] | None = None,
        camera: CameraInfo | list[CameraInfo] | None = None,
        reference: str | dict[str, Any] | ReferenceBank | None = None,
        options: InferenceOptions | None = None,
    ) -> int:
        """Submit a detect call. Returns a monotonically increasing seq id."""
        return self._submit(
            _KIND_DETECT,
            (queries, rgb),
            {
                "depth": depth,
                "camera": camera,
                "reference": reference,
                "options": options,
            },
        )

    def submit_track(
        self,
        rgb: np.ndarray,
        *,
        depth: np.ndarray | None = None,
        camera: CameraInfo | None = None,
        options: InferenceOptions | None = None,
    ) -> int:
        """Submit a track call. Returns a monotonically increasing seq id."""
        return self._submit(
            _KIND_TRACK,
            (rgb,),
            {"depth": depth, "camera": camera, "options": options},
        )

    def latest(self) -> tuple[int, PerceptionOutput] | None:
        """Return the most recent finished `(seq, output)`, or None if none yet.

        Re-raises any exception that occurred in the worker since the last call.
        """
        with self._result_cond:
            self._maybe_raise_locked()
            if self._latest is None:
                return None
            return self._latest_seq, self._latest

    def wait_next(
        self, after_seq: int = -1, timeout: float | None = None
    ) -> tuple[int, PerceptionOutput] | None:
        """Block until a result with ``seq > after_seq`` is available.

        Returns None if `timeout` elapses or the wrapper is closed. Re-raises
        any worker exception.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._result_cond:
            while self._latest_seq <= after_seq and self._last_exc is None:
                if self._stop.is_set():
                    break
                remaining: float | None = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                self._result_cond.wait(timeout=remaining)
            self._maybe_raise_locked()
            if self._latest is None or self._latest_seq <= after_seq:
                return None
            return self._latest_seq, self._latest

    def close(self, timeout: float = 5.0) -> None:
        """Stop the worker and wait for it to exit."""
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        # Wake the worker even if it's parked on _has_work.
        with self._submit_lock:
            self._pending = (_KIND_STOP, -1, (), {})
            self._has_work.set()
        # Wake any wait_next callers.
        with self._result_cond:
            self._result_cond.notify_all()
        self._worker.join(timeout=timeout)

    def __enter__(self) -> AsyncOpenPerception:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internals

    def _submit(self, kind: str, args: tuple, kwargs: dict) -> int:
        if self._closed:
            raise RuntimeError("AsyncOpenPerception is closed")
        with self._submit_lock:
            seq = next(self._seq_gen)
            if self._pending is not None:
                # Caller is overwriting a submission that never ran.
                self._n_dropped += 1
            self._pending = (kind, seq, args, kwargs)
            self._has_work.set()
        return seq

    def _run(self) -> None:
        while not self._stop.is_set():
            self._has_work.wait()
            with self._submit_lock:
                item = self._pending
                self._pending = None
                self._has_work.clear()
            if item is None:
                continue
            kind, seq, args, kwargs = item
            if kind == _KIND_STOP:
                return
            self._busy = True
            try:
                if kind == _KIND_DETECT:
                    out = self._perc.detect(*args, **kwargs)
                elif kind == _KIND_TRACK:
                    out = self._perc.track(*args, **kwargs)
                else:
                    raise RuntimeError(f"unknown work kind: {kind!r}")
            except BaseException as exc:
                logger.exception(f"[async_perception] worker {kind} raised: {exc}")
                with self._result_cond:
                    self._last_exc = exc
                    self._result_cond.notify_all()
                self._busy = False
                continue
            self._busy = False
            self._n_processed += 1
            with self._result_cond:
                if seq > self._latest_seq:
                    self._latest = out
                    self._latest_seq = seq
                    self._result_cond.notify_all()

    def _maybe_raise_locked(self) -> None:
        """Re-raise and clear a stored worker exception. Assumes result lock held."""
        if self._last_exc is not None:
            exc = self._last_exc
            self._last_exc = None
            raise exc
