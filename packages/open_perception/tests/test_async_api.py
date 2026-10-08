"""Tests for AsyncOpenPerception — mocked underlying OpenPerception, no GPU."""

from __future__ import annotations

import threading
import time
from typing import cast

import numpy as np
import pytest
from open_perception.api import OpenPerception
from open_perception.async_api import AsyncOpenPerception
from open_perception.types import InferenceStats, PerceptionOutput

_RGB = np.zeros((4, 4, 3), dtype=np.uint8)


def _empty_output() -> PerceptionOutput:
    return PerceptionOutput(detections={}, stats=InferenceStats())


class _GatedPerception:
    """Fake `OpenPerception` whose detect/track block on caller-controlled gates.

    Tests release a gate to let one call finish deterministically — avoids
    flaky sleep-based timing.
    """

    def __init__(self) -> None:
        self.detect_calls: list[tuple] = []
        self.track_calls: list[tuple] = []
        self.detect_gate = threading.Event()
        self.track_gate = threading.Event()
        self.raise_on_detect: Exception | None = None

    def detect(self, queries, rgb, **kwargs):
        self.detect_calls.append((queries, kwargs))
        assert self.detect_gate.wait(timeout=5.0), "detect gate never released"
        self.detect_gate.clear()
        if self.raise_on_detect is not None:
            raise self.raise_on_detect
        return _empty_output()

    def track(self, rgb, **kwargs):
        self.track_calls.append((id(rgb), kwargs))
        assert self.track_gate.wait(timeout=5.0), "track gate never released"
        self.track_gate.clear()
        return _empty_output()


def _wrap(fake: _GatedPerception) -> AsyncOpenPerception:
    """Cast the duck-typed fake to satisfy the static type of `perception=`."""
    return AsyncOpenPerception(perception=cast(OpenPerception, fake))


def test_submit_detect_returns_result():
    fake = _GatedPerception()
    with _wrap(fake) as perc:
        seq = perc.submit_detect(["mug"], _RGB)
        fake.detect_gate.set()
        got = perc.wait_next(timeout=2.0)
        assert got is not None
        seq_out, out = got
        assert seq_out == seq
        assert isinstance(out, PerceptionOutput)
        assert len(fake.detect_calls) == 1
        assert perc.n_processed == 1
        assert perc.n_dropped == 0


def test_submit_is_non_blocking():
    """`submit_*` must return even while the worker is blocked inside detect."""
    fake = _GatedPerception()
    with _wrap(fake) as perc:
        t0 = time.perf_counter()
        perc.submit_detect(["mug"], _RGB)  # worker will block on the gate
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert elapsed_ms < 100, f"submit_detect blocked for {elapsed_ms:.0f} ms"
        fake.detect_gate.set()
        perc.wait_next(timeout=2.0)


def test_drop_oldest_replaces_pending():
    """A submission overwritten before the worker picks it up is counted as dropped."""
    fake = _GatedPerception()
    with _wrap(fake) as perc:
        # First submission: worker will pick this up and block on detect_gate.
        seq1 = perc.submit_detect(["first"], _RGB)
        # Give the worker a moment to take seq1 out of the pending slot.
        _poll_until(lambda: perc.busy, timeout=1.0)

        # Now submit two more. The worker is busy, so both sit in the pending
        # slot — the second must evict the first.
        perc.submit_track(_RGB)  # this one will be overwritten
        seq3 = perc.submit_track(_RGB)
        assert perc.pending_seq == seq3
        assert perc.n_dropped == 1  # prior track submission was dropped by seq3

        # Unblock detect; worker finishes seq1, then picks up seq3 (skips seq2).
        fake.detect_gate.set()
        fake.track_gate.set()
        final = perc.wait_next(after_seq=seq1, timeout=2.0)
        assert final is not None
        assert final[0] == seq3
        assert len(fake.track_calls) == 1  # only seq3 ran
        assert perc.n_processed == 2


def test_worker_exception_surfaces_on_next_read():
    fake = _GatedPerception()
    fake.raise_on_detect = RuntimeError("boom")
    with _wrap(fake) as perc:
        perc.submit_detect(["q"], _RGB)
        fake.detect_gate.set()
        _poll_until(lambda: fake.raise_on_detect is None, timeout=1.0)
        # The fake's one-shot exception has fired; give the worker a beat to
        # store it before we read.
        _poll_until(lambda: perc._last_exc is not None, timeout=1.0)
        with pytest.raises(RuntimeError, match="boom"):
            perc.wait_next(timeout=1.0)
        # Exception is cleared after being raised once.
        assert perc.latest() is None


def test_close_joins_cleanly_and_rejects_further_submits():
    fake = _GatedPerception()
    perc = _wrap(fake)
    try:
        perc.close(timeout=2.0)
        assert not perc._worker.is_alive()
        with pytest.raises(RuntimeError, match="closed"):
            perc.submit_track(_RGB)
    finally:
        # Idempotent close.
        perc.close()


def test_wait_next_timeout_returns_none():
    fake = _GatedPerception()
    with _wrap(fake) as perc:
        assert perc.wait_next(timeout=0.05) is None
        # No work submitted, no state produced.
        assert perc.latest() is None


def test_latest_reflects_most_recent_seq():
    fake = _GatedPerception()
    with _wrap(fake) as perc:
        seq1 = perc.submit_detect(["a"], _RGB)
        fake.detect_gate.set()
        got1 = perc.wait_next(after_seq=-1, timeout=2.0)
        assert got1 is not None and got1[0] == seq1

        seq2 = perc.submit_track(_RGB)
        fake.track_gate.set()
        got2 = perc.wait_next(after_seq=seq1, timeout=2.0)
        assert got2 is not None and got2[0] == seq2

        latest = perc.latest()
        assert latest is not None and latest[0] == seq2


def _poll_until(pred, timeout: float, interval: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False
