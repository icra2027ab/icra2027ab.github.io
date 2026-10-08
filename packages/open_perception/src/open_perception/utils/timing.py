"""Simple stage-timing context manager."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Iterator

from ..types import InferenceStats


@contextmanager
def stage_timer(stats: InferenceStats, name: str) -> Iterator[None]:
    t0 = time.perf_counter()
    try:
        yield
    finally:
        stats.add_timing(name, (time.perf_counter() - t0) * 1000.0)
