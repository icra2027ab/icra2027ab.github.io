"""Tracker state registry: one TrackState per (query, view, instance)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrackState:
    query: str
    instance_id: int
    view_idx: int
    anchor_mask: np.ndarray
    last_mask: np.ndarray
    last_frame_idx: int = 0
    prompt_feats: Any | None = None


@dataclass
class TrackerStateRegistry:
    """Per-query lists of `TrackState` — multi-instance supported.

    Instance ids start at 1 because SAM3's video session rejects obj_id=0.
    """

    states: dict[str, list[TrackState]] = field(default_factory=dict)
    _next_instance_id: int = 1

    def issue_instance_id(self) -> int:
        issued = self._next_instance_id
        self._next_instance_id += 1
        return issued

    def register_external_id(self, instance_id: int) -> None:
        """Bump the issuance counter past an externally-injected id.

        Used by the cross-view fusion stage when a secondary view
        receives an aligned id from the main camera — without this,
        ``issue_instance_id`` could later hand out the same number.
        """
        if instance_id >= self._next_instance_id:
            self._next_instance_id = instance_id + 1

    def add(self, state: TrackState) -> None:
        self.states.setdefault(state.query, []).append(state)

    def drop(self, queries: list[str] | None = None) -> None:
        if queries is None:
            self.states.clear()
            return
        for q in queries:
            self.states.pop(q, None)

    def iter_all(self):
        for lst in self.states.values():
            yield from lst

    def __contains__(self, query: str) -> bool:
        return query in self.states and bool(self.states[query])
