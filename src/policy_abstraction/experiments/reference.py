"""The reference experiment's staged scenes, for the operator to match.

Two arms of a comparison run the same plan, and the rollouts are comparable
cell by cell only if the scene was staged the same way both times. The
overlay draws where each object's *centre* should go; it says nothing about
which way the cable was coiled, or how the banana was turned. The first
recorded frame of the reference's rollout for the same condition does — so
the window ghosts it under the live view while the operator stages, and the
two pictures converge as the objects are nudged into place.

The recording carries the frame in the convention the window displays (the
recorder applies the camera's image transform before writing), so it can be
blended pixel for pixel with the live frame from the same camera.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import Callable, Optional

import numpy as np
from loguru import logger

from policy_abstraction.experiments.outcomes import RolloutRecord
from policy_abstraction.experiments.plan import PlannedRollout
from policy_abstraction.experiments.store import ExperimentStore

GHOST_LEVELS: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)
"""How much of the reference shows through, cycled by the ``G`` hotkey: off,
a faint 30% (the default — enough to see the object outlines without hiding
the live scene), a stronger 60%, the reference alone."""


@dataclasses.dataclass(frozen=True)
class ReferenceFrame:
    """One staged scene of the reference experiment."""

    image: np.ndarray
    """``[H, W, 3]`` uint8 RGB, in the displayed convention."""
    index: int
    """The reference rollout the frame came from — its plan index."""
    timestep: int
    """Which step of the recording. 0 unless the first frames lacked camera
    parameters."""
    assignments: dict[str, str]


def _load_first_rgb(episode: pathlib.Path) -> tuple[np.ndarray, int]:
    from policy_abstraction.experiments.measure import first_frame

    frame = first_frame(episode, require_depth=False)
    return frame.rgb, frame.timestep


class ReferenceFrames:
    """Lazily reads the reference experiment's first frames, one per condition.

    A condition is matched on its factor assignments and repeat, falling back
    to the assignments alone — so a plan edited from the GUI still finds the
    reference's scene for a condition the reference staged, and a condition
    it never scored simply has no ghost.
    """

    def __init__(
        self,
        store: ExperimentStore,
        *,
        loader: Callable[[pathlib.Path], tuple[np.ndarray, int]] = _load_first_rgb,
    ) -> None:
        self.store = store
        self._loader = loader
        self._results: Optional[list[RolloutRecord]] = None
        self._cache: dict[int, Optional[ReferenceFrame]] = {}

    @property
    def name(self) -> str:
        return self.store.root.name

    def _record_for(self, planned: PlannedRollout) -> Optional[RolloutRecord]:
        if self._results is None:
            self._results = self.store.results()
        exact = [
            r
            for r in self._results
            if r.assignments == planned.assignments and r.episode_path
        ]
        if not exact:
            return None
        for r in exact:
            if r.index == planned.index:
                return r
        return exact[0]

    def frame_for(self, planned: Optional[PlannedRollout]) -> Optional[ReferenceFrame]:
        """The reference's staged scene for *planned*, or ``None``."""
        if planned is None:
            return None
        record = self._record_for(planned)
        if record is None:
            return None
        if record.index in self._cache:
            return self._cache[record.index]
        frame: Optional[ReferenceFrame] = None
        episode = self.store.root / str(record.episode_path)
        try:
            rgb, timestep = self._loader(episode)
            frame = ReferenceFrame(
                image=np.ascontiguousarray(rgb, dtype=np.uint8),
                index=record.index,
                timestep=timestep,
                assignments=dict(record.assignments),
            )
        except Exception as exc:
            logger.warning(
                f"No reference frame for rollout {record.index} of {self.name}: "
                f"{episode.name}: {exc}"
            )
        self._cache[record.index] = frame
        return frame


def ghost(live: np.ndarray, reference: np.ndarray, alpha: float) -> np.ndarray:
    """*reference* blended over *live* at *alpha* (0 = live only, 1 = reference).

    The reference is resized to the live frame if the two differ — a camera
    reconfigured between the arms — so the overlay's projection, which is for
    the live frame, still lands.
    """
    import cv2

    alpha = float(min(1.0, max(0.0, alpha)))
    if alpha <= 0.0:
        return live
    if reference.shape[:2] != live.shape[:2]:
        h, w = live.shape[:2]
        reference = cv2.resize(reference, (w, h), interpolation=cv2.INTER_AREA)
    if alpha >= 1.0:
        return np.ascontiguousarray(reference)
    return cv2.addWeighted(reference, alpha, live, 1.0 - alpha, 0)


__all__ = ["GHOST_LEVELS", "ReferenceFrame", "ReferenceFrames", "ghost"]
