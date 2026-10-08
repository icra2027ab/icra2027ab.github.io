"""Trajectory segmentation utilities."""

from __future__ import annotations

import numpy as np


def compute_subtrajectories(
    ee_pos: np.ndarray,
    speed_threshold: float = 0.005,
    min_length: int = 20,
) -> list[tuple[int, int]]:
    """Segment a trajectory into active sub-trajectories by EE speed.

    Parameters
    ----------
    ee_pos:          ``[T, 3]`` end-effector world positions.
    speed_threshold: Minimum EE speed (m/step) to be considered active.
    min_length:      Minimum sub-trajectory length in timesteps.

    Returns
    -------
    List of ``(start, end)`` index pairs (end is exclusive).
    Falls back to ``[(0, T)]`` when no valid segments are found.
    """
    T = len(ee_pos)
    if T == 0:
        return [(0, 0)]

    diffs = np.diff(ee_pos, axis=0)
    speed = np.concatenate([[0.0], np.linalg.norm(diffs, axis=1)])
    speed_smooth = np.convolve(speed, np.ones(5) / 5.0, mode="same")
    is_active = speed_smooth >= speed_threshold

    # 1. Find raw active segments
    raw: list[tuple[int, int]] = []
    i = 0
    while i < T:
        while i < T and not is_active[i]:
            i += 1
        if i >= T:
            break
        start = i
        while i < T and is_active[i]:
            i += 1
        if i - start >= min_length:
            raw.append((start, i))

    if not raw:
        return [(0, T)]

    # 2. Fill gaps so every timestep is covered.
    #    - Extend first segment to start at 0.
    #    - Extend last segment to end at T.
    #    - Split each interior gap at its midpoint.
    segments: list[tuple[int, int]] = []
    for idx, (s, e) in enumerate(raw):
        if idx == 0:
            s = 0
        else:
            prev_end = raw[idx - 1][1]
            mid = (prev_end + s) // 2
            s = mid
            # also fix the previous segment's end
            segments[-1] = (segments[-1][0], mid)
        if idx == len(raw) - 1:
            e = T
        segments.append((s, e))

    return segments
