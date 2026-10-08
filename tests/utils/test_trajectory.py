"""Tests for policy_abstraction.utils.trajectory."""

from __future__ import annotations

import numpy as np

from policy_abstraction.utils.trajectory import compute_subtrajectories

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _static(T: int) -> np.ndarray:
    """Return an all-zero (T, 3) trajectory — no motion anywhere."""
    return np.zeros((T, 3), dtype=np.float64)


def _linear(T: int, step: float = 0.1) -> np.ndarray:
    """Return a (T, 3) trajectory advancing along x by *step* per frame."""
    pos = np.zeros((T, 3), dtype=np.float64)
    pos[:, 0] = np.arange(T) * step
    return pos


def _burst(T: int, start: int, length: int, step: float = 0.1) -> np.ndarray:
    """Static trajectory with a single motion burst of *length* frames starting at *start*."""
    pos = np.zeros((T, 3), dtype=np.float64)
    for i in range(length):
        pos[start + i :, 0] += step
    return pos


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComputeSubtrajectories:
    """Tests for compute_subtrajectories."""

    def test_empty_trajectory_returns_zero(self):
        """T=0 should return [(0, 0)] without crashing."""
        result = compute_subtrajectories(np.empty((0, 3)))
        assert result == [(0, 0)]

    def test_single_frame_fallback(self):
        """T=1 has no diffs — speed is always 0, falls back to [(0, 1)]."""
        result = compute_subtrajectories(np.zeros((1, 3)))
        assert result == [(0, 1)]

    def test_static_trajectory_falls_back(self):
        """All-zero trajectory → no active frames → fallback [(0, T)]."""
        T = 50
        result = compute_subtrajectories(_static(T))
        assert result == [(0, T)]

    def test_full_linear_trajectory_is_single_segment(self):
        """Trajectory active throughout → one segment spanning [0, T)."""
        T = 40
        result = compute_subtrajectories(_linear(T, step=0.1), min_length=5)
        assert len(result) == 1
        start, end = result[0]
        assert start >= 0
        assert end == T

    def test_burst_below_min_length_filtered(self):
        """A burst shorter than min_length is dropped; fallback [(0, T)] returned."""
        T = 60
        # 10-frame burst, min_length=20 → should be dropped
        pos = _burst(T, start=20, length=10, step=0.1)
        result = compute_subtrajectories(pos, min_length=20)
        assert result == [(0, T)]

    def test_burst_above_min_length_kept(self):
        """A burst longer than min_length is kept as a segment."""
        T = 80
        pos = _burst(T, start=10, length=30, step=0.1)
        result = compute_subtrajectories(pos, min_length=20)
        assert len(result) == 1
        start, end = result[0]
        assert end - start >= 20

    def test_two_separate_bursts_detected(self):
        """Two non-overlapping motion bursts should produce two segments."""
        T = 200
        pos = np.zeros((T, 3), dtype=np.float64)
        # Burst 1: frames 10–49 (40 frames)
        for i in range(10, 50):
            pos[i:, 0] += 0.05
        # Burst 2: frames 110–159 (50 frames)
        for i in range(110, 160):
            pos[i:, 1] += 0.05

        result = compute_subtrajectories(pos, speed_threshold=0.001, min_length=30)
        assert len(result) == 2

    def test_lower_threshold_captures_slow_motion(self):
        """Lowering speed_threshold should detect slower motion."""
        T = 60
        pos = _linear(T, step=0.0005)  # very slow

        # At default threshold (0.002) this is inactive
        default_result = compute_subtrajectories(pos, speed_threshold=0.002)
        assert default_result == [(0, T)]

        # At lower threshold it should be active
        low_result = compute_subtrajectories(pos, speed_threshold=0.0001, min_length=5)
        assert len(low_result) == 1

    def test_higher_threshold_misses_slow_motion(self):
        """Raising speed_threshold above actual speed → static → fallback."""
        T = 40
        pos = _linear(T, step=0.001)
        result = compute_subtrajectories(pos, speed_threshold=0.5)
        assert result == [(0, T)]

    def test_result_is_list_of_tuples(self):
        """Return type is always a list of (int, int) tuples."""
        result = compute_subtrajectories(_static(10))
        assert isinstance(result, list)
        for seg in result:
            assert isinstance(seg, tuple) and len(seg) == 2
