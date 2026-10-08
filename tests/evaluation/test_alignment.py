"""
Tests for policy_abstraction.core.alignment.

All tests use only synthetic data (no LIBERO / GPU / network required).
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.core.alignment import MonotonicAligner
from policy_abstraction.evaluation.alignment import (
    AlignmentMode,
    MonotonicTrajectoryAligner,
)


class TestMonotonicAligner:
    def test_init_stores_reference_length(self):
        """Constructor stores the reference_length attribute."""
        aligner = MonotonicAligner(reference_length=42)
        assert aligner.reference_length == 42

    def test_init_step_starts_at_zero(self):
        """Internal step counter initialises to 0."""
        aligner = MonotonicAligner(reference_length=10)
        assert aligner._step == 0

    def test_reset_returns_step_to_zero(self):
        """reset() sets _step back to 0."""
        aligner = MonotonicAligner(reference_length=20)
        aligner._step = 15
        aligner.reset()
        assert aligner._step == 0

    def test_step_raises_not_implemented(self):
        """step() raises NotImplementedError (stub pending alignment logic)."""
        aligner = MonotonicAligner(reference_length=10)
        with pytest.raises(NotImplementedError):
            aligner.step()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _linear_ref(
    T: int, n_pairs: int, start_val: float = 1.0, end_val: float = 0.0
) -> np.ndarray:
    """Create a reference trajectory where each pair decreases linearly from
    *start_val* to *end_val* over *T* frames."""
    col = np.linspace(start_val, end_val, T, dtype=np.float32)
    return np.tile(col[:, None], (1, n_pairs))


# ---------------------------------------------------------------------------
# Tests for MonotonicTrajectoryAligner – normalize_distances
# ---------------------------------------------------------------------------


class TestMonotonicTrajectoryAlignerNormalizeDistances:
    # ------------------------------------------------------------------
    # Normalization parameter construction
    # ------------------------------------------------------------------

    def test_norm_params_scale_and_offset_linear(self):
        """For a linear ref [1→0], scale=1 and offset=0 (end value)."""
        ref = _linear_ref(T=10, n_pairs=2, start_val=1.0, end_val=0.0)
        mta = MonotonicTrajectoryAligner(ref, ["a", "b"], normalize_distances=True)

        np.testing.assert_allclose(mta._norm_scales[0], [1.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(mta._norm_offsets[0], [0.0, 0.0], atol=1e-6)

    def test_norm_params_non_unit_range(self):
        """scale = ref_start - ref_end, offset = ref_end."""
        ref = _linear_ref(T=5, n_pairs=1, start_val=3.0, end_val=1.0)
        mta = MonotonicTrajectoryAligner(ref, ["d"], normalize_distances=True)

        np.testing.assert_allclose(mta._norm_scales[0], [2.0], atol=1e-6)
        np.testing.assert_allclose(mta._norm_offsets[0], [1.0], atol=1e-6)

    def test_norm_params_constant_column_falls_back(self):
        """A column with zero variation gets scale=1, offset=0 (no normalization)."""
        ref = np.ones((8, 2), dtype=np.float32)
        ref[:, 0] = np.linspace(2.0, 0.0, 8)  # varying
        ref[:, 1] = 5.0  # constant → scale=0 → fallback
        mta = MonotonicTrajectoryAligner(ref, ["a", "b"], normalize_distances=True)

        assert mta._norm_scales[0][0] == pytest.approx(2.0, abs=1e-6)  # varying pair
        assert mta._norm_scales[0][1] == pytest.approx(1.0, abs=1e-6)  # fallback
        assert mta._norm_offsets[0][1] == pytest.approx(0.0, abs=1e-6)  # fallback

    def test_norm_params_nan_boundary_falls_back(self):
        """A pair whose start or end value is NaN falls back to scale=1, offset=0."""
        ref = np.array([[np.nan, 1.0], [0.5, 0.5], [0.0, 0.0]], dtype=np.float32)
        mta = MonotonicTrajectoryAligner(ref, ["a", "b"], normalize_distances=True)

        # Pair 0: start=NaN → invalid → fallback
        assert mta._norm_scales[0][0] == pytest.approx(1.0, abs=1e-6)
        assert mta._norm_offsets[0][0] == pytest.approx(0.0, abs=1e-6)
        # Pair 1: start=1.0, end=0.0 → scale=1, offset=0 → normal
        assert mta._norm_scales[0][1] == pytest.approx(1.0, abs=1e-6)
        assert mta._norm_offsets[0][1] == pytest.approx(0.0, abs=1e-6)

    def test_norm_params_per_subtraj(self):
        """Each sub-trajectory gets its own independent scale / offset."""
        # Segment 0: frames 0-4, range 2→0
        # Segment 1: frames 5-9, range 5→3
        ref = np.zeros((10, 1), dtype=np.float32)
        ref[:5, 0] = np.linspace(2.0, 0.0, 5)
        ref[5:, 0] = np.linspace(5.0, 3.0, 5)

        mta = MonotonicTrajectoryAligner(
            ref,
            ["d"],
            subtrajectories=[(0, 5), (5, 10)],
            normalize_distances=True,
        )

        np.testing.assert_allclose(mta._norm_scales[0], [2.0], atol=1e-6)
        np.testing.assert_allclose(mta._norm_offsets[0], [0.0], atol=1e-6)
        np.testing.assert_allclose(mta._norm_scales[1], [2.0], atol=1e-6)
        np.testing.assert_allclose(mta._norm_offsets[1], [3.0], atol=1e-6)

    # ------------------------------------------------------------------
    # Alignment behaviour: normalized vs raw should agree on ideal input
    # ------------------------------------------------------------------

    def test_perfect_live_aligns_to_correct_frame_normalized(self):
        """When the live vector exactly matches a specific reference frame,
        the aligner picks that frame regardless of normalization (time weight disabled)."""
        T = 20
        ref = _linear_ref(T, n_pairs=1, start_val=1.0, end_val=0.0)
        labels = ["d"]

        for normalize in (True, False):
            mta = MonotonicTrajectoryAligner(
                ref,
                labels,
                window=None,
                normalize_distances=normalize,
                subtraj_time_weight=0.0,
            )
            # Feed reference frame 10 as live input from t=0
            target_t = 10
            live = ref[target_t].copy()
            t = mta.align(live)
            assert t == target_t, f"normalize={normalize}: expected {target_t}, got {t}"

    def test_normalized_invariant_to_scale(self):
        """Normalized alignment is scale-invariant: scaling all reference distances
        by a constant should not change which frame is selected."""
        T = 15
        ref_base = _linear_ref(T, n_pairs=1, start_val=1.0, end_val=0.0)
        ref_scaled = ref_base * 100.0  # same shape, different magnitude

        live_base = ref_base[7].copy()
        live_scaled = ref_scaled[7].copy()

        mta_base = MonotonicTrajectoryAligner(
            ref_base, ["d"], window=None, normalize_distances=True
        )
        mta_scaled = MonotonicTrajectoryAligner(
            ref_scaled, ["d"], window=None, normalize_distances=True
        )

        assert mta_base.align(live_base) == mta_scaled.align(live_scaled)

    def test_normalize_off_sensitive_to_scale(self):
        """Without normalization, a 100× scale difference produces a 10000× MSE
        difference, which can change the winner between two candidate frames."""
        # Construct a reference where frame 5 is the ideal match in normalized
        # space but frame 3 has a smaller *absolute* distance when ref is large.
        # live = [4.5]: in raw space closest frame is 3 (diff=0); normalized would
        # still pick 3. Use a case where raw and normalized disagree.
        # ref [10→0], live at 4.5 → raw closest is frame 3 (value=4.5, diff=0)
        # Normalised: scale=10, offset=0. Normalised ref=[1,0.8,0.6,0.45,0.4,0.3]
        # normalised live=0.45 → still frame 3. Use a crafted example instead.
        #
        # ref pair 0: 100→0 (large range), pair 1: 1→0 (small range)
        # live: pair 0=50 (mid), pair 1=0.9 (near start)
        # Without norm: pair 0 dominates → closest to mid = frame 5 wins
        # With norm:    both pairs equal weight → frame 1 (norm ~[0.5,0.9]) wins
        T = 11
        ref2 = np.zeros((T, 2), dtype=np.float32)
        ref2[:, 0] = np.linspace(100.0, 0.0, T)  # large range
        ref2[:, 1] = np.linspace(1.0, 0.0, T)  # small range
        live2 = np.array([50.0, 0.9], dtype=np.float32)

        mta_norm = MonotonicTrajectoryAligner(
            ref2, ["big", "small"], window=None, normalize_distances=True
        )
        mta_raw = MonotonicTrajectoryAligner(
            ref2, ["big", "small"], window=None, normalize_distances=False
        )

        t_norm = mta_norm.align(live2)
        t_raw = mta_raw.align(live2)

        # Normalized: pair 0 norm live=0.5 → best at t=5; pair 1 norm live=0.9 → best at t=1
        # MSE norm: frame 1: ((0.9-0.9)^2+(0.9-0.9)^2)/2... let's just verify they differ
        # The key assertion: normalization changes the selected frame
        assert t_norm != t_raw, (
            f"Expected normalized ({t_norm}) and raw ({t_raw}) to disagree "
            "on this scale-asymmetric input"
        )

    def test_normalized_alignment_with_subtrajectories(self):
        """With normalize_distances=True and two sub-trajectories, each segment
        is independently normalized and alignment stays within its segment."""
        # Segment 0: frames 0–4, distance 2→0
        # Segment 1: frames 5–9, distance 5→3
        ref = np.zeros((10, 1), dtype=np.float32)
        ref[:5, 0] = np.linspace(2.0, 0.0, 5)
        ref[5:, 0] = np.linspace(5.0, 3.0, 5)

        mta = MonotonicTrajectoryAligner(
            ref,
            ["d"],
            window=None,
            subtrajectories=[(0, 5), (5, 10)],
            subtraj_end_threshold=0,  # disable early-switch for clarity
            normalize_distances=True,
            subtraj_time_weight=0.0,  # disable time dimension to test distances only
            monotonic=True,  # mandatory switch requires monotonic mode
        )

        # Feed frame 2 from segment 0 → should align to frame 2
        t = mta.align(ref[2].copy())
        assert t == 2

        # Advance cursor to segment 1 boundary to trigger mandatory switch
        mta._aligned_t = 4
        mta._current_subtraj_idx = 0
        mta.align(ref[4].copy())  # lands at end of seg 0, triggers switch

        assert mta.current_subtraj_idx == 1
        # Now feed frame 7 (segment 1) – should stay in segment 1
        t2 = mta.align(ref[7].copy())
        assert 5 <= t2 <= 9


# ---------------------------------------------------------------------------
# Tests for MonotonicTrajectoryAligner – current_mse
# ---------------------------------------------------------------------------


class TestMonotonicTrajectoryAlignerCurrentMSE:
    """current_mse exposes the MSE at the last aligned reference frame."""

    def test_current_mse_initialises_to_inf(self):
        """Before any align() call, current_mse is inf."""
        ref = _linear_ref(T=10, n_pairs=2)
        mta = MonotonicTrajectoryAligner(ref, ["a", "b"], normalize_distances=False)
        assert mta.current_mse == float("inf")

    def test_current_mse_updates_after_align(self):
        """After align(), current_mse is finite and non-negative."""
        ref = _linear_ref(T=10, n_pairs=2)
        mta = MonotonicTrajectoryAligner(
            ref, ["a", "b"], normalize_distances=False, subtraj_time_weight=0.0
        )
        mta.align(ref[0].copy())
        assert np.isfinite(mta.current_mse)
        assert mta.current_mse >= 0.0

    def test_current_mse_zero_on_perfect_match(self):
        """MSE is zero when the live vector exactly matches the aligned reference frame."""
        ref = _linear_ref(T=10, n_pairs=2, start_val=1.0, end_val=0.0)
        mta = MonotonicTrajectoryAligner(
            ref,
            ["a", "b"],
            normalize_distances=False,
            subtraj_time_weight=0.0,
            window=None,
        )
        # Feed exact reference frame at t=3; should align there with MSE=0
        mta.align(ref[3].copy())
        assert mta.current_mse == pytest.approx(0.0, abs=1e-6)

    def test_current_mse_positive_on_mismatch(self):
        """MSE is positive when live values differ from any reference frame."""
        ref = _linear_ref(T=10, n_pairs=2, start_val=1.0, end_val=0.0)
        mta = MonotonicTrajectoryAligner(
            ref,
            ["a", "b"],
            normalize_distances=False,
            subtraj_time_weight=0.0,
            window=None,
        )
        # All-zero live vec vs. linearly decreasing reference → never perfect match
        mta.align(np.full(2, 99.0, dtype=np.float32))
        assert mta.current_mse > 0.0

    def test_current_mse_reset(self):
        """reset() brings current_mse back to inf."""
        ref = _linear_ref(T=10, n_pairs=2)
        mta = MonotonicTrajectoryAligner(
            ref, ["a", "b"], normalize_distances=False, subtraj_time_weight=0.0
        )
        mta.align(ref[0].copy())
        assert np.isfinite(mta.current_mse)
        mta.reset()
        assert mta.current_mse == float("inf")

    def test_current_mse_updates_with_subtrajectories(self):
        """current_mse is updated correctly when sub-trajectories are configured."""
        ref = _linear_ref(T=10, n_pairs=2, start_val=1.0, end_val=0.0)
        mta = MonotonicTrajectoryAligner(
            ref,
            ["a", "b"],
            normalize_distances=False,
            subtraj_time_weight=0.0,
            subtrajectories=[(0, 5), (5, 10)],
            subtraj_end_threshold=0,
            window=None,
        )
        # Perfect match at frame 2
        mta.align(ref[2].copy())
        assert np.isfinite(mta.current_mse)
        assert mta.current_mse == pytest.approx(0.0, abs=1e-6)


class TestPerFeatureGuidanceWeights:
    """The temporal softmax must stay finite however far a pair lags."""

    @staticmethod
    def _mta(temperature: float) -> MonotonicTrajectoryAligner:
        ref = _linear_ref(T=300, n_pairs=3, start_val=1.0, end_val=0.0)
        return MonotonicTrajectoryAligner(
            ref,
            ["a", "b", "c"],
            subtrajectories=[(0, 300)],
            alignment_mode=AlignmentMode.PER_FEATURE,
            temporal_factor_temperature=temperature,
        )

    def test_large_lag_over_small_temperature_stays_finite(self):
        # exp(lag / temperature) overflowed to inf for a pair 290 frames behind
        # at temperature 1, and inf / inf made every weight NaN. A NaN weight
        # is not "zero", so the pair was still sent to the policy and the
        # whole guided action chunk came back NaN.
        mta = self._mta(temperature=1.0)
        w = mta._compute_guidance_weights(np.array([0, 290, 290], dtype=np.int32))
        assert np.all(np.isfinite(w))
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        # The lagging pair takes essentially all of the weight.
        assert w[0] == pytest.approx(1.0, abs=1e-6)

    def test_matches_the_unshifted_softmax_when_it_does_not_overflow(self):
        mta = self._mta(temperature=100.0)
        t = np.array([10, 40, 25], dtype=np.int32)
        w = mta._compute_guidance_weights(t)
        lag = 40 - t.astype(np.float64) + 10
        expected = np.exp(lag / 100.0)
        expected /= expected.sum()
        np.testing.assert_allclose(w, expected, atol=1e-6)
