"""Tests for pure functions in policy_abstraction.evaluation.retrieval."""

from __future__ import annotations

import numpy as np

from policy_abstraction.evaluation.retrieval import (
    _dtw_warp_to_ref,
    _fill_traj_gaps,
    _remap_pairs_for_demo,
    compute_multi_demo_stats,
)
from policy_abstraction.vlm.types import DistancePair, DistanceType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(label, from_entity="ee", to_entity=1, axis=None):
    return DistancePair(
        from_entity=from_entity,
        to_entity=to_entity,
        label=label,
        distance_type=DistanceType.CENTROID,
        axis=axis,
    )


# ---------------------------------------------------------------------------
# TestRemapPairsForDemo
# ---------------------------------------------------------------------------


class TestRemapPairsForDemo:
    """Tests for _remap_pairs_for_demo."""

    def test_basic_remapping(self):
        """Entity id 1 in source maps to entity id 5 in target by name."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        src_map = {"1": "mug"}
        tgt_map = {"5": "mug"}
        result = _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert len(result) == 1
        assert result[0].to_entity == 5

    def test_ee_entity_preserved(self):
        """'ee' entity is never remapped."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        src_map = {"1": "mug"}
        tgt_map = {"1": "mug"}
        result = _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert result[0].from_entity == "ee"

    def test_unmatchable_pair_dropped(self):
        """If entity name not in target map, pair is dropped."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        src_map = {"1": "mug"}
        tgt_map = {"5": "plate"}  # "mug" not in target
        result = _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert result == []

    def test_case_insensitive_fallback(self):
        """Name matching falls back to lowercase comparison."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        src_map = {"1": "mug"}
        tgt_map = {"7": "MUG"}  # uppercase
        result = _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert len(result) == 1
        assert result[0].to_entity == 7

    def test_empty_pairs_returns_empty(self):
        """Empty input pairs → empty output."""
        result = _remap_pairs_for_demo([], {"1": "mug"}, {"1": "mug"})
        assert result == []

    def test_identical_maps_passthrough(self):
        """When source and target maps are identical, entity IDs are unchanged."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=3)]
        seg_map = {"3": "bowl"}
        result = _remap_pairs_for_demo(pairs, seg_map, seg_map)
        assert len(result) == 1
        assert result[0].to_entity == 3

    def test_multiple_pairs_partial_match(self):
        """Matchable pairs are kept; unmatchable ones are dropped."""
        pairs = [
            _make_pair("p0", from_entity="ee", to_entity=1),
            _make_pair("p1", from_entity="ee", to_entity=2),
        ]
        src_map = {"1": "mug", "2": "bowl"}
        tgt_map = {"10": "mug"}  # only "mug" exists in target
        result = _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert len(result) == 1
        assert result[0].to_entity == 10

    def test_original_pairs_not_mutated(self):
        """The original pair list must not be modified (copy semantics)."""
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1)]
        src_map = {"1": "mug"}
        tgt_map = {"5": "mug"}
        _remap_pairs_for_demo(pairs, src_map, tgt_map)
        assert pairs[0].to_entity == 1  # unchanged


# ---------------------------------------------------------------------------
# TestDtwWarpToRef
# ---------------------------------------------------------------------------


class TestDtwWarpToRef:
    """Tests for _dtw_warp_to_ref."""

    def test_output_shape_same_length(self):
        """When ref and query have the same length, output has shape (T_ref, n_pairs)."""
        T, n = 10, 2
        ref = np.random.rand(T, n).astype(np.float32)
        qry = np.random.rand(T, n).astype(np.float32)
        result = _dtw_warp_to_ref(ref, qry)
        assert result.shape == (T, n)

    def test_output_shape_shorter_query(self):
        """Query shorter than ref → warped output still has T_ref rows."""
        ref = np.random.rand(20, 2).astype(np.float32)
        qry = np.random.rand(8, 2).astype(np.float32)
        result = _dtw_warp_to_ref(ref, qry)
        assert result.shape == (20, 2)

    def test_output_shape_longer_query(self):
        """Query longer than ref → warped output still has T_ref rows."""
        ref = np.random.rand(10, 2).astype(np.float32)
        qry = np.random.rand(30, 2).astype(np.float32)
        result = _dtw_warp_to_ref(ref, qry)
        assert result.shape == (10, 2)

    def test_identical_trajs_warps_to_same(self):
        """Identical ref and query → warped ≈ ref (up to DTW averaging)."""
        T, n = 15, 2
        traj = np.tile(np.linspace(1.0, 0.0, T).reshape(-1, 1), (1, n)).astype(
            np.float32
        )
        result = _dtw_warp_to_ref(traj, traj.copy())
        assert result.shape == (T, n)
        # Values should be close to the original (no divergence)
        valid = ~np.isnan(result)
        np.testing.assert_allclose(result[valid], traj[valid], atol=0.1)

    def test_output_dtype_is_float32(self):
        """Output dtype is always float32."""
        ref = np.random.rand(8, 2).astype(np.float64)
        qry = np.random.rand(8, 2).astype(np.float64)
        result = _dtw_warp_to_ref(ref, qry)
        assert result.dtype == np.float32

    def test_fallback_on_dtw_failure(self, monkeypatch):
        """When dtw_ndim.warping_path raises, fallback linear interp is used."""

        class _FakeDTWNdim:
            @staticmethod
            def warping_path(*args, **kwargs):
                raise RuntimeError("simulated DTW failure")

        # Patch the dtaidistance module inside the function's namespace
        import sys

        fake_module = type(sys)("dtaidistance")
        fake_module.dtw_ndim = _FakeDTWNdim()
        monkeypatch.setitem(sys.modules, "dtaidistance", fake_module)
        monkeypatch.setitem(sys.modules, "dtaidistance.dtw_ndim", _FakeDTWNdim())

        ref = np.linspace(1.0, 0.0, 10).reshape(-1, 1).astype(np.float32)
        qry = np.linspace(1.0, 0.0, 20).reshape(-1, 1).astype(np.float32)
        result = _dtw_warp_to_ref(ref, np.broadcast_to(qry, (20, 1)).copy())
        assert result.shape == (10, 1)


# ---------------------------------------------------------------------------
# TestComputeMultiDemoStats
# ---------------------------------------------------------------------------


class TestComputeMultiDemoStats:
    """Tests for compute_multi_demo_stats."""

    def test_single_demo_stats_equal_ref(self):
        """With no other_trajs, mean/std/min/max all equal ref_traj."""
        ref = np.linspace(1.0, 0.0, 20).reshape(-1, 1).astype(np.float32)
        ref = np.tile(ref, (1, 2))
        result = compute_multi_demo_stats(ref, [], ["p0", "p1"])
        np.testing.assert_allclose(result["mean"], ref, atol=1e-5)
        np.testing.assert_allclose(result["min"], ref, atol=1e-5)
        np.testing.assert_allclose(result["max"], ref, atol=1e-5)

    def test_output_keys_present(self):
        """Result contains all expected keys."""
        ref = np.ones((10, 2), dtype=np.float32)
        result = compute_multi_demo_stats(ref, [], ["p0", "p1"])
        for key in ("ref_traj", "mean", "std", "min", "max", "pair_labels", "n_demos"):
            assert key in result

    def test_n_demos_count(self):
        """n_demos = 1 (ref) + len(other_trajs) when all are valid."""
        T, n = 15, 2
        ref = np.ones((T, n), dtype=np.float32)
        others = [np.ones((T, n), dtype=np.float32) for _ in range(3)]
        result = compute_multi_demo_stats(ref, others, ["p0", "p1"])
        assert result["n_demos"] == 4

    def test_wrong_pair_count_skipped(self):
        """Traj with wrong n_pairs is skipped; n_demos not incremented for it."""
        T, n = 10, 2
        ref = np.ones((T, n), dtype=np.float32)
        wrong = np.ones((T, 3), dtype=np.float32)  # 3 pairs vs expected 2
        result = compute_multi_demo_stats(ref, [wrong], ["p0", "p1"])
        assert result["n_demos"] == 1  # only ref counted

    def test_none_traj_skipped(self):
        """None in other_trajs is silently skipped."""
        T, n = 10, 2
        ref = np.ones((T, n), dtype=np.float32)
        result = compute_multi_demo_stats(ref, [None, None], ["p0", "p1"])  # type: ignore
        assert result["n_demos"] == 1

    def test_output_shapes(self):
        """All stat arrays have shape (T_ref, n_pairs)."""
        T, n = 12, 3
        ref = np.ones((T, n), dtype=np.float32)
        other = np.ones((T, n), dtype=np.float32)
        result = compute_multi_demo_stats(ref, [other], [f"p{i}" for i in range(n)])
        for key in ("ref_traj", "mean", "std", "min", "max"):
            assert result[key].shape == (T, n), f"{key} shape mismatch"

    def test_pair_labels_preserved(self):
        """pair_labels in result matches the input."""
        ref = np.ones((5, 2), dtype=np.float32)
        labels = ["dist_ee_to_mug", "dist_ee_to_plate"]
        result = compute_multi_demo_stats(ref, [], labels)
        assert result["pair_labels"] == labels


# ---------------------------------------------------------------------------
# TestFillTrajGaps
# ---------------------------------------------------------------------------


class TestFillTrajGaps:
    """Tests for _fill_traj_gaps (sparse → dense reference trajectory)."""

    def test_gaps_between_measured_frames_carry_forward(self):
        """Depth on a stride measures t=0,2,4; the gaps hold the last value."""
        traj = np.array([[1.0], [np.nan], [2.0], [np.nan], [3.0]], dtype=np.float32)
        out = _fill_traj_gaps(traj, ["p0"])
        assert out.ravel().tolist() == [1.0, 1.0, 2.0, 2.0, 3.0]

    def test_leading_gap_back_fills_the_first_measurement(self):
        """A column first measured at t=2 must not stay NaN below it.

        The carry-forward reads row t-1, so an unseeded row 0 would poison
        every later row — the whole point of seeding first.
        """
        traj = np.array([[np.nan], [np.nan], [7.0], [np.nan]], dtype=np.float32)
        out = _fill_traj_gaps(traj, ["p0"])
        assert not np.isnan(out).any()
        assert out.ravel().tolist() == [7.0, 7.0, 7.0, 7.0]

    def test_columns_are_seeded_independently(self):
        """One pair measured late must not drag a pair measured at t=0."""
        traj = np.array(
            [[1.0, np.nan], [np.nan, np.nan], [np.nan, 5.0]], dtype=np.float32
        )
        out = _fill_traj_gaps(traj, ["a", "b"])
        assert out[:, 0].tolist() == [1.0, 1.0, 1.0]
        assert out[:, 1].tolist() == [5.0, 5.0, 5.0]

    def test_never_measured_column_is_zeroed_not_nan(self):
        """An unreconstructable pair yields zeros, so stats stay finite."""
        traj = np.array([[1.0, np.nan], [2.0, np.nan]], dtype=np.float32)
        out = _fill_traj_gaps(traj, ["a", "b"])
        assert not np.isnan(out).any()
        assert out[:, 1].tolist() == [0.0, 0.0]

    def test_fully_measured_trajectory_is_unchanged(self):
        traj = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        assert _fill_traj_gaps(traj.copy(), ["p0"]).ravel().tolist() == [1.0, 2.0, 3.0]

    def test_result_feeds_multi_demo_stats_without_nan_warnings(self):
        """The regression this fixes: all-NaN slices reaching nanmean."""
        traj = _fill_traj_gaps(
            np.array([[np.nan], [4.0], [np.nan]], dtype=np.float32), ["p0"]
        )
        with np.errstate(all="raise"):
            stats = compute_multi_demo_stats(traj, [], ["p0"])
        for key in ("mean", "std", "min", "max"):
            assert not np.isnan(stats[key]).any(), key
