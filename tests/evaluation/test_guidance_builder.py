"""Tests for policy_abstraction.evaluation.guidance_builder."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from policy_abstraction.dynamics.integration import IntegrationDynamicsModel
from policy_abstraction.evaluation.abstraction import RolloutAbstractor
from policy_abstraction.evaluation.guidance_builder import (
    GP_KEYS,
    GP_SENTINEL_KEY,
    build_distance_guidance_params,
    merge_guidance_params,
)
from policy_abstraction.features.centroid import CentroidDistance
from policy_abstraction.utils.gripper import (
    compute_finger_tip_positions,
    gripper_action_to_metric,
)
from policy_abstraction.vlm.types import DistancePair, DistanceType
from policy_steering.robot_config import load_gripper_geometry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(
    label: str,
    from_entity: int | str = "ee",
    to_entity: int | str = 1,
    dist_type: DistanceType = DistanceType.CENTROID,
    axis: list[float] | None = None,
    plane: list[float] | None = None,
) -> DistancePair:
    return DistancePair(
        from_entity=from_entity,
        to_entity=to_entity,
        label=label,
        distance_type=dist_type,
        axis=axis,
        plane=plane,
    )


def _ref_traj(T: int = 20, n_pairs: int = 2) -> np.ndarray:
    """Linearly decreasing (T, n_pairs) reference trajectory."""
    cols = [np.linspace(float(i + 1), 0.0, T, dtype=np.float32) for i in range(n_pairs)]
    return np.stack(cols, axis=1)


def _base_kwargs(
    pairs: list[DistancePair] | None = None,
    live_distances: dict | None = None,
    entity_centroids: dict | None = None,
    ref_traj: np.ndarray | None = None,
    pair_labels: list[str] | None = None,
    aligned_ts: np.ndarray | None = None,
    chunk_size: int = 8,
) -> dict:
    if pairs is None:
        pairs = [_make_pair("p0")]
    if live_distances is None:
        live_distances = {"p0": 1.0}
    if entity_centroids is None:
        entity_centroids = {
            "ee": np.array([0.0, 0.0, 1.0]),
            1: np.array([1.0, 0.0, 1.0]),
        }
    if ref_traj is None:
        ref_traj = _ref_traj(T=20, n_pairs=len(pairs))
    if pair_labels is None:
        pair_labels = [p.label for p in pairs]
    if aligned_ts is None:
        aligned_ts = np.zeros(len(pairs), dtype=np.int32)
    return dict(
        pairs=pairs,
        live_distances=live_distances,
        entity_centroids=entity_centroids,
        object_pointclouds=None,
        ref_traj=ref_traj,
        pair_labels=pair_labels,
        aligned_ts=aligned_ts,
        chunk_size=chunk_size,
    )


# ---------------------------------------------------------------------------
# TestBuildDistanceGuidanceParams
# ---------------------------------------------------------------------------


class TestBuildDistanceGuidanceParams:
    """Tests for build_distance_guidance_params."""

    def test_returns_none_when_ee_missing(self):
        """No 'ee' in entity_centroids → None."""
        kwargs = _base_kwargs()
        kwargs["entity_centroids"] = {1: np.array([1.0, 0.0, 1.0])}
        result = build_distance_guidance_params(**kwargs)
        assert result is None

    def test_returns_none_when_no_ee_pairs(self):
        """Pairs with no EE entity → None."""
        pairs = [_make_pair("p0", from_entity=1, to_entity=2)]
        live_distances = {"p0": 1.0}
        entity_centroids = {"ee": np.zeros(3), 1: np.ones(3), 2: np.ones(3) * 2}
        ref = _ref_traj(20, 1)
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances=live_distances,
            entity_centroids=entity_centroids,
            object_pointclouds=None,
            ref_traj=ref,
            pair_labels=["p0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is None

    def test_sentinel_key_present(self):
        """Result contains the GP sentinel key."""
        result = build_distance_guidance_params(**_base_kwargs())
        assert result is not None
        assert GP_SENTINEL_KEY in result

    def test_all_required_keys_present(self):
        """Core distance guidance keys are present in the output dict.

        Listed positively rather than derived from ``GP_KEYS``: that tuple is
        the server-side strip-out list, and most of what it holds is optional
        (defaulted by the guidance fns, or injected later by the episode
        runner), so subtracting an exclusion list rots on every new key.
        """
        result = build_distance_guidance_params(**_base_kwargs())
        assert result is not None
        core_keys = (
            "__gp_ee_pos_0__",
            "__gp_ee_ori_0__",
            "__gp_gripper_0__",
            "__gp_dt__",
            "__gp_guidance_factor__",
            "__gp_guidance_weights__",
            "__gp_expected_dists__",
            "__gp_expected_stds__",
            "__gp_dist_types__",
            "__gp_source_type__",
            "__gp_waypoints__",
            "__gp_has_axis__",
            "__gp_axes__",
            "__gp_has_plane__",
            "__gp_plane_normals__",
            "__gp_surface_counts__",
            "__gp_surface_pts__",
            "__gp_ee_vectors__",
        )
        for key in core_keys:
            assert key in result, f"Missing key: {key}"
        unknown = [k for k in result if k not in GP_KEYS]
        assert not unknown, f"Emitted keys the server would not strip: {unknown}"

    def test_expected_dists_shape(self):
        """__gp_expected_dists__ has shape (N, chunk_size)."""
        pairs = [_make_pair("p0"), _make_pair("p1", to_entity=2)]
        live = {"p0": 1.0, "p1": 2.0}
        centroids = {"ee": np.zeros(3), 1: np.ones(3), 2: np.ones(3) * 2}
        ref = _ref_traj(20, 2)
        kwargs = _base_kwargs(
            pairs=pairs,
            live_distances=live,
            entity_centroids=centroids,
            ref_traj=ref,
            pair_labels=["p0", "p1"],
            chunk_size=10,
        )
        result = build_distance_guidance_params(**kwargs)
        assert result is not None
        assert result["__gp_expected_dists__"].shape == (2, 10)

    def test_ref_traj_window_padded_at_end(self):
        """When aligned_t puts the window near the end, padding with last value fills chunk_size."""
        ref = _ref_traj(T=5, n_pairs=1)  # only 5 timesteps
        pairs = [_make_pair("p0")]
        live = {"p0": 1.0}
        centroids = {"ee": np.zeros(3), 1: np.ones(3)}
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances=live,
            entity_centroids=centroids,
            object_pointclouds=None,
            ref_traj=ref,
            pair_labels=["p0"],
            aligned_ts=np.array(
                [3], dtype=np.int32
            ),  # window [3,5] = 2 steps, chunk_size=8 → pad 6
            chunk_size=8,
        )
        assert result is not None
        dists = result["__gp_expected_dists__"]
        assert dists.shape == (1, 8)
        # Last two real values — rest should be padded with ref[4, 0]
        last_real = float(ref[4, 0])
        np.testing.assert_allclose(dists[0, 2:], last_real, atol=1e-5)

    def test_waypoint_equals_other_entity_centroid(self):
        """__gp_waypoints__ contains the centroid of the non-EE entity."""
        waypoint = np.array([3.0, 4.0, 5.0])
        centroids = {"ee": np.zeros(3), 1: waypoint}
        result = build_distance_guidance_params(
            **_base_kwargs(entity_centroids=centroids)
        )
        assert result is not None
        np.testing.assert_allclose(result["__gp_waypoints__"][0], waypoint, atol=1e-6)

    def test_axis_sign_ee_is_from(self):
        """When ee is from_entity, stored axis = +axis."""
        axis = [0.0, 0.0, 1.0]
        pairs = [_make_pair("p0", from_entity="ee", to_entity=1, axis=axis)]
        kwargs = _base_kwargs(pairs=pairs, live_distances={"p0": 1.0})
        result = build_distance_guidance_params(**kwargs)
        assert result is not None
        stored = result["__gp_axes__"][0]
        np.testing.assert_allclose(stored, [0, 0, 1], atol=1e-6)
        assert result["__gp_has_axis__"][0] == 1

    def test_axis_sign_ee_is_to(self):
        """When ee is to_entity, stored axis = -axis."""
        axis = [0.0, 0.0, 1.0]
        pairs = [_make_pair("p0", from_entity=1, to_entity="ee", axis=axis)]
        live = {"p0": 1.0}
        centroids = {"ee": np.zeros(3), 1: np.ones(3)}
        ref = _ref_traj(20, 1)
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances=live,
            entity_centroids=centroids,
            object_pointclouds=None,
            ref_traj=ref,
            pair_labels=["p0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is not None
        stored = result["__gp_axes__"][0]
        np.testing.assert_allclose(stored, [0, 0, -1], atol=1e-6)

    def test_centroid_pair_dist_type_is_zero(self):
        """CENTROID pair (no axis/plane) → __gp_dist_types__[0] == 0."""
        result = build_distance_guidance_params(**_base_kwargs())
        assert result is not None
        assert result["__gp_dist_types__"][0] == 0

    def test_surface_pair_dist_type_is_one(self):
        """SURFACE pair (no axis/plane) → __gp_dist_types__[0] == 1."""
        pairs = [_make_pair("p0", dist_type=DistanceType.SURFACE)]
        surf_pcd = np.random.randn(50, 7).astype(np.float32)
        surf_pcd[:, 6] = 1.0
        centroids = {"ee": np.zeros(3), 1: np.ones(3)}
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"p0": 1.0},
            entity_centroids=centroids,
            object_pointclouds={"1": surf_pcd},
            ref_traj=_ref_traj(20, 1),
            pair_labels=["p0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
            int_name_map={1: "1"},
        )
        assert result is not None
        assert result["__gp_dist_types__"][0] == 1

    def test_unobservable_pair_filtered(self):
        """Pair with NaN live distance is excluded; result is None if it's the only pair."""
        kwargs = _base_kwargs()
        kwargs["live_distances"] = {"p0": float("nan")}
        result = build_distance_guidance_params(**kwargs)
        assert result is None

    def test_none_live_distance_filtered(self):
        """Pair with None live distance is excluded."""
        kwargs = _base_kwargs()
        kwargs["live_distances"] = {}  # "p0" not present
        result = build_distance_guidance_params(**kwargs)
        assert result is None

    def test_pair_with_non_finite_weight_is_dropped(self):
        # A NaN weight reaches every action channel of the chunk through the
        # loss; the pair is dropped and the finite ones survive.
        pairs = [_make_pair("p0"), _make_pair("p1")]
        kw = _base_kwargs(pairs=pairs, live_distances={"p0": 1.0, "p1": 1.0})
        out = build_distance_guidance_params(
            **kw, guidance_weights=np.array([np.nan, 1.0], dtype=np.float32)
        )
        assert out is not None
        assert out["__gp_expected_dists__"].shape[0] == 1
        np.testing.assert_array_equal(out["__gp_guidance_weights__"], [1.0])

    def test_pair_with_non_finite_waypoint_is_dropped(self):
        kw = _base_kwargs(
            entity_centroids={
                "ee": np.array([0.0, 0.0, 1.0]),
                1: np.array([np.nan, 0.0, 1.0]),
            }
        )
        assert build_distance_guidance_params(**kw) is None

    def test_non_finite_ee_pos_disables_guidance(self):
        kw = _base_kwargs(
            entity_centroids={
                "ee": np.array([np.inf, 0.0, 1.0]),
                1: np.array([1.0, 0.0, 1.0]),
            }
        )
        assert build_distance_guidance_params(**kw) is None

    def test_non_finite_ee_ori_disables_guidance(self):
        kw = _base_kwargs()
        out = build_distance_guidance_params(
            **kw, ee_ori_0=np.array([np.nan, 0.0, 0.0], dtype=np.float32)
        )
        assert out is None

    def test_scalar_guidance_factor_stored(self):
        """Float guidance_factor=2.5 → __gp_guidance_factor__ == 2.5."""
        result = build_distance_guidance_params(**_base_kwargs(), guidance_factor=2.5)
        assert result is not None
        assert float(result["__gp_guidance_factor__"]) == pytest.approx(2.5)

    def test_guidance_factor_default_is_one(self):
        """Default guidance_factor is 1.0."""
        result = build_distance_guidance_params(**_base_kwargs())
        assert result is not None
        assert float(result["__gp_guidance_factor__"]) == pytest.approx(1.0)

    def test_surface_pts_padded_to_max(self):
        """Surface pts padded to (N, max_surface_pts, 3)."""
        pairs = [_make_pair("p0", dist_type=DistanceType.SURFACE)]
        surf_pcd = np.random.randn(50, 7).astype(np.float32)
        centroids = {"ee": np.zeros(3), 1: np.ones(3)}
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"p0": 1.0},
            entity_centroids=centroids,
            object_pointclouds={"1": surf_pcd},
            ref_traj=_ref_traj(20, 1),
            pair_labels=["p0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
            int_name_map={1: "1"},
            max_surface_pts=128,
        )
        assert result is not None
        assert result["__gp_surface_pts__"].shape == (1, 128, 3)

    def test_plane_normal_stored(self):
        """Plane pair → __gp_has_plane__[0] == 1 and __gp_plane_normals__[0] is normalised."""
        plane = [0.0, 0.0, 1.0]
        pairs = [_make_pair("p0", plane=plane)]
        result = build_distance_guidance_params(**_base_kwargs(pairs=pairs))
        assert result is not None
        assert result["__gp_has_plane__"][0] == 1
        stored_n = result["__gp_plane_normals__"][0]
        np.testing.assert_allclose(np.linalg.norm(stored_n), 1.0, atol=1e-6)

    def test_orientation_pair_with_axis(self):
        """ORIENTATION pair with axis → axes stored unsigned, has_axis=1."""
        axis = [0.0, 0.0, 1.0]
        pairs = [
            _make_pair(
                "o0",
                from_entity="ee",
                to_entity="ee",
                dist_type=DistanceType.ORIENTATION,
                axis=axis,
            )
        ]
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"o0": 0.5},
            entity_centroids={"ee": np.zeros(3)},
            object_pointclouds=None,
            ref_traj=_ref_traj(20, 1),
            pair_labels=["o0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is not None
        np.testing.assert_allclose(result["__gp_axes__"][0], [0, 0, 1], atol=1e-6)
        assert result["__gp_has_axis__"][0] == 1
        assert result["__gp_dist_types__"][0] == 2  # ORIENTATION

    def test_orientation_pair_with_plane(self):
        """ORIENTATION pair with plane (no axis) → plane normal populates
        ``plane_normals``, with ``has_plane=1`` / ``has_axis=0``.

        The JAX-side orientation dispatch uses ``has_plane`` to switch from
        the signed-cosine projection onto ``axis`` to the in-plane magnitude
        of the gripper z-axis.
        """
        pairs = [
            _make_pair(
                "o0",
                from_entity="ee",
                to_entity="ee",
                dist_type=DistanceType.ORIENTATION,
                axis=None,
                plane=[0.0, 0.0, 2.0],  # unnormalised
            )
        ]
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"o0": 0.5},
            entity_centroids={"ee": np.zeros(3)},
            object_pointclouds=None,
            ref_traj=_ref_traj(20, 1),
            pair_labels=["o0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is not None
        np.testing.assert_allclose(
            result["__gp_plane_normals__"][0], [0, 0, 1], atol=1e-6
        )
        assert result["__gp_has_plane__"][0] == 1
        assert result["__gp_has_axis__"][0] == 0
        assert result["__gp_dist_types__"][0] == 2

    def test_orientation_pair_ee_vector_stored(self):
        """ORIENTATION pair with ee_vector=[1,0,0] → __gp_ee_vectors__[0] == [1,0,0]."""
        pairs = [
            DistancePair(
                from_entity="ee",
                to_entity="ee",
                label="o0",
                distance_type=DistanceType.ORIENTATION,
                axis=[0, 0, 1],
                ee_vector=[1, 0, 0],
            )
        ]
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"o0": 0.5},
            entity_centroids={"ee": np.zeros(3)},
            object_pointclouds=None,
            ref_traj=_ref_traj(20, 1),
            pair_labels=["o0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is not None
        np.testing.assert_allclose(result["__gp_ee_vectors__"][0], [1, 0, 0], atol=1e-6)

    def test_orientation_pair_default_ee_vector(self):
        """ORIENTATION pair with no ee_vector → __gp_ee_vectors__[0] == [0,0,1] (z-axis)."""
        pairs = [
            _make_pair(
                "o0",
                from_entity="ee",
                to_entity="ee",
                dist_type=DistanceType.ORIENTATION,
                axis=[0, 0, 1],
            )
        ]
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"o0": 0.5},
            entity_centroids={"ee": np.zeros(3)},
            object_pointclouds=None,
            ref_traj=_ref_traj(20, 1),
            pair_labels=["o0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is not None
        np.testing.assert_allclose(result["__gp_ee_vectors__"][0], [0, 0, 1], atol=1e-6)

    def test_orientation_pair_without_direction_skipped(self):
        """ORIENTATION pair with no axis or plane is skipped (returns None when sole pair)."""
        pairs = [
            _make_pair(
                "o0",
                from_entity="ee",
                to_entity="ee",
                dist_type=DistanceType.ORIENTATION,
                axis=None,
                plane=None,
            )
        ]
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances={"o0": 0.5},
            entity_centroids={"ee": np.zeros(3)},
            object_pointclouds=None,
            ref_traj=_ref_traj(20, 1),
            pair_labels=["o0"],
            aligned_ts=np.zeros(1, dtype=np.int32),
            chunk_size=8,
        )
        assert result is None


# ---------------------------------------------------------------------------
# TestMergeGuidanceParams
# ---------------------------------------------------------------------------


class TestMergeGuidanceParams:
    """Tests for merge_guidance_params."""

    def _build_simple(self, n_pairs: int = 1, chunk_size: int = 8) -> dict:
        pairs = [_make_pair(f"p{i}", to_entity=i + 1) for i in range(n_pairs)]
        live = {f"p{i}": float(i + 1) for i in range(n_pairs)}
        centroids: dict[int | str, np.ndarray] = {"ee": np.zeros(3)}
        for i in range(n_pairs):
            centroids[i + 1] = np.ones(3) * (i + 1)
        ref = _ref_traj(20, n_pairs)
        result = build_distance_guidance_params(
            pairs=pairs,
            live_distances=live,
            entity_centroids=centroids,
            object_pointclouds=None,
            ref_traj=ref,
            pair_labels=[f"p{i}" for i in range(n_pairs)],
            aligned_ts=np.zeros(n_pairs, dtype=np.int32),
            chunk_size=chunk_size,
        )
        assert result is not None
        return result

    def test_none_none_returns_none(self):
        """merge(None, None) → None."""
        assert merge_guidance_params(None, None) is None

    def test_none_a_returns_b(self):
        """merge(None, b) → b."""
        b = self._build_simple(1)
        assert merge_guidance_params(None, b) is b

    def test_none_b_returns_a(self):
        """merge(a, None) → a."""
        a = self._build_simple(1)
        assert merge_guidance_params(a, None) is a

    def test_concatenated_pair_count(self):
        """merge(2-pair, 3-pair) → 5 pairs in __gp_expected_dists__."""
        a = self._build_simple(2)
        b = self._build_simple(3)
        merged = merge_guidance_params(a, b)
        assert merged is not None
        assert merged["__gp_expected_dists__"].shape[0] == 5

    def test_scalar_ee_pos_taken_from_a(self):
        """__gp_ee_pos_0__ in merged comes from a."""
        a = self._build_simple(1)
        b = self._build_simple(1)
        a["__gp_ee_pos_0__"] = np.array([9.0, 8.0, 7.0], dtype=np.float32)
        merged = merge_guidance_params(a, b)
        assert merged is not None
        np.testing.assert_array_equal(merged["__gp_ee_pos_0__"], [9.0, 8.0, 7.0])

    def test_surface_pts_s_dim_broadcast(self):
        """surface_pts S-dim is broadcast to max(S_a, S_b)."""
        a = self._build_simple(1)  # no surface pts → S_max = 256
        b = self._build_simple(1)
        # Manually shrink a's surface_pts to S=64 to test broadcast
        a["__gp_surface_pts__"] = np.zeros((1, 64, 3), dtype=np.float32)
        b["__gp_surface_pts__"] = np.zeros((1, 256, 3), dtype=np.float32)
        merged = merge_guidance_params(a, b)
        assert merged is not None
        assert merged["__gp_surface_pts__"].shape == (2, 256, 3)

    def test_chunk_size_mismatch_padded(self):
        """When chunk sizes differ, shorter expected_dists is padded with last value."""
        a = self._build_simple(1, chunk_size=4)
        b = self._build_simple(1, chunk_size=8)
        merged = merge_guidance_params(a, b)
        assert merged is not None
        assert merged["__gp_expected_dists__"].shape[1] == 8


# ---------------------------------------------------------------------------
# TestDynamicsRefChunkFingerFK
# ---------------------------------------------------------------------------


class TestDynamicsRefChunkFingerFK:
    """RolloutAbstractor.compute_dynamics_ref_chunk must derive the reference
    finger-tip positions from FK applied to the integrated reference state —
    not from the pre-stored per-frame centroids — so that the JAX-side
    guidance (which FKs from the integrated live state) sees a geometrically
    symmetric reference.
    """

    def _make_minimal_abstractor(
        self,
        *,
        ref_ee_pos: np.ndarray,
        ref_ee_ori: np.ndarray,
        ref_gripper: np.ndarray,
        ref_actions: np.ndarray,
        obj_centroids: np.ndarray,
        pair: DistancePair,
    ) -> RolloutAbstractor:
        """Build a minimally-initialised RolloutAbstractor for ref-chunk testing.

        Bypasses ``__init__`` and sets only the attributes that
        ``compute_dynamics_ref_chunk`` reads.
        """
        abstractor = RolloutAbstractor.__new__(RolloutAbstractor)
        abstractor._feature = CentroidDistance()
        abstractor._dynamics_model = IntegrationDynamicsModel()
        # Live-robot finger geometry; this test asserts the Panda numbers.
        abstractor._gripper_geometry = load_gripper_geometry("franka_panda")
        abstractor._analysis = SimpleNamespace(distance_pairs=[pair])
        abstractor._pair_labels = [pair.label]
        abstractor._ref_ee_pos = ref_ee_pos.astype(np.float32)
        abstractor._ref_ee_ori = ref_ee_ori.astype(np.float32)
        abstractor._ref_gripper = ref_gripper.astype(np.float32)
        abstractor._ref_actions = ref_actions.astype(np.float32)
        # Seed ref_traj with stand-in values so any surface-column fallback or
        # leading-NaN fill path uses a valid number.  ``float(pair_idx + 1)``
        # is large enough that the FK-derived result, if it ever lands here,
        # will not match numerically — making accidental fallback detectable.
        T_ref = ref_ee_pos.shape[0]
        abstractor._ref_traj = np.full((T_ref, 1), 99.0, dtype=np.float32)
        # Object is the other entity in the pair; stored as per-frame centroid.
        other = (
            pair.to_entity
            if pair.from_entity in ("gripper_l", "gripper_r")
            else pair.from_entity
        )
        abstractor._ref_entity_centroids = {
            "ee": ref_ee_pos.astype(np.float32),
            "gripper_l": np.full_like(ref_ee_pos, np.nan, dtype=np.float32),
            "gripper_r": np.full_like(ref_ee_pos, np.nan, dtype=np.float32),
            other: obj_centroids.astype(np.float32),
        }
        return abstractor

    def test_finger_ref_uses_fk_of_integrated_state(self):
        """Ref chunk for a ``gripper_l → obj`` pair should equal
        ``||FK(ee_dyn, ori_dyn, qpos(metric(grip_action))) - obj||``."""
        T_ref = 12
        chunk_size = 6
        dt = 1.0 / 50

        # Construct a smooth reference trajectory with known velocities.
        # Integration model: states[0] = initial_state; states[t] = states[t-1] + actions[t]*dt.
        ref_ee_pos = np.zeros((T_ref, 3), dtype=np.float32)
        ref_ee_pos[:, 0] = np.linspace(0.0, 0.1, T_ref)
        ref_ee_ori = np.zeros((T_ref, 3), dtype=np.float32)
        ref_gripper_scalar = np.linspace(-0.5, 0.5, T_ref).astype(
            np.float32
        )  # policy-space
        ref_gripper = ref_gripper_scalar[:, None]  # [T_ref, 1]

        # Ref actions that match the reference trajectory under the
        # IntegrationDynamicsModel (velocity = delta_pos / dt for EE, absolute
        # for gripper).  Size is (T_ref - 1, action_dim=7).
        ref_actions = np.zeros((T_ref - 1, 7), dtype=np.float32)
        ref_actions[:, 0] = (ref_ee_pos[1:, 0] - ref_ee_pos[:-1, 0]) / dt
        ref_actions[:, 6] = ref_gripper_scalar[1:]  # absolute gripper action

        obj_centroids = np.tile(np.array([0.5, 0.3, 0.1], dtype=np.float32), (T_ref, 1))

        pair = _make_pair(
            "gl_to_obj",
            from_entity="gripper_l",
            to_entity=5,
            dist_type=DistanceType.CENTROID,
        )

        abstractor = self._make_minimal_abstractor(
            ref_ee_pos=ref_ee_pos,
            ref_ee_ori=ref_ee_ori,
            ref_gripper=ref_gripper,
            ref_actions=ref_actions,
            obj_centroids=obj_centroids,
            pair=pair,
        )

        aligned_ts = np.array([0], dtype=np.int32)
        result = abstractor.compute_dynamics_ref_chunk(aligned_ts, chunk_size)
        assert result.shape == (chunk_size, 1)

        # Recompute the expected values the same way the method should.
        initial_state = np.concatenate(
            [ref_ee_pos[0], ref_ee_ori[0], ref_gripper[0]]
        ).astype(np.float32)
        actions_padded = np.concatenate(
            [np.zeros((1, 7), dtype=np.float32), ref_actions[:chunk_size]], axis=0
        )
        states = abstractor._dynamics_model.predict_states(
            actions_padded, initial_state, dt=dt
        )
        # Row 0 is the achieved reference state, gripper channel included —
        # not whatever the model makes of the pad action.
        states[0] = initial_state
        ee_dyn = states[:, :3].astype(np.float32)
        ori_dyn = states[:, 3:6].astype(np.float32)
        grip_metric = gripper_action_to_metric(states[:, 6]).astype(np.float32)
        q1 = grip_metric / 2.0
        q2 = -grip_metric / 2.0
        qpos = np.stack([q1, q2], axis=-1)
        left_dyn, _ = compute_finger_tip_positions(ee_dyn, ori_dyn, qpos)
        expected = np.linalg.norm(left_dyn - obj_centroids[0], axis=-1).astype(
            np.float32
        )  # [n_integrated]

        # compute_dynamics_ref_chunk returns chunk_size rows (indexed 0..chunk_size-1)
        # using _k_idx = min(k, len(ee_dyn)-1); n_integrated = chunk_size+1 here.
        np.testing.assert_allclose(result[:, 0], expected[:chunk_size], atol=1e-5)
        # Sanity: the stand-in 99.0 fallback values must not have leaked in.
        assert np.all(result[:, 0] < 2.0), (
            "Ref chunk should be FK-derived, not the 99.0 sentinel "
            "(which would mean the surface/NaN fallback was triggered)."
        )

    def test_row0_is_achieved_state_under_joint_kinematics(self):
        """With absolute joint targets the pad action must not become a pose.

        ``Ur5KinematicsDynamicsModel`` maps every action to a pose, so a zero
        pad action would put row 0 of the reference chunk at the fully
        extended arm (~0.8 m from the tool) instead of the achieved state.
        The guidance then read the first action of every chunk as metres
        off target and spiked it. Row 0 must come from the achieved state.
        """
        from policy_abstraction.dynamics.ur5_kinematics import (
            Ur5KinematicsDynamicsModel,
            ur5e_forward_kinematics,
        )

        T_ref = 12
        chunk_size = 6
        home = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, 0.0])
        # A slow elbow motion; joint targets are absolute, one per action.
        joints = np.tile(home, (T_ref, 1))
        joints[:, 2] += np.linspace(0.0, 0.05, T_ref)
        grip = 0.3
        ref_ee_pos = np.zeros((T_ref, 3), dtype=np.float32)
        ref_ee_ori = np.zeros((T_ref, 3), dtype=np.float32)
        for t in range(T_ref):
            pos, rotvec = ur5e_forward_kinematics(joints[t])
            ref_ee_pos[t] = pos
            ref_ee_ori[t] = rotvec
        ref_gripper = np.full((T_ref, 1), grip, dtype=np.float32)
        ref_actions = np.concatenate(
            [joints[1:], np.full((T_ref - 1, 1), grip)], axis=1
        ).astype(np.float32)
        obj_centroids = np.tile(ref_ee_pos[0] + [0.05, 0.0, -0.1], (T_ref, 1))

        pair = _make_pair(
            "gl_to_obj",
            from_entity="gripper_l",
            to_entity=5,
            dist_type=DistanceType.CENTROID,
        )
        abstractor = self._make_minimal_abstractor(
            ref_ee_pos=ref_ee_pos,
            ref_ee_ori=ref_ee_ori,
            ref_gripper=ref_gripper,
            ref_actions=ref_actions,
            obj_centroids=obj_centroids,
            pair=pair,
        )
        abstractor._dynamics_model = Ur5KinematicsDynamicsModel()

        result = abstractor.compute_dynamics_ref_chunk(
            np.array([0], dtype=np.int32), chunk_size
        )

        grip_metric = gripper_action_to_metric(np.array([grip])).astype(np.float32)
        qpos = np.stack([grip_metric / 2.0, -grip_metric / 2.0], axis=-1)
        left0, _ = compute_finger_tip_positions(ref_ee_pos[:1], ref_ee_ori[:1], qpos)
        expected0 = float(np.linalg.norm(left0[0] - obj_centroids[0]))
        assert result[0, 0] == pytest.approx(expected0, abs=1e-5)
        # The chunk is continuous: the achieved state and the state after the
        # first action are a few millimetres apart, not most of a metre.
        assert abs(result[0, 0] - result[1, 0]) < 0.02
