"""The source-agnostic DemoEpisode contract and its adapters."""

from __future__ import annotations

import json

import numpy as np
import pytest

from policy_abstraction.analysis.demo_episode import (
    DemoEpisode,
    LiberoHDF5Episode,
    VlaEpisode,
    build_demo_episode,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

H = W = 8
T = 6
DEPTH_STRIDE = 2


def _sidecar() -> dict:
    return {
        "task_description": "put all cables in the bag",
        "env_spec": {
            "cameras": [
                {"name": "front", "is_primary": True},
                {"name": "wrist", "is_primary": False},
            ],
            "robot": {
                "state_layout": {
                    "fields": [
                        {"op": "pos", "obs_key": "eef_pos"},
                        {"op": "quat_to_axisangle", "obs_key": "eef_quat"},
                        {"op": "gripper_aperture", "obs_key": "gripper_qpos"},
                    ]
                }
            },
        },
    }


@pytest.fixture
def vla_dir(tmp_path):
    """A .vla with RGB every step and depth on a stride, plus camera params."""
    import robodm

    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
    state = np.tile(np.arange(7, dtype=np.float32), (T, 1))
    state[:, 0] = np.arange(T)  # a moving x so ee_positions is checkable
    K = np.array([[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]])
    E = np.eye(4)

    traj = robodm.Trajectory(str(tmp_path / "episode_000.vla"), mode="w")
    for t in range(T):
        traj.add("observation/front_image", np.ascontiguousarray(rgb[t]))
        traj.add("observation/state", state[t])
        traj.add(
            "camera_params",
            np.array([{"front": {"intrinsic": K, "extrinsic": E}}], dtype=object),
        )
        if t % DEPTH_STRIDE == 0:
            # 1000 mm + t, so each depth frame is identifiable.
            traj.add(
                "observation/front_depth",
                np.full((H, W), 1000 + t, dtype=np.uint16),
            )
            traj.add("observation/depth_timestep", np.array([t], dtype=np.int32))
    traj.close()
    (tmp_path / "episode_000.json").write_text(json.dumps(_sidecar()))
    return tmp_path, rgb


@pytest.fixture
def libero_file(tmp_path):
    """A minimal LIBERO-schema HDF5 demo."""
    import h5py

    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 255, (T, H, W, 3), dtype=np.uint8)
    seg = np.zeros((T, H, W, 1), dtype=np.int32)
    seg[:, 2:5, 2:5, 0] = 7
    depth = np.full((T, H, W, 1), 0.5, dtype=np.float32)  # normalised buffer
    ee = np.tile(np.array([0.1, 0.2, 0.3], dtype=np.float32), (T, 1))
    K = np.tile(np.eye(3)[None], (T, 1, 1))
    E = np.tile(np.eye(4)[None], (T, 1, 1))

    path = tmp_path / "task_demo.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["segmentation_name_mapping"] = json.dumps({"7": "bowl"})
        obs = data.create_group("demo_0").create_group("obs")
        obs.create_dataset("agentview_rgb", data=rgb)
        obs.create_dataset("agentview_segmentation", data=seg)
        obs.create_dataset("agentview_depth", data=depth)
        obs.create_dataset("agentview_intrinsic", data=K)
        obs.create_dataset("agentview_extrinsic", data=E)
        obs.create_dataset("ee_pos", data=ee)
    return path, rgb


# ---------------------------------------------------------------------------
# .vla adapter
# ---------------------------------------------------------------------------


class TestVlaEpisode:
    def test_satisfies_the_protocol(self, vla_dir):
        directory, _ = vla_dir
        assert isinstance(VlaEpisode(str(directory), "episode_000"), DemoEpisode)

    def test_length_and_rgb_come_from_the_primary_camera(self, vla_dir):
        directory, rgb = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        assert len(ep) == T
        assert ep.rgb(0).shape == (H, W, 3)

    def test_depth_only_on_recorded_steps(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        assert ep.frames_with_depth() == [0, 2, 4]
        assert ep.depth_m(1) is None

    def test_depth_converts_millimetres_to_metres(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        np.testing.assert_allclose(ep.depth_m(2), 1.002, rtol=0, atol=1e-6)

    def test_zero_depth_is_no_return_not_zero_range(self, tmp_path):
        import robodm

        traj = robodm.Trajectory(str(tmp_path / "e.vla"), mode="w")
        traj.add("observation/front_image", np.zeros((H, W, 3), dtype=np.uint8))
        traj.add("observation/front_depth", np.zeros((H, W), dtype=np.uint16))
        traj.add("observation/depth_timestep", np.array([0], dtype=np.int32))
        traj.close()
        (tmp_path / "e.json").write_text(json.dumps(_sidecar()))

        depth = VlaEpisode(str(tmp_path), "e").depth_m(0)
        assert np.isnan(depth).all()

    def test_camera_matrices_come_from_the_recording(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        assert ep.intrinsic(0).shape == (3, 3)
        assert ep.extrinsic(0).shape == (4, 4)
        assert ep.intrinsic(0)[0, 0] == 100.0

    def test_ee_positions_read_through_the_state_layout(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        ee = ep.ee_positions()
        assert ee.shape == (T, 3)
        # The layout puts position first, so column 0 is the moving x.
        np.testing.assert_array_equal(ee[:, 0], np.arange(T, dtype=np.float32))

    def test_no_segmentation_without_a_segmenter(self, vla_dir):
        """A real recording labels none of its own pixels; say so."""
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        assert ep.segmentation(0) is None
        assert ep.seg_name_map() == {}

    def test_segmenter_supplies_labels_and_names(self, vla_dir):
        directory, _ = vla_dir
        calls: list[int] = []

        def segmenter(rgb, depth, K, E, task):
            calls.append(1)
            labels = np.zeros(rgb.shape[:2], dtype=np.int32)
            labels[1:3, 1:3] = 4
            return labels, {4: "bag"}

        ep = VlaEpisode(str(directory), "episode_000", segmenter=segmenter)
        labels = ep.segmentation(0)
        assert labels is not None and set(np.unique(labels)) == {0, 4}
        assert ep.seg_name_map() == {4: "bag"}
        # Cached: a second read must not re-run a tracking backend.
        ep.segmentation(0)
        assert len(calls) == 1

    def test_task_comes_from_the_sidecar(self, vla_dir):
        directory, _ = vla_dir
        assert VlaEpisode(str(directory), "episode_000").task == (
            "put all cables in the bag"
        )


# ---------------------------------------------------------------------------
# LIBERO adapter
# ---------------------------------------------------------------------------


class TestLiberoHDF5Episode:
    def test_satisfies_the_protocol(self, libero_file):
        path, _ = libero_file
        assert isinstance(LiberoHDF5Episode(str(path), "demo_0"), DemoEpisode)

    def test_frames_are_returned_upright(self, libero_file):
        """LIBERO stores frames 180°-rotated; the adapter owns the correction."""
        path, rgb = libero_file
        ep = LiberoHDF5Episode(str(path), "demo_0")
        np.testing.assert_array_equal(ep.rgb(0), rgb[0][::-1, ::-1, :])

    def test_normalised_depth_becomes_metres(self, libero_file):
        """The stored buffer is normalised; the adapter inverts MuJoCo's mapping."""
        from policy_abstraction.envs.libero.config import (
            LIBERO_DEPTH_FAR,
            LIBERO_DEPTH_NEAR,
        )

        path, _ = libero_file
        depth = LiberoHDF5Episode(str(path), "demo_0").depth_m(0)
        expected = LIBERO_DEPTH_NEAR / (
            1.0 - 0.5 * (1.0 - LIBERO_DEPTH_NEAR / LIBERO_DEPTH_FAR)
        )
        assert depth.shape == (H, W)
        np.testing.assert_allclose(depth, expected, rtol=1e-5)

    def test_segmentation_is_squeezed_to_2d(self, libero_file):
        path, _ = libero_file
        seg = LiberoHDF5Episode(str(path), "demo_0").segmentation(0)
        assert seg.shape == (H, W)
        assert set(np.unique(seg)) == {0, 7}

    def test_seg_name_map_read_from_the_file_attribute(self, libero_file):
        path, _ = libero_file
        assert LiberoHDF5Episode(str(path), "demo_0").seg_name_map() == {7: "bowl"}

    def test_every_frame_has_depth(self, libero_file):
        """A simulator renders depth whenever it renders colour."""
        path, _ = libero_file
        assert LiberoHDF5Episode(str(path), "demo_0").frames_with_depth() == list(
            range(T)
        )

    def test_ee_positions(self, libero_file):
        path, _ = libero_file
        assert LiberoHDF5Episode(str(path), "demo_0").ee_positions().shape == (T, 3)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestBuildDemoEpisode:
    def test_dispatches_on_source_kind(self, vla_dir, libero_file):
        directory, _ = vla_dir
        path, _ = libero_file
        assert isinstance(
            build_demo_episode(
                {
                    "source_kind": "vla",
                    "source_path": str(directory),
                    "demo_key": "episode_000",
                }
            ),
            VlaEpisode,
        )
        assert isinstance(
            build_demo_episode(
                {
                    "source_kind": "libero",
                    "source_path": str(path),
                    "demo_key": "demo_0",
                    "task": "t",
                }
            ),
            LiberoHDF5Episode,
        )

    def test_unknown_kind_names_the_extension_point(self):
        with pytest.raises(ValueError, match="demo_episode"):
            build_demo_episode(
                {"source_kind": "calvin", "source_path": "/x", "demo_key": "d"}
            )


# ---------------------------------------------------------------------------
# Calibration sanity check
# ---------------------------------------------------------------------------


class TestExtrinsicDisagreement:
    """A wrong extrinsic yields confident, wrong distances rather than an error,
    so this check is the thing standing between it and a ruined experiment."""

    class _Fake:
        """An episode whose camera sits 1 m up looking straight down."""

        source_kind = "test"
        task = "t"

        def __init__(self, depth_value=1.0, extrinsic=None, ee=(0.0, 0.0, 0.0)):
            self._depth = np.full((H, W), depth_value, dtype=np.float32)
            self._K = np.array(
                [[100.0, 0, W / 2], [0, 100.0, H / 2], [0, 0, 1]], dtype=np.float64
            )
            # Camera at z=1 in base frame, its optical axis along -z (looking
            # down), so a point at the origin is 1 m in front of it.
            self._E = (
                extrinsic
                if extrinsic is not None
                else np.array(
                    [
                        [1.0, 0, 0, 0],
                        [0, -1.0, 0, 0],
                        [0, 0, -1.0, 1.0],
                        [0, 0, 0, 1.0],
                    ]
                )
            )
            self._ee = np.array([ee], dtype=np.float32)

        def __len__(self):
            return 1

        def rgb(self, t):
            return np.zeros((H, W, 3), dtype=np.uint8)

        def depth_m(self, t):
            return self._depth

        def segmentation(self, t):
            return None

        def intrinsic(self, t):
            return self._K

        def extrinsic(self, t):
            return self._E

        def ee_positions(self):
            return self._ee

        def seg_name_map(self):
            return {}

        def frames_with_depth(self):
            return [0]

    def test_correct_calibration_reports_near_zero(self):
        from policy_abstraction.analysis.demo_episode import extrinsic_disagreement

        error = extrinsic_disagreement(self._Fake(depth_value=1.0))
        assert error is not None and error < 0.01

    def test_depth_offset_from_the_calibration_is_measured(self):
        from policy_abstraction.analysis.demo_episode import extrinsic_disagreement

        # Camera says the EE is 1.0 m away; the depth image says 1.4 m.
        error = extrinsic_disagreement(self._Fake(depth_value=1.4))
        assert error == pytest.approx(0.4, abs=1e-3)

    def test_end_effector_never_in_view_is_infinite(self):
        """A workspace camera that never sees the gripper flags the extrinsic."""
        from policy_abstraction.analysis.demo_episode import extrinsic_disagreement

        # Same pose, but the EE is 50 m to the side: it projects off-image.
        assert extrinsic_disagreement(self._Fake(ee=(50.0, 0.0, 0.0))) == float("inf")

    def test_no_depth_cannot_be_checked(self):
        from policy_abstraction.analysis.demo_episode import extrinsic_disagreement

        episode = self._Fake()
        episode.frames_with_depth = lambda: []
        assert extrinsic_disagreement(episode) is None


# ---------------------------------------------------------------------------
# Robot state beyond position
# ---------------------------------------------------------------------------


class TestRobotStateAccessors:
    """Orientation, gripper and actions, which guidance needs and only
    ``ee_positions`` used to expose.

    Without these, every pair naming ``gripper_l``/``gripper_r`` or measuring
    an orientation silently read 0.0 for the whole demo rather than failing.
    """

    def test_vla_reads_orientation_and_gripper_through_the_layout(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")

        # The fixture's layout is [pos(3), axis-angle(3), aperture(1)] over a
        # state of arange(7) with a moving x.
        ori = ep.ee_orientations()
        grip = ep.gripper_states()
        assert ori is not None and ori.shape == (T, 3)
        assert grip is not None and grip.shape == (T, 1)
        np.testing.assert_allclose(ori[0], [3.0, 4.0, 5.0])
        np.testing.assert_allclose(grip[0], [6.0])

    def test_vla_reports_no_field_the_layout_does_not_carry(self, tmp_path, vla_dir):
        directory, _ = vla_dir
        sidecar = _sidecar()
        sidecar["env_spec"]["robot"]["state_layout"]["fields"] = [
            {"op": "pos", "obs_key": "eef_pos"}
        ]
        (directory / "episode_000.json").write_text(json.dumps(sidecar))

        ep = VlaEpisode(str(directory), "episode_000")
        assert ep.ee_orientations() is None
        assert ep.gripper_states() is None
        # Position still resolves.
        assert ep.ee_positions().shape == (T, 3)

    def test_vla_names_its_own_gripper_geometry(self, vla_dir):
        directory, _ = vla_dir
        sidecar = _sidecar()
        sidecar["env_spec"]["robot"]["gripper"] = {"geometry_id": "robotiq_2f140"}
        (directory / "episode_000.json").write_text(json.dumps(sidecar))

        ep = VlaEpisode(str(directory), "episode_000")
        assert ep.gripper_geometry() == "robotiq_2f140"

    def test_libero_defaults_to_the_panda_geometry(self, libero_file):
        path, _ = libero_file
        assert LiberoHDF5Episode(str(path), "demo_0").gripper_geometry() is None

    def test_frames_preprocessed_distinguishes_the_two_conventions(
        self, vla_dir, libero_file
    ):
        """Back-projection mirrors u only for a format that stores frames
        rotated; doing it for an upright camera mirrors every object."""
        directory, _ = vla_dir
        path, _ = libero_file
        assert VlaEpisode(str(directory), "episode_000").frames_preprocessed is False
        assert LiberoHDF5Episode(str(path), "demo_0").frames_preprocessed is True


class TestExtrinsicFrameRepair:
    """An episode recorded before the base-frame correction existed carries a
    mirrored extrinsic. Repair is a sidecar key, not a rewrite of the .vla."""

    def test_absent_key_takes_the_extrinsic_at_face_value(self, vla_dir):
        directory, _ = vla_dir
        ep = VlaEpisode(str(directory), "episode_000")
        np.testing.assert_allclose(ep.extrinsic(0), np.eye(4))

    def test_base_link_is_rotated_into_the_base_frame(self, vla_dir):
        directory, _ = vla_dir
        sidecar = _sidecar()
        sidecar["extrinsic_frame"] = "base_link"
        (directory / "episode_000.json").write_text(json.dumps(sidecar))

        ep = VlaEpisode(str(directory), "episode_000")
        # 180 degrees about z: x and y negate, z is untouched.
        np.testing.assert_allclose(ep.extrinsic(0), np.diag([-1.0, -1.0, 1.0, 1.0]))

    def test_an_unknown_frame_is_refused(self, vla_dir):
        directory, _ = vla_dir
        sidecar = _sidecar()
        sidecar["extrinsic_frame"] = "world"
        (directory / "episode_000.json").write_text(json.dumps(sidecar))

        with pytest.raises(ValueError, match="extrinsic_frame"):
            VlaEpisode(str(directory), "episode_000").extrinsic(0)
