"""``ur5_env_spec()`` builds a valid EnvSpec, plus the server's pure helpers.

Neither the spec module nor these helpers touch ``rtde_*`` / ``lerobot``, so
this runs without the hardware stack — the same leaf-level property the LIBERO,
RoboCasa and CALVIN spec tests rely on.
"""

import numpy as np
import pytest

from policy_abstraction.envs.spec import EnvSpec
from policy_abstraction.envs.ur5.config import (
    DEFAULT_EXTERIOR_CAMERA,
    DEFAULT_WRIST_CAMERA,
    UR5E_HOME_JOINT_POSITIONS,
    ur5_env_spec,
)


def _fake_obs(joints=None, aperture=0.75):
    joints = np.asarray(
        joints if joints is not None else UR5E_HOME_JOINT_POSITIONS, dtype=np.float64
    )
    return {
        "joint_pos": joints,
        "gripper_qpos": np.array([aperture]),
        "eef_pos": np.array([0.1, -0.2, 0.3]),
        # 90 degrees about z.
        "eef_quat": np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
    }


class TestUr5EnvSpec:
    def test_round_trips_through_the_wire_format(self):
        spec = ur5_env_spec()
        assert isinstance(spec, EnvSpec)
        assert spec.primary_camera.name == DEFAULT_EXTERIOR_CAMERA
        assert spec.wrist_camera.name == DEFAULT_WRIST_CAMERA
        assert spec.metadata["env"] == "ur5"

        rt = EnvSpec.from_dict(spec.to_dict())
        assert rt.primary_camera.name == spec.primary_camera.name
        assert rt.robot.eef_pos_key == "eef_pos"
        assert rt.robot.joint_pos_key == "joint_pos"
        assert rt.robot.action_layout.action_dim == 7
        assert rt.robot.action_layout.space == "joint_absolute"
        assert rt.robot.action_layout.joint_pos_slice == (0, 6)
        assert rt.robot.action_layout.eef_pos_slice is None
        assert rt.robot.action_layout.gripper_dim == 6

    def test_action_space_is_joint_absolute_not_cartesian_delta(self):
        """The distinguishing property vs. every simulated env in the repo."""
        layout = ur5_env_spec().robot.action_layout
        assert layout.space == "joint_absolute"
        assert layout.integrate_eef is False

    def test_publishes_tcp_offset_and_control_rate(self):
        spec = ur5_env_spec(tcp_offset=(0.0, 0.0, 0.15, 0.0, 0.0, 0.0), control_hz=30.0)
        assert spec.metadata["tcp_offset"] == [0.0, 0.0, 0.15, 0.0, 0.0, 0.0]
        assert spec.metadata["control_hz"] == 30.0
        # Survives msgpack-style round-tripping.
        assert EnvSpec.from_dict(spec.to_dict()).metadata["tcp_offset"][2] == 0.15

    def test_defaults_tcp_offset_to_zero(self):
        assert ur5_env_spec().metadata["tcp_offset"] == [0.0] * 6

    def test_colour_only_cameras_have_no_depth_or_segmentation(self):
        """Without a RealSense SDK stream there is no registered depth."""
        for cam in ur5_env_spec().cameras:
            assert cam.depth_key is None
            assert cam.depth_is_metric is False
            assert cam.segmentation_key is None
            assert cam.image_transform == "none"

    def test_depth_cameras_publish_metric_depth(self):
        """SDK-driven cameras get a depth key flagged as already-metric.

        The flag matters: `_iter_camera_views` converts depth it believes is a
        normalised z-buffer, and a close-up real scene can be entirely under
        1 m — which the old range heuristic would have mis-converted.
        """
        spec = ur5_env_spec(depth_cameras=(DEFAULT_EXTERIOR_CAMERA,))
        by_name = {c.name: c for c in spec.cameras}

        ext = by_name[DEFAULT_EXTERIOR_CAMERA]
        assert ext.depth_key == f"{DEFAULT_EXTERIOR_CAMERA}_depth"
        assert ext.depth_is_metric is True

        # The wrist camera wasn't listed, so it stays colour-only.
        assert by_name[DEFAULT_WRIST_CAMERA].depth_key is None
        assert by_name[DEFAULT_WRIST_CAMERA].depth_is_metric is False

    def test_depth_flags_survive_the_wire_format(self):
        spec = ur5_env_spec(depth_cameras=(DEFAULT_EXTERIOR_CAMERA,))
        rt = EnvSpec.from_dict(spec.to_dict())
        ext = next(c for c in rt.cameras if c.name == DEFAULT_EXTERIOR_CAMERA)
        assert ext.depth_key == f"{DEFAULT_EXTERIOR_CAMERA}_depth"
        assert ext.depth_is_metric is True

    def test_rejects_depth_camera_not_in_cameras(self):
        with pytest.raises(ValueError, match="depth_cameras"):
            ur5_env_spec(depth_cameras=("no_such_camera",))

    def test_single_camera_cell(self):
        spec = ur5_env_spec(
            cameras=(DEFAULT_EXTERIOR_CAMERA,),
            wrist_camera="",
            primary_camera=DEFAULT_EXTERIOR_CAMERA,
        )
        assert [c.name for c in spec.cameras] == [DEFAULT_EXTERIOR_CAMERA]
        assert spec.wrist_camera is None
        assert sum(int(c.is_primary) for c in spec.cameras) == 1

    def test_extra_camera_is_routed_to_the_policy(self):
        spec = ur5_env_spec(
            cameras=(
                DEFAULT_EXTERIOR_CAMERA,
                "exterior_image_2_left",
                DEFAULT_WRIST_CAMERA,
            ),
        )
        assert len(spec.cameras) == 3
        # One policy key per camera: the raw LeRobot camera key, under the
        # ``observation/`` prefix every policy client extracts views by.
        assert spec.policy_input_keys.images == tuple(
            (n, f"observation/{n}")
            for n in (
                DEFAULT_EXTERIOR_CAMERA,
                "exterior_image_2_left",
                DEFAULT_WRIST_CAMERA,
            )
        )

    def test_rejects_unknown_primary_camera(self):
        with pytest.raises(ValueError, match="primary_camera"):
            ur5_env_spec(primary_camera="nope")

    def test_rejects_empty_camera_list(self):
        with pytest.raises(ValueError, match="at least one camera"):
            ur5_env_spec(cameras=())


class TestStateLayouts:
    def test_policy_state_is_the_7d_joint_gripper_vector(self):
        spec = ur5_env_spec()
        state = spec.policy_input_keys.state_layout.build(
            _fake_obs(), spec.robot.gripper, dtype=np.float32
        )
        assert state.shape == (7,)
        np.testing.assert_allclose(state[:6], UR5E_HOME_JOINT_POSITIONS, atol=1e-6)
        assert state[6] == pytest.approx(0.75)

    def test_internal_state_is_the_canonical_pos_axisangle_aperture_vector(self):
        spec = ur5_env_spec()
        state = spec.robot.state_layout.build(
            _fake_obs(), spec.robot.gripper, dtype=np.float32
        )
        assert state.shape == (7,)
        np.testing.assert_allclose(state[:3], [0.1, -0.2, 0.3], atol=1e-6)
        # 90 degrees about z -> axis-angle magnitude pi/2 along +z.
        np.testing.assert_allclose(state[3:6], [0.0, 0.0, np.pi / 2], atol=1e-6)
        assert state[6] == pytest.approx(0.75)

    def test_aperture_passes_the_scalar_through(self):
        """The server already flipped the driver's 0=open convention."""
        gripper = ur5_env_spec().robot.gripper
        assert gripper.aperture_from_qpos(np.array([0.25])) == pytest.approx(0.25)


class TestCameraArgParsing:
    @pytest.mark.parametrize(
        "arg, expected",
        [
            (
                "wrist_image_left=/dev/video12",
                {
                    "name": "wrist_image_left",
                    "device": "/dev/video12",
                    "width": 640,
                    "height": 480,
                    "fps": 30,
                },
            ),
            (
                "exterior_image_1_left=/dev/video4:1280x720@60",
                {
                    "name": "exterior_image_1_left",
                    "device": "/dev/video4",
                    "width": 1280,
                    "height": 720,
                    "fps": 60,
                },
            ),
            (
                "cam=/dev/video0@15",
                {
                    "name": "cam",
                    "device": "/dev/video0",
                    "width": 640,
                    "height": 480,
                    "fps": 15,
                },
            ),
        ],
    )
    def test_parses_supported_forms(self, arg, expected):
        from policy_abstraction.envs.ur5.server import parse_camera_arg

        assert parse_camera_arg(arg) == expected

    @pytest.mark.parametrize("arg", ["/dev/video4", "name=", "=/dev/video4"])
    def test_rejects_malformed_specs(self, arg):
        from policy_abstraction.envs.ur5.server import parse_camera_arg

        with pytest.raises(ValueError):
            parse_camera_arg(arg)


class TestRotvecToQuat:
    def test_matches_the_state_layouts_axis_angle_round_trip(self):
        """``rotvec -> quat -> quat2axisangle`` must be the identity.

        The server publishes ``eef_quat`` from RTDE's rotation vector and the
        state layout converts it straight back, so any convention slip here
        would silently rotate every end-effector pose.
        """
        from policy_abstraction.envs.spec import _quat2axisangle
        from policy_abstraction.envs.ur5.server import rotvec_to_quat

        rng = np.random.default_rng(3)
        for _ in range(20):
            axis = rng.normal(size=3)
            axis /= np.linalg.norm(axis)
            rotvec = axis * rng.uniform(0.0, np.pi - 1e-3)
            np.testing.assert_allclose(
                _quat2axisangle(rotvec_to_quat(rotvec)), rotvec, atol=1e-9
            )

    def test_zero_rotation_is_the_identity_quaternion(self):
        from policy_abstraction.envs.ur5.server import rotvec_to_quat

        np.testing.assert_allclose(rotvec_to_quat([0, 0, 0]), [0, 0, 0, 1])
