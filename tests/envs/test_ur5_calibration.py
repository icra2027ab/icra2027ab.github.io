"""Calibration loading for the UR5 env server.

Both failure modes here are silent: a mirrored extrinsic relocates every
reconstructed object without erroring, and adopting a calibration's intrinsics
at the wrong resolution skews reconstruction subtly. So they get tests.
"""

import json

import numpy as np
import pytest

from policy_abstraction.envs.ur5.server import Ur5EnvServer

# Kalib's real convention string; the loader keys off it.
KALIB_CONVENTION = "X_cam = T_cam_base @ X_base_homog  (Kalib / OpenCV solvePnP)"


def _rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _pose(R: np.ndarray, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _kalib_file(tmp_path, cam_to_base: np.ndarray, meta: dict | None = None):
    """Write a Kalib-shaped file for a given camera-to-base pose."""
    base_to_cam = np.linalg.inv(cam_to_base)
    payload = {
        "T_cam_base": base_to_cam.tolist(),  # Kalib's naming: base -> cam
        "T_base_cam": cam_to_base.tolist(),
        "convention": KALIB_CONVENTION,
        "meta": meta or {"width": 1280, "height": 720, "avg_reprojection_err_px": 2.9},
    }
    path = tmp_path / "T_cam_base.json"
    path.write_text(json.dumps(payload))
    return path


class TestKalibLayout:
    def test_picks_the_camera_to_base_direction(self, tmp_path):
        """The key named `T_cam_base` is base->cam; we must return its inverse."""
        cam_to_base = _pose(_rot_z(0.7), [0.07, -0.39, 0.45])
        path = _kalib_file(tmp_path, cam_to_base)

        out = Ur5EnvServer._load_calibration(str(path), "exterior_image_1_left", "base")

        assert set(out) == {"exterior_image_1_left"}
        np.testing.assert_allclose(
            out["exterior_image_1_left"]["extrinsic"], cam_to_base, atol=1e-9
        )

    def test_does_not_adopt_kalib_intrinsics(self, tmp_path):
        """Kalib calibrates at its own resolution; K must come from the device."""
        path = _kalib_file(tmp_path, _pose(np.eye(3), [0.0, 0.0, 0.5]))
        out = Ur5EnvServer._load_calibration(str(path), "exterior_image_1_left", "base")
        assert "intrinsic" not in out["exterior_image_1_left"]

    def test_falls_back_to_inverting_when_only_one_direction_present(self, tmp_path):
        cam_to_base = _pose(_rot_z(-0.4), [0.1, 0.2, 0.6])
        payload = {
            "T_cam_base": np.linalg.inv(cam_to_base).tolist(),
            "convention": KALIB_CONVENTION,
        }
        path = tmp_path / "k.json"
        path.write_text(json.dumps(payload))

        out = Ur5EnvServer._load_calibration(str(path), "cam", "base")
        np.testing.assert_allclose(out["cam"]["extrinsic"], cam_to_base, atol=1e-9)

    def test_rejects_an_unknown_convention_rather_than_guessing(self, tmp_path):
        payload = {
            "T_cam_base": np.eye(4).tolist(),
            "convention": "something else entirely",
        }
        path = tmp_path / "k.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="Unrecognised Kalib convention"):
            Ur5EnvServer._load_calibration(str(path), "cam", "base")

    def test_rejects_a_non_rigid_extrinsic(self, tmp_path):
        bad = np.eye(4)
        bad[:3, :3] *= 2.0  # scaled, not a rotation
        payload = {
            "T_base_cam": bad.tolist(),
            "T_cam_base": np.linalg.inv(bad).tolist(),
            "convention": KALIB_CONVENTION,
        }
        path = tmp_path / "k.json"
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="not a rigid transform"):
            Ur5EnvServer._load_calibration(str(path), "cam", "base")


class TestEnvLayout:
    def test_loads_extrinsic_and_optional_intrinsic(self, tmp_path):
        E = _pose(_rot_z(0.2), [1.0, 2.0, 3.0])
        K = np.array([[600.0, 0, 320.0], [0, 600.0, 240.0], [0, 0, 1.0]])
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps(
                {
                    "cam_a": {"extrinsic": E.tolist(), "intrinsic": K.tolist()},
                    "cam_b": {"extrinsic": E.tolist()},
                }
            )
        )

        out = Ur5EnvServer._load_calibration(str(path), "cam_a", "base")
        np.testing.assert_allclose(out["cam_a"]["intrinsic"], K)
        # cam_b omits K, so the device supplies it.
        assert "intrinsic" not in out["cam_b"]

    def test_extrinsic_is_required(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"cam_a": {"intrinsic": np.eye(3).tolist()}}))

        with pytest.raises(ValueError, match="no 'extrinsic'"):
            Ur5EnvServer._load_calibration(str(path), "cam_a", "base")


def test_no_path_means_no_calibration():
    """Absent calibration degrades to 'skip projection', never to a crash."""
    assert Ur5EnvServer._load_calibration(None, "cam") == {}
    assert Ur5EnvServer._load_calibration("", "cam") == {}


class TestBaseFrameCorrection:
    """UR ships two base frames 180 degrees apart about z.

    ``base`` is what RTDE reports poses in; ``base_link`` is the ROS/URDF
    convention that URDF-based calibration tools produce. Mixing them mirrors
    the reconstructed scene in x and y but leaves the table horizontal, so a
    plane-fit check still passes and only a comparison against the robot's own
    reported TCP catches it. That is the whole reason these tests exist.
    """

    def test_base_is_the_identity(self):
        np.testing.assert_array_equal(
            Ur5EnvServer._base_frame_correction("base"), np.eye(4)
        )

    def test_base_link_is_a_180_degree_turn_about_z(self):
        correction = Ur5EnvServer._base_frame_correction("base_link")
        np.testing.assert_allclose(correction[:3, :3], _rot_z(np.pi), atol=1e-12)
        np.testing.assert_array_equal(correction[:3, 3], np.zeros(3))

    def test_unknown_frame_is_rejected_rather_than_guessed(self):
        with pytest.raises(ValueError, match="base_frame must be"):
            Ur5EnvServer._base_frame_correction("world")

    def test_base_link_mirrors_x_and_y_but_not_z(self):
        """The signature of the mistake: height survives, handedness does not."""
        correction = Ur5EnvServer._base_frame_correction("base_link")
        point = np.array([0.494, 0.087, 0.270, 1.0])
        np.testing.assert_allclose(
            (correction @ point)[:3], [-0.494, -0.087, 0.270], atol=1e-12
        )

    def test_env_layout_applies_the_correction(self, tmp_path):
        cam_to_base = _pose(_rot_z(0.3), [0.07, -0.39, 0.45])
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"cam": {"extrinsic": cam_to_base.tolist()}}))

        as_base = Ur5EnvServer._load_calibration(str(path), "cam", "base")
        as_base_link = Ur5EnvServer._load_calibration(str(path), "cam", "base_link")

        np.testing.assert_allclose(as_base["cam"]["extrinsic"], cam_to_base)
        np.testing.assert_allclose(
            as_base_link["cam"]["extrinsic"],
            Ur5EnvServer._base_frame_correction("base_link") @ cam_to_base,
        )
        # The camera position is mirrored in x/y, keeping its height.
        origin = as_base_link["cam"]["extrinsic"][:3, 3]
        np.testing.assert_allclose(origin, [-0.07, 0.39, 0.45], atol=1e-12)

    def test_kalib_layout_applies_the_correction(self, tmp_path):
        cam_to_base = _pose(_rot_z(0.3), [0.07, -0.39, 0.45])
        path = _kalib_file(tmp_path, cam_to_base)

        as_base_link = Ur5EnvServer._load_calibration(str(path), "cam", "base_link")
        np.testing.assert_allclose(
            as_base_link["cam"]["extrinsic"],
            Ur5EnvServer._base_frame_correction("base_link") @ cam_to_base,
            atol=1e-12,
        )


class TestBaseFrameMustBeDeclared:
    """The frame is never guessed.

    A wrong base frame mirrors the scene in x and y while leaving the table
    horizontal, so it survives every cheap sanity check and only shows up as
    guidance measured in a frame the robot does not use. An episode was in fact
    recorded that way. The cost of never guessing is one key in the file.
    """

    def test_undeclared_and_unflagged_is_an_error(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"cam": {"extrinsic": np.eye(4).tolist()}}))

        with pytest.raises(ValueError, match="does not declare which base frame"):
            Ur5EnvServer._load_calibration(str(path), "cam")

    def test_the_file_can_declare_it(self, tmp_path):
        cam_to_base = _pose(_rot_z(0.3), [0.07, -0.39, 0.45])
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps(
                {
                    "base_frame": "base_link",
                    "cam": {"extrinsic": cam_to_base.tolist()},
                }
            )
        )

        out = Ur5EnvServer._load_calibration(str(path), "cam")

        # "base_frame" is metadata, not a camera.
        assert set(out) == {"cam"}
        np.testing.assert_allclose(
            out["cam"]["extrinsic"],
            Ur5EnvServer._base_frame_correction("base_link") @ cam_to_base,
        )

    def test_the_flag_overrides_the_file(self, tmp_path):
        cam_to_base = _pose(_rot_z(0.3), [0.07, -0.39, 0.45])
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps(
                {"base_frame": "base_link", "cam": {"extrinsic": cam_to_base.tolist()}}
            )
        )

        out = Ur5EnvServer._load_calibration(str(path), "cam", "base")
        np.testing.assert_allclose(out["cam"]["extrinsic"], cam_to_base)

    def test_a_kalib_file_can_declare_it_too(self, tmp_path):
        cam_to_base = _pose(_rot_z(0.3), [0.07, -0.39, 0.45])
        path = _kalib_file(tmp_path, cam_to_base)
        raw = json.loads(path.read_text())
        raw["base_frame"] = "base_link"
        path.write_text(json.dumps(raw))

        out = Ur5EnvServer._load_calibration(str(path), "cam")
        np.testing.assert_allclose(
            out["cam"]["extrinsic"],
            Ur5EnvServer._base_frame_correction("base_link") @ cam_to_base,
            atol=1e-12,
        )


def test_the_shipped_calibration_declares_its_frame():
    """The cell's own calibration file must say which frame it is in.

    It is a `base_link` extrinsic produced by Kalib. Before it said so, the
    server's default treated it as `base` and every recorded extrinsic came out
    mirrored.
    """
    from policy_abstraction import PROJECT_ROOT

    path = PROJECT_ROOT / "configs" / "ur5_camera_calibration.json"
    raw = json.loads(path.read_text())
    assert raw.get("base_frame") in ("base", "base_link"), (
        f"{path} must declare 'base_frame'; loading it would otherwise raise."
    )


class TestTcpParentedEntries:
    """A wrist camera's entry is a pose in the TCP frame; the server composes it
    with the live TCP pose on every get_camera_params call."""

    def _file(self, tmp_path, base_frame="base", parent="tcp", meta=None):
        T_tcp_cam = _pose(_rot_z(0.3), [0.0, -0.06, -0.12])
        entry = {"extrinsic": T_tcp_cam.tolist(), "intrinsic": np.eye(3).tolist()}
        if parent is not None:
            entry["parent"] = parent
        if meta is not None:
            entry["meta"] = meta
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps({"base_frame": base_frame, "wrist_image_left": entry})
        )
        return path, T_tcp_cam

    def test_loads_without_the_base_frame_correction(self, tmp_path):
        path, T = self._file(tmp_path, base_frame="base_link")
        out = Ur5EnvServer._load_calibration(str(path), "wrist_image_left")
        np.testing.assert_allclose(out["wrist_image_left"]["extrinsic"], T)
        assert out["wrist_image_left"]["parent"] == "tcp"

    def test_meta_is_kept_and_unknown_parent_rejected(self, tmp_path):
        path, _ = self._file(tmp_path, meta={"tcp_offset": [0, 0, 0.2, 0, 0, 0]})
        out = Ur5EnvServer._load_calibration(str(path), "wrist_image_left")
        assert out["wrist_image_left"]["meta"]["tcp_offset"] == [0, 0, 0.2, 0, 0, 0]
        path, _ = self._file(tmp_path, parent="flange")
        with pytest.raises(ValueError, match="parent"):
            Ur5EnvServer._load_calibration(str(path), "wrist_image_left")

    def test_camera_params_compose_with_the_live_tcp(self, tmp_path):
        from policy_abstraction.calibration.geometry import pose_to_matrix
        from policy_abstraction.envs.ur5.fake_driver import FakeUr5Driver

        path, T_tcp_cam = self._file(tmp_path)
        tcp_offset = [0.0, 0.0, 0.22, 0.0, 0.0, 0.0]
        server = Ur5EnvServer(
            tasks=["t"],
            cameras=["wrist_image_left=/dev/video0"],
            primary_camera="wrist_image_left",
            wrist_camera="wrist_image_left",
            simulate_hardware=True,
            tcp_offset=tcp_offset,
            calibration_path=str(path),
        )
        # Before the arm exists the camera cannot be placed, so it is omitted.
        assert server._camera_params(["wrist_image_left"]) == {}
        server._robot = FakeUr5Driver([], control_period=0.02, tcp_offset=tcp_offset)
        server._robot.set_joints([0.3, -1.2, 1.4, -1.7, -1.5, 0.4])
        params = server._camera_params(["wrist_image_left"])
        T_base_tcp = pose_to_matrix(server._robot.rtde_rec.getActualTCPPose())
        np.testing.assert_allclose(
            params["wrist_image_left"]["extrinsic"], T_base_tcp @ T_tcp_cam, atol=1e-9
        )
        # And it follows the arm.
        server._robot.set_joints([0.0, -1.5, 1.5, -1.5, -1.5, 0.0])
        params2 = server._camera_params(["wrist_image_left"])
        assert not np.allclose(
            params2["wrist_image_left"]["extrinsic"],
            params["wrist_image_left"]["extrinsic"],
        )

    def test_base_parented_entries_are_unchanged_by_the_new_keys(self, tmp_path):
        E = _pose(_rot_z(0.2), [1.0, 2.0, 3.0])
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps(
                {
                    "base_frame": "base_link",
                    "cam": {"extrinsic": E.tolist(), "meta": {"anything": 1}},
                }
            )
        )
        out = Ur5EnvServer._load_calibration(str(path), "cam")
        np.testing.assert_allclose(
            out["cam"]["extrinsic"],
            Ur5EnvServer._base_frame_correction("base_link") @ E,
        )
        assert "parent" not in out["cam"]
