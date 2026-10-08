"""The capture on disk (Kalib's layout) and the calibration file (the env's)."""

import json

import numpy as np
import pytest

from policy_abstraction.calibration.dataset import (
    CaptureFrame,
    CaptureMeta,
    KalibDataset,
)
from policy_abstraction.calibration.output import (
    base_link_to_base,
    camera_entry,
    write_calibration,
)


def _meta(mode="exterior"):
    return CaptureMeta(
        camera_name="cam",
        camera_device="rs:0",
        width=64,
        height=48,
        fps=30,
        intrinsic=np.eye(3).tolist(),
        mode=mode,
        tcp_offset=[0, 0, 0.2, 0, 0, 0],
    )


def _frame(i, settled=True, depth=True):
    rgb = np.full((48, 64, 3), i * 3, np.uint8)
    return CaptureFrame(
        rgb=rgb,
        joints=np.arange(6) * 0.1 + i,
        joint_velocities=np.zeros(6),
        tcp_pose=np.array([0.1 * i, 0.2, 0.3, 0, 0, 0]),
        keypoint=np.array([0.1 * i, 0.2, 0.3]),
        settled=settled,
        timestamp=float(i),
        depth_m=np.full((48, 64), 1.234, np.float32) if depth else None,
    )


class TestKalibDataset:
    def test_writes_kalib_pairs_and_reads_them_back(self, tmp_path):
        ds = KalibDataset.create(tmp_path / "cap", _meta())
        ds.append(_frame(0))
        ds.append(_frame(1, settled=False))
        assert ds.n_frames == 2
        recs = ds.records()
        assert [r.index for r in recs] == [0, 1]
        assert recs[0].settled and not recs[1].settled
        np.testing.assert_allclose(recs[1].keypoint, [0.1, 0.2, 0.3])
        assert recs[1].tcp_pose is not None
        np.testing.assert_allclose(recs[1].tcp_pose, [0.1, 0.2, 0.3, 0, 0, 0])
        # Kalib's own schema is present.
        d = json.loads((tmp_path / "cap" / "000000.json").read_text())
        obj = d["objects"][0]
        assert obj["keypoints"][0]["location"] == [0.0, 0.2, 0.3]
        assert obj["camera_intrinsics"] == np.eye(3).tolist()
        assert len(obj["joint_positions"]) == 6
        # Images come back RGB at the declared size; depth in metres.
        assert ds.load_rgb(1).shape == (48, 64, 3)
        depth = ds.load_depth_m(1)
        assert depth is not None
        assert depth[10, 10] == pytest.approx(1.234, abs=1e-3)

    def test_depth_is_kept_out_of_kalibs_glob(self, tmp_path):
        ds = KalibDataset.create(tmp_path / "cap", _meta())
        ds.append(_frame(0))
        assert not list((tmp_path / "cap").glob("*.png"))
        assert (tmp_path / "cap" / "depth" / "000000.png").exists()

    def test_create_wipes_a_previous_capture_but_nothing_else(self, tmp_path):
        root = tmp_path / "cap"
        ds = KalibDataset.create(root, _meta())
        ds.append(_frame(0))
        (root / "Kalib").mkdir()
        (root / "Kalib" / "stale.pkl").write_text("x")
        (root / "unrelated.txt").write_text("keep me")
        ds = KalibDataset.create(root, _meta())
        assert ds.n_frames == 0
        assert not (root / "Kalib").exists()
        assert (root / "unrelated.txt").exists()

    def test_meta_roundtrip(self, tmp_path):
        ds = KalibDataset.create(tmp_path / "cap", _meta("wrist"))
        assert ds.meta.mode == "wrist" and ds.meta.parent_frame == "tcp"
        assert ds.meta.width == 64

    def test_rejects_a_frame_of_the_wrong_size(self, tmp_path):
        ds = KalibDataset.create(tmp_path / "cap", _meta())
        f = _frame(0)
        f.rgb = np.zeros((10, 10, 3), np.uint8)
        with pytest.raises(ValueError, match="declared"):
            ds.append(f)


def _pose(z_deg, t):
    T = np.eye(4)
    a = np.deg2rad(z_deg)
    T[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    T[:3, 3] = t
    return T


class TestWriteCalibration:
    def test_new_file_declares_base_and_holds_the_entry(self, tmp_path):
        path = tmp_path / "calib.json"
        E = _pose(30, [1, 2, 3])
        out = write_calibration(
            path, "cam_a", camera_entry(E, parent="base", meta={"k": 1})
        )
        raw = json.loads(path.read_text())
        assert raw == out
        assert raw["base_frame"] == "base"
        np.testing.assert_allclose(raw["cam_a"]["extrinsic"], E)
        assert "parent" not in raw["cam_a"]
        assert raw["cam_a"]["meta"] == {"k": 1}

    def test_wrist_entry_is_parented_to_the_tcp(self):
        e = camera_entry(_pose(0, [0, 0, 0.1]), parent="tcp")
        assert e["parent"] == "tcp"
        with pytest.raises(ValueError):
            camera_entry(_pose(0, [0, 0, 0]), parent="flange")

    def test_merge_keeps_other_cameras_and_backs_up(self, tmp_path):
        path = tmp_path / "calib.json"
        E1 = _pose(10, [1, 0, 0])
        write_calibration(path, "cam_a", camera_entry(E1, parent="base"))
        E2 = _pose(20, [0, 1, 0])
        write_calibration(path, "cam_b", camera_entry(E2, parent="tcp"))
        raw = json.loads(path.read_text())
        assert set(raw) == {"base_frame", "cam_a", "cam_b"}
        np.testing.assert_allclose(raw["cam_a"]["extrinsic"], E1)
        assert raw["cam_b"]["parent"] == "tcp"
        assert len(list(tmp_path.glob("calib.json.bak-*"))) == 1

    def test_merging_into_a_base_link_file_rotates_the_old_entries(self, tmp_path):
        """The shipped config was in base_link; the new tool writes base. One file,
        one frame: the old fixed-camera entry is rotated, a tcp entry is not."""
        path = tmp_path / "calib.json"
        old_fixed = _pose(0, [0.07, -0.39, 0.45])
        old_wrist = _pose(0, [0, 0, 0.1])
        path.write_text(
            json.dumps(
                {
                    "base_frame": "base_link",
                    "old_fixed": {"extrinsic": old_fixed.tolist()},
                    "old_wrist": {"extrinsic": old_wrist.tolist(), "parent": "tcp"},
                }
            )
        )
        new = _pose(5, [1, 1, 1])
        raw = write_calibration(path, "new_cam", camera_entry(new, parent="base"))
        assert raw["base_frame"] == "base"
        np.testing.assert_allclose(
            raw["old_fixed"]["extrinsic"], base_link_to_base() @ old_fixed
        )
        np.testing.assert_allclose(raw["old_wrist"]["extrinsic"], old_wrist)
        np.testing.assert_allclose(raw["new_cam"]["extrinsic"], new)

    def test_refuses_an_undeclared_frame_in_the_old_file(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(json.dumps({"cam": {"extrinsic": np.eye(4).tolist()}}))
        with pytest.raises(ValueError, match="base_frame"):
            write_calibration(path, "new", camera_entry(np.eye(4), parent="base"))

    def test_a_raw_kalib_export_is_replaced(self, tmp_path):
        path = tmp_path / "calib.json"
        path.write_text(
            json.dumps({"T_cam_base": np.eye(4).tolist(), "convention": "x"})
        )
        raw = write_calibration(path, "cam", camera_entry(np.eye(4), parent="base"))
        assert set(raw) == {"base_frame", "cam"}
