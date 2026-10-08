"""Depth recording in RobodmRecorder (metres in, uint16 millimetres out)."""

from __future__ import annotations

import numpy as np
import pytest
import robodm

from policy_abstraction.data.recorder import RobodmRecorder


def _read(path) -> dict:
    traj = robodm.Trajectory(str(path), mode="r")
    try:
        return traj.load()
    finally:
        traj.close()


@pytest.fixture
def recorder(tmp_path):
    rec = RobodmRecorder()
    rec.initialize(str(tmp_path / "episode_000.vla"), {"task_description": "t"})
    return rec, tmp_path / "episode_000.vla"


class TestDepthRecording:
    def test_depth_round_trips_as_millimetres(self, recorder):
        rec, path = recorder
        depth = np.full((8, 8), 1.234, dtype=np.float32)
        for t in range(3):
            rec.record(
                camera_depth_streams={"cam": depth},
                depth_timestep=t * 5,
                state=np.zeros(7, dtype=np.float32),
            )
        rec.save()

        data = _read(path)
        got = np.asarray(data["observation/cam_depth"])
        assert got.dtype == np.uint16
        assert got.shape == (3, 8, 8)
        # 1.234 m -> 1234 mm, exactly.
        assert np.all(got == 1234)

    def test_depth_timestep_index_is_written(self, recorder):
        rec, path = recorder
        depth = np.zeros((4, 4), dtype=np.float32)
        for t in (0, 5, 10):
            rec.record(camera_depth_streams={"cam": depth}, depth_timestep=t)
        rec.save()

        idx = np.asarray(_read(path)["observation/depth_timestep"]).ravel()
        assert idx.tolist() == [0, 5, 10]

    def test_non_finite_depth_becomes_zero(self, recorder):
        """NaN/inf mean "no return"; 0 is the uint16 convention for that."""
        rec, path = recorder
        depth = np.array(
            [[np.nan, np.inf], [-np.inf, 0.5]],
            dtype=np.float32,
        )
        rec.record(camera_depth_streams={"cam": depth}, depth_timestep=0)
        rec.save()

        got = np.asarray(_read(path)["observation/cam_depth"])[0]
        assert got.tolist() == [[0, 0], [0, 500]]

    def test_depth_beyond_uint16_range_is_clipped(self, recorder):
        """Clipping keeps a stray far reading from wrapping to a near one."""
        rec, path = recorder
        rec.record(
            camera_depth_streams={"cam": np.full((2, 2), 100.0, dtype=np.float32)},
            depth_timestep=0,
        )
        rec.save()

        got = np.asarray(_read(path)["observation/cam_depth"])[0]
        assert np.all(got == 65535)

    def test_multiple_cameras_get_their_own_streams(self, recorder):
        rec, path = recorder
        rec.record(
            camera_depth_streams={
                "front": np.full((4, 4), 1.0, dtype=np.float32),
                "wrist": np.full((4, 4), 2.0, dtype=np.float32),
            },
            depth_timestep=0,
        )
        rec.save()

        data = _read(path)
        assert np.all(np.asarray(data["observation/front_depth"])[0] == 1000)
        assert np.all(np.asarray(data["observation/wrist_depth"])[0] == 2000)

    def test_no_depth_records_no_streams(self, recorder):
        rec, path = recorder
        rec.record(state=np.zeros(7, dtype=np.float32))
        rec.save()

        data = _read(path)
        assert not [k for k in data if k.endswith("_depth")]
        assert "observation/depth_timestep" not in data
