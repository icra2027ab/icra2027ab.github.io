"""Reconstruction and input handling for scripts/debug/perception_playground.py.

Runs the playground's pure functions against a synthetic RGBD view built from
the real UR5 env spec — no hardware, no VLM call — so a shape or
frame-convention mistake fails here rather than in front of the robot.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from policy_abstraction import PROJECT_ROOT
from policy_abstraction.envs.ur5.config import ur5_env_spec

sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "debug"))

from perception_playground import (  # noqa: E402
    Args,
    _iter_camera_views,
    _object_clouds,
    _parse_queries,
    _scene_cloud,
    _settings_md,
    _stats_md,
)

CAM = "exterior_image_1_left"
H, W = 48, 64
BLOB = (slice(10, 20), slice(10, 20))
DEAD_ROWS = 3


@pytest.fixture
def spec():
    return ur5_env_spec(
        cameras=(CAM,), wrist_camera="", primary_camera=CAM, depth_cameras=(CAM,)
    )


@pytest.fixture
def frame():
    """One synthetic RGBD observation plus its calibration."""
    rng = np.random.default_rng(0)
    depth = np.full((H, W), 0.8, dtype=np.float32)
    depth[BLOB] = 0.6  # a nearer blob standing in for an object
    depth[:DEAD_ROWS, :] = 0.0  # invalid pixels, as a real sensor emits
    obs = {
        CAM: rng.integers(0, 255, (H, W, 3), dtype=np.uint8),
        f"{CAM}_depth": depth,
        "joint_pos": np.zeros(6),
        "gripper_qpos": np.array([0.5]),
        "eef_pos": np.array([-0.4, -0.2, 0.15]),
        "eef_axis_angle": np.array([0.1, 0.2, 0.3]),
    }
    K = np.array([[600.0, 0, W / 2], [0, 600.0, H / 2], [0, 0, 1.0]])
    E = np.eye(4)
    E[:3, 3] = [0.5, 0.0, 0.6]
    return obs, {CAM: {"intrinsic": K, "extrinsic": E}}


@pytest.fixture
def views(spec, frame):
    return _iter_camera_views(spec, *frame)


def _masks():
    mask = np.zeros((H, W), bool)
    mask[BLOB] = True
    return [{7: ("cable", mask)}]


class TestCameraViews:
    def test_depth_camera_is_yielded(self, views):
        assert len(views) == 1
        assert views[0][0].name == CAM

    def test_metric_depth_is_not_reconverted(self, views):
        """A close-up real scene is legitimately all under 1 m; converting it as
        a z-buffer would silently rescale the whole cloud."""
        _cam, _rgb, dep, _K, _E = views[0]
        assert dep.max() == pytest.approx(0.8)

    def test_camera_without_depth_is_skipped(self, frame):
        """Colour-only cameras can't reconstruct, so they must not reach fusion."""
        obs, params = frame
        colour_only = ur5_env_spec(
            cameras=(CAM,), wrist_camera="", primary_camera=CAM, depth_cameras=()
        )
        assert _iter_camera_views(colour_only, obs, params) == []


class TestReconstruction:
    def test_instance_cloud_matches_mask_area(self, views):
        clouds, _ = _object_clouds(views, _masks(), [], flip_u=False, depth_max=2.0)
        assert set(clouds) == {"cable#7"}
        assert clouds["cable#7"].shape[0] == 100  # 10x10 blob, all valid depth

    def test_centroid_sits_at_the_blob_depth(self, views):
        """Blob is at 0.6 m in front of a camera 0.6 m up, looking down +z."""
        _, centroids = _object_clouds(views, _masks(), [], flip_u=False, depth_max=2.0)
        assert centroids["cable#7"][2] == pytest.approx(1.2, abs=1e-3)

    def test_scene_cloud_drops_invalid_depth(self, views):
        scene = _scene_cloud(views, flip_u=False, depth_max=2.0, cap=0)
        assert scene.shape[0] == H * W - DEAD_ROWS * W

    def test_scene_cloud_respects_cap(self, views):
        assert _scene_cloud(views, flip_u=False, depth_max=2.0, cap=500).shape[0] <= 500

    def test_depth_max_trims_the_background(self, views):
        """Everything is beyond 0.5 m, so a tight cap must empty the cloud."""
        assert _scene_cloud(views, flip_u=False, depth_max=0.5, cap=0).shape[0] == 0

    def test_flip_u_changes_the_geometry(self, views):
        """Guards the flag against becoming decorative: mirroring u must move
        the reconstruction, which is exactly the failure that put every UR5
        cloud 0.8 m from where the robot said it was."""
        a, _ = _object_clouds(views, _masks(), [], flip_u=False, depth_max=2.0)
        b, _ = _object_clouds(views, _masks(), [], flip_u=True, depth_max=2.0)
        shift = np.linalg.norm(
            a["cable#7"][:, :3].mean(0) - b["cable#7"][:, :3].mean(0)
        )
        assert shift > 0.01

    def test_empty_masks_yield_no_objects(self, views):
        clouds, centroids = _object_clouds(views, [{}], [], flip_u=False, depth_max=2.0)
        assert clouds == {} and centroids == {}


class TestQueryParsing:
    @pytest.mark.parametrize(
        "text, current, expected",
        [
            ("cable, bag", [], ["cable", "bag"]),
            ("cable bag", [], ["cable", "bag"]),
            ("+ mug", ["cable"], ["cable", "mug"]),
            ("+ cable", ["cable"], ["cable"]),  # no duplicates
            ("   ", ["cable"], None),  # blank leaves the set alone
        ],
    )
    def test_parse(self, text, current, expected):
        assert _parse_queries(text, current) == expected


class TestPanels:
    def test_settings_names_the_active_gripper_and_pipeline(self, spec, views):
        md = _settings_md(Args(), spec, False, views)
        assert "robotiq_2f140" in md
        assert "sam2" in md

    def test_stats_lists_each_instance(self, views):
        clouds, centroids = _object_clouds(
            views, _masks(), [], flip_u=False, depth_max=2.0
        )
        md = _stats_md(3, ["cable"], clouds, centroids, 4070.0, 51.0, 19.6, 900, 12)
        assert "cable#7" in md and "4070" in md

    def test_stats_says_so_when_nothing_matched(self):
        md = _stats_md(0, ["ghost"], {}, {}, 4070.0, 51.0, 0.0, 0, 0)
        assert "nothing matched" in md
