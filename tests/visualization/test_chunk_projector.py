"""Where the action-chunk overlay lands on a camera frame.

The overlay is drawn on top of the frame the operator watches and the video
keeps, so "wrong by a mirror" looks exactly like "the policy is planning to go
the other way" — a plausible, wrong, and very expensive thing to believe about a
rollout. These pin the projection against plain pinhole arithmetic on both
camera conventions in the tree: a simulator's pre-flipped square frames, and a
real camera's upright non-square ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from policy_abstraction.envs.spec import CameraSpec
from policy_abstraction.visualization.chunk_projector import ChunkProjector

# Camera 1 m above the table looking straight down: x right, y down in image.
LOOKING_DOWN = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def intrinsic(width: int, height: int, f: float = 600.0) -> np.ndarray:
    return np.array([[f, 0.0, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]])


def pinhole(point, K, E=LOOKING_DOWN) -> tuple[float, float]:
    """Textbook projection, independent of the code under test."""
    cam = np.linalg.inv(E) @ np.array([*point, 1.0])
    uv = K @ cam[:3]
    return uv[0] / uv[2], uv[1] / uv[2]


def camera(name: str, transform: str) -> CameraSpec:
    return CameraSpec(
        name=name, role="world", image_key=f"{name}_img", image_transform=transform
    )


def states(points: np.ndarray, gripper: float = 0.0) -> np.ndarray:
    """``[1, T, 7]`` predicted states: xyz, axis-angle, aperture."""
    padding = np.zeros((len(points), 4))
    padding[:, -1] = gripper
    return np.concatenate([points, padding], axis=1)[np.newaxis]


def projector(cam: CameraSpec, K: np.ndarray, **kwargs) -> ChunkProjector:
    proj = ChunkProjector(
        camera_height=kwargs.pop("request_height", 256),
        camera_width=kwargs.pop("request_width", 256),
        cameras=[cam],
        eef_pos_slice=(0, 3),
        gripper_dim=6,
        **kwargs,
    )
    proj._camera_params = {cam.name: {"intrinsic": K, "extrinsic": LOOKING_DOWN}}
    return proj


def drawn_pixels(proj: ChunkProjector, cam_name: str, width: int) -> np.ndarray:
    """The pixels :meth:`draw` would use, without needing cv2 or an image."""
    raw, _valid, _gs = proj._projected[cam_name][0]
    return proj._to_frame(raw, cam_name, width)


TABLE_POINTS = np.array([[0.05, 0.0, 0.0], [0.15, 0.0, 0.0], [0.25, 0.0, 0.0]])


class TestUprightCamera:
    """A real camera publishes frames as they come off the sensor."""

    W, H = 640, 480

    @pytest.fixture
    def proj(self):
        cam = camera("exterior_image_1_left", "none")
        proj = projector(cam, intrinsic(self.W, self.H))
        proj.update(states(TABLE_POINTS))
        return proj

    def test_the_overlay_lands_where_the_pinhole_says(self, proj):
        K = intrinsic(self.W, self.H)
        pixels = drawn_pixels(proj, "exterior_image_1_left", self.W)
        for point, (u, v) in zip(TABLE_POINTS, pixels, strict=True):
            expected = pinhole(point, K)
            assert (u, v) == pytest.approx(expected, abs=1.0)

    def test_it_is_not_mirrored(self, proj):
        """The regression: u was mirrored for every camera, real ones included."""
        pixels = drawn_pixels(proj, "exterior_image_1_left", self.W)
        assert (pixels[:, 0] > 0).all(), "the overlay fell off the left of the frame"
        # The EEF moves right across the table, so the overlay must too.
        assert (np.diff(pixels[:, 0]) > 0).all()

    def test_the_frames_width_decides_not_the_requested_one(self):
        """The real cell's server ignores the requested size and returns native.

        Mirroring about a 256 px request on a 640 px frame is how the overlay
        ended up off the image entirely.
        """
        cam = camera("exterior_image_1_left", "flip_vh")
        proj = projector(cam, intrinsic(self.W, self.H), request_width=256)
        proj.update(states(TABLE_POINTS))
        K = intrinsic(self.W, self.H)
        pixels = drawn_pixels(proj, "exterior_image_1_left", self.W)
        for point, (u, _v) in zip(TABLE_POINTS, pixels, strict=True):
            assert u == pytest.approx((self.W - 1) - pinhole(point, K)[0], abs=1.0)


class TestPreFlippedCamera:
    """A simulator's frames arrive 180°-rotated; only u needs undoing."""

    SIZE = 256

    @pytest.fixture
    def proj(self):
        cam = camera("agentview", "flip_vh")
        proj = projector(cam, intrinsic(self.SIZE, self.SIZE))
        proj.update(states(TABLE_POINTS))
        return proj

    def test_u_is_mirrored_about_the_frame_width(self, proj):
        K = intrinsic(self.SIZE, self.SIZE)
        pixels = drawn_pixels(proj, "agentview", self.SIZE)
        for point, (u, v) in zip(TABLE_POINTS, pixels, strict=True):
            eu, ev = pinhole(point, K)
            assert u == pytest.approx((self.SIZE - 1) - eu, abs=1.0)
            assert v == pytest.approx(ev, abs=1.0), "v must be left alone"

    def test_the_two_conventions_are_mirror_images(self, proj):
        """Same geometry, both conventions: the only difference is the u flip."""
        upright = projector(
            camera("agentview", "none"), intrinsic(self.SIZE, self.SIZE)
        )
        upright.update(states(TABLE_POINTS))
        flipped_px = drawn_pixels(proj, "agentview", self.SIZE)
        upright_px = drawn_pixels(upright, "agentview", self.SIZE)
        assert np.allclose(flipped_px[:, 0], (self.SIZE - 1) - upright_px[:, 0], atol=1)
        assert np.array_equal(flipped_px[:, 1], upright_px[:, 1])


class TestMixedCameras:
    def test_each_camera_uses_its_own_convention(self):
        """A rig is allowed to mix them; the flag is per camera, not per run."""
        cams = [camera("upright", "none"), camera("flipped", "flip_vh")]
        proj = ChunkProjector(
            camera_height=480,
            camera_width=640,
            cameras=cams,
            eef_pos_slice=(0, 3),
            gripper_dim=6,
        )
        K = intrinsic(640, 480)
        proj._camera_params = {
            c.name: {"intrinsic": K, "extrinsic": LOOKING_DOWN} for c in cams
        }
        proj.update(states(TABLE_POINTS))
        upright = drawn_pixels(proj, "upright", 640)
        flipped = drawn_pixels(proj, "flipped", 640)
        assert np.allclose(flipped[:, 0], 639 - upright[:, 0], atol=1)


class TestDegenerateInput:
    def test_no_states_clears_the_cache(self):
        proj = projector(camera("cam", "none"), intrinsic(640, 480))
        proj.update(states(TABLE_POINTS))
        proj.update(None)
        assert proj._projected == {}

    def test_a_flat_chunk_is_accepted_as_one_chunk(self):
        proj = projector(camera("cam", "none"), intrinsic(640, 480))
        proj.update(states(TABLE_POINTS)[0])  # [T, D] rather than [N, T, D]
        assert len(proj._projected["cam"]) == 1
        assert proj.chunk_length == len(TABLE_POINTS)

    def test_points_behind_the_camera_are_marked_invalid(self):
        proj = projector(camera("cam", "none"), intrinsic(640, 480))
        behind = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 3.0]])  # above the camera
        proj.update(states(behind))
        _pts, valid, _gs = proj._projected["cam"][0]
        assert not valid.any()

    def test_an_uncalibrated_camera_simply_gets_no_overlay(self):
        proj = projector(camera("cam", "none"), intrinsic(640, 480))
        proj._camera_params = {}
        proj.update(states(TABLE_POINTS))
        assert proj._projected == {}
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        assert proj.draw(img, "cam") is img


class TestCameraSpecRule:
    """One definition of "are this camera's pixels mirrored"."""

    def test_a_pre_flipped_camera_mirrors(self):
        assert camera("agentview", "flip_vh").pixels_mirrored

    def test_an_upright_camera_does_not(self):
        assert not camera("exterior_image_1_left", "none").pixels_mirrored


class TestSecondChunkInOneColour:
    """A counterfactual chunk goes on a frame that already carries the
    executed one, so it must be told apart by colour and must not need a
    live env for its camera matrices."""

    W, H = 640, 480
    MAGENTA = (255, 60, 220)

    def _projector(self) -> ChunkProjector:
        cam = camera("exterior_image_1_left", "none")
        proj = ChunkProjector(
            camera_height=self.H,
            camera_width=self.W,
            cameras=[cam],
            eef_pos_slice=(0, 3),
            gripper_dim=None,
        )
        proj.set_camera_params(
            {
                cam.name: {
                    "intrinsic": intrinsic(self.W, self.H),
                    "extrinsic": LOOKING_DOWN,
                }
            }
        )
        proj.update(states(TABLE_POINTS))
        return proj

    def test_recorded_matrices_project_like_fetched_ones(self):
        proj = self._projector()
        K = intrinsic(self.W, self.H)
        pixels = drawn_pixels(proj, "exterior_image_1_left", self.W)
        for point, (u, v) in zip(TABLE_POINTS, pixels, strict=True):
            assert (u, v) == pytest.approx(pinhole(point, K), abs=1.0)

    def test_a_fixed_colour_replaces_the_gripper_gradient(self):
        pytest.importorskip("cv2")
        proj = self._projector()
        frame = np.zeros((self.H, self.W, 3), dtype=np.uint8)
        out = proj.draw(frame, "exterior_image_1_left", color=self.MAGENTA)
        touched = out[(out != 0).any(axis=-1)].astype(int)
        assert len(touched) > 0
        # Anti-aliasing dims the edges but never changes the hue: the gripper
        # gradient would put green or red on top with blue lowest, magenta
        # puts green lowest everywhere.
        assert (touched[:, 1] < touched[:, 0]).all()
        assert (touched[:, 1] < touched[:, 2]).all()
        brightest = touched[touched.sum(axis=1).argmax()]
        expected = np.round(np.array(self.MAGENTA) * proj._alpha)
        assert np.abs(brightest - expected).max() <= 1

    def test_a_camera_without_matrices_is_left_alone(self):
        proj = self._projector()
        frame = np.full((self.H, self.W, 3), 7, dtype=np.uint8)
        assert proj.draw(frame, "wrist_image_left", color=self.MAGENTA) is frame
