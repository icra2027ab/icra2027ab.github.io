"""Projects predicted action-chunk EEF trajectories onto camera images."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from policy_abstraction.envs.spec import CameraSpec
from policy_abstraction.utils.pcd import project_points_to_pixels


class ChunkProjector:
    """Projects action-chunk EEF trajectories into camera image space.

    Usage
    -----
    ::

        projector = ChunkProjector(
            camera_height=H, camera_width=W, alpha=0.65,
            cameras=env_spec.cameras, eef_pos_slice=(0, 3), gripper_dim=6,
        )
        projector.fetch_params(env)                             # once per replan
        projector.update(predicted_action_chunk_states)         # [N, T, D]

        annotated_agent = projector.draw(agentview_img, "agentview")
        annotated_wrist = projector.draw(wrist_img,     "robot0_eye_in_hand")

    Camera matrices
    ---------------
    ``get_camera_params`` from the env server returns robosuite's matrices:

    * **intrinsic** ``K`` — standard 3×3 pinhole (pixels, top-left origin,
      OpenCV convention).
    * **extrinsic** ``E`` — *camera-to-world* 4×4 pose (camera frame in world
      coordinates, with robosuite's axis correction applied).

    We invert ``E`` to world-to-camera before projecting.

    Image convention
    ----------------
    A camera that declares ``image_transform="flip_vh"`` has its frames
    rotated 180° before anyone sees them, which converts OpenGL bottom-left
    to a top-left-origin image that is also horizontally mirrored. The
    axis-corrected K already gives a top-left-origin v, so only u needs
    mirroring: ``col = (W - 1) - u``.

    That correction is **per camera** and comes from
    :attr:`~policy_abstraction.envs.spec.CameraSpec.pixels_mirrored`. It used
    to be assumed, which is right for the simulators and wrong for every real
    camera -- the UR5e publishes upright frames, and mirroring them put the
    whole overlay on the other side of the image.

    The mirror is applied in :meth:`draw`, against the width of the frame
    being drawn on, rather than in :meth:`update` against the requested
    render size. The two are not always the same number: the real cell's
    server ignores the requested size and returns native frames with a
    matching K, so a mirror computed from the request would be wrong there
    even with the right camera flag.
    """

    # Gripper colour: interpolates green (>= +0.5) → red (<= -0.5).
    _COLOR_OPEN = (200, 50, 50)  # red  – fully open  (gs >= +0.5)
    _COLOR_CLOSE = (50, 200, 50)  # green – fully closed (gs <= -0.5)
    _MAX_GRIPPER_VALUE = 0.9
    _MIN_GRIPPER_VALUE = -0.9

    def __init__(
        self,
        camera_height: int,
        camera_width: int,
        alpha: float = 0.65,
        *,
        cameras: Sequence[CameraSpec],
        eef_pos_slice: tuple[int, int],
        gripper_dim: int | None,
    ) -> None:
        """Build a projector keyed to the env's cameras + action layout.

        Args:
            camera_height: Frame height to request from ``get_camera_params``.
            camera_width: Frame width to request. Both are the *request*: a
                server is free to ignore them and return native-resolution
                matrices, which the real cell does. Never use them as the size
                of a frame being drawn on -- read that off the frame.
            cameras: The env's camera specs. Their names key
                ``get_camera_params``, :meth:`draw` and :meth:`update`'s
                output, and their ``pixels_mirrored`` decides whether a
                projected u is mirrored onto that camera's frames.
            eef_pos_slice: Half-open ``[start, stop)`` action-dim slice for
                the 3-D EEF position. Typically ``(0, 3)``.
            gripper_dim: Action-dim index for the scalar gripper state used
                to colour overlays. ``None`` disables gripper colouring.
        """
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._alpha = alpha  # trajectory overlay opacity
        self._camera_names = [c.name for c in cameras]
        self._mirrored = {c.name: c.pixels_mirrored for c in cameras}
        self._eef_pos_slice = eef_pos_slice
        self._gripper_dim = gripper_dim
        self._camera_params: dict = {}  # {cam_name: {intrinsic, extrinsic}}
        self._projected: dict[str, list] = {}  # {cam_name: [(pts_2d, valid, gs), ...]}
        self._chunk_length: int | None = None  # number of steps in each chunk

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fetch_params(self, env) -> None:
        """Fetch camera intrinsic/extrinsic matrices from *env*.

        Calls ``env.get_camera_params`` (supported by :class:`WebsocketEnvClient`
        via the ``get_camera_params`` RPC).  Silently skips on failure so that
        projection is simply skipped if the command is unavailable.

        Must be called at every replan step because the wrist-camera extrinsic
        changes as the EEF moves.
        """
        try:
            self._camera_params = env.get_camera_params(
                camera_names=self._camera_names,
                camera_height=self._camera_height,
                camera_width=self._camera_width,
            )
        except Exception as exc:
            logging.debug(f"[ChunkProjector] Could not fetch camera params: {exc}")
            self._camera_params = {}

    def set_camera_params(self, params: dict | None) -> None:
        """Use camera matrices from somewhere other than a live env.

        ``params`` maps camera name to ``{"intrinsic", "extrinsic"}`` in the
        same convention :meth:`fetch_params` returns; a recording carries
        exactly this dict per timestep. Cameras missing from it are simply
        not drawn on.
        """
        self._camera_params = dict(params or {})

    def update(self, predicted_action_chunk_states: np.ndarray) -> None:
        """Project all action chunks to every known camera and cache the result.

        Args:
            predicted_action_chunk_states: ``[N, T, D]`` or ``[T, D]`` state
                array where the first 3 dimensions are EEF XYZ in world frame
                and the last dimension is the gripper state.
        """
        if predicted_action_chunk_states is None:
            self._projected = {}
            return

        states = np.asarray(predicted_action_chunk_states)
        if states.ndim == 2:  # [T, D] legacy → [1, T, D]
            states = states[np.newaxis]
        if states.ndim != 3 or states.shape[0] == 0:
            self._projected = {}
            return

        lo, hi = self._eef_pos_slice
        positions = states[:, :, lo:hi]  # [N, T, 3]
        if self._gripper_dim is not None:
            gripper = states[:, :, self._gripper_dim]  # [N, T] – scalar gripper state
        else:
            gripper = np.zeros(states.shape[:2], dtype=np.float32)
        self._chunk_length = states.shape[1]

        self._projected = {}
        for cam_name, params in self._camera_params.items():
            K = np.asarray(params["intrinsic"], dtype=np.float64)
            E = np.asarray(params["extrinsic"], dtype=np.float64)

            # Raw pinhole pixels, no mirror: the frame these land on is not
            # known until draw(), and its width is the only thing the mirror
            # may be computed from.
            self._projected[cam_name] = [
                (*self._project_points(chunk, K, E), gs)
                for chunk, gs in zip(
                    positions, gripper, strict=False
                )  # each chunk: [T, 3]
            ]

    def draw(
        self,
        img: np.ndarray,
        cam_name: str,
        *,
        color: tuple[int, int, int] | None = None,
    ) -> np.ndarray:
        """Return a copy of *img* with projected chunk trajectories overlaid.

        Trajectory segments are coloured by gripper state:
        green = open, red = closed.  The overlay is blended at ``self._alpha``
        opacity.  A small square marks the end of each chunk.

        Args:
            img:      RGB uint8 ``[H, W, 3]`` image.
            cam_name: Camera name key matching one returned by ``fetch_params``.
            color:    Draw every segment in this one RGB colour instead of the
                      gripper gradient. Used to put a second, differently
                      coloured chunk (e.g. the unguided policy output) on a
                      frame that already carries the executed one.

        Returns:
            Annotated RGB uint8 copy.  Returns *img* unchanged if no
            projection data is available or if ``cv2`` is not installed.
        """
        projected = self._projected.get(cam_name)
        if not projected:
            return img
        try:
            import cv2
        except ImportError:
            logging.debug("[ChunkProjector] cv2 not available; skipping overlay.")
            return img

        out = img.copy()
        overlay = out.copy()
        H, W = img.shape[:2]

        for chunk_idx, (raw_2d, valid, gs) in enumerate(projected):
            pts_2d = self._to_frame(raw_2d, cam_name, W)
            in_bounds = (
                valid
                & (pts_2d[:, 0] >= 0)
                & (pts_2d[:, 0] < W)
                & (pts_2d[:, 1] >= 0)
                & (pts_2d[:, 1] < H)
            )

            T = len(pts_2d)
            washed = chunk_idx > 0  # non-first chunks get grayish colors

            # Draw each segment coloured by the gripper value at its start.
            for t in range(T - 1):
                if not (in_bounds[t] and in_bounds[t + 1]):
                    continue
                seg_color = (
                    color
                    if color is not None
                    else self._gripper_color(gs[t], washed=washed)
                )
                cv2.line(
                    overlay,
                    tuple(pts_2d[t]),
                    tuple(pts_2d[t + 1]),
                    seg_color,
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )

            # Small filled square at the end of the chunk.
            visible_indices = np.where(in_bounds)[0]
            if len(visible_indices) > 0:
                last_idx = visible_indices[-1]
                end_color = (
                    color
                    if color is not None
                    else self._gripper_color(gs[last_idx], washed=washed)
                )
                lx, ly = int(pts_2d[last_idx, 0]), int(pts_2d[last_idx, 1])
                sq = 4
                cv2.rectangle(
                    overlay,
                    (lx - sq, ly - sq),
                    (lx + sq, ly + sq),
                    end_color,
                    thickness=-1,
                )

        # Blend the annotated overlay with the original at the configured opacity.
        cv2.addWeighted(overlay, self._alpha, out, 1.0 - self._alpha, 0, out)
        return out

    @property
    def chunk_length(self) -> int | None:
        """Number of steps in each action chunk, or ``None`` if not yet updated."""
        return self._chunk_length

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gripper_color(
        self, gs: float, *, washed: bool = False
    ) -> tuple[int, int, int]:
        """Interpolate between red (gs <= _MIN_GRIPPER_VALUE) and green (gs >= _MAX_GRIPPER_VALUE).

        The mapping is linear and saturates outside [_MIN_GRIPPER_VALUE, _MAX_GRIPPER_VALUE].
        ``t = 0`` → fully red, ``t = 1`` → fully green.
        If *washed* is True, blends the result toward gray for a subdued look.
        """
        span = self._MAX_GRIPPER_VALUE - self._MIN_GRIPPER_VALUE
        if not np.isfinite(gs):
            gs = self._MIN_GRIPPER_VALUE
        t = float(np.clip((gs - self._MIN_GRIPPER_VALUE) / span, 0.0, 1.0))  # 0.0 … 1.0
        r = int(self._COLOR_CLOSE[0] * (1 - t) + self._COLOR_OPEN[0] * t)
        g = int(self._COLOR_CLOSE[1] * (1 - t) + self._COLOR_OPEN[1] * t)
        b = int(self._COLOR_CLOSE[2] * (1 - t) + self._COLOR_OPEN[2] * t)
        if washed:
            # Blend 40% original + 60% gray(140) for a grayish wash-down
            r = int(r * 0.4 + 140)
            g = int(g * 0.4 + 140)
            b = int(b * 0.4 + 140)
        return (r, g, b)

    def _to_frame(self, pts_2d: np.ndarray, cam_name: str, width: int) -> np.ndarray:
        """Map raw pinhole pixels onto the frame *cam_name* actually displays.

        A no-op for a camera that publishes upright frames. For one whose
        frames are pre-flipped, u is mirrored about the frame's own width --
        measured from the frame, never from the requested render size.
        """
        if not self._mirrored.get(cam_name, False):
            return pts_2d
        mirrored = pts_2d.copy()
        mirrored[:, 0] = (width - 1) - mirrored[:, 0]
        return mirrored

    @staticmethod
    def _project_points(
        points_3d: np.ndarray,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world-frame points to raw pinhole pixel coordinates.

        Deliberately ``preprocessed=False``: the per-camera mirror belongs to
        the frame being drawn on and is applied in :meth:`_to_frame`. The size
        arguments are unused by the helper in that mode, and passing them here
        is what let a width/height swap sit undetected -- it is invisible on a
        simulator's square frames.
        """
        return project_points_to_pixels(
            points_3d,
            intrinsic,
            extrinsic,
            image_height=0,
            image_width=0,
            preprocessed=False,
        )
