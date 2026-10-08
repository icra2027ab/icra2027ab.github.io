"""VisualizationHandler: centralized image overlay and annotation orchestrator."""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from policy_abstraction.envs.spec import CameraSpec, EnvSpec
from policy_abstraction.evaluation.abstraction import PerceptionMode
from policy_abstraction.visualization.chunk_projector import ChunkProjector
from policy_abstraction.visualization.constraint_panel import render_constraint_panel
from policy_abstraction.visualization.frame_bundle import FrameBundle

_HEADER_HEIGHT = 30
_HEADER_BG = (40, 40, 40)
_HEADER_FG = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.7
_FONT_THICKNESS = 2


def _add_header(img: np.ndarray, text: str) -> np.ndarray:
    """Prepend a small text header bar above *img*."""
    h, w = img.shape[:2]
    header = np.full((_HEADER_HEIGHT, w, 3), _HEADER_BG, dtype=np.uint8)
    cv2.putText(
        header,
        text,
        (8, _HEADER_HEIGHT - 8),
        _FONT,
        _FONT_SCALE,
        _HEADER_FG,
        _FONT_THICKNESS,
        cv2.LINE_AA,
    )
    return np.concatenate([header, img], axis=0)


class VisualizationHandler:
    """Centralizes all image overlay and annotation generation.

    Owns the :class:`ChunkProjector` and calls annotation methods on the
    :class:`~policy_abstraction.evaluation.abstraction.RolloutAbstractor`
    when both a result and an abstractor are provided.

    Annotation is **lazy**: when *enabled* is ``False`` (no visualizer or video
    saver active), :meth:`render` returns a minimal :class:`FrameBundle` with
    only the raw camera frames — no drawing code runs.

    The handler is camera-count agnostic: it draws chunk overlays for every
    camera the projector has intrinsics for and stores them in
    :attr:`FrameBundle.chunk_overlays` keyed by camera name. The legacy 2×2
    replay grid generalises to ``len(cameras) × {1,2} (live + reference)``.
    """

    def __init__(
        self,
        camera_height: int,
        camera_width: int,
        alpha: float = 0.65,
        *,
        env_spec: EnvSpec,
        enabled: bool = True,
    ) -> None:
        self._env_spec = env_spec
        self._primary_camera: CameraSpec = env_spec.primary_camera
        self._wrist_camera: CameraSpec | None = env_spec.wrist_camera
        # The projector consumes *predicted states* (from the dynamics model),
        # not raw actions. For the Cartesian-action envs the two layouts
        # coincide, so the action layout's slices are the right values; for
        # joint-space envs (UR5e) the action layout carries no EEF slice and we
        # fall back to the canonical robot-state layout —
        # ``[pos(3), axis-angle(3), aperture(1)]`` — which every env publishes.
        _layout = env_spec.robot.action_layout
        self._projector = ChunkProjector(
            camera_height=camera_height,
            camera_width=camera_width,
            alpha=alpha,
            cameras=env_spec.cameras,
            eef_pos_slice=_layout.eef_pos_slice or (0, 3),
            gripper_dim=_layout.gripper_dim,
        )
        self._enabled = enabled
        self._constraints: list = []

    def set_constraints(self, constraints: list | None) -> None:
        """Register the constraint list used to render the constraint panel.

        Must be called once after retrieval/abstraction setup so that the
        constraint panel can be grouped by ``subtraj_idx``.
        """
        self._constraints = list(constraints) if constraints else []

    # ------------------------------------------------------------------
    # Camera-parameter management
    # ------------------------------------------------------------------

    def fetch_camera_params(self, env) -> None:
        """Fetch camera intrinsic/extrinsic matrices from *env*.

        Must be called every replan step because the wrist-camera extrinsic
        changes as the EEF moves.
        """
        self._projector.fetch_params(env)

    def update_projections(self, predicted_action_chunk_states: np.ndarray) -> None:
        """Project action-chunk EEF trajectories to all known cameras."""
        self._projector.update(predicted_action_chunk_states)

    @property
    def camera_params(self) -> dict:
        """Expose the underlying camera parameter dict (name → intrinsic/extrinsic)."""
        return self._projector._camera_params

    # ------------------------------------------------------------------
    # Main rendering entry point
    # ------------------------------------------------------------------

    def render(
        self,
        images: dict[str, np.ndarray],
        *,
        obs: dict | None = None,
        camera_params: dict | None = None,
        abstraction_result=None,
        abstractor=None,
    ) -> FrameBundle:
        """Produce a :class:`FrameBundle` for one timestep.

        ``images`` maps camera name → upright RGB ``[H, W, 3]`` uint8.
        Every key must be a valid camera in the env spec.
        """
        primary_name = self._primary_camera.name
        if primary_name not in images:
            raise KeyError(
                f"render() missing primary camera image {primary_name!r}; "
                f"got keys {list(images)}"
            )

        bundle = FrameBundle(images=dict(images), primary_name=primary_name)

        if not self._enabled:
            return bundle

        # --- Per-camera chunk overlays -----------------------------------
        for cam in self._env_spec.cameras:
            if cam.name not in images:
                continue
            try:
                bundle.chunk_overlays[cam.name] = self._projector.draw(
                    images[cam.name], cam.name
                )
            except Exception as exc:
                logger.debug(
                    f"[VisualizationHandler] chunk projector skipped {cam.name!r}: {exc}"
                )

        # Default the annotated live view to the primary chunk overlay so the
        # "Agentview (annotated)" view always shows action chunks even when
        # no abstractor/guidance is active. The abstractor branch below
        # replaces this with a chunk+annotation composite when available.
        primary_overlay = bundle.chunk_overlays.get(primary_name)
        if primary_overlay is not None:
            bundle.live_image_annotated = primary_overlay

        # --- Abstraction annotation (lazy) -------------------------------
        if abstraction_result is not None and abstractor is not None:
            try:
                ref_primary, ref_wrist = abstractor.draw_annotated_ref_image(
                    abstraction_result.aligned_t,
                    abstraction_result.distance_errors,
                    abstraction_result.current_subtraj_idx,
                    chunk_length=self._projector.chunk_length,
                )
                if ref_primary is not None:
                    bundle.ref_images[primary_name] = ref_primary
                    bundle.ref_image_annotated = ref_primary
                if ref_wrist is not None and self._wrist_camera is not None:
                    bundle.ref_images[self._wrist_camera.name] = ref_wrist
            except Exception as exc:
                logger.debug(
                    f"[VisualizationHandler] draw_annotated_ref_image failed: {exc}"
                )

            if obs is not None and camera_params is not None:
                try:
                    seg: np.ndarray | None = None
                    primary = self._primary_camera
                    if abstractor._perception_mode == PerceptionMode.REAL_PERCEPTION:
                        seg_pred = abstractor._latest_agentview_predicted_seg
                        if seg_pred is not None:
                            seg = np.ascontiguousarray(seg_pred)
                    else:
                        seg_raw = (
                            obs.get(primary.segmentation_key)
                            if primary.segmentation_key is not None
                            else None
                        )
                        if seg_raw is not None:
                            seg = np.ascontiguousarray(
                                primary.apply_image_transform(np.asarray(seg_raw))
                            )

                    cam_info = camera_params.get(primary.name, {})
                    K = cam_info.get("intrinsic")
                    E = cam_info.get("extrinsic")
                    ee_pos = np.asarray(
                        obs.get(self._env_spec.robot.eef_pos_key, [0.0, 0.0, 0.0])
                    )
                    if seg is not None and K is not None and E is not None:
                        # Annotate on top of the chunk overlay when available
                        # so users see both action chunks and abstraction
                        # annotations together; fall back to the raw frame.
                        annotated_source = bundle.chunk_overlays.get(
                            primary.name, images[primary.name]
                        )
                        annotated = abstractor.draw_annotated_live_image(
                            annotated_source,
                            seg,
                            np.asarray(K),
                            np.asarray(E),
                            ee_pos,
                            abstraction_result,
                        )
                        if annotated is not None:
                            bundle.live_image_annotated = annotated
                except Exception as exc:
                    logger.debug(
                        f"[VisualizationHandler] draw_annotated_live_image failed: {exc}"
                    )

        # --- Constraint-satisfaction panel -------------------------------
        if (
            abstraction_result is not None
            and abstraction_result.constraint_costs
            and self._constraints
        ):
            try:
                bundle.constraint_panel = render_constraint_panel(
                    abstraction_result.constraint_costs, self._constraints
                )
            except Exception as exc:
                logger.debug(
                    f"[VisualizationHandler] render_constraint_panel failed: {exc}"
                )

        # --- Compose combined_replay (video frame) -----------------------
        bundle.combined_replay = self._compose_replay(bundle)

        return bundle

    # ------------------------------------------------------------------
    # Video-frame composition
    # ------------------------------------------------------------------

    def _compose_replay(self, bundle: FrameBundle) -> np.ndarray | None:
        """Build the combined replay frame with annotated images.

        Layout (cameras tiled left-to-right; reference row appears below the
        live row only when ref frames are available; alignment plot and
        constraint panel are appended on the right when available)::

            ┌─────────┬─────────┬─────────┐──────────┬───────────┐
            │  Live   │  Live   │  Live   │          │           │
            │  cam-0  │  cam-1  │  cam-N  │ Alignment│ Constraint│
            ├─────────┼─────────┼─────────┤   Plot   │   Panel   │
            │  Ref    │  Ref    │  Ref    │          │           │
            │  cam-0  │  cam-1  │  cam-N  │          │           │
            └─────────┴─────────┴─────────┘──────────┴───────────┘

        Falls back to a single row when no retrieved frames are available.
        The constraint panel is omitted when no constraints are active.
        """
        cams: list[CameraSpec] = [
            c for c in self._env_spec.cameras if c.name in bundle.images
        ]
        if not cams:
            return None

        # Pick the largest source image as the per-tile target so we don't
        # upsample tiny wrist frames unnecessarily.
        primary_img = bundle.primary_image
        h, w = primary_img.shape[:2]

        live_tiles: list[np.ndarray] = []
        for cam in cams:
            tile = bundle.chunk_overlays.get(cam.name, bundle.images[cam.name])
            if (
                cam.name == bundle.primary_name
                and bundle.live_image_annotated is not None
            ):
                tile = bundle.live_image_annotated
            if tile.shape[:2] != (h, w):
                tile = cv2.resize(tile, (w, h))
            label = f"Live {cam.name}"
            if (
                cam.name == bundle.primary_name
                and bundle.live_image_annotated is not None
            ):
                label = f"Live {cam.name} (annotated)"
            live_tiles.append(_add_header(tile, label))
        live_row = np.concatenate(live_tiles, axis=1)

        ref_tiles: list[np.ndarray] = []
        for cam in cams:
            if (
                cam.name == bundle.primary_name
                and bundle.ref_image_annotated is not None
            ):
                tile = bundle.ref_image_annotated
            else:
                tile = bundle.ref_images.get(cam.name)
            if tile is None:
                continue
            tile = cv2.resize(tile, (w, h))
            ref_tiles.append(_add_header(tile, f"Retrieved {cam.name}"))

        if ref_tiles and len(ref_tiles) == len(live_tiles):
            ref_row = np.concatenate(ref_tiles, axis=1)
            combined = np.concatenate([live_row, ref_row], axis=0).astype(np.uint8)
        else:
            combined = live_row.astype(np.uint8)

        if bundle.alignment_plot is not None:
            plot_img = bundle.alignment_plot
            target_h = combined.shape[0]
            plot_w = int(plot_img.shape[1] * target_h / plot_img.shape[0])
            plot_resized = cv2.resize(plot_img, (plot_w, target_h))
            combined = np.concatenate([combined, plot_resized], axis=1).astype(np.uint8)

        # --- Append constraint-satisfaction panel to the right ---
        if bundle.constraint_panel is not None:
            panel_img = bundle.constraint_panel
            target_h = combined.shape[0]
            max_panel_w = 400
            panel_w = max(1, int(panel_img.shape[1] * target_h / panel_img.shape[0]))
            if panel_w <= max_panel_w:
                panel_resized = cv2.resize(panel_img, (panel_w, target_h))
            else:
                # Cap at max width: scale by width and vertically center on a
                # background that matches the panel's dark fill (RGB 30).
                scaled_h = max(
                    1, int(panel_img.shape[0] * max_panel_w / panel_img.shape[1])
                )
                scaled_h = min(scaled_h, target_h)
                scaled = cv2.resize(panel_img, (max_panel_w, scaled_h))
                panel_resized = np.full((target_h, max_panel_w, 3), 30, dtype=np.uint8)
                y0 = (target_h - scaled_h) // 2
                panel_resized[y0 : y0 + scaled_h] = scaled
            combined = np.concatenate([combined, panel_resized], axis=1).astype(
                np.uint8
            )

        return combined
