"""Multi-view segmentation client backed by the in-repo ``open_perception`` package.

Wraps :class:`open_perception.OpenPerception` with a VLM-detection pipeline:
Gemini 3 Flash (via OpenRouter) proposes one bounding box per requested object
(``detect.gemini`` grounded), SAM2 segments each box (``sam2`` bbox), and the
masks are size/erode-filtered, depth-reconstructed and fused across
views. Full order
``detect.gemini → sam2 → filter.size → filter.erode →
reconstruct → fuse_views``, ``flip_u`` on both ``reconstruct`` and
``fuse_views`` (set from the env's camera transform — see ``flip_u``), ``primary_guided`` fusion with ``main_camera`` = the env's
primary view. Grounded VLM boxes already carry the query label, so no DINO
reference bank is needed (unlike the earlier SAM3 + DINO-disambiguation path).

Used by :class:`policy_abstraction.evaluation.abstraction.RolloutAbstractor`
when ``perception_mode == REAL_PERCEPTION``. One instance is built
once at startup and reset between episodes; tracker backend is SAM2 so
per-frame propagation is cheap (frame-0 detection runs the full VLM→SAM2
pipeline).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np

from policy_abstraction import PROJECT_ROOT

# open_perception ships in-repo with its own pyproject; it is not installed
# into the main workspace. Add its src to sys.path before the import so this
# module is usable from the main eval venv. Same pattern as
# scripts/ablations/perception/op_sam3_rerank_eval.py.
_OP_SRC = PROJECT_ROOT / "packages" / "open_perception" / "src"
if str(_OP_SRC) not in sys.path:
    sys.path.insert(0, str(_OP_SRC))

from open_perception import (  # noqa: E402
    InferenceOptions,
    OpenPerception,
    PerceptionConfig,
    TrackedFrame,
    overlay_masks,
    render_debug_image,
    render_segmentation_video,
)
from open_perception.utils.images import normalize_name  # noqa: E402
from open_perception.visualization import (  # noqa: E402
    _query_color_map,
    _render_snapshot_frame,
    draw_label,
)

DEFAULT_REFERENCE_HDF5 = PROJECT_ROOT / "datasets" / "all_objects.hdf5"


DEFAULT_VLM_MODEL = "google/gemini-3-flash-preview"


def _build_config(
    main_camera: int = 0,
    *,
    detector_backend: Literal["gemini", "openrouter"] = "openrouter",
    detector_model: str | None = None,
    flip_u: bool = True,
) -> PerceptionConfig:
    """Default real-perception pipeline: VLM box detection → SAM2 segmentation.

    A vision-language model proposes one bounding box per requested object
    (``detect.gemini`` in *grounded* mode), SAM2 turns each proposed box into a
    precise mask (``sam2`` in *bbox* mode), and the masks are size/erode-filtered,
    back-projected to 3D with depth (``reconstruct``) and aligned across cameras
    (``fuse_views``). Grounded VLM boxes already carry the query label, so no DINO
    reference bank is needed (unlike the earlier SAM3 + DINO-disambiguation path).

    *detector_backend* picks the box-proposal transport:

    - ``"openrouter"`` (default) — any OpenAI-wire model (``OPENROUTER_API_KEY``),
      e.g. ``google/gemini-3-flash-preview``.
    - ``"gemini"`` — Google's native API (``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``),
      required for the robotics-ER models (``gemini-robotics-er-1.6-preview``),
      which are NOT served on OpenRouter.

    *detector_model* overrides the model id for the chosen backend (``None`` →
    the backend's config default). *main_camera* is the index of the primary
    view; ``fuse_views`` re-aliases secondary views to it, so it must match
    :attr:`EnvSpec.primary_camera_idx`.
    """
    cfg = PerceptionConfig().order(
        [
            "detect.gemini",
            "sam2",
            "filter.size",
            "filter.erode",
            "reconstruct",
            "fuse_views",
        ]
    )
    # Per-frame mask propagation goes through the cheap SAM2 video tracker;
    # the first frame's detect() runs the full VLM→SAM2 detection pipeline.
    cfg.tracker_backend = "sam2"
    cfg.box_detector_backend = detector_backend
    if detector_model is not None:
        if detector_backend == "openrouter":
            cfg.openrouter_model = detector_model
        else:
            cfg.gemini_model = detector_model
    # Grounded: locate exactly the caller's queries (one box per object), then
    # let SAM2 segment each proposed box.
    cfg.update("detect.gemini", stage="grounded", proposal="box")
    cfg.update("sam2", mode="bbox")
    # `flip_u` compensates for the abstractor pre-flipping the frames: LIBERO
    # (and RoboCasa/CALVIN) RGB+depth arrive 180°-rotated and are `[::-1, ::-1]`d
    # before they get here, so back-projection has to mirror u to land on the
    # right pixels. Envs whose CameraSpec.image_transform is "none" — the real
    # UR5e among them — must NOT flip: doing so mirrors every reconstructed
    # object about the optical axis, which measured ~0.8 m of displacement on
    # the UR5 cell. Hence the caller decides, from the env spec.
    cfg.update("reconstruct", flip_u=flip_u)
    cfg.update(
        "fuse_views", mode="primary_guided", main_camera=main_camera, flip_u=flip_u
    )
    return cfg


class OpenPerceptionClient:
    """Multi-view callable wrapping :class:`OpenPerception`.

    Lifecycle: construct once at startup (loads SAM3/SAM2/DINO with
    ``eager_load=True``); call :meth:`reset` at episode start; call
    ``__call__`` once per replan with the latest per-view RGB/depth/cameras.
    The first ``__call__`` after a ``reset`` runs ``detect()`` (seeds the
    trackers); subsequent calls run ``track()``.
    """

    def __init__(
        self,
        reference_hdf5: str | Path | None = None,
        device: str | None = None,
        *,
        primary_view_idx: int = 0,
        debug_perception: bool = False,
        detector_backend: Literal["gemini", "openrouter"] = "openrouter",
        detector_model: str | None = None,
        vlm_model: str | None = None,
        flip_u: bool = True,
    ) -> None:
        # Resolve the box-detector model. ``detector_model`` wins; otherwise the
        # legacy ``vlm_model`` arg seeds the OpenRouter model (back-compat), and
        # the native backend falls back to its config default (ER-1.6).
        if detector_model is None and detector_backend == "openrouter":
            detector_model = vlm_model if vlm_model is not None else DEFAULT_VLM_MODEL
        cfg = _build_config(
            main_camera=primary_view_idx,
            detector_backend=detector_backend,
            detector_model=detector_model,
            flip_u=flip_u,
        )
        if device is not None:
            cfg.device = device
        self._primary_view_idx = primary_view_idx
        self._debug_perception = debug_perception

        # Fail fast with a clear message if the chosen backend's key is absent.
        import os

        if detector_backend == "openrouter":
            if not (
                os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
            ):
                raise RuntimeError(
                    "detector_backend='openrouter' requires OPENROUTER_API_KEY "
                    "(or OPENAI_API_KEY) in the environment (e.g. via .env)."
                )
        elif not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise RuntimeError(
                "detector_backend='gemini' requires GEMINI_API_KEY or "
                "GOOGLE_API_KEY in the environment. The robotics-ER models are "
                "native-only (not on OpenRouter); uncomment the key in .env, or "
                "use --perception.detector-backend openrouter with an "
                "OpenRouter-served model."
            )

        # The detect.gemini → sam2 pipeline doesn't use DINO disambiguation, so
        # no reference bank is required. ``reference_hdf5`` is accepted for
        # backward compatibility but ignored.
        self._perc = OpenPerception(cfg)
        self._opts = InferenceOptions(top_k_per_query=None, match_threshold=0.2)
        self._first_call = True
        self._last_frame: TrackedFrame | None = None
        # First detect() captures the full per-view stage history. Cache that
        # frame so render_history_grids always renders the detect output, not
        # the (mostly empty) track output of subsequent calls.
        self._first_detect_frame: TrackedFrame | None = None
        self._first_detect_grids_emitted: bool = False
        self._first_detect_debug_emitted: bool = False

    def reset(self) -> None:
        """Clear tracker state and force the next call to re-``detect``."""
        self._perc.reset()
        self._first_call = True
        self._last_frame = None
        self._first_detect_frame = None
        self._first_detect_grids_emitted = False
        self._first_detect_debug_emitted = False

    @property
    def last_frame(self) -> TrackedFrame | None:
        """The :class:`TrackedFrame` from the most recent ``__call__``."""
        return self._last_frame

    @property
    def history(self):
        """Full :class:`TrackHistory` accumulated since the last :meth:`reset`."""
        return self._perc.history

    def __call__(
        self,
        rgbs: list[np.ndarray],
        depths: list[np.ndarray],
        cameras: list[Any],
        object_names: list[str],
    ) -> list[dict[int, tuple[str, np.ndarray]]]:
        """Run detect (first call) or track (subsequent), demerge masks per view.

        Returns one ``{instance_id: (query_name, [H, W] bool mask)}`` dict per
        view. Multiple entries can share a query name when the same query
        matched several physical instances; ``instance_id`` is the perception
        tracker's stable per-object identifier (matches the IDs shown in
        :meth:`render_overlays` / ``fuse_views`` snapshots).

        Caller queries are normalized to open_perception's canonical form
        (``normalize_name``: digits stripped, ``_`` → space) before running
        detect/track. Without this, seg-name-style queries like
        ``"alphabet_soup_1"`` miss every entry in the canonical reference bank
        (keyed e.g. ``"alphabet soup"``), and ``disambiguate.dino``'s
        ``fallback_to_text`` route picks visually-similar wrong objects
        (cream_cheese for alphabet_soup, etc.). Returned entries carry the
        caller's original query strings so downstream lookups still work.
        """
        # Build canonical query list and a canonical→original lookup so the
        # SAM3/DINO/CLIP stages see names that match the reference bank, while
        # the result dict is still keyed by what the caller asked for.
        canonical_queries: list[str] = []
        canonical_to_orig: dict[str, str] = {}
        for orig in object_names:
            canon = normalize_name(orig) or orig
            if canon not in canonical_to_orig:
                canonical_to_orig[canon] = orig
                canonical_queries.append(canon)

        if self._first_call:
            out = self._perc.detect(
                canonical_queries,
                rgbs,
                depth=depths,
                camera=cameras,
                options=self._opts,
            )
            self._first_call = False
            just_detected = True
        else:
            out = self._perc.track(
                rgbs,
                depth=depths,
                camera=cameras,
                options=self._opts,
            )
            just_detected = False

        n_views = len(rgbs)
        result: list[dict[int, tuple[str, np.ndarray]]] = [{} for _ in range(n_views)]
        # out.detections is dict[canonical_query, list[Detection]] flattened
        # across views. We expose every detection (multi-instance), keyed by
        # its perception ``instance_id`` so callers can correlate masks across
        # views and frames. ``fuse_views`` aliases secondary-view tracker IDs
        # to the primary view (agentview, main_camera=0); to defend against
        # the rare case where it fails (or runs before alias propagation), we
        # drop any secondary-view detection whose ``instance_id`` doesn't
        # match *some* primary-view detection for the same query.
        primary_idx = self._primary_view_idx
        for canon_query, dets in out.detections.items():
            orig_query = canonical_to_orig.get(canon_query)
            if orig_query is None:
                continue
            primary_ids = {
                d.instance_id
                for d in dets
                if d.view_idx == primary_idx and d.instance_id is not None
            }
            for det in dets:
                v = det.view_idx
                if not (0 <= v < n_views) or det.mask is None:
                    continue
                if det.instance_id is None:
                    continue
                if (
                    v != primary_idx
                    and primary_ids
                    and det.instance_id not in primary_ids
                ):
                    continue
                result[v][det.instance_id] = (orig_query, det.mask.astype(bool))

        # Surface the latest TrackedFrame for visualization/recording. The
        # underlying TrackHistory keeps every frame; we just expose the tail.
        history = self._perc.history
        self._last_frame = history[-1] if history else None
        if just_detected and self._last_frame is not None:
            self._first_detect_frame = self._last_frame
        return result

    # ------------------------------------------------------------------
    # Visualization / debug surfaces
    # ------------------------------------------------------------------

    def render_overlays(
        self,
        *,
        alpha: float = 0.5,
        show_id: bool = True,
        show_label: bool = True,
    ) -> list[np.ndarray]:
        """Per-view RGB overlay of the latest detections (one image per view).

        Renders every detection ``open_perception`` returned — keyed by
        ``instance_id`` so multi-instance queries (``assign`` runs with
        ``top_k=-1``) get one mask each, matching what :meth:`__call__`
        exposes downstream. Empty list if no call has happened since the
        last :meth:`reset`.
        """
        frame = self._last_frame
        if frame is None:
            return []
        out: list[np.ndarray] = []
        for v, rgb in enumerate(frame.rgbs):
            mask_dict: dict[int, np.ndarray] = {}
            ann_dict: dict[int, str] = {}
            for q, dets in frame.detections.items():
                for d in dets:
                    if d.view_idx != v or d.instance_id is None or d.mask is None:
                        continue
                    mask_dict[d.instance_id] = d.mask
                    parts: list[str] = []
                    if show_id:
                        parts.append(f"[{d.instance_id}]")
                    if show_label:
                        parts.append(q)
                    ann_dict[d.instance_id] = " ".join(parts) if parts else ""
            out.append(overlay_masks(rgb, mask_dict, annotations=ann_dict, alpha=alpha))
        return out

    def render_history_grids(
        self,
        *,
        alpha: float = 0.5,
        show_id: bool = True,
        show_label: bool = True,
    ) -> list[np.ndarray]:
        """Per-view per-stage overlay grid for the first ``detect()`` call.

        Concatenates the per-pipeline-stage snapshots horizontally so callers
        can see the mask evolution through ``sam3 → filter.* → assign →
        reconstruct → fuse_views`` at a glance. Renders only the first detect
        frame (subsequent ``track()`` calls don't run the per-view stages, so
        their stage history would be near-empty and confusing). Returns the
        rendered grids on the first call after reset; ``[]`` thereafter so
        downstream consumers (rerun, recorder) only emit it once per episode.

        Delegates to ``open_perception.visualization._render_snapshot_frame``
        — the same helper backing ``render_history_gif`` — so early stages
        (``sam3``, ``filter.*``) that only populate ``snap.instances`` still
        get visible masks rather than rendering as bare RGB.
        """
        if self._first_detect_grids_emitted:
            return []
        frame = self._first_detect_frame
        if frame is None:
            return []
        out: list[np.ndarray] = []
        for v, rgb in enumerate(frame.rgbs):
            view_snaps = [s for s in frame.stage_history if s.view_idx == v]
            if not view_snaps:
                out.append(rgb)
                continue
            query_colors = _query_color_map(view_snaps, None)
            tiles = [draw_label(rgb, "input")]
            for snap in view_snaps:
                tiles.append(
                    _render_snapshot_frame(
                        rgb,
                        snap,
                        alpha=alpha,
                        query_colors=query_colors,
                        show_id=show_id,
                        show_label=show_label,
                    )
                )
            out.append(np.concatenate(tiles, axis=1))
        self._first_detect_grids_emitted = True
        return out

    def render_debug_images(
        self,
        *,
        alpha: float = 0.5,
    ) -> list[np.ndarray]:
        """Per-view per-stage debug panel for the first ``detect()`` frame.

        Wraps :func:`open_perception.render_debug_image`, which writes a PNG to
        disk; we render to a tempfile and load the bytes back so callers can
        log the panel as an in-memory image (e.g. to a rerun ``Image`` view).
        Returns ``[]`` when ``debug_perception`` is disabled, when no detect
        frame has been captured yet, or when none of the views have stage
        snapshots.
        """
        if not self._debug_perception or self._first_detect_debug_emitted:
            return []
        frame = self._first_detect_frame
        if frame is None:
            return []

        import tempfile

        from PIL import Image as _PILImage

        out: list[np.ndarray] = []
        for v, rgb in enumerate(frame.rgbs):
            view_snaps = [s for s in frame.stage_history if s.view_idx == v]
            if not view_snaps:
                continue
            with tempfile.NamedTemporaryFile(suffix=".png", delete=True) as tmp:
                render_debug_image(
                    rgb,
                    view_snaps,
                    tmp.name,
                    config=self._perc.config,
                    alpha=alpha,
                    title=f"detect frame {frame.frame_idx} view {v}",
                )
                with _PILImage.open(tmp.name) as im:
                    out.append(np.asarray(im.convert("RGB")))
        self._first_detect_debug_emitted = True
        return out

    def last_timings_md(self) -> str:
        """Markdown-formatted per-stage timings of the latest call (for rerun text panel)."""
        frame = self._last_frame
        if frame is None or not frame.stats.timings_ms:
            return "_no perception call yet_"
        lines = ["| stage | ms |", "|---|---:|"]
        for stage, ms in sorted(frame.stats.timings_ms.items()):
            lines.append(f"| `{stage}` | {ms:.1f} |")
        return "\n".join(lines)

    def save_segmentation_video(self, path: str | Path) -> Path | None:
        """Write a single-mp4 of every frame's per-view detection overlay.

        Returns the output path on success, or ``None`` if no perception
        frames have been recorded yet (e.g. episode failed before the first
        replan).
        """
        if not self._perc.history:
            return None
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        return render_segmentation_video(
            self._perc.history, out, show_id=True, show_fps=False
        )


__all__ = ["OpenPerceptionClient", "DEFAULT_REFERENCE_HDF5"]
