"""PerceptionConfig — declarative pipeline shape."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .stages.base import Stage


# Stage dataclass fields that are infrastructure (service getters, cached flags,
# the stage's own name) rather than user-tunable parameters — hidden from the
# summary to keep it focused on knobs.
_HIDDEN_STAGE_FIELDS = {"name", "runs_on_track"}


def _is_stage_param_field(f) -> bool:
    if f.name in _HIDDEN_STAGE_FIELDS:
        return False
    if f.name.startswith("_"):
        return False
    if f.name.endswith("_getter"):
        return False
    return True


class _SummaryServices:
    """Minimal stand-in for ``_Services`` used only when formatting the summary.

    Stage factories only *store* the service getters — they don't invoke them
    at build time — so returning ``None`` here is safe and avoids triggering
    model loads just to print the config.
    """

    def __init__(self, config: "PerceptionConfig") -> None:
        self.config = config

    def _noop(self, *_a, **_k):
        return None

    get_sam3 = _noop
    get_sam2 = _noop
    get_clip = _noop
    get_dino = _noop
    get_vlm_client = _noop
    get_box_detector = _noop
    get_reference_bank = _noop


DEFAULT_ORDER: list[str] = ["sam3", "filter.overlap"]


@dataclass
class PerceptionConfig:
    # Pipeline in execution order. Strings are resolved against the stage
    # registry at build time using params from `_params[name]`. `Stage`
    # instances are used directly. Duplicates allowed — repeated names share
    # `_params[name]`; for per-occurrence params, pass a pre-built instance.
    _order: "list[str | Stage]" = field(default_factory=lambda: list(DEFAULT_ORDER))
    _params: dict[str, dict[str, Any]] = field(default_factory=dict)

    device: str = "cuda"
    sam3_model: str = "facebook/sam3"
    sam2_model: str = "facebook/sam2-hiera-tiny"
    # Which video-tracker backend propagates masks between frames.
    # `"sam3"` reuses the same SAM3 weights (text + tracking); `"sam2"` loads
    # a smaller SAM2 video model that tracks given masks much faster but
    # cannot segment from text — initial detection still goes through SAM3.
    tracker_backend: Literal["sam3", "sam2"] = "sam3"
    clip_model: str = "openai/clip-vit-base-patch16"
    dino_model: str = "facebook/dinov2-base"
    vlm_model: str = "gpt-5.4"
    # Box/point-proposal detector for the ``detect.gemini`` stage.
    # ``box_detector_backend`` selects the transport:
    #   "gemini"     -> Google native API (``gemini_model``; [gemini] extra)
    #   "openrouter" -> any OpenAI-wire model via OpenRouter
    #                   (``openrouter_model``; OPENROUTER_API_KEY)
    box_detector_backend: Literal["gemini", "openrouter"] = "gemini"
    gemini_model: str = "gemini-robotics-er-1.6-preview"
    gemini_temperature: float | None = None
    gemini_max_objects: int = 25
    gemini_thinking_budget: int = 0
    openrouter_model: str = "google/gemini-3-flash-preview"
    openrouter_base_url: str | None = None
    # Preload every model the configured pipeline uses (plus the tracker) at
    # construction time. Keeps the first `detect()` call's stage timings free
    # of one-shot model-load latency. Set False for tests that patch
    # `LazyModel._instance` directly.
    eager_load: bool = True

    # SAM3 prompts (mirror of sam3_clip_rerank_eval.py defaults)
    sam3_prompts: list[str] = field(default_factory=lambda: ["<object_names>"])
    sam3_score_threshold: float = 0.15
    sam3_mask_threshold: float = 0.4

    # SAM2 image-segmentation defaults (used by stages/sam2.py). SAM2 has
    # no text-conditioning, so these only matter when the ``sam2`` stage
    # is in the pipeline.
    sam2_points_per_side: int = 32
    sam2_points_per_batch: int = 64
    sam2_pred_iou_thresh: float = 0.7
    sam2_stability_score_thresh: float = 0.92
    sam2_crops_nms_thresh: float = 0.7

    # Default filter thresholds
    default_min_area: int = 10
    default_max_instances: int = 40
    default_bbox_pad: int = 4
    default_clip_mask_bg: bool = True

    # Point tracking (CoTracker, optional dependency).
    # OFF by default — enabling requires installing the `[cotracker]` extras
    # AND passing depth at detect/track time. CoTracker tracks 2D pixel
    # queries seeded from each detection's reconstructed-point pixels.
    enable_point_tracking: bool = False
    cotracker_model: str = "cotracker3_online"
    cotracker_max_points_per_detection: int = 64
    cotracker_min_points_per_detection: int = 8
    cotracker_erode_kernel_size: int = 3
    # On SAM3 drift recovery: "freeze" leaves cotracker untouched (recovered
    # instance keeps its old trail); "reseed_view" resets the whole view's
    # cotracker session and re-enters warm-up with new queries (cotracker
    # online cannot add queries mid-stream, so per-instance reseed isn't
    # possible).
    cotracker_drift_action: Literal["freeze", "reseed_view"] = "freeze"

    def order(self, items: "list[str | Stage]") -> "PerceptionConfig":
        self._order = list(items)
        return self

    def update(self, name: str, **params: Any) -> "PerceptionConfig":
        self._params.setdefault(name, {}).update(params)
        return self

    def validate(
        self,
        registry: set[str],
        *,
        input_fields: set[str] | None = None,
    ) -> None:
        """Check stage names + warn (don't raise) on unmet stage dependencies.

        ``input_fields`` describes what the caller is supplying as input
        (e.g. ``{"rgb", "depth", "camera.intrinsics"}``). When omitted,
        the validator assumes a permissive baseline of every input field
        — useful at config build time, before we know what the user will
        pass at ``detect()`` time. ``OpenPerception`` calls validate()
        again with the *actual* per-call inputs so warnings like
        "reconstruct in pipeline but no depth supplied" still fire.
        """
        from loguru import logger

        for item in self._order:
            if isinstance(item, str) and item not in registry:
                raise ValueError(
                    f"unknown stage {item!r}; registered stages: {sorted(registry)}"
                )

        if input_fields is None:
            input_fields = {
                "rgb",
                "depth",
                "camera.intrinsics",
                "camera.extrinsics",
            }
        available: set[str] = set(input_fields)

        seen_names: list[str] = []
        for item in self._order:
            stage = self._build_for_validation(item, registry)
            if stage is None:
                continue
            stage_name = getattr(stage, "name", str(item))
            requires = tuple(getattr(stage, "requires", ()) or ())
            provides = tuple(getattr(stage, "provides", ()) or ())
            missing = [r for r in requires if r not in available]
            if missing:
                logger.warning(
                    f"[config] stage {stage_name!r} (position {len(seen_names)}) "
                    f"requires {missing!r} but no earlier stage or input "
                    f"provides them; available so far: {sorted(available)}"
                )
            available.update(provides)
            seen_names.append(stage_name)

    def _build_for_validation(self, item, registry: set[str]):
        """Materialize a stage just enough to read its requires/provides.

        Returns the existing instance for pre-built stages, or builds a
        fresh one via the registry using the same ``_SummaryServices``
        stand-in the summary code uses (no model loads triggered).
        """
        from .stages.base import REGISTRY

        if not isinstance(item, str):
            return item
        if item not in registry:
            return None
        services = _SummaryServices(self)
        try:
            return REGISTRY.get(item)(services, **self._params.get(item, {}))
        except Exception:
            # If a stage's factory chokes on the stand-in services, skip
            # validation for that stage rather than failing config build.
            return None

    @property
    def summary(self) -> str:
        from .stages.base import REGISTRY

        services = _SummaryServices(self)

        lines = ["PerceptionConfig"]
        lines.append(f"  device: {self.device}")
        lines.append(
            f"  models: sam3={self.sam3_model} tracker={self.tracker_backend} "
            f"clip={self.clip_model} dino={self.dino_model} vlm={self.vlm_model} "
            f"gemini={self.gemini_model}"
        )
        counts: dict[str, int] = {}
        for item in self._order:
            if isinstance(item, str):
                counts[item] = counts.get(item, 0) + 1
        seen: dict[str, int] = {}
        lines.append(f"  pipeline ({len(self._order)} stage(s)):")
        for i, item in enumerate(self._order):
            if isinstance(item, str):
                if counts[item] > 1:
                    idx = seen.get(item, 0)
                    seen[item] = idx + 1
                    display = f"{item}#{idx}"
                else:
                    display = item
                overrides = self._params.get(item, {})
                stage_obj: Any | None = None
                build_error: str | None = None
                if item in REGISTRY.names():
                    try:
                        stage_obj = REGISTRY.get(item)(services, **overrides)
                    except Exception as exc:
                        build_error = f"{type(exc).__name__}: {exc}"
                lines.append(f"    [{i}] {display}")
                self._append_stage_params(lines, stage_obj, overrides, build_error)
            else:
                inst_name = getattr(item, "name", item.__class__.__name__)
                lines.append(
                    f"    [{i}] {inst_name} <instance: {item.__class__.__name__}>"
                )
                self._append_stage_params(lines, item, {}, None)
        return "\n".join(lines)

    @staticmethod
    def _append_stage_params(
        lines: list[str],
        stage_obj: Any | None,
        overrides: dict[str, Any],
        build_error: str | None,
    ) -> None:
        if stage_obj is not None and is_dataclass(stage_obj):
            params = [f for f in fields(stage_obj) if _is_stage_param_field(f)]
            if not params:
                return
            for f in params:
                value = getattr(stage_obj, f.name)
                marker = " *" if f.name in overrides else ""
                lines.append(f"        {f.name}={value!r}{marker}")
            return
        if build_error is not None:
            lines.append(f"        <could not introspect params: {build_error}>")
            if overrides:
                for k, v in overrides.items():
                    lines.append(f"        {k}={v!r} *")
            return
        if overrides:
            for k, v in overrides.items():
                lines.append(f"        {k}={v!r} *")
