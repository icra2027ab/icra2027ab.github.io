# open_perception — V2

Open-vocabulary, **multi-camera**, multi-instance segmentation and tracking
built on [SAM3](https://huggingface.co/facebook/sam3). Give it natural-language
queries and one or more RGB(D) views; it returns per-instance masks, fused
3D point clouds, stable `instance_id`s across cameras and time, optional
2D point trajectories from [CoTracker](https://github.com/facebookresearch/co-tracker),
and a `TrackHistory` accumulator that drives the visualization helpers
directly.

`open_perception` is a standalone Python package with no dependency on any
host project — clone the repo, install it, import `open_perception`.

The processing pipeline is built from composable stages
(`sam3` / `sam2`, `filter.*`, `disambiguate.*`, `assign`, `reconstruct`,
`fuse_views`, `encode`). Declare the pipeline you want with
`cfg.order([...])` — anything omitted is excluded.

---

## What's new in V2

- **Native multi-view input.** Pass a list of RGBs / depths / `CameraInfo`s to
  a single `detect()` / `track()`. The pipeline runs each view through the same
  stage list and the `fuse_views` stage aligns instance IDs across cameras.
- **Cross-camera ID fusion.** `fuse_views` in `primary_guided` mode projects
  the main camera's 3D point clouds into the secondaries, matches by occlusion-aware
  IoU, and reuses (or seeds) tracker IDs so the same physical object carries
  the same `instance_id` and colour everywhere.
- **SAM2 as a first-class segmentation stage.** Beyond the SAM2 *tracker* backend
  (`tracker_backend="sam2"`), the new `sam2` stage runs SAM2 image segmentation
  in `auto`, `bbox`, or `point` mode — drop-in replacement for `sam3` when you
  don't need text conditioning or already have prompts.
- **New filter stages.** `filter.erode`, `filter.score`, `filter.top_k`,
  `filter.pointcloud_outlier` round out the filtering palette alongside
  `filter.overlap` and `filter.size`.
- **CoTracker streaming.** Enable `cfg.enable_point_tracking=True` and every
  `Detection` carries a `PointTracks` handle with per-instance 2D trajectories
  through the video (queries seeded from each detection's mask pixels).
- **TrackHistory accumulator.** `perc.history` collects a `TrackedFrame` after
  every detect/track call — visualization helpers
  (`render_pointcloud_orbit_gif`, `render_segmentation_video`) read it directly,
  no manual aggregation.
- **Richer visualization.** New `render_pointcloud_orbit_gif`,
  `render_segmentation_video`, `render_disambiguation_panel`, and
  `render_debug_image` helpers in addition to `overlay_*` / `render_history_gif`.

---

## Installation

Clone the repo on the **`v2`** branch and sync its own environment — no
parent project is required.

```bash
git clone <anonymized-repository-url> open_perception
cd open_perception
uv sync --extra dev --extra examples
```

`--extra dev` pulls pytest; `--extra examples` pulls `imageio[ffmpeg]` for the
mp4-writing examples. Add `--extra vlm` for the optional VLM disambiguator,
or `--extra cotracker` for point tracking.

To use it as a dependency in another project, pin the `v2` branch directly,
e.g. with `uv`:

```bash
uv add "open-perception @ <anonymized-repository-url>"
```

## Quick start

```python
from open_perception import OpenPerception
import imageio.v3 as iio

perc = OpenPerception()                       # sensible defaults
rgb = iio.imread("scene.png")

out = perc.detect(["mug", "plate"], rgb)
for det in out.detections["mug"]:
    ...  # det.mask (H, W) bool, det.score float, det.bbox [y0, x0, y1, x1]
```

### Streaming

```python
for i, out in enumerate(perc.stream(["mug", "plate"], frames)):
    # First frame: full detect. Remaining frames: video-tracker steps.
    print(i, sum(len(lst) for lst in out.detections.values()), "detections")
```

### Multi-camera

Pass lists in lockstep — one entry per view. `fuse_views` aligns ids across
cameras when enabled.

```python
from open_perception import CameraInfo, PerceptionConfig

cfg = PerceptionConfig().order(
    ["sam3", "filter.overlap", "filter.erode", "reconstruct", "fuse_views"]
)
cfg.update("fuse_views", mode="primary_guided", main_camera=0)
perc = OpenPerception(cfg)

rgbs   = [rgb_cam0, rgb_cam1]
depths = [d_cam0,   d_cam1]
cams   = [CameraInfo(K0, E0, name="agentview"),
          CameraInfo(K1, E1, name="eye_in_hand")]
out = perc.detect(["mug", "plate"], rgbs, depth=depths, camera=cams)
```

## Public API

| Entry point | Purpose |
|---|---|
| `OpenPerception(config=None, vlm_client=None, reference=None)` | Construct; eager-loads models for the configured pipeline (set `cfg.eager_load=False` to defer) |
| `.detect(queries, rgb, *, depth=None, camera=None, reference=None, options=None)` | Full pipeline, seeds the tracker(s). Accepts a single view or a list of views. |
| `.track(rgb, *, depth=None, camera=None, options=None)` | Propagate tracked masks one (multi-view) frame |
| `.stream(queries, frames, *, reference=None, options=None)` | Iterator yielding one output per frame |
| `.reset(queries=None)` | Clear tracker state + `history` (all or by name) |
| `.register_reference(refs, space="clip")` | Attach reference images (HDF5 path or dict) for disambiguation |
| `.history` | `TrackHistory` accumulator — pass directly to renderers |
| `AsyncOpenPerception(...)` | Non-blocking wrapper: `.submit_detect`, `.submit_track`, `.latest`, `.wait_next` — see [examples/advanced/async_*.py](examples/advanced/) |

Both `detect()` and `track()` accept either positional args or a
`PerceptionInput` dataclass for full control.

Visualization helpers re-exported from the package:
`overlay_detections`, `overlay_masks`, `draw_label`, `render_history_gif`,
`render_pointcloud_orbit_gif`, `render_segmentation_video`,
`render_disambiguation_panel`, `render_debug_image`, `DEFAULT_COLORS`.

### Output shape

```python
@dataclass
class Detection:
    query: str
    mask: np.ndarray         # (H, W) bool
    score: float             # combined disambiguation score
    view_idx: int            # which camera this detection came from
    bbox: np.ndarray | None  # (4,) [y0, x0, y1, x1]
    instance_id: int | None  # tracker id; stable across frames and (with fuse_views) views
    drifted: bool            # set under drift_policy="surface"
    pointcloud: np.ndarray | None    # (K, 3) world-frame points
    embeddings: dict[str, np.ndarray] | None
    point_tracks: PointTracks | None  # CoTracker 2D trails (when enabled)

class PerceptionOutput:
    detections: dict[str, list[Detection]]    # multi-instance per query, merged across views
    stats: InferenceStats                     # timings, counts, recoveries
    history: list[PipelineSnapshot]           # per-stage snapshots (when record_history=True)

class TrackedFrame:
    frame_idx: int
    queries: list[str]
    rgbs: list[np.ndarray]                    # per-view
    depths: list[np.ndarray | None]
    cameras: list[CameraInfo | None]
    detections: dict[str, list[Detection]]
    stage_history: list[PipelineSnapshot]
    stats: InferenceStats

class TrackHistory:                           # perc.history
    frames: list[TrackedFrame]
    queries: list[str]
    n_views: int
```

## Configuration

`PerceptionConfig` declares the pipeline as an ordered list of stage names.
Two methods shape it:

- `cfg.order([...])` — set the exact pipeline. Strings are resolved against
  the stage registry; anything omitted is excluded. Duplicates are allowed.
- `cfg.update(name, **params)` — merge per-stage params in one line.

Default order (used when `order()` is never called):

```python
["sam3", "filter.overlap"]
```

This emits raw SAM3 masks keyed by resolved prompt (no disambiguation,
no per-query assignment). Add `filter.size`, a `disambiguate.*` stage,
and `assign` when you want one-mask-per-query semantics — see
[examples/04–06](examples/) and [examples/advanced/reorder_stages.py](examples/advanced/reorder_stages.py).

Registered stages:

| Stage | Purpose |
|---|---|
| `vlm_pre` | VLM prunes/hints queries before SAM3 |
| `sam3` | SAM3 text-prompted instance segmentation |
| `sam2` | SAM2 image segmentation — `mode="auto"` / `"bbox"` / `"point"` (no text conditioning) |
| `filter.overlap` | Greedy NMS — drop masks with IoU ≥ threshold against a kept mask |
| `filter.size` | Drop masks below `min_mask_area` px; cap total at `max_instances` |
| `filter.erode` | Morphologically erode masks (denoise mask borders before reconstruction) |
| `filter.score` | Threshold by raw SAM score / per-source score |
| `filter.top_k` | Keep top-K by score per query (or overall) |
| `filter.pointcloud_outlier` | Drop statistical outlier points from each reconstructed cloud |
| `disambiguate.clip` | CLIP image/text similarity (needs reference images — see below) |
| `disambiguate.dino` | DINOv2/3 image similarity (needs reference images — see below) |
| `disambiguate.vlm` | VLM candidate reranker |
| `assign` | Threshold-based multi-match (or top-1 Hungarian) |
| `reconstruct` | RGBD → per-mask world-frame point cloud |
| `fuse_views` | Cross-view ID alignment — `mode="independent"` (default) or `"primary_guided"` |
| `encode` | Attach CLIP/DINO embeddings to outputs |

Usage:

```python
cfg = PerceptionConfig()
cfg.order(
    ["sam3", "filter.overlap", "filter.size",
     "disambiguate.dino", "disambiguate.clip", "assign",
     "reconstruct", "fuse_views"]
)
cfg.update("disambiguate.dino", fallback_to_text=False)
cfg.update("assign", top_k=1, weights={"clip": 1.0, "dino": 0.7})
cfg.update("fuse_views", mode="primary_guided", main_camera=0)
perc = OpenPerception(cfg)

print(cfg.summary)   # multi-line overview of the effective pipeline
```

Minimal pipelines are one-liners — e.g. raw SAM3 masks, no filters, no
disambiguation:

```python
cfg.order(["sam3"])
cfg.update("sam3", prompts=["<object_names>"])  # token expands 1-1 against queries
```

For per-occurrence params (e.g. running `assign` twice with different
weights), pass an instantiated `Stage` in place of the string:

```python
from open_perception.stages.assign import AssignStage

cfg.order([
    "sam3", "filter.overlap", "disambiguate.dino", "assign",
    "disambiguate.vlm", AssignStage(weights={"vlm": 1.0}, top_k=1),
])
```

Duplicate names share `_params` and show up in `stats.stages_run` /
`stats.timings_ms` with a `#<index>` suffix (e.g. `assign#0`, `assign#1`).

### Reference images for image-based disambiguation

`disambiguate.clip` and `disambiguate.dino` score candidate crops against a
**reference bank** of images per query. Register references before calling
`detect()` / `track()`:

```python
perc.register_reference({"white mug": mug_img, "plate": plate_img}, space="clip")
# or from an HDF5 file: perc.register_reference("refs.h5", space="clip")
```

If no bank is registered, `disambiguate.clip` logs a warning and falls back
to CLIP **text** embeddings (`fallback_to_text=True` by default).
`disambiguate.dino` also warns, and with its default `fallback_to_text=False`
its scores will be zero — register DINO references or enable the text
fallback.

### Per-call options

`InferenceOptions` overrides config defaults for a single call:

```python
from open_perception import InferenceOptions

opts = InferenceOptions(
    match_threshold=0.3,          # per-query similarity cutoff
    top_k_per_query=None,         # None = unlimited multi-instance; 1 = default (top-1)
    filter_overlap=False,         # override the stage default for this call
    overlap_iou_threshold=0.85,
    drift_policy="surface",       # "auto" | "surface" (default) | "off"
    drift_iou_threshold=0.2,
    export_embeddings=True,
    record_history=True,          # capture a PipelineSnapshot after each stage
    record_track_history=True,    # append a TrackedFrame to perc.history
)
out = perc.detect(["mug"], rgb, options=opts)
```

Defaults favor the common case (one mask per query, tight drift threshold,
history retention on). For multi-instance detection — e.g. "find every mug
in the scene" — set `top_k_per_query=None` and lower `match_threshold` to
taste. For long video runs, set `record_track_history=False` to avoid
retaining per-frame RGB/depth.

## Examples

Under `examples/` — one minimal file per core capability:

| File | What it shows |
|---|---|
| [01_detect.py](examples/01_detect.py) | Open-vocabulary detection on a single image |
| [02_track.py](examples/02_track.py) | Stream tracking through a video (detect + propagate) |
| [03_reconstruct.py](examples/03_reconstruct.py) | RGBD + camera intrinsics/extrinsics → world-frame point clouds |
| [04_disambiguate_clip_text.py](examples/04_disambiguate_clip_text.py) | Pick between similar instances using CLIP text embeddings |
| [05_disambiguate_clip_image.py](examples/05_disambiguate_clip_image.py) | Same, but CLIP scores candidates against reference images |
| [06_disambiguate_vlm.py](examples/06_disambiguate_vlm.py) | VLM reranker (minimal OpenAI client, requires `OPENAI_API_KEY`) |
| [07_track_and_reconstruct.py](examples/07_track_and_reconstruct.py) | Tracking + per-frame reconstruction → orbiting 3D GIF |
| [08_multi_camera.py](examples/08_multi_camera.py) | Two cameras in a single `detect()`/`track()` — independent ids |
| [09_camera_fusion.py](examples/09_camera_fusion.py) | `fuse_views` `primary_guided` mode — same id across cameras |
| [10_camera_fusion_complex.py](examples/10_camera_fusion_complex.py) | Multi-camera fusion with DINO disambiguation + reference bank |
| [11_sam2_segment.py](examples/11_sam2_segment.py) | SAM2 as a drop-in segmentation stage (`auto` / `bbox` / `point` modes) |
| [12_cotracker_points.py](examples/12_cotracker_points.py) | CoTracker per-detection 2D trails, lifted to 3D for visualization |

Customization / benchmarking recipes live under [examples/advanced/](examples/advanced/):

| File | What it shows |
|---|---|
| [async_detect.py](examples/advanced/async_detect.py) | Non-blocking one-shot detect via `AsyncOpenPerception` (control loop keeps ticking) |
| [async_track.py](examples/advanced/async_track.py) | Drop-oldest streaming tracking from a faster-than-SAM3 control loop |
| [reorder_stages.py](examples/advanced/reorder_stages.py) | Combine CLIP + DINO disambiguators, tune the assigner |
| [raw_sam3_masks.py](examples/advanced/raw_sam3_masks.py) | Skip disambiguation; render the per-stage history GIF |
| [sam2_tracker.py](examples/advanced/sam2_tracker.py) | Use the SAM2 backend for faster per-frame propagation |
| [decouple_detect_track.py](examples/advanced/decouple_detect_track.py) | Explicit two-stage pipeline (`Sam3Wrapper` → `Sam3VideoTracker`) |
| [benchmark_sam2_vs_sam3.py](examples/advanced/benchmark_sam2_vs_sam3.py) | Per-frame latency benchmark across SAM3 / SAM2 / decoupled |
| [validate_tracking.py](examples/advanced/validate_tracking.py) | Mean IoU vs frame-0 GT + tracker stability |

Run each from the repo root:

```bash
uv run python examples/01_detect.py
```

Assets under `assets/` ship with the repo so every example runs without
external dataset setup. They were originally extracted from a robotics
HDF5 demo; regenerate them (optional) from your own HDF5 with:

```bash
uv run python scripts/extract_demo_assets.py --hdf5 <path-to-demo.hdf5>
```

## Tracking details

`OpenPerception` wraps a HuggingFace streaming video session, one per view.
The default backend is `Sam3TrackerVideoModel` + `Sam3TrackerVideoProcessor` —
SAM3 does both text-prompted segmentation *and* mask propagation, so one set
of weights covers detection and tracking. Switch to SAM2 for faster per-frame
tracking when initial detection latency is not the bottleneck:

```python
cfg = PerceptionConfig(tracker_backend="sam2")  # default: "sam3"
cfg.sam2_model = "facebook/sam2-hiera-tiny"     # or a larger variant
perc = OpenPerception(cfg)
```

SAM2 cannot segment from text, so initial `detect()` calls still run SAM3
for text-conditioned segmentation — the SAM2 session is seeded with those
masks and used only for cheap mask propagation on subsequent `track()` /
`stream()` frames. Both backends share the same streaming API, so behaviour
(drift detection, recoveries, `instance_id` assignment) is identical.

Detect/track flow (applies to either backend):

1. `detect()` seeds one `TrackState` per final `Detection` and initializes
   the video session conditioned on those masks at frame 0 — per view.
2. Each subsequent `track(frame)` calls `session.add_new_frame(...)` followed
   by `model.forward(session, frame_idx=i)` — the memory bank propagates
   masks through motion and brief occlusions.
3. If a propagated mask's IoU against its anchor falls below
   `drift_iou_threshold`, the behavior is governed by `drift_policy`:
   - `"surface"` (default): return the propagated mask with
     `Detection.drifted=True` so the caller decides.
   - `"auto"`: silently re-run `detect()` for that query and re-anchor
     the session.
   - `"off"`: keep propagating whatever the tracker returns without flagging.

Recoveries are listed in `output.stats.recoveries`.

## Point tracking (CoTracker)

Enable per-detection 2D point trajectories — useful for dense motion cues
or seeding downstream 3D point tracking. Requires the `cotracker` extra:

```bash
uv sync --extra cotracker
```

```python
cfg = PerceptionConfig()
cfg.enable_point_tracking = True
cfg.cotracker_max_points_per_detection = 64
perc = OpenPerception(cfg)
```

Each detection's reconstructed mask pixels seed CoTracker queries at frame 0.
After a warm-up of `2 * model.step - 1` frames (7 for `cotracker3_online`),
every subsequent `track()` returns `Detection.point_tracks` populated with
the per-instance 2D trail (`for_instance(instance_id)` slices the shared
`PointTracks` object to one detection's columns). See
[examples/12_cotracker_points.py](examples/12_cotracker_points.py) for a full
detect → track → lift-to-3D → orbit-GIF flow.

## Multi-camera fusion (`fuse_views`)

Two modes:

- **`independent`** (default) — each view's tracker mints its own ids. The
  same physical object will get different `instance_id`s in different
  cameras.
- **`primary_guided`** — after `reconstruct`, the main camera's 3D point
  clouds are reprojected (with occlusion-aware visibility) into every other
  view. Matching by dilated-footprint IoU either *aligns* an existing
  secondary tracker id to the primary's, or *seeds* the secondary tracker
  with the projected mask when no candidate matches. Net effect: stable
  `instance_id` across cameras.

Tunable params on the stage: `iou_align_threshold`, `occlusion_tolerance`,
`min_projected_points`, `max_seed_points`, `footprint_dilate`, `flip_u`
(for sources where rgb/depth are flipped relative to the intrinsics
convention). See [examples/09_camera_fusion.py](examples/09_camera_fusion.py)
for the minimal flow and
[examples/10_camera_fusion_complex.py](examples/10_camera_fusion_complex.py)
for fusion combined with DINO disambiguation + a reference bank.

## Testing

From the repo root:

```bash
uv run pytest tests/
```

The default suite runs without a GPU or network — external models are stubbed
via `tests/_stubs.py`. Examples are smoke-tested at three tiers in
`tests/test_examples_smoke.py`: byte-compile, import (executes top-level
`from open_perception import ...`), and full end-to-end run under
`pytest -m slow` (loads real models and hits the GPU).
