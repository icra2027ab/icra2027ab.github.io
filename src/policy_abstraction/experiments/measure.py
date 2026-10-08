"""Measuring where the objects actually were, from each rollout's recording.

A positional experiment records the *nominal* position of every staged object
— the centre of the grid cell the operator was shown. The cable never sits
exactly there: it lands a few centimetres off, it is a long object with no
single centre, and the operator's eye is not a ruler. The report's dots were
drawn at the nominal position, so they showed the plan, not the experiment.

This module reads each rollout's first recorded frame — colour, metric depth
and the calibrated camera, all of which the recorder already writes — and asks
the same perception pipeline the guided policy runs on (``detect -> SAM2 ->
reconstruct``) where the object is. The mask is back-projected through the
depth image into the robot's base frame with the very call the abstractor
makes, so the measured position is the one the policy would have reasoned
about, not an approximation of it.

The result is written beside the experiment as ``observed_positions.json``,
keyed by rollout index and by the same keys ``RolloutRecord.target_positions``
uses, so a measured position lands next to the nominal one it checks. The
report draws the measured positions when they exist and falls back to the
cell centre when they do not.

The heavy dependency — the perception client, its models and its VLM detector
— is injected as a callable, so everything here runs in a unit test with a
mask drawn by hand.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
from loguru import logger

from policy_abstraction.experiments.config import ExperimentConfig
from policy_abstraction.experiments.outcomes import RolloutRecord
from policy_abstraction.experiments.store import ExperimentStore

Perceive = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]],
    dict[int, tuple[str, np.ndarray]],
]
"""``(rgb, depth_m, K3, E, queries) -> {instance_id: (query, mask)}`` for one
view. What :class:`OpenPerceptionClient` returns for its primary view, minus
the client — see :func:`open_perception_measurer`."""


@dataclasses.dataclass
class Measurement:
    """Where one queried object was found in one frame."""

    position: list[float]
    """Base-frame centroid ``[x, y, z]`` of the object's point cloud."""
    query: str
    points: int
    """How many depth returns the centroid averages. A handful means the mask
    landed on a depth hole, and the position is not to be trusted."""
    instances: int
    """How many instances the detector found for the query. Above one, the
    instance nearest the nominal position was taken."""
    offset_m: Optional[float] = None
    """Distance from the nominal position, when one was recorded."""
    timestep: int = 0
    """Which frame of the recording was measured — 0 unless it had no depth."""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class FirstFrame:
    """The inputs perception needs, lifted out of a recording."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsic: np.ndarray
    extrinsic: np.ndarray
    timestep: int
    pixels_mirrored: bool
    """Whether the frame is in the simulator's flipped convention. Real
    cameras are upright; the reconstruction must be told which."""
    camera: str = ""
    """Name of the camera the frame came from — the recording's primary."""


# ---------------------------------------------------------------------------
# Reading the recording
# ---------------------------------------------------------------------------


def first_frame(
    episode: pathlib.Path,
    *,
    start: int = 0,
    max_lookahead: int = 50,
    require_depth: bool = True,
) -> FirstFrame:
    """The first frame of *episode* at or after *start* with a calibrated camera.

    Depth is required too unless *require_depth* is off — a frame chosen as
    the report's background needs no depth, a frame to measure from does.
    Frame 0 normally qualifies: the UR5e records depth every ``depth_stride``
    steps starting at the first executed one. A recording whose first depth
    row is missing is searched forward a little, since a frame a few steps in
    still shows the object where it was staged — the arm has not reached it.
    """
    from policy_abstraction.analysis.demo_episode import VlaEpisode

    ep = VlaEpisode(str(episode.parent), episode.stem)
    if len(ep) == 0:
        raise ValueError(f"{episode.name} holds no frames")
    if start < 0 or start >= len(ep):
        raise ValueError(
            f"{episode.name} has {len(ep)} frames; step {start} is not one of them"
        )
    for t in range(start, min(len(ep), start + max_lookahead + 1)):
        depth = ep.depth_m(t)
        if depth is None and require_depth:
            continue
        K, E = ep.intrinsic(t), ep.extrinsic(t)
        if K is None or E is None:
            continue
        K = np.asarray(K, dtype=np.float64)
        if depth is None:
            depth = np.full(ep.rgb(t).shape[:2], np.nan, dtype=np.float32)
        return FirstFrame(
            rgb=np.ascontiguousarray(ep.rgb(t)),
            depth_m=np.ascontiguousarray(np.asarray(depth, dtype=np.float32)),
            intrinsic=K[:3, :3] if K.shape == (4, 4) else K,
            extrinsic=np.asarray(E, dtype=np.float64),
            timestep=t,
            pixels_mirrored=_pixels_mirrored(ep),
            camera=ep._primary_camera,
        )
    raise ValueError(
        f"{episode.name} has no frame with "
        f"{'depth and ' if require_depth else ''}camera parameters within "
        f"{max_lookahead} steps of step {start}"
    )


def _pixels_mirrored(ep) -> bool:
    """Whether the recording's primary camera frames are in the flipped convention."""
    for cam in ep._sidecar.get("env_spec", {}).get("cameras", []):
        if cam.get("is_primary"):
            return str(cam.get("image_transform", "none")) != "none"
    return bool(getattr(ep, "frames_preprocessed", False))


# ---------------------------------------------------------------------------
# From masks to positions
# ---------------------------------------------------------------------------


def positions_from_masks(
    frame: FirstFrame,
    masks: dict[int, tuple[str, np.ndarray]],
    queries: dict[str, str],
    *,
    nominal: Optional[Mapping[str, Sequence[float]]] = None,
    depth_max: float = 3.0,
    min_points: int = 20,
) -> dict[str, Measurement]:
    """Back-project each query's mask into a base-frame centroid.

    Instance ids are stamped into a synthetic segmentation image and fed
    through ``build_segmented_pointcloud`` — the abstractor's own call — so
    the geometry is the deployment path, not a re-derivation of it. A query
    the detector found several of resolves to the instance nearest its
    nominal position; with no nominal position, to the one with the most
    depth returns.

    Returns one :class:`Measurement` per key in *queries* that was found with
    at least *min_points* returns. Keys the detector missed are absent.
    """
    from policy_abstraction.utils.segmentation import build_segmented_pointcloud

    nominal = nominal or {}
    seg = np.zeros(frame.depth_m.shape[:2], dtype=np.int32)
    by_query: dict[str, list[int]] = {}
    for inst_id, (name, mask) in masks.items():
        if mask is None or int(inst_id) <= 0:
            continue
        seg[np.asarray(mask, dtype=bool)] = int(inst_id)
        by_query.setdefault(_canon(name), []).append(int(inst_id))
    if not by_query:
        return {}

    depth = frame.depth_m.copy()
    depth[~np.isfinite(depth)] = 0.0
    pcd = build_segmented_pointcloud(
        frame.rgb,
        depth,
        seg,
        frame.intrinsic,
        frame.extrinsic,
        depth_max=depth_max,
        preprocessed=frame.pixels_mirrored,
    )
    if pcd.shape[0] == 0:
        return {}
    ids = pcd[:, 6].astype(np.int32)

    out: dict[str, Measurement] = {}
    for key, query in queries.items():
        candidates = by_query.get(_canon(query), [])
        clouds = [
            (inst, pcd[ids == inst, :3])
            for inst in candidates
            if int((ids == inst).sum()) >= min_points
        ]
        if not clouds:
            continue
        target = nominal.get(key)
        if target is not None and len(clouds) > 1:
            ref = np.asarray(target, dtype=np.float64)
            inst, pts = min(
                clouds, key=lambda c: float(np.linalg.norm(c[1].mean(axis=0) - ref))
            )
        else:
            inst, pts = max(clouds, key=lambda c: c[1].shape[0])
        centroid = pts.mean(axis=0)
        out[key] = Measurement(
            position=[float(v) for v in centroid],
            query=query,
            points=int(pts.shape[0]),
            instances=len(candidates),
            offset_m=(
                float(np.linalg.norm(centroid - np.asarray(target, dtype=np.float64)))
                if target is not None
                else None
            ),
            timestep=frame.timestep,
        )
    return out


def _canon(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


# ---------------------------------------------------------------------------
# Over an experiment
# ---------------------------------------------------------------------------


def measure_rollout(
    record: RolloutRecord,
    config: ExperimentConfig,
    store: ExperimentStore,
    perceive: Perceive,
    *,
    queries: Optional[dict[str, str]] = None,
    retries: int = 1,
) -> tuple[dict[str, Measurement], FirstFrame, dict[int, tuple[str, np.ndarray]]]:
    """Measure one recorded rollout. Raises when the recording cannot be read."""
    if not record.episode_path:
        raise FileNotFoundError(f"rollout {record.index} recorded no episode")
    episode = store.root / record.episode_path
    if not episode.exists():
        raise FileNotFoundError(f"rollout {record.index}: {episode} is missing")
    queries = queries or config.measurement_queries(record.assignments)
    if not queries:
        raise ValueError(f"rollout {record.index} stages nothing positional to measure")
    frame = first_frame(episode)
    names = sorted(set(queries.values()))
    found: dict[str, Measurement] = {}
    masks: dict[int, tuple[str, np.ndarray]] = {}
    # The detector is a VLM asked for boxes; it occasionally answers with
    # none for an object in plain view, and asking again is usually all it
    # takes. Retry only for what is still missing.
    for attempt in range(1 + max(0, retries)):
        masks = perceive(
            frame.rgb, frame.depth_m, frame.intrinsic, frame.extrinsic, names
        )
        found.update(
            positions_from_masks(
                frame, masks, queries, nominal=record.target_positions or None
            )
        )
        if set(found) >= set(queries):
            break
        if attempt < retries:
            logger.debug(
                f"rollout {record.index}: nothing for "
                f"{sorted(set(queries) - set(found))}, asking again"
            )
    return found, frame, masks


def measure_experiment(
    store: ExperimentStore,
    perceive: Perceive,
    *,
    config: Optional[ExperimentConfig] = None,
    indices: Optional[Iterable[int]] = None,
    force: bool = False,
    queries: Optional[dict[str, str]] = None,
    retries: int = 1,
    method: str = "open_perception",
    debug_dir: Optional[pathlib.Path] = None,
    on_measured: Optional[
        Callable[[RolloutRecord, dict[str, Measurement]], None]
    ] = None,
) -> dict:
    """Measure every scored rollout of an experiment and update its sidecar.

    Args:
        perceive: The detector, see :data:`Perceive`.
        indices: Restrict to these plan indices. ``None`` means every result.
        force: Re-measure rollouts that already have an entry.
        queries: Override the per-factor object queries — ``{key: name}`` with
            the keys of ``target_positions``. For when the noun the config
            derived is not the one the detector understands.
        retries: How many more times to ask the detector when it finds
            nothing; see :func:`measure_rollout`.
        method: Recorded in the sidecar so a later reader knows what produced
            the numbers.
        debug_dir: Where to write one annotated frame per rollout showing the
            mask outline and the measured point, for eyeballing. ``None``
            writes none.
        on_measured: Called after each rollout, measured or not, with what was
            found — for a progress line.

    The sidecar is rewritten after every rollout, so a crash halfway keeps
    what was measured; re-running skips those unless *force*.

    Returns the sidecar dict as saved.
    """
    config = config or store.load_config()
    data = store.load_observed_positions()
    rollouts: dict = data.setdefault("rollouts", {})
    data["method"] = method
    data["frame"] = "robot base"
    wanted = set(int(i) for i in indices) if indices is not None else None

    records = store.results()
    for record in records:
        if wanted is not None and record.index not in wanted:
            continue
        key = str(record.index)
        if key in rollouts and not force:
            continue
        try:
            found, frame, masks = measure_rollout(
                record, config, store, perceive, queries=queries, retries=retries
            )
        except Exception as exc:
            logger.warning(f"rollout {record.index}: not measured ({exc})")
            rollouts[key] = {"error": f"{type(exc).__name__}: {exc}"}
            data["measured_at"] = time.time()
            store.save_observed_positions(data)
            if on_measured is not None:
                on_measured(record, {})
            continue
        expected = queries or config.measurement_queries(record.assignments)
        missing = sorted(set(expected) - set(found))
        if missing:
            logger.warning(
                f"rollout {record.index}: detector found nothing for "
                f"{[expected[k] for k in missing]}"
            )
        rollouts[key] = {k: m.to_dict() for k, m in found.items()}
        if missing:
            rollouts[key]["missing"] = missing
        data["measured_at"] = time.time()
        store.save_observed_positions(data)
        # Written for a miss too: the frame is what the detector was shown,
        # and that is what tells a bad query from a bad recording.
        if debug_dir is not None:
            try:
                write_debug_image(
                    debug_dir / f"rollout_{record.index:04d}.png",
                    frame,
                    masks,
                    found,
                    nominal=record.target_positions,
                )
            except Exception as exc:  # a debug picture must never stop the run
                logger.debug(f"rollout {record.index}: no debug image ({exc})")
        if on_measured is not None:
            on_measured(record, found)
    return data


# ---------------------------------------------------------------------------
# Choosing the report's background
# ---------------------------------------------------------------------------


def set_reference_view(
    store: ExperimentStore, record: RolloutRecord, *, timestep: int = 0
) -> FirstFrame:
    """Make a frame of *record*'s recording the view the report projects onto.

    The frame saved at the start of the session shows the table as it was
    before the first rollout, which is sometimes the wrong picture: the object
    may be in the way of the grid, or missing from it altogether. Any frame of
    any rollout carries the same calibrated camera, so it can stand in — one
    with the object staged at a mid-grid cell and the arm on its way to it
    reads best. *timestep* picks the frame; the first one with camera
    parameters at or after it is used. The frame it replaces is kept as
    ``reference_frame.original.png``.
    """
    if not record.episode_path:
        raise FileNotFoundError(f"rollout {record.index} recorded no episode")
    episode = store.root / record.episode_path
    if not episode.exists():
        raise FileNotFoundError(f"rollout {record.index}: {episode} is missing")
    frame = first_frame(episode, start=timestep, require_depth=False)
    store.save_reference_view(
        frame.rgb,
        intrinsic=frame.intrinsic,
        extrinsic=frame.extrinsic,
        camera_name=frame.camera,
        preprocessed=frame.pixels_mirrored,
        replace=True,
    )
    logger.info(
        f"Reference view is now rollout {record.index}'s frame at step "
        f"{frame.timestep} ({record.assignments})"
    )
    return frame


# ---------------------------------------------------------------------------
# Eyeballing
# ---------------------------------------------------------------------------


def write_debug_image(
    path: pathlib.Path,
    frame: FirstFrame,
    masks: dict[int, tuple[str, np.ndarray]],
    found: dict[str, Measurement],
    *,
    nominal: Optional[Mapping[str, Sequence[float]]] = None,
) -> pathlib.Path:
    """The measured frame with every mask outlined and each position marked.

    Measured positions are filled dots, nominal ones hollow, joined by a line:
    the length of that line is the staging error the report now accounts for.
    """
    import cv2

    from policy_abstraction.experiments.overlay import OverlayCanvas

    canvas = OverlayCanvas(
        frame.rgb[..., :3],
        intrinsic=frame.intrinsic,
        extrinsic=frame.extrinsic,
        preprocessed=frame.pixels_mirrored,
        scale=1.0,
    )
    for _inst, (_name, mask) in masks.items():
        if mask is None:
            continue
        contours, _ = cv2.findContours(
            np.asarray(mask, dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas.image, contours, -1, (255, 220, 40), 1, cv2.LINE_AA)
    for key, m in found.items():
        px, ok = canvas.project(np.asarray([m.position]))
        if not ok[0]:
            continue
        target = (nominal or {}).get(key)
        if target is not None:
            npx, nok = canvas.project(np.asarray([target], dtype=np.float64))
            if nok[0]:
                cv2.line(
                    canvas.image,
                    tuple(int(v) for v in npx[0]),
                    tuple(int(v) for v in px[0]),
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
                cv2.circle(
                    canvas.image,
                    tuple(int(v) for v in npx[0]),
                    6,
                    (230, 230, 230),
                    1,
                    cv2.LINE_AA,
                )
        cv2.circle(
            canvas.image,
            tuple(int(v) for v in px[0]),
            5,
            (60, 235, 90),
            -1,
            cv2.LINE_AA,
        )
        text = f"{key}: {m.query} ({m.points} pts"
        text += f", {m.offset_m * 100:.1f} cm off)" if m.offset_m is not None else ")"
        # Keep the label on the frame: to the left of the point when the
        # point is near the right edge, and clear of the bottom edge.
        width, height = canvas.image.shape[1], canvas.image.shape[0]
        est = int(9.5 * len(text))
        x = int(px[0][0]) + 8
        if x + est > width:
            x = max(4, int(px[0][0]) - 8 - est)
        y = max(14, min(int(px[0][1]) - 8, height - 8))
        canvas._text(text, (x, y), (255, 255, 255), outline=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), canvas.image[:, :, ::-1])
    return path


# ---------------------------------------------------------------------------
# The real detector
# ---------------------------------------------------------------------------


def open_perception_measurer(
    *,
    device: Optional[str] = None,
    detector_backend: str = "openrouter",
    detector_model: Optional[str] = None,
    vlm_model: Optional[str] = None,
    pixels_mirrored: bool = False,
) -> Perceive:
    """A :data:`Perceive` backed by :class:`OpenPerceptionClient`.

    Built once — the models take a while to load — and reset before every
    frame, because detection is stateful: without the reset the second
    recording would be *tracked* from the first one's masks.
    """
    # The client puts the in-repo open_perception package on sys.path when it
    # is imported, so it has to come first.
    from open_perception import CameraInfo  # noqa: I001

    from policy_abstraction.perception.open_perception_client import (
        OpenPerceptionClient,
    )

    client = OpenPerceptionClient(
        device=device,
        primary_view_idx=0,
        detector_backend=detector_backend,  # type: ignore[arg-type]
        detector_model=detector_model,
        vlm_model=vlm_model,
        flip_u=pixels_mirrored,
    )

    def perceive(rgb, depth, K, E, names):
        client.reset()
        views = client([rgb], [depth], [CameraInfo(intrinsics=K, extrinsics=E)], names)
        return views[0] if views else {}

    return perceive


__all__ = [
    "FirstFrame",
    "Measurement",
    "Perceive",
    "first_frame",
    "measure_experiment",
    "measure_rollout",
    "open_perception_measurer",
    "positions_from_masks",
    "set_reference_view",
    "write_debug_image",
]
