"""Where the objects stood across a task's training demos, drawn on the camera view.

Each checkpoint was fine-tuned on a few dozen teleoperated demos. How far
the object was moved between them is the spread the policy has seen, and
the first thing to hold a positional or appearance experiment against: a
cell the demos never covered failing says something different from one they
did. The raw demos carry no depth and no camera calibration, so the spread
is measured in the exterior camera's own pixels — the camera was bolted in
place, so the pixels line up across demos and the overlay is faithful even
without a metric frame.

Every demo's first frame is run through the same detect -> SAM2 pipeline the
guided policy uses, one mask per object instance, and each mask's centroid
is drawn as a dot on one demo's frame — one colour per object, a faint
outline of the mask behind it, a legend in the corner. One PNG and one JSON
of the pixel positions per task, plus a side-by-side of every task. ::

    uv run python scripts/experiments/plot_training_positions.py

    # one task, other nouns for the detector, and a debug image per demo
    uv run python scripts/experiments/plot_training_positions.py \\
        --datasets put_all_cables_in_bag --objects "usb cable" bag --debug

    # redraw from the saved positions without running the detector again
    uv run python scripts/experiments/plot_training_positions.py --redraw

Needs the perception models and the detector's API key
(``OPENROUTER_API_KEY`` or ``GEMINI_API_KEY``), as a guided run does. No
robot, no servers.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import time
from typing import Literal, Optional, cast

import cv2
import numpy as np
import tyro
from loguru import logger

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.overlay import OverlayCanvas  # noqa: E402

DEFAULT_DEMO_ROOT = "data/demonstration_data"

DEFAULT_QUERIES: dict[str, list[str]] = {
    "put_all_cables_in_bag": ["cable", "bag"],
    "pour_contents_of_blue_cup_into_green_cup": ["blue cup", "green cup"],
    "place_the_pen_in_blue_cup_on_paper": ["pen", "blue cup", "white paper"],
}
"""What the detector is asked for, per training set. The manipulated object
first, then what it goes to; a task not listed here needs ``--objects``."""

# One colour per object, in query order. RGB, chosen to read against the
# grey table and the black bag: the manipulated object is always the first.
PALETTE: list[tuple[int, int, int]] = [
    (255, 120, 40),  # orange
    (70, 200, 255),  # sky blue
    (255, 230, 60),  # yellow
    (200, 90, 255),  # violet
    (90, 240, 120),  # green
]

# The exterior D435 at 640x480, as the experiments' reference cameras record
# it. Perception's reconstruction step wants *some* intrinsics; with no
# depth it produces nothing, and nothing here reads what it produces.
NOMINAL_K = np.array(
    [[607.24, 0.0, 323.36], [0.0, 607.66, 244.76], [0.0, 0.0, 1.0]], dtype=np.float64
)


@dataclasses.dataclass
class Args:
    datasets: list[str] = dataclasses.field(
        default_factory=lambda: list(DEFAULT_QUERIES)
    )
    """Training sets to draw: names under ``--demo-root``, or paths to
    directories of ``.vla`` files."""
    demo_root: str = DEFAULT_DEMO_ROOT
    objects: Optional[list[str]] = None
    """What to look for, for every dataset given. Default: per task, see
    ``DEFAULT_QUERIES``."""
    camera: str = "camera_0"
    """Which recorded camera. ``camera_0`` is the exterior one."""
    timestep: int = 0
    """Which frame of each demo. 0 is the staged scene before the arm moves."""
    out: str = "data/ur5_experiments/training_positions"
    """Directory for the PNGs and JSONs."""
    scale: float = 2.0
    """Upscale factor for the drawn frame."""
    background: int = 0
    """Index (in sorted order) of the demo whose frame is drawn under the
    dots."""
    outlines: bool = True
    """Draw each mask's outline, faintly, behind its dot."""
    limit: Optional[int] = None
    """Only the first N demos of each set. For a quick look."""
    redraw: bool = False
    """Reuse the saved positions JSON instead of running the detector. The
    outlines need the masks, which are not saved, so this draws dots only."""
    debug: bool = False
    """Write one annotated frame per demo under ``<out>/debug/<task>/``."""

    device: Optional[str] = None
    detector_backend: Literal["gemini", "openrouter"] = "openrouter"
    detector_model: Optional[str] = None
    vlm_model: Optional[str] = None


# ---------------------------------------------------------------------------
# Reading a demo
# ---------------------------------------------------------------------------


def demo_frame(
    path: pathlib.Path, camera: str, timestep: int
) -> tuple[np.ndarray, str]:
    """Frame *timestep* of *camera* in the raw demo at *path*, and its instruction.

    Raw ``rio`` demos have no sidecar, so :class:`VlaEpisode` cannot open
    them; the file is read directly. A sliced load is tried first — it is
    fifty times faster — and the whole file decoded only when the slice
    comes back empty, which robodm does for a slice starting at frame 0.
    """
    import robodm

    key = f"observation/cameras/{camera}/rgb"

    def _load(data_slice) -> dict:
        traj = robodm.Trajectory(str(path), mode="r")
        try:
            return cast(dict, traj.load(data_slice=data_slice))
        finally:
            traj.close()

    data = _load(slice(timestep, timestep + 1))
    frames = np.asarray(data.get(key, ()))
    if frames.ndim != 4 or frames.shape[0] == 0:
        data = _load(None)
        frames = np.asarray(data.get(key, ()))
        if frames.ndim != 4:
            raise KeyError(f"{path.name}: no {key!r} stream")
        if timestep >= frames.shape[0]:
            raise ValueError(f"{path.name} has {frames.shape[0]} frames")
        rgb = frames[timestep]
    else:
        rgb = frames[0]
    instruction = ""
    raw = data.get("instruction")
    if raw is not None and len(raw):
        instruction = str(np.asarray(raw).ravel()[0])
    return np.ascontiguousarray(rgb[..., :3]), instruction


# ---------------------------------------------------------------------------
# Finding the objects
# ---------------------------------------------------------------------------


def _clean(mask: np.ndarray) -> np.ndarray:
    """Holes closed, specks dropped — but every component kept: a coiled
    cable is one object in several pieces on screen."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:
            keep |= labels == i
    return keep


def find_objects(
    perceive,
    rgb: np.ndarray,
    queries: list[str],
    *,
    min_area: int = 80,
    retries: int = 1,
) -> dict[str, list[dict]]:
    """``{query: [instance, ...]}`` for one frame, every instance the detector
    found — "all cables" means all of them. Each instance carries its pixel
    centroid, box, area and (not serialised) mask.

    The detector is a VLM asked for boxes, and it occasionally answers with
    none for an object in plain view; asking again is usually all it takes,
    so a query that came back empty is asked for again up to *retries*
    times, as the experiment measurer does.
    """
    depth = np.full(rgb.shape[:2], np.nan, dtype=np.float32)
    found: dict[str, list[dict]] = {q: [] for q in queries}
    pending = list(queries)
    for _attempt in range(1 + max(0, retries)):
        masks = perceive(rgb, depth, NOMINAL_K, np.eye(4), pending)
        for _inst, (name, mask) in masks.items():
            if mask is None or name not in pending:
                continue
            mask = _clean(np.asarray(mask, dtype=bool))
            area = int(mask.sum())
            if area < min_area:
                continue
            ys, xs = np.nonzero(mask)
            found[name].append(
                {
                    "centroid_px": [float(xs.mean()), float(ys.mean())],
                    "bbox": [
                        int(xs.min()),
                        int(ys.min()),
                        int(xs.max()),
                        int(ys.max()),
                    ],
                    "area": area,
                    "mask": mask,
                }
            )
        pending = [q for q in pending if not found[q]]
        if not pending:
            break
    for name in found:
        found[name].sort(key=lambda d: -d["area"])
    return found


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _outline(canvas: OverlayCanvas, mask: np.ndarray, color, *, alpha: float) -> None:
    scaled = cv2.resize(
        mask.astype(np.uint8),
        (canvas.image.shape[1], canvas.image.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    contours, _ = cv2.findContours(scaled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    layer = canvas.image.copy()
    cv2.drawContours(layer, contours, -1, color, canvas.line_width, cv2.LINE_AA)
    cv2.addWeighted(layer, alpha, canvas.image, 1 - alpha, 0, canvas.image)


def _dot(canvas: OverlayCanvas, centroid_px, color, *, radius: float) -> None:
    x, y = (int(round(v * canvas.scale)) for v in centroid_px)
    r = max(3, int(round(radius * canvas.scale)))
    cv2.circle(canvas.image, (x, y), r, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(
        canvas.image, (x, y), max(2, r - canvas.line_width), color, -1, cv2.LINE_AA
    )


def draw_overlay(
    background: np.ndarray,
    demos: dict[str, dict[str, list[dict]]],
    queries: list[str],
    *,
    title: str,
    scale: float,
    outlines: bool,
) -> np.ndarray:
    """Every demo's object positions on *background*, with a legend."""
    canvas = OverlayCanvas(background, scale=scale, preprocessed=False)
    colors = {q: PALETTE[i % len(PALETTE)] for i, q in enumerate(queries)}
    # Outlines first, all of them, so no dot is hidden under a later demo's
    # outline; then the dots.
    if outlines:
        for found in demos.values():
            for q in queries:
                for inst in found.get(q, []):
                    if inst.get("mask") is not None:
                        _outline(canvas, inst["mask"], colors[q], alpha=0.35)
    for found in demos.values():
        for q in queries:
            for inst in found.get(q, []):
                _dot(canvas, inst["centroid_px"], colors[q], radius=4.0)

    canvas.draw_badge(title, line=0)
    n_demos = len(demos)
    for i, q in enumerate(queries):
        n_inst = sum(len(f.get(q, [])) for f in demos.values())
        n_seen = sum(1 for f in demos.values() if f.get(q))
        text = f"{q}: {n_inst} in {n_seen}/{n_demos} demos"
        canvas.draw_badge(text, color=colors[q], line=i + 1)
    return canvas.image


def draw_debug(
    rgb: np.ndarray, found: dict[str, list[dict]], queries: list[str], *, scale: float
) -> np.ndarray:
    canvas = OverlayCanvas(rgb, scale=scale, preprocessed=False)
    colors = {q: PALETTE[i % len(PALETTE)] for i, q in enumerate(queries)}
    for q in queries:
        for inst in found.get(q, []):
            if inst.get("mask") is not None:
                _outline(canvas, inst["mask"], colors[q], alpha=0.9)
            _dot(canvas, inst["centroid_px"], colors[q], radius=4.0)
    for i, q in enumerate(queries):
        canvas.draw_badge(f"{q}: {len(found.get(q, []))}", color=colors[q], line=i)
    return canvas.image


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _resolve_dataset(name: str, root: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(name)
    if path.is_dir():
        return path
    if (root / name).is_dir():
        return root / name
    raise SystemExit(f"No training set at {name!r} or under {root}")


def _strip_masks(demos: dict[str, dict[str, list[dict]]]) -> dict:
    return {
        demo: {
            q: [{k: v for k, v in inst.items() if k != "mask"} for inst in insts]
            for q, insts in found.items()
        }
        for demo, found in demos.items()
    }


def main(args: Args) -> int:
    root = pathlib.Path(args.demo_root)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    perceive = None
    if not args.redraw:
        from policy_abstraction.experiments.measure import open_perception_measurer

        perceive = open_perception_measurer(
            device=args.device,
            detector_backend=args.detector_backend,
            detector_model=args.detector_model,
            vlm_model=args.vlm_model,
            pixels_mirrored=False,
        )

    panels: list[np.ndarray] = []
    for name in args.datasets:
        dataset = _resolve_dataset(name, root)
        task = dataset.name
        queries = list(args.objects or DEFAULT_QUERIES.get(task) or [])
        if not queries:
            raise SystemExit(f"{task}: not in DEFAULT_QUERIES, pass --objects")
        files = sorted(dataset.glob("*.vla"))
        if args.limit is not None:
            files = files[: args.limit]
        if not files:
            raise SystemExit(f"{dataset}: no .vla files")
        json_path = out / f"{task}_training_positions.json"
        png_path = out / f"{task}_training_positions.png"

        demos: dict[str, dict[str, list[dict]]] = {}
        instruction = ""
        background: Optional[np.ndarray] = None
        bg_index = min(max(0, args.background), len(files) - 1)

        if args.redraw:
            if not json_path.exists():
                raise SystemExit(f"{json_path} is missing; run without --redraw first")
            saved = json.loads(json_path.read_text())
            demos = saved["demos"]
            instruction = saved.get("instruction", "")
            queries = list(saved.get("queries", queries))
            background, _ = demo_frame(files[bg_index], args.camera, args.timestep)
        else:
            assert perceive is not None
            t_start = time.time()
            for i, path in enumerate(files):
                rgb, instr = demo_frame(path, args.camera, args.timestep)
                instruction = instruction or instr
                if i == bg_index:
                    background = rgb
                found = find_objects(perceive, rgb, queries)
                demos[path.stem] = found
                counts = ", ".join(f"{q}={len(found[q])}" for q in queries)
                logger.info(f"{task} {path.stem} ({i + 1}/{len(files)}): {counts}")
                if args.debug:
                    dbg = out / "debug" / task / f"{path.stem}.png"
                    dbg.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(dbg), draw_debug(rgb, found, queries, scale=1.0)[:, :, ::-1]
                    )
            logger.info(f"{task}: {len(files)} demos in {time.time() - t_start:.0f} s")
            json_path.write_text(
                json.dumps(
                    {
                        "dataset": str(dataset),
                        "instruction": instruction,
                        "camera": args.camera,
                        "timestep": args.timestep,
                        "image_size": [int(v) for v in background.shape[:2]]  # type: ignore[union-attr]
                        if background is not None
                        else None,
                        "queries": queries,
                        "background_demo": files[bg_index].stem,
                        "demos": _strip_masks(demos),
                    },
                    indent=2,
                )
            )
            print(f"Wrote {json_path}")

        assert background is not None
        title = instruction or task
        image = draw_overlay(
            background,
            demos,
            queries,
            title=f"{title}  ({len(demos)} demos)",
            scale=args.scale,
            outlines=args.outlines and not args.redraw,
        )
        cv2.imwrite(str(png_path), image[:, :, ::-1])
        print(f"Wrote {png_path}")
        panels.append(image)

    if len(panels) > 1:
        h = min(p.shape[0] for p in panels)
        row = np.concatenate(
            [
                cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
                if p.shape[0] != h
                else p
                for p in panels
            ],
            axis=1,
        )
        all_path = out / "training_positions_all.png"
        cv2.imwrite(str(all_path), row[:, :, ::-1])
        print(f"Wrote {all_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
