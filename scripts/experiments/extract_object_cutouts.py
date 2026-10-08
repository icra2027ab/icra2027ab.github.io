"""Cut the staged objects out of a recording, with a transparent background.

For figures: each object the experiment stages, isolated from the table. A
rollout's first recorded frame is run through the same detect -> SAM2
pipeline the guided policy uses, and each object's mask becomes the alpha
channel of a tight crop. One PNG per object per rollout, plus a contact
sheet, so the cleanest cutout can be picked by eye. ::

    uv run python scripts/experiments/extract_object_cutouts.py \\
        --name cable_disambiguation_baseline_v3 --run ur5_experiments \\
        --indices 3 6 --out data/ur5_experiments/objects
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys
from typing import Literal, Optional

import cv2
import numpy as np
import tyro
from loguru import logger

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.measure import (  # noqa: E402
    first_frame,
    open_perception_measurer,
)
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)


@dataclasses.dataclass
class Args:
    name: str
    """The experiment, as given to ``--experiment.experiment-name``."""
    run: str = "ur5_experiments"
    base: str = "data"
    indices: Optional[list[int]] = None
    """Plan indices whose recordings to cut from. Default: the first scored
    rollout of each layout."""
    timestep: int = 0
    """Which frame of each recording. 0 is the staged scene before the arm
    moves, which is the one with nothing in the way."""
    objects: Optional[list[str]] = None
    """What to cut out. Default: the experiment's staged objects."""
    out: str = "data/ur5_experiments/objects"
    """Directory for the PNGs."""
    scale: float = 2.0
    """Upscale factor for the crops (cubic). The cameras are 640x480 and a
    spoon is a hundred pixels long; 2x makes a usable figure element."""
    margin: int = 6
    """Transparent border around each crop, in source pixels."""
    feather: int = 1
    """Blur radius on the alpha edge, in source pixels. 0 for a hard edge."""

    device: Optional[str] = None
    detector_backend: Literal["gemini", "openrouter"] = "openrouter"
    detector_model: Optional[str] = None
    vlm_model: Optional[str] = None


def _clean(mask: np.ndarray) -> np.ndarray:
    """Largest connected component, holes closed — one object, no specks."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m.astype(bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def _cutout(
    rgb: np.ndarray, mask: np.ndarray, *, margin: int, feather: int, scale: float
) -> np.ndarray:
    """RGBA crop of *rgb* where *mask* is the object."""
    ys, xs = np.where(mask)
    y0, y1 = max(0, ys.min() - margin), min(rgb.shape[0], ys.max() + margin + 1)
    x0, x1 = max(0, xs.min() - margin), min(rgb.shape[1], xs.max() + margin + 1)
    crop = rgb[y0:y1, x0:x1]
    alpha = mask[y0:y1, x0:x1].astype(np.float32)
    if feather > 0:
        k = 2 * feather + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    rgba = np.dstack([crop.astype(np.float32), alpha * 255.0])
    if scale != 1.0:
        rgba = cv2.resize(rgba, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return np.clip(rgba, 0, 255).astype(np.uint8)


def _contact_sheet(tiles: list[tuple[str, np.ndarray]], path: pathlib.Path) -> None:
    """Every cutout on a checkerboard, labelled, in one PNG."""
    if not tiles:
        return
    h = max(t.shape[0] for _, t in tiles) + 40
    w = sum(t.shape[1] + 20 for _, t in tiles) + 20
    sheet = np.full((h, w, 3), 200, np.uint8)
    yy, xx = np.mgrid[0:h, 0:w]
    sheet[((yy // 12) + (xx // 12)) % 2 == 0] = 235
    x = 20
    for label, tile in tiles:
        th, tw = tile.shape[:2]
        alpha = tile[..., 3:4].astype(np.float32) / 255.0
        region = sheet[10 : 10 + th, x : x + tw].astype(np.float32)
        sheet[10 : 10 + th, x : x + tw] = (
            alpha * tile[..., :3] + (1 - alpha) * region
        ).astype(np.uint8)
        cv2.putText(
            sheet,
            label,
            (x, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
        x += tw + 20
    cv2.imwrite(str(path), sheet[:, :, ::-1])


def main(args: Args) -> int:
    store = ExperimentStore(experiment_root(args.run, args.name, base=args.base))
    if not store.exists:
        raise SystemExit(f"No experiment at {store.root}. Check --name and --run.")
    config = store.load_config()
    results = store.results()
    objects = args.objects or [label for _id, label in config.scene_objects()]
    if not objects:
        raise SystemExit("Nothing to cut out: pass --objects.")

    if args.indices is None:
        # The first scored rollout of each layout: every layout shows the
        # objects in a different arrangement, so one of them is usually clear.
        target = config.target_factor()
        skip = target.name if target is not None else None
        seen: set[str] = set()
        picked = []
        for r in results:
            key = "|".join(
                f"{k}={v}" for k, v in sorted(r.assignments.items()) if k != skip
            )
            if key not in seen:
                seen.add(key)
                picked.append(r)
    else:
        wanted = set(args.indices)
        picked = [r for r in results if r.index in wanted]
    if not picked:
        raise SystemExit("No scored rollouts matched.")

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    perceive = open_perception_measurer(
        device=args.device,
        detector_backend=args.detector_backend,
        detector_model=args.detector_model,
        vlm_model=args.vlm_model,
        pixels_mirrored=False,
    )

    tiles: list[tuple[str, np.ndarray]] = []
    for record in picked:
        if not record.episode_path:
            continue
        frame = first_frame(
            store.root / record.episode_path, start=args.timestep, require_depth=False
        )
        masks = perceive(
            frame.rgb, frame.depth_m, frame.intrinsic, frame.extrinsic, objects
        )
        by_name: dict[str, np.ndarray] = {}
        for _inst, (name, mask) in masks.items():
            if mask is None or not mask.any():
                continue
            prev = by_name.get(name)
            if prev is None or mask.sum() > prev.sum():
                by_name[name] = np.asarray(mask, dtype=bool)
        for name in objects:
            mask = by_name.get(name)
            if mask is None:
                logger.warning(f"rollout {record.index}: no mask for {name!r}")
                continue
            rgba = _cutout(
                frame.rgb,
                _clean(mask),
                margin=args.margin,
                feather=args.feather,
                scale=args.scale,
            )
            path = out / f"{name.replace(' ', '_')}_rollout{record.index:04d}.png"
            cv2.imwrite(str(path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
            tiles.append((f"{name} (r{record.index})", rgba))
            print(f"  {path}  {rgba.shape[1]}x{rgba.shape[0]}")
    _contact_sheet(tiles, out / f"contact_sheet_{args.name}.png")
    print(f"Contact sheet: {out / f'contact_sheet_{args.name}.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
