"""The training demos' object positions drawn onto an experiment's camera-view grid.

Where the checkpoint saw the object during training, laid over where the
experiment staged it, so a cell that failed can be read against the demos'
coverage of it. The camera was moved between the demo recordings and the
experiments, so the demos' pixels cannot simply be copied across; instead
each demo's proprioception is read — it is in the same robot base frame as
the grid and the experiment's measured positions, as the identical home
pose in both recordings confirms — and every moment the gripper closed near
the table is taken as a grasp, i.e. as where a cable was. Where it opened
again, high, is where the bag was. Those base-frame points are projected
through the experiment's own calibrated camera onto its reference frame,
on top of the report's ``<factor>_on_camera.png`` (cells tinted by success
rate, one dot per rollout at its measured position). ::

    uv run python scripts/experiments/plot_training_on_grid.py \\
        --name cable_position_baseline_v3 --run ur5_experiments

    # rows within reach only, as the report's 3x3 variant
    uv run python scripts/experiments/plot_training_on_grid.py \\
        --name cable_position_guided_v3 --exclude-rows D

    # a heat map of the grasps on the table, with the cables' orientations
    uv run python scripts/experiments/plot_training_on_grid.py \\
        --name cable_position_baseline_v3_merged --heatmap --orientations \\
        --dataset ~/dev/rio/data/put_all_cables_in_the_bag \\
        --out-name put_all_cables_in_the_bag_training_heatmap

Writes ``<out>/<name>_with_training.png``, plus the grasp and release
positions as ``<out>/<dataset>_grasp_positions.json`` (base frame, metres).
The cable's orientation at each grasp is the gripper's yaw: the 2F-140's
fingers close along the tool's x axis (the wrist camera's horizontal, as
its optical flow under tool-frame motion shows), so the cable runs between
the pads along the tool's y axis, and its direction on the table is that
axis projected onto the plane, modulo 180 degrees, in the robot base frame.
No robot, no servers, no detector: only the recordings.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
from typing import Optional, cast

import cv2
import numpy as np
import tyro
from loguru import logger

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from policy_abstraction.experiments.factors import GridPositionFactor  # noqa: E402
from policy_abstraction.experiments.overlay import OverlayCanvas  # noqa: E402
from policy_abstraction.experiments.store import (  # noqa: E402
    ExperimentStore,
    experiment_root,
)
from policy_abstraction.experiments.viz import plot_grid_success_projected  # noqa: E402

DEFAULT_DATASET = "data/demonstration_data/put_all_cables_in_bag"

COLOR_GRASP = (255, 120, 40)  # RGB: the training overlay's colour for the cable
COLOR_RELEASE = (70, 200, 255)  # and for the bag


@dataclasses.dataclass
class Args:
    name: str
    """The experiment, as given to ``--experiment.experiment-name``."""
    run: str = "ur5_experiments"
    base: str = "data"
    dataset: str = DEFAULT_DATASET
    """Directory of the raw ``.vla`` demos the checkpoint was trained on."""
    factor: Optional[str] = None
    """Which grid factor to draw. Default: the experiment's only one."""
    exclude_rows: Optional[list[str]] = None
    """Drop whole rows of the grid by letter, e.g. ``D``."""
    out: str = "data/ur5_experiments/training_positions"
    """Directory for the PNG and JSON."""
    scale: float = 3.0
    """Upscale factor for the drawn frame, as the report uses."""
    releases: bool = True
    """Also draw where the gripper opened high — the bag."""
    open_above: float = 0.35
    """Aperture at or above which the gripper counts as open."""
    closed_below: float = 0.25
    """Aperture at or below which it counts as closed."""
    grasp_max_z: float = 0.12
    """A closing above this height (metres, base frame) is not a grasp from
    the table."""
    release_min_z: float = 0.14
    """An opening below this height is not a drop into the bag: the bag's rim
    is about that high, and lower openings are the operator letting go to
    regrasp a cable."""
    draw_z: float = 0.01
    """Height the grasp points are projected at: the cable's centroid height,
    not the tool's, since the tool sits a few centimetres above what it
    holds and the camera is oblique."""

    heatmap: bool = False
    """Instead of the success map with dots: the bare grid lines over a
    continuous density of the training grasps on the table — where the
    checkpoint saw cables, as a heat map. Writes ``<name>_training_heatmap.png``."""
    bandwidth_cm: float = 3.0
    """Gaussian kernel width of the heat map, in centimetres on the table."""
    heat_alpha: float = 0.85
    """Opacity of the heat map at its peak; it fades to nothing with the
    density."""
    heat_floor: float = 0.04
    """Density below this fraction of the peak is not drawn at all, so the
    tail does not tint the whole table."""
    labels: bool = False
    """Letter the cells on the heat map."""
    orientations: bool = False
    """On the heat map, draw each grasp's cable direction as a short bar on
    the table, and a rose of all the directions in the corner."""
    bar_cm: float = 4.0
    """Length of an orientation bar, in centimetres on the table."""
    out_name: Optional[str] = None
    """Stem of the PNG. Default ``<name>_training_heatmap`` or
    ``<name>_with_training``; set it when drawing another dataset onto the
    same experiment's view, so the default is not overwritten."""


# ---------------------------------------------------------------------------
# Grasp and release events from proprioception
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Events:
    grasps: list[list[float]]
    """Base-frame ``[x, y, z]`` of the tool each time the gripper closed near
    the table."""
    releases: list[list[float]]
    """Same, each time it opened high — over the bag."""
    cable_yaws: list[float] = dataclasses.field(default_factory=list)
    """One per grasp: the grasped cable's direction on the table, degrees in
    ``[0, 180)`` from the base x axis towards y. The tool's y axis (across
    the finger pads) projected onto the table plane."""


def demo_events(
    path: pathlib.Path,
    *,
    open_above: float,
    closed_below: float,
    grasp_max_z: float,
    release_min_z: float,
) -> Events:
    """Read one raw demo and find its grasps and releases.

    The gripper signal is the aperture: high open, low closed, resting
    closed. A grasp is the aperture falling from open to closed while the
    tool is low; a release is it rising from closed to open while the tool
    is high, and only the first opening after a grasp counts — the operator
    often works the gripper several times over the bag. Single-step glitches in the signal (it spikes to fully open for
    one sample now and then) are ignored by requiring the new state to hold
    for a few samples.
    """
    import robodm

    traj = robodm.Trajectory(str(path), mode="r")
    try:
        data = cast(dict, traj.load())
    finally:
        traj.close()
    pose = np.asarray(data["observation/proprio_eef"], dtype=np.float64)
    eef = pose[:, :3]
    g = np.asarray(data["observation/gripper_position"], dtype=np.float64).ravel()
    n = min(len(eef), len(g))
    pose, eef, g = pose[:n], eef[:n], g[:n]

    hold = 5
    state = 0  # 1 open, -1 closed, 0 unknown
    holding = False
    grasps: list[list[float]] = []
    releases: list[list[float]] = []
    yaws: list[float] = []
    for t in range(n - hold):
        window = g[t : t + hold]
        if state != 1 and bool((window >= open_above).all()):
            if holding and eef[t, 2] >= release_min_z:
                releases.append(eef[t].tolist())
                holding = False
            state = 1
        elif state != -1 and bool((window <= closed_below).all()):
            if state == 1 and eef[t, 2] <= grasp_max_z:
                grasps.append(eef[t].tolist())
                yaws.append(cable_yaw_deg(pose[t, 3:6]))
                holding = True
            state = -1
    return Events(grasps=grasps, releases=releases, cable_yaws=yaws)


def cable_yaw_deg(rotvec: np.ndarray) -> float:
    """The grasped cable's direction on the table, from the tool's rotation
    vector: the tool y axis, which runs across the finger pads, projected
    onto the plane. Degrees in ``[0, 180)`` from base x towards base y."""
    R, _ = cv2.Rodrigues(np.asarray(rotvec, dtype=np.float64))
    ty = R[:, 1]
    return float(np.degrees(np.arctan2(ty[1], ty[0])) % 180.0)


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _dot(image, px, color, *, radius: int, edge: int) -> None:
    cv2.circle(image, px, radius + edge, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(image, px, radius, color, -1, cv2.LINE_AA)


def training_heatmap(
    canvas: OverlayCanvas,
    grid,
    grasps: np.ndarray,
    *,
    z: float,
    bandwidth: float,
    alpha_max: float,
    floor: float,
    margin: float = 0.06,
    resolution: float = 0.0025,
) -> np.ndarray:
    """Blend a kernel density of *grasps* on the table plane into *canvas*.

    The density is rendered top-down in the grid's own ``(u, v)`` metres,
    then warped into the camera through the plane-to-image homography — the
    same projection the grid lines use, so the two line up exactly. The
    warm end of the colour map is where the demos put the cables most; the
    tail fades to transparent so it does not tint the whole table.

    Returns the ``[H, W]`` density (peak 1) in canvas pixels, for a caller
    that wants contours or a colour bar.
    """
    from matplotlib import colormaps

    from policy_abstraction.utils.pcd import project_points_to_pixels

    o = np.asarray(grid.origin, dtype=np.float64)
    u_axis = np.asarray(grid.u_axis, dtype=np.float64)
    v_axis = np.asarray(grid.v_axis, dtype=np.float64)
    rel = grasps[:, :3] - o
    gu, gv = rel @ u_axis, rel @ v_axis

    # Top-down raster over the grid plus a margin, bandwidth-sized cells.
    u0, u1 = -margin, grid.extent_u + margin
    v0, v1 = -margin, grid.extent_v + margin
    us = np.arange(u0, u1, resolution)
    vs = np.arange(v0, v1, resolution)
    uu, vv = np.meshgrid(us, vs)  # [len(vs), len(us)]
    dens = np.zeros_like(uu)
    inv = 1.0 / (2.0 * bandwidth * bandwidth)
    for pu, pv in zip(gu, gv, strict=True):
        dens += np.exp(-((uu - pu) ** 2 + (vv - pv) ** 2) * inv)
    dens /= dens.max()

    # Plane -> canvas homography from the raster's corners.
    corners_uv = np.array([[u0, v0], [u1, v0], [u1, v1], [u0, v1]])
    corners_3d = np.stack([o + cu * u_axis + cv * v_axis for cu, cv in corners_uv])
    corners_3d[:, 2] = z
    src_h, src_w = canvas._src_h, canvas._src_w
    K, E = canvas._K, canvas._E
    if K is None or E is None:
        raise ValueError("the canvas has no camera to project the heat map with")
    px, ok = project_points_to_pixels(
        corners_3d,
        K,
        E,
        src_h,
        src_w,
        preprocessed=canvas._preprocessed,
    )
    if not ok.all():
        raise ValueError("the grid's corners do not all project into the camera")
    dst = (px.astype(np.float64) * canvas.scale).astype(np.float32)
    src = np.array(
        [
            [0, 0],
            [len(us) - 1, 0],
            [len(us) - 1, len(vs) - 1],
            [0, len(vs) - 1],
        ],
        dtype=np.float32,
    )
    H = cv2.getPerspectiveTransform(src, dst)
    h, w = canvas.image.shape[:2]
    warped = cv2.warpPerspective(
        dens.astype(np.float32), H, (w, h), flags=cv2.INTER_LINEAR
    )

    cmap = colormaps["inferno"]
    rgb = (cmap(np.clip(warped, 0, 1))[..., :3] * 255).astype(np.float32)
    alpha = np.clip((warped - floor) / (1.0 - floor), 0, 1) * alpha_max
    alpha = alpha[..., None]
    base = canvas.image.astype(np.float32)
    canvas.image[:] = np.clip(alpha * rgb + (1 - alpha) * base, 0, 255).astype(np.uint8)
    return warped


COLOR_BAR = (255, 255, 255)  # RGB: the orientation bars, over the inferno map


def draw_orientation_bars(
    canvas: OverlayCanvas,
    grasps: np.ndarray,
    yaws_deg: np.ndarray,
    *,
    z: float,
    length: float,
) -> int:
    """A bar through each grasp along the cable's direction, on the table.

    Both ends are projected through the camera, so the bars foreshorten as
    the table does and read as lying on it. Dark edge under a light bar, so
    they show over both the map's yellow peak and the bare table.
    """
    if len(grasps) != len(yaws_deg):
        raise ValueError(f"{len(grasps)} grasps but {len(yaws_deg)} yaws")
    theta = np.radians(yaws_deg)
    d = np.stack([np.cos(theta), np.sin(theta), np.zeros_like(theta)], axis=1)
    centres = grasps[:, :3].copy()
    centres[:, 2] = z
    a, ok_a = canvas.project(centres - 0.5 * length * d)
    b, ok_b = canvas.project(centres + 0.5 * length * d)
    width = max(1, int(round(0.6 * canvas.scale)))
    n = 0
    for pa, pb, good in zip(a, b, ok_a & ok_b, strict=True):
        if not good:
            continue
        pa_i, pb_i = (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1]))
        cv2.line(canvas.image, pa_i, pb_i, (0, 0, 0), width + 2, cv2.LINE_AA)
        cv2.line(canvas.image, pa_i, pb_i, COLOR_BAR, width, cv2.LINE_AA)
        n += 1
    return n


def rose_image(
    yaws_deg: np.ndarray, *, size_px: int, bin_deg: float = 15.0
) -> np.ndarray:
    """An RGBA rose of the cable directions, drawn top-down in the base frame.

    Orientations are axial (a cable has no front), so each is counted at its
    angle and at its angle plus 180 degrees, and the rose is symmetric. Zero
    is base +x (east), angles run counter-clockwise towards +y, as the table
    is seen from above with z up.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    edges = np.arange(0.0, 360.0 + bin_deg, bin_deg)
    both = np.concatenate([yaws_deg % 180.0, yaws_deg % 180.0 + 180.0])
    counts, _ = np.histogram(both, bins=edges)
    centres = np.radians(edges[:-1] + bin_deg / 2.0)

    dpi = 100
    fig = plt.figure(figsize=(size_px / dpi, size_px / dpi), dpi=dpi)
    ax = fig.add_axes((0.14, 0.14, 0.72, 0.72), projection="polar")
    fig.patch.set_alpha(0.0)
    ax.set_facecolor((0, 0, 0, 0.55))
    ax.bar(
        centres,
        counts,
        width=np.radians(bin_deg) * 0.92,
        bottom=0.0,
        color="#fca636",
        edgecolor="black",
        linewidth=0.6,
        alpha=0.95,
    )
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_xticks(np.radians([0, 90, 180, 270]))
    ax.set_xticklabels(["+x", "+y", "-x", "-y"], color="white", fontsize=11)
    ax.set_yticks([])
    ax.spines["polar"].set_color("white")
    ax.grid(color="white", alpha=0.3, linewidth=0.6)
    ax.set_title("cable yaw, base frame", color="white", fontsize=11, pad=6)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba()).copy()
    plt.close(fig)
    return rgba


def paste_rose(
    canvas: OverlayCanvas,
    yaws_deg: np.ndarray,
    *,
    below_lines: int = 3,
    frac: float = 0.32,
) -> None:
    """Composite the rose into the canvas's top-left corner, under the
    first *below_lines* badge lines, where the frame is empty."""
    h, w = canvas.image.shape[:2]
    size = int(round(min(h, w) * frac))
    rgba = rose_image(yaws_deg, size_px=size)
    rh, rw = rgba.shape[:2]
    # The badges' own geometry, so the rose starts just under them.
    unit = canvas._font_unit
    pad = max(4, int(6 * unit))
    (_, th), base = cv2.getTextSize(
        "Ag", cv2.FONT_HERSHEY_SIMPLEX, unit, max(1, int(round(2 * unit)))
    )
    y0 = min(h - rh, pad + below_lines * (th + base + pad))
    x0 = pad
    region = canvas.image[y0 : y0 + rh, x0 : x0 + rw].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    blended = alpha * rgba[..., :3].astype(np.float32) + (1 - alpha) * region
    canvas.image[y0 : y0 + rh, x0 : x0 + rw] = np.clip(blended, 0, 255).astype(np.uint8)


def main(args: Args) -> int:
    store = ExperimentStore(experiment_root(args.run, args.name, base=args.base))
    if not store.exists:
        raise SystemExit(f"No experiment at {store.root}. Check --name and --run.")
    config = store.load_config()
    grids = [f for f in config.factors if isinstance(f, GridPositionFactor)]
    if args.factor:
        grids = [f for f in grids if f.name == args.factor]
    if len(grids) != 1:
        raise SystemExit(
            f"Pass --factor: grid factors here are {[f.name for f in grids]}"
        )
    factor = grids[0]
    keep = list(factor.cells)
    if args.exclude_rows:
        rows = {
            factor.grid.parse_label(f"{r.strip()}{factor.grid.first_col}")[0]
            for r in args.exclude_rows
        }
        keep = [c for c in keep if c[0] not in rows]
    if not keep:
        raise SystemExit("No cells left after the filters.")
    factor = GridPositionFactor(
        factor.name, factor.grid, cells=keep, object=factor.object
    )
    records = store.attach_observed_positions(store.results())

    # The training set: one pass over the demos, cached beside the output so
    # a second experiment does not decode 28 large files again.
    dataset = pathlib.Path(args.dataset)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    events_path = out / f"{dataset.name}_grasp_positions.json"
    params = dict(
        open_above=args.open_above,
        closed_below=args.closed_below,
        grasp_max_z=args.grasp_max_z,
        release_min_z=args.release_min_z,
    )
    cached = json.loads(events_path.read_text()) if events_path.exists() else None
    if (
        cached is not None
        and cached.get("params") == params
        and all("cable_yaws" in v for v in cached["demos"].values())
    ):
        demos = {k: Events(**v) for k, v in cached["demos"].items()}
    else:
        files = sorted(dataset.glob("*.vla"))
        if not files:
            raise SystemExit(f"{dataset}: no .vla files")
        demos = {}
        for i, path in enumerate(files):
            ev = demo_events(path, **params)
            demos[path.stem] = ev
            logger.info(
                f"{path.stem} ({i + 1}/{len(files)}): "
                f"{len(ev.grasps)} grasps, {len(ev.releases)} releases"
            )
        events_path.write_text(
            json.dumps(
                {
                    "dataset": str(dataset),
                    "frame": "robot base",
                    "params": params,
                    "demos": {k: dataclasses.asdict(v) for k, v in demos.items()},
                },
                indent=2,
            )
        )
        print(f"Wrote {events_path}")
    grasps = np.asarray([p for ev in demos.values() for p in ev.grasps], dtype=float)
    releases = np.asarray(
        [p for ev in demos.values() for p in ev.releases], dtype=float
    )
    if grasps.size == 0:
        raise SystemExit("No grasps found in the demos; loosen the thresholds.")

    if args.heatmap:
        ref_image, camera = store.load_reference_view()
        if ref_image is None or "intrinsic" not in camera:
            raise SystemExit("No reference frame or camera calibration to draw on.")
        canvas = OverlayCanvas(
            np.asarray(ref_image)[..., :3],
            intrinsic=np.asarray(camera["intrinsic"]),
            extrinsic=np.asarray(camera["extrinsic"]),
            preprocessed=bool(camera.get("preprocessed", True)),
            scale=args.scale,
        )
        z = float(factor.grid.origin[2]) + args.draw_z
        # Grid first, density over it: the map is what the figure is about,
        # and the lines and letters are there to locate it, so they may sit
        # under its peak.
        canvas.draw_grid(factor.grid, labels=False, cells=factor.cells)
        training_heatmap(
            canvas,
            factor.grid,
            grasps,
            z=z,
            bandwidth=args.bandwidth_cm / 100.0,
            alpha_max=args.heat_alpha,
            floor=args.heat_floor,
        )
        if args.labels:
            # Letters over the map, outlined, so the cells under the peak
            # can still be named; the lines stay under it.
            centres = np.stack([factor.grid.cell_center(c) for c in factor.cells])
            centres[:, 2] = z
            px, ok = canvas.project(centres)
            for cell, point, good in zip(factor.cells, px, ok, strict=True):
                if good:
                    canvas._text(
                        factor.grid.label(cell),
                        point,
                        (255, 255, 255),
                        centred=True,
                        scale=0.55,
                        outline=True,
                    )
        if args.orientations:
            yaws = np.asarray(
                [y for ev in demos.values() for y in ev.cable_yaws], dtype=float
            )
            draw_orientation_bars(canvas, grasps, yaws, z=z, length=args.bar_cm / 100.0)
            paste_rose(canvas, yaws, below_lines=3)
        canvas.draw_badge(
            f"training cable positions: {len(grasps)} grasps in {len(demos)} demos",
            line=0,
        )
        canvas.draw_badge(
            f"kernel density on the table, {args.bandwidth_cm:g} cm bandwidth",
            line=1,
        )
        if args.orientations:
            canvas.draw_badge(
                "bars: cable direction at each grasp (across the finger pads)",
                line=2,
            )
        png_path = out / f"{args.out_name or f'{args.name}_training_heatmap'}.png"
        cv2.imwrite(str(png_path), canvas.image[:, :, ::-1])
        print(f"Wrote {png_path}")
        return 0

    # The report's own camera-view success map, then the training set on top.
    png_path = out / f"{args.out_name or f'{args.name}_with_training'}.png"
    written = plot_grid_success_projected(
        records, factor, store, png_path, scale=args.scale
    )
    if written is None:
        raise SystemExit("No reference frame or camera calibration to draw on.")
    bgr = cv2.imread(str(written))
    if bgr is None:
        raise SystemExit(f"Could not read back {written}")
    image = bgr[:, :, ::-1].copy()
    ref_image, camera = store.load_reference_view()
    if ref_image is None:
        raise SystemExit("No reference frame to draw on.")
    canvas = OverlayCanvas(
        np.asarray(ref_image)[..., :3],
        intrinsic=np.asarray(camera["intrinsic"]),
        extrinsic=np.asarray(camera["extrinsic"]),
        preprocessed=bool(camera.get("preprocessed", True)),
        scale=args.scale,
    )
    # Draw onto the report's rendering, projecting with its camera.
    np.copyto(canvas.image, image)

    z = float(factor.grid.origin[2]) + args.draw_z
    radius = max(3, int(round(2.2 * args.scale)))
    edge = max(1, int(round(0.5 * args.scale)))
    n_drawn = 0
    if args.releases and releases.size:
        pts = releases.copy()
        pts[:, 2] = z
        px, ok = canvas.project(pts)
        for p, good in zip(px, ok, strict=True):
            if good:
                _dot(
                    canvas.image,
                    (int(p[0]), int(p[1])),
                    COLOR_RELEASE,
                    radius=radius,
                    edge=edge,
                )
    pts = grasps.copy()
    pts[:, 2] = z
    px, ok = canvas.project(pts)
    for p, good in zip(px, ok, strict=True):
        if good:
            _dot(
                canvas.image,
                (int(p[0]), int(p[1])),
                COLOR_GRASP,
                radius=radius,
                edge=edge,
            )
            n_drawn += 1

    # Legend lines, below the report's own.
    line = 2 + (1 if any(r.observed_positions for r in records) else 0)
    canvas.draw_badge(
        f"training: {len(grasps)} cable grasps in {len(demos)} demos",
        color=COLOR_GRASP,
        line=line,
    )
    if args.releases and releases.size:
        canvas.draw_badge(
            f"training: {len(releases)} releases over the bag",
            color=COLOR_RELEASE,
            line=line + 1,
        )
    cv2.imwrite(str(png_path), canvas.image[:, :, ::-1])
    print(f"Wrote {png_path} ({n_drawn}/{len(grasps)} grasps in view)")
    return 0


if __name__ == "__main__":
    sys.exit(main(tyro.cli(Args)))
