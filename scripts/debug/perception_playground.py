"""Interactive real-perception playground with a live Rerun 3D scene.

Type an object name in the terminal and watch the framework locate, track and
reconstruct it in 3D — the same ``OpenPerceptionClient`` call sequence, the same
back-projection and the same :class:`RerunVisualizer` entity paths that
``evaluate.py`` drives during a rollout. Nothing here re-implements perception;
the point is that what you see is what deployment sees.

What runs per frame
-------------------
``detect`` on the first frame after a query change (VLM box proposal → SAM2
mask, a few seconds), then ``track`` on every frame after that (SAM2
propagation, tens of milliseconds). That asymmetry *is* the deployment
behaviour: a rollout detects once at replan and tracks in between, so the
steady-state rate you see here is the rate guidance would get.

Terminal commands (typed while the stream runs)
-----------------------------------------------
``cable, bag``   replace the query set and re-detect
``+ mug``        add a query and re-detect
``!redetect``    re-run detection with the current queries
``!settings``    reprint the active perception configuration
``!quit``        stop (also ``Ctrl-C``)

Requires a running UR5 env server; this script never touches the arm — it only
reads observations. Depth comes from the RealSense cameras, so at least one
camera must be addressed as ``rs:<serial>`` for reconstruction to produce
anything. By default it attaches to a Rerun viewer already listening on
``0.0.0.0:9876`` (the address ``evaluate.py`` uses), so the viewer outlives the
run and keeps your camera angle across restarts::

    # 1. viewer — the `rerun` binary is not on PATH, take it from the venv
    .venv/bin/rerun --port 9876

    # 2. env server
    python scripts/servers/serve_ur5_env.py \\
        --tasks "perception playground" \\
        --cameras exterior_image_1_left=rs:262722073432:640x480@30 \\
        --wrist-camera '' \\
        --calibration-path configs/ur5_camera_calibration.json \\
        --calibration-base-frame base_link --dry-run

    # 3. playground
    python scripts/debug/perception_playground.py --queries cable bag

Headless, with no viewer to attach to, record instead::

    python scripts/debug/perception_playground.py --queries cable bag \\
        --connect-addr None --max-frames 60 --save-rrd tmp/playground.rrd
"""

from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import tyro
from loguru import logger

from policy_abstraction.envs.spec import CameraSpec, EnvSpec
from policy_abstraction.envs.websocket_client import WebsocketEnvClient
from policy_abstraction.evaluation.abstraction import AbstractionResult
from policy_abstraction.perception.open_perception_client import (
    DEFAULT_VLM_MODEL,
    OpenPerceptionClient,
)
from policy_abstraction.utils.depth import mujoco_depth_to_metric
from policy_abstraction.utils.segmentation import build_segmented_pointcloud
from policy_abstraction.visualization.rerun import RerunVisualizer

# Distinct colours per tracked instance, reused by the 3D clouds so an object's
# points, centroid and 2D overlay all read as the same thing.
PALETTE = np.array(
    [
        [235, 70, 70],
        [70, 215, 110],
        [90, 160, 250],
        [245, 195, 60],
        [215, 110, 230],
        [90, 220, 220],
    ],
    dtype=np.float32,
)


@dataclass
class Args:
    """Interactive perception playground."""

    queries: list[str] = field(default_factory=lambda: ["cable", "bag"])
    """Objects to locate at startup. Retype any time in the terminal."""

    env_host: str = "127.0.0.1"
    env_port: int = 8765
    """Where the UR5 env server is listening (``8765 + --instance N``)."""

    task_id: int = 0
    """Task index passed to ``make_env``; only picks the prompt string."""

    robot_description: str = "ur5e_mj_description"
    """``robot_descriptions`` entry rendered as the robot body. The menagerie
    UR5e is arm-only (6 joints, no gripper), so the fingers are drawn from our
    own 2F-140 geometry instead — see ``_log_gripper``."""

    flip_u: Literal["auto", "true", "false"] = "auto"
    """Undo the u-axis mirror when back-projecting. ``auto`` reads the primary
    camera's ``image_transform``, matching ``evaluate.py``. Getting this wrong
    mirrors every cloud about the image centre-line."""

    detector_backend: Literal["openrouter", "gemini"] = "openrouter"
    detector_model: Optional[str] = None
    """VLM that proposes boxes. Defaults to ``DEFAULT_VLM_MODEL``."""

    device: Optional[str] = None
    """Torch device for SAM2. ``None`` lets open_perception choose."""

    max_hz: float = 10.0
    """Upper bound on the tracking loop rate. Perception is the bottleneck well
    before this matters; it exists so a fast tracker can't spin the CPU."""

    max_points: int = 60000
    """Cap on the logged scene cloud. Stride-subsampled, not cropped."""

    depth_max: float = 2.0
    """Discard depth beyond this (metres) — trims the room behind the table."""

    connect: Optional[str] = "0.0.0.0:9876"
    """``host:port`` of a Rerun viewer already listening — same flag shape and
    default as ``evaluate.py``'s ``--viz.connect``, and wrapped into a
    ``rerun+http://.../proxy`` URL the same way. A viewer you started yourself
    outlives the run, so re-running the playground doesn't cost you your camera
    angle and layout. Pass ``--connect None`` to detach (then ``--spawn`` or
    ``--save-rrd``)."""

    spawn: bool = False
    """Launch a throwaway viewer instead of connecting. Only consulted when
    ``--connect`` is ``None``, and needs a display — on a headless host use
    ``--save-rrd``."""

    redetect_every: int = 0
    """Force a re-detect every N frames (0 = track until you ask). Useful for
    watching detection cost and drift recovery back-to-back."""

    max_frames: int = 0
    """Stop after N frames (0 = run until you quit). Setting this also stops
    end-of-stdin from ending the run, so a scripted/headless invocation isn't
    cut short the instant it finds nothing to read."""

    save_rrd: Optional[str] = None
    """Also write the recording to this ``.rrd``. Needed on a headless host,
    where ``--spawn`` has no display to draw on: record here, open it in a
    local viewer with ``rerun <file>``."""


# --------------------------------------------------------------------------- #
# Terminal input                                                              #
# --------------------------------------------------------------------------- #


def _stdin_reader(q: "queue.Queue[str]", quit_on_eof: bool) -> None:
    """Feed terminal lines to the main loop.

    A daemon thread rather than a select() poll so the loop never blocks on
    input: perception keeps streaming while you decide what to type.

    End-of-stdin ends an interactive run (Ctrl-D), but must not end a run with
    a frame budget — a headless invocation has no stdin to speak of and would
    otherwise stop before the first detection returned.
    """
    for line in sys.stdin:
        q.put(line.strip())
    if quit_on_eof:
        q.put("!quit")


def _parse_queries(text: str, current: list[str]) -> list[str] | None:
    """Interpret one input line as a new query set, or ``None`` for no change."""
    if text.startswith("+"):
        added = [t.strip() for t in text[1:].replace(",", " ").split() if t.strip()]
        return current + [a for a in added if a not in current]
    parts = [t.strip() for t in text.replace(",", " ").split() if t.strip()]
    return parts or None


# --------------------------------------------------------------------------- #
# Frame assembly                                                              #
# --------------------------------------------------------------------------- #


def _iter_camera_views(
    spec: EnvSpec, obs: dict, camera_params: dict
) -> list[tuple[CameraSpec, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """``(cam, rgb, depth_m, K3, E)`` per depth-capable camera, primary first.

    Mirrors ``RolloutAbstractor._iter_camera_views``: same ordering (the fusion
    stage treats view 0 as the main camera), same image transform, same
    metric-depth rule. Cameras without depth are skipped — they can't
    reconstruct.
    """
    ordered = [spec.primary_camera] + [c for c in spec.cameras if not c.is_primary]
    out = []
    for cam in ordered:
        if cam.depth_key is None or cam.depth_key not in obs:
            continue
        if cam.image_key not in obs or cam.name not in camera_params:
            continue
        params = camera_params[cam.name]
        K, E = params.get("intrinsic"), params.get("extrinsic")
        if K is None or E is None:
            continue
        K, E = np.asarray(K), np.asarray(E)
        rgb = np.ascontiguousarray(
            cam.apply_image_transform(np.asarray(obs[cam.image_key]))
        )
        dep = np.ascontiguousarray(
            cam.apply_image_transform(np.asarray(obs[cam.depth_key]))
        )
        if not cam.depth_is_metric and dep.max() <= 1.0:
            dep = mujoco_depth_to_metric(dep, near=cam.depth_near, far=cam.depth_far)
        out.append((cam, rgb, dep, K[:3, :3] if K.shape == (4, 4) else K, E))
    return out


def _object_clouds(
    views, masks_per_view, labels: list[str], flip_u: bool, depth_max: float
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Back-project each instance mask into base-frame clouds + centroids.

    Instance IDs are stamped into a synthetic segmentation image and fed
    through ``build_segmented_pointcloud`` — the same call the abstractor makes
    — so the geometry here is not an approximation of the deployment path, it
    is that path.
    """
    per_label: dict[str, list[np.ndarray]] = {}
    for v, (_cam, rgb, dep, K3, E) in enumerate(views):
        masks = masks_per_view[v] if v < len(masks_per_view) else {}
        if not masks:
            continue
        seg = np.zeros(dep.shape[:2], dtype=np.int32)
        id_to_label: dict[int, str] = {}
        for inst_id, (name, mask) in masks.items():
            if mask is None:
                continue
            seg[np.asarray(mask, dtype=bool)] = inst_id
            id_to_label[inst_id] = f"{name}#{inst_id}"
        pcd = build_segmented_pointcloud(
            rgb, dep, seg, K3, E, depth_max=depth_max, preprocessed=flip_u
        )
        if pcd.shape[0] == 0:
            continue
        ids = pcd[:, 6].astype(np.int32)
        for inst_id, label in id_to_label.items():
            sel = pcd[ids == inst_id]
            if sel.shape[0]:
                per_label.setdefault(label, []).append(sel)

    clouds, centroids = {}, {}
    for label, parts in per_label.items():
        pts = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
        if label not in labels:
            labels.append(label)
        colour = PALETTE[labels.index(label) % len(PALETTE)]
        tinted = pts.copy()
        tinted[:, 3:6] = 0.45 * pts[:, 3:6] + 0.55 * colour
        clouds[label] = tinted
        centroids[label] = pts[:, :3].mean(axis=0)
    return clouds, centroids


def _scene_cloud(views, flip_u: bool, depth_max: float, cap: int) -> np.ndarray:
    """Every valid-depth pixel across all views, stride-capped."""
    parts = []
    for _cam, rgb, dep, K3, E in views:
        seg_zero = np.zeros(dep.shape[:2], dtype=np.int32)
        pcd = build_segmented_pointcloud(
            rgb, dep, seg_zero, K3, E, depth_max=depth_max, preprocessed=flip_u
        )
        if pcd.shape[0]:
            parts.append(pcd)
    if not parts:
        return np.zeros((0, 7), dtype=np.float32)
    scene = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=0)
    if cap and scene.shape[0] > cap:
        scene = scene[:: int(np.ceil(scene.shape[0] / cap))]
    return scene


def _log_gripper(spec: EnvSpec, obs: dict) -> None:
    """Draw the 2F-140 fingertips, which the arm-only URDF cannot show.

    Uses the gripper geometry the env spec selects, so the rendered opening is
    the real 140 mm stroke rather than a stand-in.
    """
    import rerun as rr

    from policy_abstraction.utils.gripper import compute_finger_tip_positions_single
    from policy_steering.robot_config import load_gripper_geometry

    if "eef_pos" not in obs or "eef_axis_angle" not in obs:
        return
    geom = load_gripper_geometry(spec.robot.gripper.geometry_id)
    tips = compute_finger_tip_positions_single(
        np.asarray(obs["eef_pos"], dtype=np.float64),
        np.asarray(obs["eef_axis_angle"], dtype=np.float64),
        np.asarray(obs[spec.robot.gripper_qpos_key], dtype=np.float64),
        geom,
    )
    rr.log(
        "robot/gripper_fingers",
        rr.Points3D(
            [tips["gripper_l"].tolist(), tips["gripper_r"].tolist()],
            radii=0.008,
            colors=[[255, 140, 40, 255], [255, 140, 40, 255]],
        ),
    )


# --------------------------------------------------------------------------- #
# Panels                                                                      #
# --------------------------------------------------------------------------- #


def _settings_md(args: Args, spec: EnvSpec, flip_u: bool, views) -> str:
    cams = "<br>".join(
        f"`{c.name}` ({c.role}){' depth' if c.depth_key else ' **no depth**'}"
        for c in spec.cameras
    )
    return "\n".join(
        [
            "## Perception settings",
            "",
            "| setting | value |",
            "|---|---|",
            "| pipeline | `detect.gemini(grounded)` → `sam2(bbox)` → "
            "`filter.size` → `filter.erode` → `reconstruct` → `fuse_views` |",
            f"| detector backend | `{args.detector_backend}` |",
            f"| detector model | `{args.detector_model or DEFAULT_VLM_MODEL}` |",
            "| tracker | `sam2` (video propagation) |",
            f"| device | `{args.device or 'auto'}` |",
            f"| flip_u | `{flip_u}` (`--flip-u {args.flip_u}`) |",
            f"| primary view | `{spec.primary_camera.name}` (idx 0 for fusion) |",
            f"| depth cameras | {len(views)} of {len(spec.cameras)} |",
            f"| depth cap | {args.depth_max:.2f} m |",
            f"| gripper geometry | `{spec.robot.gripper.geometry_id}` |",
            f"| cameras | {cams} |",
        ]
    )


def _stats_md(
    frame: int,
    queries: list[str],
    clouds: dict[str, np.ndarray],
    centroids: dict[str, np.ndarray],
    detect_ms: float,
    last_ms: float,
    track_rate: float,
    n_scene: int,
    since_detect: int,
) -> str:
    rows = [
        "## Live stats",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| frame | {frame} |",
        f"| queries | {', '.join(queries) or '—'} |",
        f"| detect (frame 0) | {detect_ms:.0f} ms |",
        f"| last track | {last_ms:.0f} ms |",
        f"| track rate | {track_rate:.1f} Hz |",
        f"| frames since detect | {since_detect} |",
        f"| scene points | {n_scene} |",
        f"| instances found | {len(clouds)} |",
        "",
        "### Tracked instances",
        "",
    ]
    if not clouds:
        rows.append("_nothing matched — try a different word, or `!redetect`_")
        return "\n".join(rows)
    rows += ["| instance | points | centroid (base frame, m) |", "|---|---:|---|"]
    for label, pts in sorted(clouds.items()):
        c = centroids[label]
        rows.append(
            f"| `{label}` | {pts.shape[0]} | ({c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}) |"
        )
    return "\n".join(rows)


def _send_blueprint(spec: EnvSpec) -> None:
    """Layout built for this tool: 3D scene left, perception panels right.

    ``RerunVisualizer.send_blueprint`` targets a rollout (reference distances,
    alignment charts) and would leave the settings/stats panels unplaced, so
    this picks the subset that matters here.
    """
    import rerun as rr
    import rerun.blueprint as rrb

    cam_names = [c.name for c in spec.cameras]
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(name="Scene", origin="/"),
                rrb.Vertical(
                    rrb.Tabs(
                        *[
                            rrb.Spatial2DView(
                                name=f"{n} overlay", origin=f"perception/{n}/overlay"
                            )
                            for n in cam_names
                        ],
                        *[
                            rrb.Spatial2DView(name=f"{n} raw", origin=f"camera/{n}")
                            for n in cam_names
                        ],
                    ),
                    rrb.Tabs(
                        rrb.TextDocumentView(name="Stats", origin="playground/stats"),
                        rrb.TextDocumentView(
                            name="Settings", origin="playground/settings"
                        ),
                        rrb.TextDocumentView(
                            name="Stage timings", origin="perception/timings"
                        ),
                    ),
                    row_shares=[3, 2],
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main(args: Args) -> None:
    import rerun as rr

    env = WebsocketEnvClient(host=args.env_host, port=args.env_port, retry_interval=2.0)
    suite = env.setup_suite()
    spec = EnvSpec.from_dict(suite["env_spec"])
    env.make_env(task_id=args.task_id, seed=0)
    camera_params = env.get_camera_params(camera_names=[c.name for c in spec.cameras])
    base = env.get_robot_base()

    flip_u = (
        spec.primary_camera.image_transform != "none"
        if args.flip_u == "auto"
        else args.flip_u == "true"
    )
    primary_idx = next((i for i, c in enumerate(spec.cameras) if c.is_primary), 0)

    # Probe one observation before building anything else: a cell with no depth
    # camera can still stream images, and the failure would otherwise surface
    # as a silently empty 3D view.
    obs = env.get_obs()
    views = _iter_camera_views(spec, obs, camera_params)
    if not views:
        raise SystemExit(
            "No camera publishes depth + calibration, so nothing can be "
            "reconstructed. Address a RealSense as `rs:<serial>` and pass "
            "--calibration-path to the env server."
        )
    logger.info(
        f"[playground] {len(views)} depth view(s): "
        f"{', '.join(c.name for c, *_ in views)}  flip_u={flip_u}"
    )

    perception = OpenPerceptionClient(
        primary_view_idx=primary_idx,
        device=args.device,
        detector_backend=args.detector_backend,
        detector_model=args.detector_model,
        flip_u=flip_u,
    )

    # rerun 0.30 rejects a bare host:port ("Invalid or missing scheme"), so the
    # flag stays a plain address and gets wrapped here — same shape as
    # evaluate.py, which builds the identical URL from --viz.connect.
    connect_url = f"rerun+http://{args.connect}/proxy" if args.connect else None
    viz = RerunVisualizer(
        application_id="perception_playground",
        spawn=args.spawn and connect_url is None,
        connect_addr=connect_url,
        robot_description_name=args.robot_description,
        log_action_chunks=False,
        log_predicted_states=False,
        log_action_chunk_states=False,
    )
    viz.set_env_spec(spec)
    viz.setup("perception playground", policy_label="none (perception only)")
    if args.save_rrd:
        # Attach the file sink before anything but the task text is logged, so
        # the .rrd holds the whole run rather than its tail.
        rr.save(args.save_rrd)
        logger.info(f"[playground] recording to {args.save_rrd}")
    viz.set_robot_base(np.asarray(base["base_pos"]), np.asarray(base["base_ori"]))
    _send_blueprint(spec)

    settings_md = _settings_md(args, spec, flip_u, views)
    rr.log(
        "playground/settings", rr.TextDocument(settings_md, media_type="text/markdown")
    )

    q: "queue.Queue[str]" = queue.Queue()
    threading.Thread(
        target=_stdin_reader, args=(q, args.max_frames == 0), daemon=True
    ).start()

    queries = list(args.queries)
    labels: list[str] = []
    detect_ms = 0.0
    track_ms: list[float] = []
    frame = 0
    since_detect = 0
    pending_detect = True  # first frame always detects

    print(_banner(queries))
    min_dt = 1.0 / args.max_hz if args.max_hz > 0 else 0.0

    try:
        while True:
            loop_start = time.perf_counter()

            # --- terminal input ------------------------------------------
            while not q.empty():
                line = q.get_nowait()
                if not line:
                    continue
                if line in ("!quit", "!q"):
                    return
                if line == "!redetect":
                    pending_detect = True
                    continue
                if line == "!settings":
                    print(settings_md)
                    continue
                new_queries = _parse_queries(line, queries)
                if new_queries is None:
                    continue
                queries = new_queries
                pending_detect = True
                print(f"[playground] queries -> {', '.join(queries)} (re-detecting)")

            if args.redetect_every and since_detect >= args.redetect_every:
                pending_detect = True

            # --- observation ---------------------------------------------
            obs = env.get_obs()
            views = _iter_camera_views(spec, obs, camera_params)
            if not views:
                logger.warning("[playground] frame with no depth view; skipping")
                continue

            # --- perception ----------------------------------------------
            from open_perception import CameraInfo  # lazy: sys.path-mounted

            if pending_detect:
                # reset() forces the next call down the detect branch, which is
                # what a replan does in deployment.
                perception.reset()
                labels = []
                pending_detect = False
                is_detect = True
            else:
                is_detect = False

            cams = [
                CameraInfo(intrinsics=K, extrinsics=E) for _c, _r, _d, K, E in views
            ]
            t0 = time.perf_counter()
            masks_per_view = perception(
                [v[1] for v in views], [v[2] for v in views], cams, queries
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if is_detect:
                detect_ms = elapsed_ms
                since_detect = 0
                track_ms.clear()
            else:
                track_ms.append(elapsed_ms)
                since_detect += 1
            recent = track_ms[-20:]
            track_rate = 1000.0 / (sum(recent) / len(recent)) if recent else 0.0

            # --- 3D reconstruction ---------------------------------------
            clouds, centroids = _object_clouds(
                views, masks_per_view, labels, flip_u, args.depth_max
            )
            scene = _scene_cloud(views, flip_u, args.depth_max, args.max_points)

            # --- log via the deployment visualiser ------------------------
            result = AbstractionResult(
                live_distances={},
                aligned_t=0,
                ref_distances={},
                distance_errors={},
                mta_progress=0.0,
                pair_labels=[],
                object_pointclouds=clouds,
                object_centroids=centroids,
                scene_pointcloud=scene,
                # Overlays are logged separately below: the visualiser maps
                # them positionally onto ``spec.cameras``, but our view list is
                # primary-first and skips depth-less cameras, so the two orders
                # can disagree and mislabel a panel.
                perception_overlays=None,
                perception_timings_md=perception.last_timings_md(),
            )
            joint_pos = np.asarray(obs.get("joint_pos", np.zeros(6)), dtype=np.float64)
            grip = np.asarray(obs.get("gripper_qpos", [0.0]), dtype=np.float64)
            robot_state = np.concatenate(
                [
                    np.asarray(obs.get("eef_pos", np.zeros(3)), dtype=np.float64),
                    np.asarray(
                        obs.get("eef_axis_angle", np.zeros(3)), dtype=np.float64
                    ),
                    grip,
                ]
            )
            viz.update(
                timestep=frame,
                obs=obs,
                robot_state=robot_state,
                # 7 values against the arm-only model's 6 qpos: the visualiser
                # fills the first 6 and ignores the aperture, which is what we
                # want — see _log_gripper for the fingers.
                robot_joints=np.concatenate([joint_pos, grip]),
                camera_params=camera_params,
                abstraction_result=result,
            )
            _log_gripper(spec, obs)

            # Keyed by the camera each overlay actually came from.
            for (cam, *_), overlay in zip(
                views, perception.render_overlays(), strict=False
            ):
                rr.log(f"perception/{cam.name}/overlay", rr.Image(overlay))

            stats_md = _stats_md(
                frame,
                queries,
                clouds,
                centroids,
                detect_ms,
                elapsed_ms,
                track_rate,
                int(scene.shape[0]),
                since_detect,
            )
            rr.log(
                "playground/stats",
                rr.TextDocument(stats_md, media_type="text/markdown"),
            )

            if is_detect:
                # Centroids on the console too: on a headless run this is the
                # only chance to notice a calibration or flip_u problem, which
                # shows up as plausible-looking clouds in an impossible place.
                found = (
                    ", ".join(
                        f"{k} at ({centroids[k][0]:+.2f}, {centroids[k][1]:+.2f}, "
                        f"{centroids[k][2]:+.2f})"
                        for k in sorted(clouds)
                    )
                    or "nothing"
                )
                print(f"[playground] detect {elapsed_ms:.0f} ms -> {found}")

            frame += 1
            if args.max_frames and frame >= args.max_frames:
                print(f"[playground] reached --max-frames {args.max_frames}")
                return
            spare = min_dt - (time.perf_counter() - loop_start)
            if spare > 0:
                time.sleep(spare)
    except KeyboardInterrupt:
        print()
    finally:
        print(_summary(frame, detect_ms, track_ms, args.save_rrd))
        logger.info("[playground] disconnecting")
        env.disconnect()
        viz.close()


def _summary(
    frame: int, detect_ms: float, track_ms: list[float], rrd: str | None
) -> str:
    """Report what the run actually cost, rather than leaving it to be inferred.

    Track times are per perception call only; the end-to-end loop also
    reconstructs and logs, so the loop rate is lower than ``1000 / median``.
    """
    lines = ["", f"  frames: {frame}", f"  detect: {detect_ms:.0f} ms"]
    if track_ms:
        arr = np.array(track_ms)
        lines += [
            f"  track:  {np.median(arr):.0f} ms median "
            f"({arr.min():.0f}-{arr.max():.0f} ms over {len(arr)} calls)"
            f"  =  {1000.0 / np.median(arr):.1f} Hz",
        ]
    if rrd:
        lines += ["", f"  recording: {rrd}", "  open with: rerun <file>"]
    return "\n".join(lines) + "\n"


def _banner(queries: list[str]) -> str:
    return (
        "\n"
        "  perception playground — the robot is not commanded, only observed\n"
        f"  tracking: {', '.join(queries)}\n"
        "\n"
        "  type object names ......  cable, bag\n"
        "  add one ................  + mug\n"
        "  re-detect ..............  !redetect\n"
        "  show settings ..........  !settings\n"
        "  quit ...................  !quit  (or Ctrl-C)\n"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
