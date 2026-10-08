"""Track one clicked point through a capture with Kalib, non-interactively.

Runs with **Kalib's** interpreter (the ``kalib`` conda env), not this repo's:
it imports ``easycalib_demo`` from the Kalib checkout and drives the same
``Kalib`` class its own demo uses, with two changes:

* the operator's click comes from ``--click U V`` instead of Kalib's OpenCV
  window, so the GUI in ``policy_abstraction.calibration.gui`` owns the
  interaction;
* the result is written as plain JSON (tracked pixel + visibility per frame,
  plus Kalib's own PnP pose as a cross-check) instead of a pickle.

Invoked by :class:`policy_abstraction.calibration.kalib_runner.KalibTracker`;
not meant to be run by hand, but it can be::

    ~/miniconda3/envs/kalib/bin/python scripts/setup/_kalib_track_driver.py \\
        --root-dir data/calibration/exterior --kalib-root ~/dev/Kalib \\
        --click 312.5 244.0 --result data/calibration/exterior/Kalib/tracking_result.json

Only the standard library and numpy are imported before the Kalib checkout is
on ``sys.path``; nothing from ``policy_abstraction`` is used here.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import sys
import time


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    p.add_argument(
        "--root-dir", required=True, help="capture directory (frames + JSON)"
    )
    p.add_argument("--kalib-root", required=True, help="the Kalib checkout")
    p.add_argument(
        "--click",
        nargs=2,
        type=float,
        required=True,
        metavar=("U", "V"),
        help="pixel of the tracked point in the first frame",
    )
    p.add_argument("--device", type=int, default=0, help="CUDA device index")
    p.add_argument(
        "--tracker", default="spatial_tracker", choices=["spatial_tracker", "cotracker"]
    )
    p.add_argument("--result", required=True, help="where to write the result JSON")
    p.add_argument(
        "--skip-kalib-pnp",
        action="store_true",
        help="only track; do not run Kalib's own PnP cross-check",
    )
    return p.parse_args()


def _write_manipulator_config(
    root_dir: pathlib.Path, kalib_root: pathlib.Path
) -> pathlib.Path:
    """Kalib's argument parser insists on a manipulator config; only the
    keypoint *count* matters for tracking + PnP (rendering is skipped)."""
    urdf = kalib_root / "assets" / "ur5e" / "ur5e.urdf"
    cfg = {
        "manipulator": {
            "mesh_paths": [],
            "urdf_path": str(urdf),
            "keypoints": ["tcp"],
            "name": "ur5e",
        }
    }
    out = root_dir / "Kalib" / "manipulator_config.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg, indent=2))
    return out


def main() -> int:
    args = _parse()
    t0 = time.monotonic()
    root_dir = pathlib.Path(args.root_dir).resolve()
    kalib_root = pathlib.Path(os.path.expanduser(args.kalib_root)).resolve()
    result_path = pathlib.Path(args.result).resolve()
    if not (kalib_root / "easycalib_demo.py").exists():
        print(f"ERROR: {kalib_root} is not a Kalib checkout", file=sys.stderr)
        return 2
    n_images = len(sorted(root_dir.glob("[0-9][0-9][0-9][0-9][0-9][0-9].jpg")))
    if n_images == 0:
        print(f"ERROR: no frames in {root_dir}", file=sys.stderr)
        return 2

    # Kalib resolves its third_party paths relative to its own root.
    os.chdir(kalib_root)
    if str(kalib_root) not in sys.path:
        sys.path.insert(0, str(kalib_root))

    stale = root_dir / "Kalib" / "Kalib_outputs" / "pnp_inference_res.pkl"
    if stale.exists():
        # Kalib silently skips tracking when this exists.
        stale.unlink()

    cfg_path = _write_manipulator_config(root_dir, kalib_root)
    spat_root = kalib_root / "third_party" / "spatial_tracker"
    sys.argv = [
        "easycalib_demo.py",
        "--root_dir",
        str(root_dir),
        "--manipulator_config_path",
        str(cfg_path),
        "--keypoint_ids",
        "0",
        "--caliberate_method",
        "pnp",
        "--pnp_refinement",
        "true",
        "--use_pnp_ransac",
        "false",
        "--win_len",
        "1",
        "--verbose",
        "--cut_off",
        "-1",
        "--keypoint_tracking_method",
        args.tracker,
        "--spatial_tracker_repo_path",
        str(spat_root) + "/",
        "--renderer_device_id",
        str(args.device),
        "--tracking_device_id",
        str(args.device),
        "--mask_inference_device_id",
        str(args.device),
        "--var_x",
        "0",
        "--var_y",
        "0",
        "--skip_rendering",
    ]
    print(
        f"[driver] {n_images} frames in {root_dir}; click=({args.click[0]:.1f}, {args.click[1]:.1f})"
    )
    print("[driver] importing Kalib ...", flush=True)
    import easycalib_demo as ed  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
    import numpy as np

    kargs = ed.parse_easycalib_default_args(cli_input=True)
    kalib = ed.Kalib(kargs)

    # The click, instead of Kalib's PointDrawer window.
    click = [[float(args.click[0]), float(args.click[1])]]
    kalib.mask_end_effector_position = lambda first_frame_path: click  # type: ignore[method-assign]

    already = kalib.preprocess(ed.SAVE_TAG)
    if already:
        print(
            "ERROR: Kalib found a previous result and skipped tracking", file=sys.stderr
        )
        return 3
    print("[driver] tracking ...", flush=True)
    kalib.kpt_tracking()

    data_dir = root_dir / ed.SAVE_TAG / "saved_frames_visibility" / "tracker_data"
    tracks = np.load(data_dir / "pred_tracks.npy")  # (1, T, N, 2)
    vis = np.load(data_dir / "pred_visibility.npy")  # (1, T, N)
    tracks = np.asarray(tracks)[0, :, 0, :2].astype(float)
    vis = np.asarray(vis)[0, :, 0]
    vis = vis > 0.5 if vis.dtype != bool else vis
    if tracks.shape[0] != n_images:
        print(
            f"WARNING: tracker returned {tracks.shape[0]} frames for {n_images} images",
            file=sys.stderr,
        )

    kalib_T = None
    kalib_reproj = None
    if not args.skip_kalib_pnp:
        print("[driver] Kalib PnP cross-check ...", flush=True)
        try:
            kalib.run_caliberate_camera()
            pkl = (
                root_dir
                / ed.SAVE_TAG
                / f"{ed.SAVE_TAG}_outputs"
                / "pnp_inference_res.pkl"
            )
            with open(pkl, "rb") as f:
                res = pickle.load(f)
            mats = np.asarray(res["pnp_transform_predicted_mats"], dtype=float)
            kalib_T = (mats[0] if mats.ndim == 3 else mats).reshape(4, 4).tolist()
            kalib_reproj = float(res["avg_reprojection_err"])
        except Exception as exc:  # the cross-check is optional
            print(f"WARNING: Kalib PnP failed: {exc}", file=sys.stderr)

    out = {
        "tracks": [
            [None, None] if not np.all(np.isfinite(p)) else [float(p[0]), float(p[1])]
            for p in tracks
        ],
        "visibility": [bool(v) for v in vis],
        "kalib_T_cam_parent": kalib_T,
        "kalib_reproj_px": kalib_reproj,
        "elapsed_s": time.monotonic() - t0,
        "tracker": args.tracker,
        "click": click[0],
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(out))
    visible = int(np.sum(vis))
    print(
        f"[driver] done: {visible}/{len(vis)} frames visible, {out['elapsed_s']:.0f} s -> {result_path}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
