"""Extract RGB/depth/cam frames + a short video from a LIBERO demo.

Dumps into ``assets/`` (next to this script's parent) for use by the
examples and tracker validation. Run from the package root:

    uv run python scripts/extract_demo_assets.py --hdf5 <path-to-libero-demo.hdf5>

Only needed if you want to regenerate the bundled assets from a fresh
LIBERO dump — the package already ships with the required files under
``assets/``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

DEFAULT_HDF5 = ""  # required — user supplies path to a LIBERO demo HDF5.
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "assets" / "demo"

# Optional LIBERO object name → reference-crop filename overrides (used by the
# disambiguation examples to give human-readable / positional names like
# "left_plate"). Anything in the demo's segmentation map but absent here gets
# auto-named by stripping the trailing ``_<digits>`` suffix.
REF_CROP_OVERRIDES = {
    "porcelain_mug_1": "white_mug",
    "white_yellow_mug_1": "yellow_and_white_mug",
    "red_coffee_mug_1": "red_mug",
    "plate_1": "left_plate",
    "plate_2": "right_plate",
}

# Segmentation entries that aren't task objects (robot, mounts, etc.).
NON_OBJECT_SUBSTRINGS = ("Panda", "NullMount", "Mount", "Robot")


def _resolve_ref_crops(seg_map: dict[str, str]) -> dict[str, str]:
    """Build a {libero_name → ref_crop_name} mapping for the present objects."""
    out: dict[str, str] = {}
    for libero_name in seg_map.values():
        if any(s in libero_name for s in NON_OBJECT_SUBSTRINGS):
            continue
        if libero_name in REF_CROP_OVERRIDES:
            out[libero_name] = REF_CROP_OVERRIDES[libero_name]
        else:
            out[libero_name] = re.sub(r"_\d+$", "", libero_name)
    return out


def _task_from_filename(hdf5_path: str) -> str:
    """Best-effort task description recovered from a LIBERO demo filename."""
    stem = Path(hdf5_path).stem
    # e.g. "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_..._demo"
    m = re.match(r"^[A-Z_]+?\d+_(?P<task>.+?)_demo$", stem)
    task = m.group("task") if m else stem
    return task.replace("_", " ")


def _flip(img: np.ndarray) -> np.ndarray:
    """LIBERO stores images upside down (OpenGL convention). Flip to display."""
    return np.ascontiguousarray(img[::-1, ::-1])


def _crop_mask(rgb: np.ndarray, mask: np.ndarray, pad: int = 4) -> np.ndarray:
    ys, xs = np.where(mask)
    H, W = mask.shape
    y0 = max(ys.min() - pad, 0)
    y1 = min(ys.max() + pad + 1, H)
    x0 = max(xs.min() - pad, 0)
    x1 = min(xs.max() + pad + 1, W)
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=DEFAULT_HDF5, required=not DEFAULT_HDF5)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--stride", type=int, default=4, help="take every Nth frame for the video"
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument(
        "--n-frames-saved", type=int, default=4, help="individual PNGs to write"
    )
    parser.add_argument(
        "--task",
        default=None,
        help="task description for the README (defaults to one parsed from the HDF5 filename)",
    )
    parser.add_argument(
        "--video-name",
        default="demo_video.mp4",
        help="filename for the rendered RGB video",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.hdf5, "r") as f:
        obs = f["data"][args.demo]["obs"]
        rgbs = np.asarray(obs["agentview_rgb"][:])
        depth = np.asarray(obs["agentview_depth"][:])
        seg = np.asarray(obs["agentview_segmentation"][:])
        K = np.asarray(obs["agentview_intrinsic"][0])
        E = np.asarray(obs["agentview_extrinsic"][0])
        seg_map = json.loads(f["data"].attrs["segmentation_name_mapping"])
        # Both cameras get the full extrinsic sequence — agentview happens to
        # be static in LIBERO but the multi-camera demo shouldn't bake that in.
        E_seq = np.asarray(obs["agentview_extrinsic"][:])
        rgbs_wrist = np.asarray(obs["eye_in_hand_rgb"][:])
        depth_wrist = np.asarray(obs["eye_in_hand_depth"][:])
        K_wrist = np.asarray(obs["eye_in_hand_intrinsic"][0])
        E_wrist_seq = np.asarray(obs["eye_in_hand_extrinsic"][:])

    # Display convention — row 0 at top.
    rgbs = np.stack([_flip(r) for r in rgbs], axis=0)
    depth = np.stack([_flip(d) for d in depth], axis=0)
    seg = np.stack([_flip(s) for s in seg], axis=0)
    rgbs_wrist = np.stack([_flip(r) for r in rgbs_wrist], axis=0)
    depth_wrist = np.stack([_flip(d) for d in depth_wrist], axis=0)

    # Save full video at stride.
    frames = rgbs[:: args.stride]
    video_path = out / args.video_name
    iio.imwrite(video_path, frames, fps=args.fps, codec="libx264")
    print(f"wrote {video_path} ({len(frames)} frames @ {args.fps} fps)")

    # Save all RGB frames as a compressed .npy bundle so examples can iterate
    # without re-decoding the mp4.
    np.savez_compressed(
        out / "demo_frames.npz",
        rgb=frames,
        intrinsics=K.astype(np.float32),
        extrinsics=E.astype(np.float32),
        stride=np.asarray(args.stride, dtype=np.int32),
    )
    print(f"wrote {out / 'demo_frames.npz'}  rgb shape={frames.shape}")

    # Save a handful of individual frames as PNG for quick-start examples.
    indices = np.linspace(0, len(frames) - 1, args.n_frames_saved, dtype=int)
    for i, idx in enumerate(indices):
        p = out / f"frame_{i:02d}.png"
        iio.imwrite(p, frames[idx])
        print(f"wrote {p}")

    # Save a single RGBD frame + camera info as a reusable fixture.
    np.savez_compressed(
        out / "demo_rgbd_frame.npz",
        rgb=rgbs[0],
        depth=depth[0].astype(np.float32),
        intrinsics=K.astype(np.float32),
        extrinsics=E.astype(np.float32),
    )
    print(f"wrote {out / 'demo_rgbd_frame.npz'}")

    # Multi-camera bundle: agentview + eye-in-hand RGB/depth/intrinsics, plus
    # the per-frame wrist extrinsics (the wrist moves with the gripper).
    wrist_d = depth_wrist[:: args.stride]
    if wrist_d.ndim == 4 and wrist_d.shape[-1] == 1:
        wrist_d = wrist_d[..., 0]
    cam0_d = depth[:: args.stride]
    if cam0_d.ndim == 4 and cam0_d.shape[-1] == 1:
        cam0_d = cam0_d[..., 0]
    np.savez_compressed(
        out / "demo_multi_camera.npz",
        rgb_cam0=frames,
        depth_cam0=cam0_d.astype(np.float32),
        intrinsics_cam0=K.astype(np.float32),
        extrinsics_cam0=E_seq[:: args.stride].astype(np.float32),
        rgb_cam1=rgbs_wrist[:: args.stride],
        depth_cam1=wrist_d.astype(np.float32),
        intrinsics_cam1=K_wrist.astype(np.float32),
        extrinsics_cam1=E_wrist_seq[:: args.stride].astype(np.float32),
        stride=np.asarray(args.stride, dtype=np.int32),
    )
    print(f"wrote {out / 'demo_multi_camera.npz'}")

    # Save GT masks as a reference for validation.
    # Collapse (H, W, 1) -> (H, W) and split per object.
    seg2d = seg[0][..., 0] if seg.ndim == 4 else seg[0]
    per_object_masks = {}
    for sid_s, name in seg_map.items():
        sid = int(sid_s)
        m = seg2d == sid
        if m.any():
            per_object_masks[name] = m
    np.savez_compressed(
        out / "demo_gt_masks_frame0.npz",
        **{k: v.astype(bool) for k, v in per_object_masks.items()},
    )
    print(f"wrote {out / 'demo_gt_masks_frame0.npz'}  objects={list(per_object_masks)}")

    # Reference crops for the disambiguation examples.
    ref_crops = _resolve_ref_crops(seg_map)
    refs_dir = out / "refs"
    refs_dir.mkdir(parents=True, exist_ok=True)
    for libero_name, ref_name in ref_crops.items():
        if libero_name not in per_object_masks:
            continue
        p = refs_dir / f"{ref_name}.png"
        iio.imwrite(p, _crop_mask(rgbs[0], per_object_masks[libero_name]))
        print(f"wrote {p}")

    # Small README so the assets are self-describing.
    task = args.task if args.task is not None else _task_from_filename(args.hdf5)
    (out / "README.md").write_text(
        f"""# `open_perception` demo assets

Extracted from [LIBERO demo]({args.hdf5}) demo={args.demo}.

Task: {task}.

Objects in `demo_gt_masks_frame0.npz`: {sorted(per_object_masks)}

| File | Purpose |
|---|---|
| `{args.video_name}` | short clip ({len(frames)} frames @ {args.fps} fps) for tracker examples |
| `demo_frames.npz` | same frames as a numpy array + camera info |
| `frame_00..03.png` | a few still frames for single-image detect examples |
| `demo_rgbd_frame.npz` | RGB + metric depth + intrinsics/extrinsics of frame 0 |
| `demo_multi_camera.npz` | agentview + eye-in-hand RGBD streams (per-frame wrist extrinsics) |
| `demo_gt_masks_frame0.npz` | per-object GT masks for frame 0 (validation) |
| `refs/*.png` | per-object reference crops for image/vlm disambiguation examples |

Re-run ``uv run python scripts/extract_demo_assets.py --hdf5 <path>`` from
the package root to regenerate after changing the source demo.
"""
    )
    print(f"wrote {out / 'README.md'}")


if __name__ == "__main__":
    main()
