# `open_perception` demo assets

Extracted from [LIBERO demo](datasets/libero_10_new/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5) demo=demo_0.

Task: put the white mug on the left plate and put the yellow-and-white mug on the right plate.

Objects in `demo_gt_masks_frame0.npz`: ['OnTheGroundPanda0', 'plate_1', 'plate_2', 'porcelain_mug_1', 'red_coffee_mug_1', 'white_yellow_mug_1']

| File | Purpose |
|---|---|
| `demo_mugs_plates.mp4` | short clip (69 frames @ 15 fps) for tracker examples |
| `demo_frames.npz` | same frames as a numpy array + camera info |
| `frame_00..03.png` | a few still frames for single-image detect examples |
| `demo_rgbd_frame.npz` | RGB + metric depth + intrinsics/extrinsics of frame 0 |
| `demo_multi_camera.npz` | agentview + eye-in-hand RGBD streams (per-frame wrist extrinsics) |
| `demo_gt_masks_frame0.npz` | per-object GT masks for frame 0 (validation) |
| `refs/*.png` | per-object reference crops for image/vlm disambiguation examples |

Re-run `uv run python scripts/extract_demo_assets.py --hdf5 <path>` from
the package root to regenerate after changing the source demo.
