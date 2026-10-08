# `open_perception` demo assets

Extracted from `datasets/libero_10_new/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5` demo=demo_1 (path is relative to the parent `policy_abstraction` repo root).

Task: put both the alphabet soup and the cream cheese box in the basket.

Objects in `demo_gt_masks_frame0.npz`: ['OnTheGroundPanda0', 'alphabet_soup_1', 'basket_1', 'cream_cheese_1', 'ketchup_1', 'tomato_sauce_1']

| File | Purpose |
|---|---|
| `demo_alphabet_basket.mp4` | short clip (73 frames @ 15 fps) for tracker examples |
| `demo_frames.npz` | same frames as a numpy array + camera info |
| `frame_00..03.png` | a few still frames for single-image detect examples |
| `demo_rgbd_frame.npz` | RGB + metric depth + intrinsics/extrinsics of frame 0 |
| `demo_multi_camera.npz` | agentview + eye-in-hand RGBD streams (per-frame wrist extrinsics) |
| `demo_gt_masks_frame0.npz` | per-object GT masks for frame 0 (validation) |
| `refs/*.png` | per-object reference crops for image/vlm disambiguation examples |

Re-run ``uv run python scripts/extract_demo_assets.py --hdf5 <path>`` from
the package root to regenerate after changing the source demo.
