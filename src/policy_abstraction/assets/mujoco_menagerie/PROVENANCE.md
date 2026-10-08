# Vendored MuJoCo Menagerie assets

Models that upstream [mujoco_menagerie][mm] does not ship yet, kept here so a
checkout renders identically on any machine. Everything in this directory is
third-party; the per-model `LICENSE` file governs it, not the repo's licence.

[mm]: https://github.com/google-deepmind/mujoco_menagerie

## `robotiq_2f140/`

The 2F-140 is the gripper on the UR5e cell, and menagerie ships only the 2F-85.

| | |
|---|---|
| Source | [mujoco_menagerie PR #161][pr] — "Add Robotiq 2F-140 model assets and documentation" |
| Branch | `2f140_PR` on `kaikwan/mujoco_menagerie` |
| Commit | `8766b62bf82f3a55b016783186143a6d4a816c97` |
| Retrieved | 2026-08-27 |
| Status | **Open, not merged.** Re-check before assuming it is upstream. |
| Licence | BSD-2-Clause (see `robotiq_2f140/LICENSE`) |

[pr]: https://github.com/google-deepmind/mujoco_menagerie/pull/161

Files are byte-for-byte as fetched. Fixes belong in
`policy_abstraction.visualization.robot_assembly`, so that when the PR merges the
directory can be deleted and `robot_descriptions` used instead — with no other
edit to reconcile.

### What was checked before trusting it

Both independently of the PR:

* **Tool length.** The fingertip sits 221.3 mm from the UR5e flange along the
  tool axis. The TCP taught on the real arm is 223.47 mm — a 2.2 mm agreement,
  which is the pad-mesh tip versus the taught contact plane.
* **Knuckle pivot.** `right_driver pos="0 0.030601 0.054905"` puts the pivot
  30.601 mm off the centreline. `policy_steering`'s four-bar constants, derived
  from robosuite's MJCF of the official Robotiq URDF, say 30.60 mm. Two
  unrelated derivations of the same hardware, agreeing to a micron.

### Known quirks

* `README.md` is adapted from the 2F-85's and still claims the collision pads
  were "broken up into two pads". This model has one (`pad_box1`) per finger.
* The two sides are not exact mirrors: at full open the pads sit 67.08 mm and
  67.75 mm off centre, a 0.67 mm asymmetry, and the `connect` anchors differ in
  the 4th decimal. Harmless for rendering; do not use this model to calibrate.
* The `driver` joint range is `[-0.8, 0]`, and 0 — nominally "fully open" —
  is a 128.6 mm aperture, not the datasheet's 140 mm stroke. See
  `robot_assembly.py`, which drives the linkage from the aperture rather than
  from this range.
