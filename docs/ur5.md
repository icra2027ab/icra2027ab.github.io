# Running on the real UR5e

The UR5e adapter makes a physical arm look like any other env in this repo. It
speaks the same WebSocket protocol as the LIBERO-PRO server,
publishes an `EnvSpec`, and plugs into the same `EpisodeRunner`, so switching
between simulation and hardware is a subcommand change:

```bash
python scripts/evaluate.py ... env:libero policy:openpi     # simulator
python scripts/evaluate.py ... env:ur5    policy:lerobot    # real arm
```

Read the [Safety](#safety) section before the first powered run.

---

## Architecture

Three processes, two venvs:

| Process | Venv | Talks |
|---|---|---|
| `scripts/evaluate.py` | this repo's `.venv` | WebSocket → env, ZMQ → policy |
| `scripts/servers/serve_ur5_env.py` | `physical-ai-sdk/.venv` | RTDE → UR controller, V4L2 → cameras |
| `scripts/servers/serve_lerobot.py` | `physical-ai-sdk/.venv` | torch → GPU |

Both servers run from the physical-ai-sdk venv because `lerobot`, `rtde_control`
and the SDK's UR5e driver need a torch/numpy stack this repo pins differently —
the same split already used for GR00T.

### What is different from the simulated envs

| | LIBERO-PRO | UR5e |
|---|---|---|
| Action space | normalised Cartesian deltas for an OSC controller | **absolute joint targets (rad) + gripper**, streamed via `servoJ` |
| `ActionLayout.space` | `eef_delta` | `joint_absolute` |
| Dynamics model | `IntegrationDynamicsModel` (Euler-integrate the deltas) | `Ur5KinematicsDynamicsModel` (forward kinematics) |
| Depth / segmentation | rendered by the sim | none — plain UVC colour cameras |
| Initial state | savable sim state | homing move + the operator staging the scene |
| Episode success | `done` from the sim | never, or an operator verdict (`--env.success-mode stdin`) |
| Guidance | supported | supported — see [Guidance](#guidance) |

`evaluate.py` picks the right dynamics model and camera resolution for you when
`env:ur5` is selected; nothing else in the pipeline special-cases the robot.

### Gripper convention

`lerobot_robot_ur5e` reports the Robotiq position as **0 = open, 1 = closed**.
The pi0/pi05 UR5e checkpoints were trained on the opposite (openness), and every
other env in this repo treats a gripper value as an *aperture*. The env server
therefore flips it in both directions — observations are republished as openness
under `gripper_qpos`, and the policy's gripper channel is flipped back before
being commanded. This is the `FLIP_GRIPPER` flag from the reference deployment;
turn it off with `--env.no-flip-gripper` if your checkpoint used the raw driver
convention.

### Gripper geometry

The cell runs a **Robotiq 2F-140** on the standard coupler. Its geometry lives in
`packages/policy_steering/src/policy_steering/configs/robotiq_2f140.yaml` and is
selected by `GripperSpec.geometry_id` in `envs/ur5/config.py` — no call site
hardcodes a robot, so switching grippers is a one-line spec change plus a YAML.

Three things that config pins down, all of which are easy to get wrong:

* **Finger axis.** `finger_axis: x` says the jaws open along the tool frame's
  x axis (left pad at −x, right at +x), which is how the gripper is bolted to
  this cell's coupler and how menagerie's `robotiq_2f140` sits on the UR5e's
  `attachment_site` at its default mount yaw. Both the fingertip FK guidance
  measures from and the rendered mesh read this: `--viz.gripper-mount-yaw-deg`
  (0° for `env:ur5`) has to agree with it, and
  `tests/visualization/test_gripper_mount.py` checks that it does. Before
  2026-09-02 the FK used the Panda's y axis and the viewer compensated with a
  90° mount, so the two agreed with each other and were both a quarter turn
  about the tool axis from the real gripper.
* **Stroke.** Training used the default (full) opening, so the normalised
  `[0, 1]` aperture spans the whole 140 mm. Feeding it through the Panda's
  `[-1, 1]` bounds instead maps a *closed* gripper to 40 mm open and squeezes
  the range into a 40–80 mm band.
* **TCP position.** The controller reports a 223.47 mm tool offset, matching the
  2F-140 body (209.3 mm) plus the standard coupler (13.9 mm) to 0.3 mm — 48 mm
  away from the equivalent 2F-85 stack, so the offset alone tells you which
  gripper is fitted. Because the TCP is configured at the fingertip plane, the
  grip-site-to-pad offsets are zero.

Note that a rollout is *cross-embodiment*: the retrieved demo is a Franka Panda
from LIBERO while the live arm is the UR5e, so the two sides are measured with
different finger geometry at the same instant. `RolloutAbstractor` resolves the
live geometry from the env spec and leaves the reference trajectory on Panda
geometry deliberately.

Caveat for guidance work: the 2F-140's fingers ride a four-bar linkage rather
than prismatic rails, so they stay parallel only through the middle of the
stroke and splay near the extremes. The linear stroke → fingertip model is a
good approximation mid-range and degrades towards fully open/closed.

---

## One-time setup

The env and policy servers need three transport packages inside the SDK venv
(they are not part of `physical-ai-sdk`'s own dependencies):

```bash
uv pip install --python ~/dev/physical-ai-sdk/.venv websockets msgpack-numpy loguru
```

Then find your camera device nodes — one camera per `/dev/videoN`; reusing a
node fails on the second `connect()`:

```bash
v4l2-ctl --list-devices
v4l2-ctl --device /dev/video4 --list-formats-ext
```

The camera *names* must match the checkpoint's keys. For
`pi05_put_all_cables_in_bag_a100` those are `exterior_image_1_left` and
`wrist_image_left` (the server derives the expected list from the checkpoint's
own `input_features`, so a mismatch fails loudly rather than silently feeding
the wrong view).

---

## Bring-up, in order

Use `scripts/run_ur5_rollout.sh` (see [real_deployment.md](real_deployment.md));
each stage is one flag. Escalate only when the previous stage is clean.

| Stage | Flag | Exercises |
|---|---|---|
| 1. No hardware | `--simulate` | Protocol, env spec, policy, recorder — arm powered off |
| 2. Real arm, no motion | `--dry-run` | RTDE connect, joints/TCP/gripper, camera frames, TCP-offset check |
| 3. Real arm, moving | *(none)* | The rollout |

Stage 2 logs a warning if `--env.tcp-offset` disagrees with the controller;
pass the controller's value if it fires. On this cell it is
`0.00052 0.00109 0.22347 0 0 0` — a 223 mm tool offset, without which every
predicted end-effector pose is wrong by exactly that much.

To own the server's terminal (and use `--success-mode stdin`, where `s`/`f` +
ENTER ends an episode as success/failure), start it by hand and attach with
`--env.no-launch-server`:

```bash
~/dev/physical-ai-sdk/.venv/bin/python scripts/servers/serve_ur5_env.py \
    --port 8770 --tasks "Put all cables in the bag." \
    --cameras exterior_image_1_left=rs:262722073432:640x480@30 \
    --stage-prompt --success-mode stdin
```

## Safety

The env server enforces the limits, so nothing upstream can bypass them:

- **Joint rate limit** — every commanded target is clamped to at most
  `--env.max-joint-delta` radians (default 0.05, ≈2.5 rad/s at 50 Hz) from the
  *measured* joint positions. A single malformed action cannot throw the arm;
  clamping is logged.
- **Joint limits** — targets are clamped to `UR5E_JOINT_LIMITS`.
- **Non-finite actions** are rejected outright.
- **Sustained clamping is flagged.** One clamped step is a spiky action; being
  clamped for a full second of steps means the arm is rate-limited toward a
  pose far from where it is. In practice that means the arm started somewhere
  the demonstrations never did, and it will creep toward the first chunk at
  0.05 rad/step. The server logs this explicitly — stage the arm near the
  demonstrations' start pose.
- **All-zeros joint targets are treated as "hold"**. `EpisodeRunner` emits an
  all-zeros dummy action during its settle phase; read as *absolute* joint
  targets that would mean the fully-extended singular pose. `Ur5EnvArgs`
  defaults `num_steps_wait` to 0 so the phase is skipped, and the server
  special-cases the vector as a belt-and-braces second line of defence.
- **Client disconnect stops motion and parks the arm** (`servoStop`, then a
  homing move), and SIGINT/SIGTERM stop, home, and disconnect. Parking happens
  on disconnect rather than only on shutdown because `evaluate.py` gives its
  child servers just a few seconds to exit — not enough for a blocking
  `moveJ`. Disable with `--env.no-home-on-disconnect`.

None of that substitutes for a hand on the e-stop the first time a new
checkpoint runs.

---

## Control rate

`step` paces itself to `--env.control-hz` (default 50, matching the rate the
pi0/pi05 UR5e datasets were recorded at). Actions must be executed at the rate
they were recorded, otherwise the arm trails the trajectory the policy
predicted and never reaches the object.

Holding 50 Hz means every step — WebSocket round-trip, camera grab, image
resize, visualisation, recording — has to fit in 20 ms. The server separates the
two ways that deadline gets missed, because they have different fixes:

- **`late_step_fraction`** — the *server* overran its own step. Fix it here.
- **`stall_step_fraction`** — the *caller* was still thinking (policy
  inference). Expected at every replan boundary; raise
  `--policy.replan-steps` if it happens too often.

If `late_step_fraction` climbs, in rough order of impact:

1. `--viz.no-enable` (Rerun logging is the biggest per-step cost)
2. `--output.no-save-episodes`
3. run the env server on the machine the cameras are plugged into
4. lower `--env.control-hz` (the arm stays in sync with itself, but the policy
   is now being replayed slower than it was trained)

### Replanning and RTC

`EpisodeRunner` is synchronous: it drains the whole action plan, *then* asks for
a new chunk, so the arm is stationary while inference runs. That makes
`--policy.replan-steps` a direct trade-off between reactivity and stalling — at
50 Hz with ~200 ms inference, replanning every 5 steps would stall the arm more
often than it moves. The LeRobot backend therefore defaults to 25 (pi0/pi05 emit
50-step chunks).

Real-Time Chunking is on by default and does something slightly different from
the reference `rtc_inference_example.py`. There, inference runs on a background
thread while a 50 Hz executor drains the queue, so RTC has to blend around a
nonzero inference delay. Here the delay is exactly zero steps — but RTC still
earns its keep, because each new chunk is conditioned on the *unexecuted tail*
of the previous one, so plans stitch together instead of jumping at every
replan boundary. The client reports how many steps it consumed and the server
derives the leftover from its own copy of the previous chunk, exactly as
`ActionQueue.get_left_over` would.

Turn it off with `--policy.no-rtc`. `--policy.execution-horizon` must be at
least `--policy.replan-steps`; `evaluate.py` checks this up front.

### Measured timings

`pi05_put_all_cables_in_bag_a100`, RTX 5090, RTC on, `replan_steps=25`:

| | |
|---|---|
| Model load | ~10 s (was ~2.5 min before the policy was built on the GPU, see [install_lerobot.md](install_lerobot.md#load-time)) |
| Warmup (both compiled paths) | ~40 s |
| Steady-state inference | **~120 ms** |

At 50 Hz that means the arm stands still for ~6 control steps at each replan —
i.e. ~120 ms of stall per 500 ms of motion. The server logs each stall.

Two details keep the steady state fast, and both are worth knowing if you touch
the policy server:

- **CUDA graphs are disabled when RTC is on.** RTC differentiates through the
  denoising step, which is a second graph for a function inductor has already
  captured; adding it to a live CUDA-graph pool crashes on the *second*
  inference. The server rewrites the checkpoint's `compile_mode`
  (`max-autotune` → `max-autotune-no-cudagraphs`) and says so in its log.
- **Warmup compiles the RTC path at the real leftover length.** The leftover's
  length is part of the compiled shape, so the server needs to know
  `replan_steps` (passed through automatically). Get it wrong and the first
  replan of the run pays a ~4 s recompile with the arm stopped.

### Checkpoint compatibility

The `pi05_*_a100` checkpoints were saved by a LeRobot version that named a
processor step `relative_actions_processor`; current LeRobot calls the same
class `delta_actions_processor`. The server registers the old name as an alias
before loading the processor pipeline (and logs that it did) — without it,
loading fails with a bare `KeyError`.

---

## Camera calibration

Calibration is what turns depth into a reconstruction: without it,
`get_camera_params` returns `{}`, the visualiser skips projecting action chunks,
and `_iter_camera_views` drops the camera so perception sees no views at all.

### Calibrating a camera

One command, one window:

```bash
scripts/setup/calibrate_ur5_cameras.sh                     # both cameras, exterior then wrist
scripts/setup/calibrate_ur5_cameras.sh --camera exterior   # one camera only
scripts/setup/calibrate_ur5_cameras.sh --camera wrist
scripts/setup/calibrate_ur5_cameras.sh --simulate          # fake arm, cameras and tracker
scripts/setup/calibrate_ur5_cameras.sh --help
```

The window walks through capture (the arm moving itself inside a box you
set, or you in freedrive), point selection, tracking, solving, checking and
saving, and writes straight into
`configs/ur5_camera_calibration.json` — the file the rollout launchers default
to — so a guided run works on the new set-up with no conversion step. Both
cameras are calibrated against the **controller's TCP**, which on this cell is
the closed fingertip plane (`--env.tcp-offset 0.00052 0.00109 0.22347 0 0 0`,
now also `DEFAULT_TCP_OFFSET` in `envs/ur5/config.py`). The arm is only ever
*read* over RTDE; you move it in freedrive.

**Autonomous capture (default).** The window shows a **safe box** in the
robot base frame — a top-down x/y plot with a z bar, the live TCP, and once
planned, every pose the arm will visit. It follows the TCP until you edit it:
±15 cm in x and y, and in z from the current height up 20 cm — never below
where you parked the arm, so a TCP parked near the table stays away from it.
Edit the six ranges or press *Centre box on TCP*. For the wrist camera the
box must also sit at least 5 cm above the recorded mark, or planning refuses. *Start
autonomous capture* plans stratified positions over the box (orientations
jittered by the tilt/roll settings, or aimed at the mark for the wrist
camera), drops any pose the controller's `isPoseWithinSafetyLimits` rejects,
confirms with you, then drives the arm with slow asynchronous `moveL`s
(default 0.08 m/s), pausing at each so a still frame is taken. The arm must
already be inside the box, in **Remote Control** and out of freedrive. The
live TCP is checked against the box (plus 2 cm) on every poll: an excursion
stops the arm and the box turns red. The red **STOP MOTION** button and
`Esc` stop it at any time; the e-stop stays in your hand regardless.
`--no-motion` removes the motion controls and never connects RTDE control.

**Manual capture.** Close the gripper. Press *Start manual capture* and move
the arm slowly through the camera's whole view — near and far, side to side, high and
low, rotating the wrist — pausing about a second at each pose. A *still pose*
is stored when the arm has stopped and the TCP has moved at least 2 cm since
the previous one; frames while moving are stored too, so the tracker sees a
continuous video, but only still poses become PnP correspondences (an RTDE
reading and a camera frame only describe the same instant when nothing
moves). Aim for 40 still poses spread over 30 cm or more in every direction,
then *Finish*. In the first frame, click the exact spot where the closed pads
meet — the magnifier follows the cursor. *Track and solve* runs Kalib's
SpaTracker in its conda env, then a RANSAC PnP over the still frames. Scrub
the result: the green ring is the tracker, the magenta cross is where the
solved pose puts the TCP; they must coincide in every still frame. Save.

**Wrist camera.** By default it is the second stage of the same window: when
the exterior entry is saved, *Next camera* switches to the wrist stage (a
camera that already has a saved entry can be skipped). One extra step comes
first: choose a fixed, high-contrast mark on the table (a tape corner, a
printed cross) whose base-frame position is needed. Two ways to get it:

* **Pick it in the exterior image** — the freshly calibrated exterior view is
  frozen, you click the mark's centre, and its depth plus the exterior
  extrinsic place it in the base frame (good to a few millimetres; refused
  where the depth sensor has no reading).
* **Touch it** — bring the closed fingertips exactly onto the mark and press
  *Record*; the controller's TCP is the position. Most accurate.

From then on the point tracked in the wrist image is that mark, expressed in
the TCP frame at every pose, and the solve yields the camera's pose **in the
TCP frame**. Autonomous capture aims the tool axis at the mark from poses
spread over the box (which must start at least 5 cm above the mark); manual
capture means moving the camera around the mark, 15–60 cm away, keeping it in
view the whole time.

**What you get.** The file has one entry per camera:

```json
{
  "base_frame": "base",
  "exterior_image_1_left": {"extrinsic": [[...]], "meta": {...}},
  "wrist_image_left": {"extrinsic": [[...]], "parent": "tcp", "meta": {...}}
}
```

`extrinsic` is the camera's pose in its parent frame. A fixed camera's parent
is the robot base — RTDE's `base`, because the tracked point came straight
from `getActualTCPPose()`. A wrist entry is flagged `"parent": "tcp"`, and
the env server composes it with the live TCP pose on **every**
`get_camera_params` call; `ChunkProjector.fetch_params` and the guided runner
re-fetch at each replan, so guidance sees the camera move with the arm.
`meta` records how the entry was made (resolution, inlier count, reprojection
error, depth agreement, the TCP offset, the capture folder); the server warns
at start-up if its `--tcp-offset` differs from the one recorded, since a wrist
extrinsic is expressed in that very frame. Recalibrating one camera merges
into the file and keeps the other's entry; the previous file is kept as
`*.bak-<timestamp>`. An older file written in `base_link` is rotated into
`base` on merge, so the file stays in one frame.

**Checks before you trust it.** The verdict in the window combines three
things: the PnP inlier fraction, the mean reprojection error, and the
RealSense's *own depth* at the tracked pixel against the solved distance —
depth never entered the solve, so it catches a self-consistent-but-wrong
answer such as a mirrored frame. Kalib's own PnP runs as well and the
difference is shown; more than a couple of centimetres apart means the tracker
slipped somewhere, and the frame scrubber will show where. A guided run
repeats the depth check at start-up as the `[calibration]` line.

**Intrinsics** are queried from the RealSense at the streaming resolution
(640x480 by default, matching the env server) and are *not* written into the
file — the server publishes the SDK's aligned-colour intrinsics, which are the
right matrix for the depth it registers. Keep `--resolution` equal to what the
server streams. The exterior camera is the D435I with serial `262722073432`
(fitted 2026-09-22; the earlier `035422072667`, whose factory intrinsics were
`fx=607.244, fy=607.659, cx=323.365, cy=244.755`, distortion all-zero, is no
longer on the cell). Its intrinsics are read from the device at run time.

**Where things live.** Frames go to `data/calibration/<camera_name>/` in the
layout Kalib reads (`NNNNNN.jpg` + `NNNNNN.json`), plus `depth/` and
`meta/capture.json`; *Load a previous capture* re-opens one to re-click or
re-solve without the robot. Kalib is expected at `~/dev/Kalib` with its conda
env at `~/miniconda3/envs/kalib` (`KALIB_ROOT`, `KALIB_PYTHON`,
`KALIB_DEVICE` override); the GUI runs in the `physical-ai-sdk` venv, which
needs PyQt6 once:

```bash
uv pip install --python ~/dev/physical-ai-sdk/.venv PyQt6
PYTHON=~/dev/physical-ai-sdk/.venv/bin/python scripts/setup/vendor_qt_xcb_cursor.sh
```

Only the camera being calibrated is opened, colour+depth, so the USB limit on
two depth streams does not bite here. It may once the wrist camera has a
calibration: the server then streams depth from *both* cameras, which one
controller cannot carry (see the pre-flight). Move one camera to another
controller, or leave the wrist uncalibrated for runs that do not need it.

### Older layouts

`--env.calibration-path` still accepts a raw Kalib `T_cam_base.json`
(attached to `--primary-camera`, transform direction taken from its
`convention` field) and files written in `base_link`, provided the frame is
declared — see below. `scripts/setup/ur5_calibration_from_kalib.py` remains
for converting such a file by hand. Mind the trap that motivated all of this:
Kalib's headline key `T_cam_base` is the *opposite* of what the name suggests
— its own `convention` field says `X_cam = T_cam_base @ X_base`. The
camera-to-base pose is `T_base_cam`. (Measured on this cell: the correct
direction puts the table plane 2.9° off base-z; the reversed one tilts it 63°.)

### Which base frame — you must say

UR ships two base frames 180° apart about z. `base` is what RTDE reports poses
in, and what the new tool writes; `base_link` is the ROS/URDF convention, and
it is what a URDF-based pipeline (Kalib's own FK included) produces. Mixing
them mirrors the scene in x and y while leaving the table perfectly horizontal
— a plane-fit sanity check still passes, the recording still looks right, and
only a comparison against the robot's own TCP catches it.

So the frame is never inferred. Declare it in the calibration file:

```json
{
  "base_frame": "base",
  "exterior_image_1_left": { "extrinsic": [[...]] }
}
```

`--env.calibration-base-frame` overrides the file. If neither says, loading
raises rather than guessing. (A `"parent": "tcp"` entry is in the TCP frame
and the question does not arise for it.)

**Repairing an episode recorded before this existed.** Such a `.vla` holds a
mirrored extrinsic. Rather than rewrite a multi-gigabyte recording, add one key
to its JSON sidecar and the reader corrects it on load:

```json
{ "extrinsic_frame": "base_link" }
```

To tell whether an episode needs it, watch the `[calibration]` line the guided
run prints at startup. It projects the end-effector — whose position
proprioception knows exactly — into the camera and measures how far the
reconstructed surface there is from it:

```
[calibration] extrinsic agrees with depth to 0.7 cm at the end-effector.
```

A mirrored episode instead reports that the extrinsic *never* places the
end-effector inside the camera image, which is the signature of this exact
mistake.

---

## Guidance

Supported. `scripts/run_ur5_rollout.sh --guided` is the entry point.

The correction lands on the model's *normalized* action tensor, so
`--guidance.factor` is in normalized units for any checkpoint whose ACTION
normalization is not IDENTITY — the server logs which at startup. The first
guided inference also pays a `torch.compile` of the guided graph while the arm
stands still. Guided runs need a retrieved demo, so point `--retrieval` at a
dataset covering the task, and `--perception.mode REAL_PERCEPTION`.

### How the loss reads the chunk

The guidance loss has to turn an action chunk into a tool trajectory before it
can measure distances on it. On the simulators that is Euler integration of
the first three channels, which are Cartesian deltas. This arm's actions are
absolute joint targets, and read as velocities they put the loss somewhere the
arm is not going: the base, shoulder and elbow angles become x/y/z speeds, and
because an early "speed" moves every later position the whole correction
lands on the *first* action of the chunk, as a fixed-direction bias. Measured
on a real guided rollout, every chunk started 2.5–4.9 cm from the tool in the
same joint direction, and that is what "the arm goes back when approaching"
looks like.

`--policy.guidance-action-space` controls this and defaults to `auto`, which
resolves to `ur5e-joints` for `env:ur5`: the server unnormalizes the chunk
with the checkpoint's own action statistics (read off its postprocessor, so
the stats are the ones the model's output would be put through) and pushes
every joint target through a differentiable UR5e forward kinematics
(`policy_steering.ur5e_kinematics_torch`, the torch twin of the JAX one the
fake policy uses), composed with `--env.tcp-offset` so it measures the same
point RTDE reports. A chunk that holds the current joints is then scored at
the current tool pose, and a correction is a function of where the arm is
rather than of the joint angles' numeric values. `--policy.gripper` follows
the env too: finger-sourced pairs are measured from the 2F-140's pads, not
the Panda's. The server prints the resolved kinematics at startup and echoes
them in `info` (`guidance_action_space`, `guidance_tcp_offset`,
`guidance_gripper_geometry`), and `evaluate.py` warns if the kinematics are
requested without a TCP offset, because the flange sits ~22 cm above the
tool the perception stack measures from.

The correction itself still lands on the normalized tensor: with QUANTILES
normalization, `--guidance.max-alpha 1.0` is half a channel's 1st–99th
percentile span, which for a joint can be tenths of a radian. Retune it per
checkpoint if the corrections look too large.

### When the correction is not finite

A NaN anywhere in the guidance gradient would land in every action channel of
the chunk, and the env server rejects non-finite actions outright, so the run
would stop with an error that points at the policy. Both ends of the wire now
guard against that:

- `guidance_builder` drops a pair whose weight, expected distance, waypoint or
  surface points are not finite (and disables guidance for the chunk if the
  shared `ee_pos_0` / `ee_ori_0` / factor are), logging which value was bad.
- The server refuses parameters that still carry a NaN, naming the key.
- A denoising step whose *corrected velocity* comes out non-finite is served
  with the network's own velocity instead. The server logs the first such
  step with which stage went bad (input state, loss or gradient), counts
  them, and the client logs a warning per inference — so a run that reports
  itself as guided while some steps were not is visible in
  `evaluation.log`, not silent.
- If the chunk is still non-finite after that, the server raises with the
  RTC / compile state, because the network or the RTC blend produced it, not
  guidance; `--compile-model off` rules out the compiled graph.

One known source is fixed at the root: the 2F-140's four-bar model clamped the
pad height for apertures beyond the linkage's reach but left the square root's
gradient infinite there, so any normalized gripper action outside the reach
(routine when denoising from noise) poisoned the whole gradient.

### Guiding from an earlier real rollout

A folder of `.vla` rollouts is a retrieval source in its own right, so a real
run can be guided by an earlier real run rather than by a simulator demo. Three
things have to be true of the recording, and all three are checked at startup:

1. **It carries depth.** Guidance measures 3-D distances, and a rollout
   recorded with `--output.depth-stride 0` has none. The default stride is 5.
2. **It carries a camera extrinsic**, i.e. it was recorded with
   `--env.calibration-path` — and in the frame it claims (see
   [Which base frame](#which-base-frame--you-must-say)).
3. **It has been encoded**, which is what builds the retrieval index:

```bash
python scripts/encode_datasets.py \
    --source-path data/ur5_demo_depth/ur5/<task>/episodes

RETRIEVAL_DIR=data/ur5_demo_depth/ur5/<task>/episodes \
    scripts/run_ur5_rollout.sh --guided
```

### Sub-trajectory segmentation

`compute_subtrajectories()` splits a demo wherever the end-effector is moving
faster than `--subtraj.gripper-speed-threshold`, and its units are **metres per
step, not per second**. That makes the shared default wrong here twice over:
this arm records at 50 Hz against LIBERO's 20 Hz, so the same physical speed is
2.5x smaller per step, and a hand-paced real demo is slower than a scripted
simulator one to begin with.

Measured on `data/ur5_demo_depth/.../episode_000` (1250 steps, 25 s, 1.2 m of
path):

| threshold (m/step) | m/s at 50 Hz | share of episode active | sub-trajectories |
|---|---|---|---|
| 0.0005 | 0.025 | 32% | 3 |
| **0.001** | **0.05** | **24%** | **3** |
| 0.002  | 0.10 | 17% | 3 |
| 0.005 (shared default) | 0.25 | 4% | **1 — the fallback** |
| 0.010 (`GripperSpec`) | 0.50 | 0% | **1 — the fallback** |

At 0.005 no active run survives `--subtraj.min-length`, so the function returns
its `[(0, T)]` fallback and the decomposition silently does nothing. *Raising*
the threshold cannot fix that — it is already past the collapse. The launcher
therefore sets 0.001 m/step and a 50-step minimum (1 s at 50 Hz, matching the
duration LIBERO's 20-step default buys at 20 Hz); both are overridable via
`SUBTRAJ_SPEED_THRESHOLD` / `SUBTRAJ_MIN_LENGTH`.

Two caveats worth knowing before tuning further:

- **`GripperSpec.speed_threshold` is not what you are looking for.** The UR5e
  spec declares `speed_threshold=0.01` and the field is commented as the
  "default sub-trajectory segmentation threshold", but nothing reads it —
  the live value is `--subtraj.gripper-speed-threshold`. Changing the spec
  field has no effect.
- **An idle tail dominates whatever you pick.** In the episode above, all real
  motion is over by t≈400; the remaining 850 steps are sub-centimetre drift.
  The gap-filling rule extends the last segment to `T`, so that idle tail is
  glued onto the final sub-trajectory (930+ steps) at *every* threshold. Trim
  recordings at the end of the motion, or lower `--env.max-steps`, if the last
  sub-trajectory looks absurdly long.

### Debugging the pipeline without the checkpoint

Loading the pi05 checkpoint costs around two minutes and a GPU, and none of it
is involved in perception, retrieval, demo analysis, alignment or guidance. To
iterate on those, swap the policy for an in-process fake:

```bash
RETRIEVAL_DIR=data/ur5_demo_depth/ur5/<task>/episodes \
    scripts/run_ur5_rollout.sh --guided --fake-policy --spawn-viewer
```

Everything upstream of the policy still runs against the real cell: the cameras
stream, perception reconstructs, the demo is retrieved and analysed, and the
guidance gradient is computed and visualised. Only the network's velocity is
missing — `--fake-policy` is `--guidance.debug`, the same switch the real
server honours to inspect the guidance field in isolation, so the actions you
see *are* the guidance correction. `--fake-policy circular` substitutes a
circle-tracing fake instead, for shaking out the visualiser.

On this arm the noop fake speaks joint space. Its chunk starts as the *observed
joints* held over the horizon — "don't move" for an arm whose actions are
absolute joint targets, where a zero chunk would be the fully-extended pose —
and the guidance loss sees that chunk through a differentiable UR5e forward
kinematics (`dynamics/ur5_kinematics_jax.py`) rather than the simulators'
Euler integration, which would read joint radians as tool velocities. So the
predicted trajectory in Rerun starts at the end-effector and the correction
moves the joints that move the tool. `--fake-policy circular` still emits
Cartesian deltas and is drawn by integration, as on the simulators.

It implies `--dry-run`. Even a held chunk plus a guidance correction is a set
of joint targets nobody has vetted, so the arm must not obey it; the sensors
are still read, so nothing you are debugging is affected. Add `--simulate` to
run with the arm powered off entirely — the fake driver then reports the TCP
pose with the configured `--env.tcp-offset`, exactly as the controller would,
so the published end-effector and the FK-predicted chunks agree.

A real recording labels no pixels of its own, so its objects are named by a VLM
and segmented by the same open-vocabulary stack the live scene uses, on its own
tracker instance (`analysis/demo_segmenter.py`). Everything downstream —
sub-trajectories, distance pairs, the reference trajectory guidance steers
toward — then reads through the one `DemoEpisode` contract in
`analysis/demo_episode.py`, so a simulator demo and a real rollout take the
same path.

### Perception status (measured)

| Stage | Status |
|---|---|
| Open-vocabulary **detection** | **works** — localizes `bag` and `cable` in the real workcell frame |
| **Segmentation** (SAM2) | **works** — bag 0.96, coiled cable 0.95; ~3.7 s per `detect()` on GPU, dominated by the VLM round-trip |
| **Tracking** (SAM2 video) | **works** — tracker loads and seeds from the detect frame |
| **Reconstruction** / `fuse_views` | needs a camera **extrinsic**; depth and intrinsics are now published |

Reconstruction additionally needs a camera **extrinsic** — see
[Camera calibration](#camera-calibration). Depth and intrinsics
are published by the `rs:` camera path.

### Running with real perception

Perception is env-agnostic: the same flags drive sim and hardware, and the
env-specific parts (camera resolution, dynamics model, `flip_u`) resolve from
the published `EnvSpec`.

```bash
# simulator
python scripts/evaluate.py --instance 9 --guidance.no-enable \
    --perception.mode REAL_PERCEPTION \
  env:libero --env.task-suite-name libero_spatial \
  policy:openpi --policy.gpus 0

# real UR5e — same shape, different subcommand arguments
python scripts/evaluate.py --instance 9 --guidance.no-enable \
    --perception.mode REAL_PERCEPTION \
  env:ur5 --env.cameras exterior_image_1_left=rs:262722073432:640x480@30 \
          --env.wrist-camera '' \
          --env.calibration-path configs/ur5_camera_calibration.json \
  policy:lerobot --policy.model-path <checkpoint> --policy.gpus 0
```

`--guidance.no-enable` keeps retrieval, demo analysis and perception running
without applying the guidance gradient — an honest unguided baseline with the
same perception path. `--policy.guidance off` goes further and makes the server
refuse guided requests outright.

#### Interactive playground

`scripts/debug/perception_playground.sh` runs the perception half of the pipeline on
its own, with a Rerun 3D scene and the robot URDF loaded, and lets you retype
the object list in the terminal while it streams. It never commands the arm —
it only reads observations — so it is safe to leave running while you rearrange
the table.

Launch it through the wrapper, which starts the env server, waits for it, and —
the reason to prefer it — stops it again on the way out, so the RealSense is
released. A server left holding a camera is what makes the *next* run fail with
`Device or resource busy`, usually long after you've forgotten it is running.

```bash
# optional: a viewer to stream into (the `rerun` binary is not on PATH)
.venv/bin/rerun --port 9876

scripts/debug/perception_playground.sh                 # cable + bag
scripts/debug/perception_playground.sh cable bag mug   # your own queries
scripts/debug/perception_playground.sh --help          # everything else
```

`--wrist` adds the wrist camera (uncalibrated today, so it contributes nothing
until it has an extrinsic), `--simulate` runs with the arm powered off but the
cameras real, `--no-server` attaches to a server you started yourself, and
anything after `--` is forwarded to the Python script. Config is overridable
from the environment:

```bash
EXTERIOR_SERIAL=207222070875 scripts/debug/perception_playground.sh cable
```

Before starting the server the wrapper names any process already holding a
video device, so a busy camera is reported with a PID rather than surfacing
seconds later as a bare `Device or resource busy`.

To drive the Python script directly (three terminals: viewer, `serve_ur5_env.py`
with `--dry-run`, then the script) see `--help` on both — but remember that
stopping the server is then your job.

Like `evaluate.py`, the playground defaults to `--connect 0.0.0.0:9876` and
attaches to a viewer you started yourself, so the viewer outlives the run and
keeps your camera angle across restarts. The flag takes a bare `host:port` and
is wrapped into `rerun+http://<addr>/proxy` internally — rerun 0.30 rejects an
unschemed address. With nothing listening the connection is a no-op rather than
an error, so a stray run won't fail; it just goes nowhere.

Then type at the prompt: `cable, bag` replaces the query set, `+ mug` adds one,
`!redetect` re-runs detection, `!settings` reprints the configuration, `!quit`
exits. Changing the queries resets the tracker, so the next frame goes down the
`detect` branch exactly as a replan does in deployment — which is also why the
first frame after a change costs seconds and the ones after it cost
milliseconds.

The viewer shows the fused scene cloud, one coloured cloud and centroid per
tracked instance, the 2D mask overlays, and text panels for the live stats
(detect/track timings, per-instance point counts and base-frame centroids) and
the active perception settings. Reconstruction uses the same
`build_segmented_pointcloud` call the abstractor makes, so what you see is the
geometry guidance would consume rather than a look-alike.

Two things it will tell you loudly:

* **No depth camera** — it exits rather than showing an empty 3D view. At least
  one camera must be addressed as `rs:<serial>` *and* have calibration.
* **Nothing matched** — the stats panel says so instead of silently drawing
  nothing, which is usually a wording problem rather than a pipeline problem.

The menagerie UR5e model is arm-only (6 joints, no gripper), so the perception
playground draws the fingers separately from the `robotiq_2f140` geometry rather
than rendering them. A rollout does render them — see
[Rendering the gripper](#rendering-the-gripper).

### Rendering the gripper

Menagerie publishes its arms bare, so a UR5e used to render as a tool flange
with nothing on the end of it. `env:ur5` now bolts a Robotiq 2F-140 on
automatically; `--viz.gripper-model None` renders the arm alone.

The model is **vendored**, because menagerie has no 2F-140 — only the 2F-85. It
comes from [PR #161][2f140pr], which is open and unmerged; see
`src/policy_abstraction/assets/mujoco_menagerie/PROVENANCE.md` for the exact
commit and for the two independent checks that it is the right hardware. If the
PR lands, delete that directory: the loader prefers a real menagerie checkout
and will pick the upstream copy up with no code change.

[2f140pr]: https://github.com/google-deepmind/mujoco_menagerie/pull/161

Posing it needs more than writing a joint angle. The 2F-140 is a *closed* chain:
each finger is a four-bar whose loop MuJoCo holds shut with `connect` equality
constraints, and those are solved in the dynamics, not in kinematics. The
visualiser only ever calls `mj_kinematics`, so setting the driver joint and
stopping there renders each finger as four disconnected links. Instead
`visualization/robot_assembly.py` solves the free-air pose once per model over a
grid of apertures, and the render path interpolates that table — about 20 µs a
frame.

Two consequences worth knowing:

* **The input is millimetres between the pads, not a joint angle.** The observed
  `[0, 1]` openness is mapped onto the 140 mm stroke by
  `robotiq_2f140.yaml`, the same config `policy_steering`'s FK uses. That is
  deliberate: the drawn pads and the fingertips guidance measures distances from
  are then the same points, so an overlay that disagrees with the render means a
  real bug rather than two conventions.
* **The model's authored driver range stops at a 128.6 mm aperture**, short of
  the datasheet's 140 mm stroke. The linkage closes cleanly past that limit
  (residuals stay at 1e-14 m), so the renderer poses through it rather than
  drawing a gripper that can never fully open. Beyond what the mechanism can
  reach it clamps and warns once.

To check any of this without a robot, replay a recorded episode:

```bash
python scripts/replay_episode.py \
    --episode data/ur5_baseline/ur5/Put_all_cables_in_the_bag/episodes/<ep>.vla \
    --robot-model ur5e_mj_description --gripper-model robotiq_2f140 \
    --save-rrd tmp/ur5_2f140.rrd
.venv/bin/rerun tmp/ur5_2f140.rrd
```

`--gripper-model` defaults to `auto`, which reads the gripper out of the
episode's own env spec and falls back to the 2F-140 for UR5e recordings that
pre-date that metadata.

With no viewer to attach to and no display for `--spawn`, record instead and
open the file from a machine that has one:

```bash
python scripts/debug/perception_playground.py --queries cable bag \
    --connect None --max-frames 60 --save-rrd tmp/playground.rrd
.venv/bin/rerun tmp/playground.rrd
```

`--max-frames` also stops end-of-stdin from ending the run, so a scripted
invocation isn't cut short the instant it finds nothing to read. Every run
prints a summary and the detect line carries base-frame centroids, which is the
headless way to catch a calibration or `flip_u` problem — those show up as
plausible-looking clouds in an impossible place, not as an error.

Measured on this cell (one D435i, 640×480, RTX 5090):

| stage | cost |
|---|---|
| `detect` (VLM box → SAM2 mask) | ~3.5–4.0 s |
| `track` (SAM2 propagation) | 51–66 ms median → **15–20 Hz** |

Detection is a network round-trip to the VLM, so it dominates and varies; the
tracking figure is the one that matters for a rollout, since a replan detects
once and tracks in between.

#### `--perception.flip-u`

`reconstruct`/`fuse_views` can mirror the u axis when back-projecting. It exists
to compensate for the abstractor pre-flipping frames: the simulator adapters
publish 180°-rotated RGB+depth (`image_transform: flip_vh`) which the abstractor
`[::-1, ::-1]`s before perception sees it. Envs publishing upright frames — the
real UR5e — must **not** flip; doing so mirrors every reconstructed object about
the optical axis, measured at ~0.8 m of displacement on this cell.

| Value | Behaviour |
|---|---|
| `auto` (default) | derive from the primary camera's `image_transform` — correct for every env in-tree |
| `true` / `false` | force it |

### Depth capture

Address a camera as `rs:<serial>` instead of `/dev/videoN` and the server
streams it through the RealSense SDK, publishing `<name>_depth` (float32
metres) alongside the colour image:

```bash
--env.cameras exterior_image_1_left=rs:262722073432:640x480@30
```

Find serials with `rs-enumerate-devices` or:

```python
import pyrealsense2 as rs
[d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
```

Raw V4L2 depth (`Z16` on its own `/dev/videoN`) is deliberately *not* used: it
sits in the depth sensor's frame at a device-configured scale, so pairing it
with colour pixels mis-associates depth by the stereo baseline. The SDK path
also supplies the depth scale and the aligned-colour intrinsics.

Two things worth knowing about the depth itself, measured on this cell: ~94% of
pixels are valid, and the dark IR-absorbing bag is where most of the invalid
ones are — that object reconstructs thinner than it looks.

`--simulate-hardware` still opens real RealSense cameras when you address them
as `rs:`, so perception can be developed with the arm powered off.

### Where the gradient lands

`__gp_*__` params steer the flow-matching integration server-side. lerobot
already routes every denoising step through `rtc_processor.denoise_step`, so a
`GuidedRTCProcessor` subclass sees them all without copying the loop per model
— and composes with RTC rather than competing with it. `pi0_fast` has no
denoising loop to hook and is refused rather than answered unguided.

The geometry side: the env publishes the same `[pos, axis-angle, aperture]`
state as every other env, and `Ur5KinematicsDynamicsModel` maps joint-space
action chunks to end-effector trajectories.

---

## Reference

- Robot driver: `physical_ai_sdk/robots/lerobot_robot_ur5` (`UR5E`, `UR5EConfig`).
  Older SDK checkouts put it at `physical_ai_sdk/lerobot_robot_ur5e`; the
  server tries both names, newest first.
- Deployment this adapter is modelled on:
  `physical_ai_sdk/inference/rtc_inference_example.py`
- Env spec: `src/policy_abstraction/envs/ur5/config.py`
- Env server: `src/policy_abstraction/envs/ur5/server.py`
- Policy backend: `src/policy_abstraction/policies/lerobot.py`
- Forward kinematics: `src/policy_abstraction/dynamics/ur5_kinematics.py`
