# Live guidance demo on the UR5e

Showing the guidance framework working, on the real arm, to people who are
standing next to it.

One policy, loaded once. One rollout at a time, started and stopped from a
window. And a switch that turns guidance on and off **under the running
policy**: the arm keeps moving, and the next chunk it executes is steered or
not. While guidance is on, the exterior camera view shows the tracked objects
and the distance pairs the policy is being pulled towards, a plot shows the
live distances against the retrieved demo's, and a Rerun viewer alongside
shows the same thing in 3-D.

Everything below the window is the evaluation stack unchanged — the same
`EvalBackends`, the same `EpisodeRunner`, the same flags as
`scripts/evaluate.py` — so what the demo shows is what the experiments
measured. Background on the arm: [ur5.md](ur5.md); the experiments the demo
sits beside: [real_experiments.md](real_experiments.md).

---

## The 60-second version

```bash
# the guided pipeline up, switch off at the first rollout
RETRIEVAL_DIR=data/reference_dataset_cable scripts/run_ur5_demo.sh

# switch on from the first rollout
RETRIEVAL_DIR=data/reference_dataset_cable scripts/run_ur5_demo.sh --start-guided

# with a Rerun viewer spawned for the run (needs a display on this machine)
RETRIEVAL_DIR=data/reference_dataset_cable scripts/run_ur5_demo.sh --spawn-viewer   # 3-D scene in a separate native window instead of the panel

# rehearse everything but the arm and the checkpoint: real cameras and
# perception, retrieval, alignment, guidance, the window and the viewer
RETRIEVAL_DIR=data/reference_dataset_cable \
    scripts/run_ur5_demo.sh --simulate --fake-policy circular

# streaming-only: the policy alone, no retrieval, the switch has nothing to open
scripts/run_ur5_demo.sh --unguided
```

`RETRIEVAL_DIR` is the folder of encoded demo rollouts guidance retrieves
from — the same variable, and the same folder, as
`scripts/run_ur5_rollout.sh --guided` and the experiment launchers use. The
guided configuration is the launcher's **default**, unlike the other two
launchers, because the switch needs the whole chain (retrieval, demo
analysis, real perception, guidance) to have anything to open; without
`RETRIEVAL_DIR` it refuses before starting any server rather than paying the
checkpoint load and then retrieving from simulator datasets that are not on
this host. See [Guiding from an earlier real
rollout](ur5.md#guiding-from-an-earlier-real-rollout) for how a folder of
`.vla` files becomes a retrieval source.

The launcher pre-flights the same things the experiment launcher does: the
arm's dashboard (unless `--simulate`), the two RealSense serials, the
calibration file and the base frame it declares, the viewer.
`scripts/run_ur5_demo.sh --help` prints the flags; `-- <flags>` passes
anything else through to `scripts/demo_ur5.py`, routed to the right side of
the `env:ur5` / `policy:lerobot` subcommands by prefix.

| Launcher flag | Effect |
|---|---|
| (default) / `--guided` | Retrieval, demo analysis, real perception and guidance on; needs `RETRIEVAL_DIR` |
| `--unguided` | `--no-guide`: the policy alone |
| `--start-guided` | The switch is on at the first rollout (`--demo.guided`) |
| `--name X`, `--max-steps N` | Output sub-folder; per-rollout cap (0 = the env's horizon) |
| `--no-viz` / `--full-viewer` | No Rerun at all; the full analysis layout instead of the 3-D scene |
| `--spawn-viewer` / `--external-viewer` | Instead of the in-window panel: a separate native Rerun window for the run, or stream to a viewer already at `VIEWER_ADDR` (falls back to the panel if nothing listens) |
| `COMPILE_MODEL=off` (default) | No `torch.compile`: start-up is seconds instead of minutes and the first flip of the switch compiles nothing; `auto` honours the checkpoint for a faster steady state |
| `CACHE_LIVE_OBJECTS=0` (default) | Always ask the VLM which objects are on the table (`--demo-analysis.no-cache-live-objects`); the scene is re-staged between rollouts, and a cached match from another staging silently loses the target |
| `CACHE_PAIRS=0` (default) | Re-run the VLM's distance-pair analysis of the retrieved demo at every rollout (`--demo-analysis.no-cache-pairs`) rather than loading the cached answer. Retrieval, the analysis and the perception set-up already restart with every rollout; this makes the analysis itself fresh too |
| `RECORD=0` (default) | No episode recording (`--output.no-save-episodes`): encoding two camera streams per step into a `.vla` costs time inside the control loop. Retrieval, the demo analysis and guidance run exactly the same without it. `RECORD=1` records every rollout as the experiments do |
| `TRAINING_POSITIONS=<json>` | The training demos' grasp positions (`scripts/experiments/plot_training_on_grid.py --dataset <demos>` writes `<dataset>_grasp_positions.json`); the window can then overlay their density on the live table as the training distribution — the **Training heat map** checkbox under KEYS in the side column, or `H`. Defaults to the cable dataset's file under `data/training_positions/`; a missing file only warns |
| `--simulate` / `--dry-run` | Arm off, synthetic observations; or connect and read sensors, never command motion |
| `--fake-policy [noop\|circular]` | In-process fake instead of the checkpoint; forces `--dry-run` unless `--simulate` |

---

## What happens when you flip the switch

The switch (`G`) is a flag the runner polls at every replan. Nothing is
rebuilt when it changes: the retrieved demo, the perception tracker and the
alignment keep running whether guidance is on or off, and the abstractor
still produces its distances, its pairs and its alignment cursor every step.
What the switch changes is one thing — whether the `__gp_*__` guidance
parameters are sent with the policy request. Off, they are withheld and the
policy server samples its chunk unsteered; on, the server steers the
denoising loop with them exactly as a guided evaluation run does.

Because the policy replans every `REPLAN_STEPS` steps with RTC on, the
first chunk after a flip is blended into the unexecuted tail of the last
one, the same way every other replan is. A flip is therefore a blend over
the RTC horizon, not a jump.

The recorded episode says what was *applied*: while the switch is off,
`guidance_factor` is logged as `0` and the value the gate withheld is kept
alongside it as `requested_factor`, with `gated: true`. So a recording can be
read back to see when guidance was on and what it would have done when it
was not.

The first guided request compiles a second graph on the policy server (the
guidance loss and its gradient through every denoising step), and
`torch.compile` would otherwise meet it the first time the switch is
flipped, with the arm standing still. The session pays that at start-up
instead: after the plain warm-up it sends one guided request shaped like a
rollout's, against a synthetic single-pair target, and drops the actions
(`--demo.guided-warmup`, on by default; the log reports how long each took).
One caveat: the warm-up request has one pair, and a live scene with a
differently shaped guidance set may still trigger one more recompile on the
first flip.

---

## The window

A PyQt6 window, opened *before* the servers come up so the checkpoint load
is watched from a window and not a still terminal. The panels:

- **Exterior view** — the primary camera, upright, with the overlays below
  drawn on it when they are on. Across its top, in large type, the
  **instruction** the policy is conditioned on. Click it to edit: Enter
  applies the new prompt — at the next replan if a rollout is running,
  since the runner reads its task string at every policy query — and Escape
  reverts. While the field has focus the hotkeys are off, so typing "stop"
  does not stop the arm.
- **3-D panel** — the Rerun scene, beside the exterior view; see [The Rerun
  view](#the-rerun-view). `V` hides and shows it.
- **Bottom row**, the full width under the centre: the **wrist camera**
  (bare), the **retrieved demo's frame** at the aligned time with the pairs
  drawn on it, the **trend plots** — see [The trend plot](#the-trend-plot)
  — and, to their right, the **DISTANCE PAIRS** panel, which is the plots'
  legend: one row per pair with a dot in the pair's colour, the label, and
  the live / reference distance with the error, and a last line naming the
  plots' view mode (`window: ±15 steps …` or `full trajectory`). `T` hides
  and shows the whole row.
- **Banner** along the bottom of the exterior view, coloured by state:
  grey while idle (arm parked, the view streams a preview), blue while a
  rollout runs unguided, green while it runs guided, amber while a stop is
  pending (the rollout ends at its next step boundary and the arm homes),
  and orange-red when the switch is on but guidance is *unavailable*.
- **Badges** top-left of the exterior view say what is not working and
  why: a run launched `--unguided` ("the switch has nothing to open"), or an
  abstractor that could not be set up on this scene (nothing detected, or
  the VLM found no pairs — "running unguided"). The status strip at the top
  says `WARMING UP THE POLICY` while the checkpoint compiles, and during a
  rollout counts the steps against the budget and names the mode of the
  chunk on the arm — `RUNNING · GUIDED · step 143/3000 · 12.4 s`, with
  `switching at next replan` appended in the half-second after a flip,
  while the chunk on the arm is still the old kind.
- **Side panels**: the detector's own rendering of its masks (PERCEPTION),
  the guidance state (GUIDANCE: switch, whether the current chunk was
  steered, the applied and withheld factor, alignment progress,
  sub-trajectory, mean relevance), and the keys.
- **Dimming.** Everything only guidance feeds — the retrieved demo's frame,
  the trend plots, the pairs, PERCEPTION and GUIDANCE — sits under a
  translucent dark sheet captioned `guidance off` whenever guidance *is*
  off: during a rollout, when the chunk on the arm is unguided; between
  rollouts, when the switch is off. It never dims while a guided chunk
  executes, and not in the half second after the switch is flipped on
  while the arm still runs the old chunk — the picture is about to change.
  The sheet is see-through, so the last frame and traces stay readable
  underneath; the point is that the mode is visible from across the room.
- **A fresh start per rollout.** When a rollout starts, the demo frame, the
  detector's masks, the pairs and the plots go back to their placeholders
  before this rollout's first analysis arrives; the previous rollout's
  frame is never on screen at the start of the next one. Between rollouts
  the last analysis stays up.

The window opens maximised and its layout follows the window: the exterior
view and the 3-D panel take the width, the side column keeps its 340 px.
The bottom row is deliberately tall — it starts at about 42 % of the height
and never drops under 260 px, because at a strip's height the trend traces
cannot be told apart — and within it the two camera views take about a
fifth of the width each, the pairs pane a fixed 240–320 px, and the trends
the rest; both splitters can be dragged, and their handles are wide gaps in
the window's background so the wrist image, the demo frame and the plots
never touch. The two views scale with the row's height and keep their
aspect ratio. Nothing in it grows the window — a long status line wraps, a
long pair label wraps inside its pane, the side column scrolls — so it
stays on a small display and on a projector alike. `F11` is true full
screen; `--demo.no-fullscreen` opens it as a plain window. The application
runs the Fusion style with a dark palette (`#101014` window, `#d8d8e0`
text), so dialogs, scroll bars and anything else Qt draws match the panels
rather than the desktop.

Hotkeys:

| Key | Does |
|---|---|
| `Space` | Start a rollout; while one runs, pause it (the arm holds between two steps) and resume it |
| `S` | Stop it at the next step and home the arm |
| `G` | The switch: guidance on/off, from the next replan |
| `O` | Overlays: auto → on → off |
| `T` | Show/hide the bottom row (wrist camera, retrieved demo frame, trend plots, pairs) |
| `W` | Trend window: a snapshot around now, or the full trajectory |
| `H` | Training heat map on/off (also a checkbox in the button row; needs `TRAINING_POSITIONS`) |
| `N` | Distance numbers on the pair lines on/off (also a checkbox under KEYS) |
| `V` | Show/hide the 3-D panel |
| `L` | Viewer layout: the 3-D scene alone, or the full analysis layout |
| `F11` | Full screen on/off (off returns to the maximised window) |
| `Esc` | Quit — stops any rollout, parks the arm, shuts the servers down |

The `KEYS` panel in the side column lists the same table, and every key is
also a button along the bottom; a greyed button's key does nothing.

Between rollouts the arm is parked by the session itself, and the operator
re-stages the scene while it is parked. The env server is told not to home
again on its own reset (`--env.no-home-on-reset`), for the same reason the
experiment launcher does.

---

## Overlays

`--demo.overlays auto` (the default) draws the overlays while guidance is
on and the bare frame while it is off, so the picture itself says which
mode the arm is in. `on` and `off` pin them; `O` cycles.

- **Masks** — the tracked objects the distance pairs reference, each filled
  in its own colour with its name beside it. Objects the pairs do not use
  are not drawn.
- **Pairs** — a line between the two entities each pair measures, in the
  pair's colour. That palette is shared with the 3-D scene and the trend
  plot, so a line on the camera view, a line in Rerun and a trace on the
  plot in the same colour are the same pair.
- **Readouts** — a chip at each pair's midpoint with the live distance and
  the retrieved demo's distance at the aligned time, in metres, plus a dot
  in the error colour: green on the demo, red far from it.

Pairs are projected into the camera through the calibration the env server
publishes. With no `--env.calibration-path` the masks still draw (they are
image-space) but the pairs cannot be placed; see
[Troubleshooting](#troubleshooting).

---

## The trend plot

The reference plot (the retrieved demo over its own time) sits on the left and the live plot (the rollout over control steps) on the right, so the eye reads from what the demo did to what the arm is doing.
Two plots stacked in the bottom row, sharing the pair palette. The plots
draw no legend of their own: the **DISTANCE PAIRS** panel to their right is
the legend — one row per pair, a dot in the pair's colour, the label and the
live / reference values — and its last line names the plots' view mode.

- **Top: the retrieved demo.** Each pair's distance over the demo's time,
  with a standard-deviation band where the retrieval had several demos to
  average, and the sub-trajectory bands the demo analysis segmented it into
  shaded behind. A cursor marks the aligned time — where the alignment
  thinks the live rollout currently is in the demo — and the stretch the
  current action chunk covers is highlighted.
- **Bottom: the live rollout.** The same pairs' live distances over the
  rollout's steps, with the stretches where guidance was on shaded. This
  is the picture the switch is for: flip it and watch the traces move
  towards, or drift from, the demo's.

By default the plots show a **snapshot around now**, not the whole
trajectory: the reference plot spans three action chunks either side of
the aligned time, the live plot the last six chunks up to the current step
with one chunk of room ahead, and each y-axis fits the data in view. That
is what makes a trace's shape legible at the row's height — over a whole
rollout the same traces flatten into lines. `W` switches to the **full
trajectory** (the whole demo above, the whole rollout so far below) and
back; the pairs panel's last line says which is showing (`window: ±15
steps …` or `full trajectory`). The whole history is kept either way —
only the visible range moves. A third row with the per-object relevances exists but
is off by default (`TrendsPanel.show_relevance`), so the two plots share
the row's height.

The plot draws from the same `AbstractionResult` the runner produces per
step, so it lags the camera by nothing the camera does not lag by too.

---

## The Rerun view

The 3-D scene is a panel of the demo window, between the exterior view and
the side column. It shows the robot model, the object point clouds, the
distance pairs and the action chunks in 3-D, and is where the audience
sees the correction as a change of the arm's planned path.

How it gets there: the visualiser *serves* its recording from the demo
process (`--viz.serve`, the launcher's default) — a gRPC stream and Rerun's
web viewer on the usual ports — and the panel is an embedded browser
(QtWebEngine) pointed at that web viewer. Nothing else has to be running,
and there is no second window to place on the projector. The URL is
printed at start-up, so the same scene can also be opened in any browser on
the network. The panel comes up as soon as the visualiser exists, which is
after the servers, and until then says so. `V` hides and shows it.

The panel follows the window's dark theme. The Rerun viewer draws its UI on
a canvas and reads light/dark from `prefers-color-scheme`, which
QtWebEngine answers as light whatever the Qt palette says, so the window
installs a script at document creation that answers it as dark. The scene's
background is a grey-blue, not black, so black objects — the cable, the pen
— stay visible against it.

`--viz.layout scene` (what the launcher passes) collapses every panel but
the 3-D scene. `L` flips to the full analysis layout (cameras, retrieval,
charts) and back.

Without QtWebEngine — or with `--no-viz` — the panel is a note saying why,
and `V` puts it away. The alternative is a separate native viewer:
`--spawn-viewer` launches one for the run (needs a display here; dies with
the run), or start one yourself from the venv (`rerun` is not on `PATH`)
and point the run at it with `VIEWER_ADDR` (`127.0.0.1:9876`):

```bash
.venv/bin/rerun --port 9876
```

Over SSH, run that viewer on your laptop and forward the port to this
machine: `ssh -R 9876:localhost:9876 <host>`. With an external viewer the
scene streams there and the panel in the window stays empty.

---

## Performance notes

- **Rerun logging is the largest per-step cost** in this loop, and the loop
  already runs below its 50 Hz target on this cell (see [Control
  rate](ur5.md#control-rate)), so the arm moves more slowly with it on.
  `--no-viz` gives the fastest loop; the demo window still shows the
  annotated live view.
- **Perception runs every step** while the guided pipeline is up, whether
  the switch is on or off — that is what lets the switch be instant. The
  cost is the same as a guided evaluation run's.
- **The wrist stream** is a second RealSense on the same USB bus. Only one
  camera streams depth; the wrist is colour-only, and a wrist frame timeout
  after a successful connect is usually the camera falling off the bus
  (see [ur5.md](ur5.md)).
- **Start-up** is the checkpoint load, then the plain warm-up, then the
  guided warm-up. Each is announced in the window and in the log with how
  long it took.

---

## Pausing and stopping

`Space` during a rollout pauses it: the loop holds between two control
steps, no action is sent, and the arm keeps its last target. The plan, the
alignment and the RTC state are left exactly as they were, so `Space` again
resumes the same chunk where it stopped. While paused the exterior view
keeps streaming (the env is asked for frames), the banner reads `PAUSED`,
and `S` still stops and homes.


`S` ends the rollout at its next control step (or at once while paused) — the gate is checked before
every step, so nothing of the chunk is finished first — and the arm homes
immediately. Only then is the episode written: encoding a long rollout's
video streams into the `.vla` takes a minute or more, and that used to
happen *before* the homing move, which read as an arm that had stopped but
would not go home. The write now runs after the arm is parked, the status
strip says `writing the last episode to disk`, and the next rollout can be
started while it finishes.

---

## Outputs

```
data/<RUN_NAME>/<name>/          # RUN_NAME defaults to ur5_demo, --name to demo
  demo.log                       # DEBUG-level log of the whole session
  provenance.json                # git state, flags, environment
  episodes/demo_NNN_<stamp>.vla  # one recording per rollout, NNN counting from 000
  videos/                        # only with --demo.save-video
  *.log                          # the servers' stdout/stderr
```

Every rollout is recorded as a `.vla` episode, exactly as an evaluation
episode is — including depth on the usual stride and the camera extrinsic,
so a demo rollout can itself be encoded and retrieved from later. Replay
one with `python scripts/replay_episode.py`.

---

## Troubleshooting

- **Pairs are not drawn, masks are.** No camera extrinsic: the launcher
  found no file at `CALIBRATION` (it warned at start-up), or the file does
  not cover this camera. See [Camera calibration](ur5.md#camera-calibration).
- **`GUIDANCE UNAVAILABLE` on the frame with the switch on.** The
  abstractor could not be set up on this scene: nothing was detected, or
  the VLM found no pairs, and the rollout runs unguided. `demo.log` has the
  detector's and the VLM's own messages. A run launched `--unguided` shows
  it permanently, and `demo_ur5.py` warns about that at launch. Retrieval
  and demo analysis run on a rollout's first frame, so the first chunk of a
  rollout started with the switch on is already guided.
- **The window never opens / Qt aborts.** `demo_ur5.py` probes Qt in a
  subprocess before anything loads and prints Qt's own diagnostic with the
  fix; the usual one is `libxcb-cursor0` (or
  `scripts/setup/vendor_qt_xcb_cursor.sh` without root), and over SSH an
  unset `DISPLAY`. Same as the experiment GUI: see [Running the GUI over
  SSH](real_experiments.md#running-the-gui-over-ssh).
- **The 3-D panel is blank or says "unavailable".** The panel is Rerun's web
  viewer served from the demo process on `WEB_PORT` (9090) and `VIEWER_ADDR`'s
  port (9876); if either port is taken by an earlier run, `demo.log` says so
  at start-up — set `WEB_PORT` / `VIEWER_ADDR` or stop the other process. The
  placeholder names the URL, which any browser on this machine can open. A
  missing `PyQt6-WebEngine` also lands here; `--spawn-viewer` is the way
  round it.
- **`--policy.guidance off` refused.** That setting makes the policy server
  refuse guided requests, so every flip would fail at the server;
  `demo_ur5.py` exits before loading anything. Drop it, or run `--unguided`.
- **The first flip stalls the arm.** The guided warm-up did not cover this
  scene's guidance shape and the server recompiled once. It does not
  recur within the session.
