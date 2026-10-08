# Real-world experiments — operator-driven collection

Running a policy many times with an object in different places, and getting
a number out of it.

One operator, one policy setting, many rollouts, **one policy load**. The
GUI shows the live camera with a grid drawn on the table and the next
target cell highlighted; the operator stages the object there, presses
SPACE, watches, and calls the result with a hotkey. Everything is recorded
per rollout, and the whole thing resumes by name if it is interrupted.

Two experiments are built:

1. **Positional distribution shift** on "put the cable in the box" — the
   cable starts at each cell of a grid, several times, first with the
   policy alone and then with guidance.
2. **Target disambiguation** — several named objects on the table at once
   and a prompt that names one of them, to see whether the policy goes to
   the object it was *told* to. See
   [Disambiguating the target](#disambiguating-the-target).

3. **Appearance shift** on "pour the contents of the left cup into the right cup" — the
   same pour from several different cups, a few rollouts each, with the
   prompt held fixed. See
   [Varying the object's appearance](#varying-the-objects-appearance).
4. **Appearance shift** on "place the pen in blue cup on paper" — the same
   fetch with several different pens, a few rollouts each, under the
   training prompt. Same factor, same report, a different checkpoint; see
   [the same section](#varying-the-objects-appearance).

All run through the same GUI, store and report, and differ only in a YAML
file. Orientation is a further axis; adding one is a stanza in the same
file (see [Later experiments](#later-experiments)).

Background on the arm itself: [ur5.md](ur5.md).

---

## The 60-second version

```bash
# 0. rehearse the whole workflow with no hardware at all
MUJOCO_GL=egl python scripts/experiments/collect_experiment.py \
    --experiment.experiment-config configs/experiments/libero_rehearsal.yaml \
    --experiment.experiment-name rehearsal \
    --guidance.debug --retrieval.no-enable --demo-analysis.no-enable \
  env:libero --env.perturb.no-swap policy:openpi

# 1. get the grid right (projects it onto one camera frame, then exits)
python scripts/experiments/collect_experiment.py \
    --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \
    --experiment.calibrate \
  env:ur5 --env.no-launch-server policy:lerobot --policy.model-path <ckpt>

# 2. collect the baseline
python scripts/experiments/collect_experiment.py \
    --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \
    --experiment.experiment-name cable_position_baseline \
    --experiment.policy-setting baseline --no-guide \
  env:ur5 --env.no-launch-server \
  policy:lerobot --policy.model-path <ckpt> --policy.no-launch-server

# 3. same plan, guidance on. The retrieval source is a folder of encoded
#    demo rollouts (scripts/encode_datasets.py); without it retrieval falls
#    back to the LIBERO simulator datasets, so the script refuses to start.
python scripts/experiments/collect_experiment.py \
    --experiment.experiment-config configs/experiments/cable_in_box_position.yaml \
    --experiment.experiment-name cable_position_guided \
    --experiment.policy-setting guided --perception.mode REAL_PERCEPTION \
    --retrieval.dataset-dirs data/reference_dataset_cable \
  env:ur5 ... policy:lerobot ...

# 4. report (any time, including mid-experiment)
python scripts/experiments/collect_experiment.py --experiment.report-only \
    --experiment.experiment-name cable_position_baseline
```

On the real cell there is a launcher per experiment that fills in the
checkpoint, the camera serials, the calibration and the pre-flight, so the
whole of the above is:

```bash
scripts/experiments/run_cable_position_experiment.sh --calibrate
scripts/experiments/run_cable_position_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_cable \
    scripts/experiments/run_cable_position_experiment.sh --guided
scripts/experiments/run_cable_position_experiment.sh --report

scripts/experiments/run_object_disambiguation_experiment.sh --calibrate
scripts/experiments/run_object_disambiguation_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_cable \
    scripts/experiments/run_object_disambiguation_experiment.sh --guided

scripts/experiments/run_pour_appearance_experiment.sh --calibrate
scripts/experiments/run_pour_appearance_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_pour \
    scripts/experiments/run_pour_appearance_experiment.sh --guided

scripts/experiments/run_pen_appearance_experiment.sh --calibrate
scripts/experiments/run_pen_appearance_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_pen \
    scripts/experiments/run_pen_appearance_experiment.sh --guided
```

`--guided` needs `RETRIEVAL_DIR`, the folder of encoded demo rollouts the
guidance retrieves from, and the launcher refuses to start without it. The
guided arm otherwise runs exactly what `scripts/run_ur5_rollout.sh --guided`
runs: real perception, the retrieved demo's distance pairs, and a guidance
loss that reads the chunk as UR5e joint targets through the forward
kinematics (`--policy.guidance-action-space auto`, see
[ur5.md](ur5.md#how-the-loss-reads-the-chunk)). `collect_experiment.py`
refuses a `guided` setting whose guidance, retrieval or demo analysis is
switched off, and a `baseline` setting with guidance on, so a rollout cannot
be filed under the wrong arm.

`--help` on either prints its own usage. Everything about the *cell* —
serials, TCP offset, pre-flight, the tyro flag routing — lives once in
`scripts/experiments/_ur5_experiment.sh`, which both source; a launcher is
a config path, a default prompt and its help text.

Every evaluation flag is spelled exactly as in `scripts/evaluate.py` —
`env:ur5`, `--policy.model-path`, `--guidance.factor` — because it is the
same argument tree. Only `--experiment.*` is new.

---

## While the servers come up

The window opens **first** — within a second of the command — and the
servers, the env and the ~3 minute checkpoint load run on a worker thread
behind it. The start-up screen shows the log as it happens, so a slow
compile and a wedged socket look different from across the cell, and the
operator window replaces it by itself when the backends are ready.

* **Closing the start-up screen cancels**, and cancelling waits. The
  servers are mid-launch; walking away would leave them running with
  nothing owning them. The window says it is cancelling and closes itself
  once start-up finishes and can be shut down cleanly. Same for Ctrl-C.
* **A start-up failure is shown in the window**, not only in the terminal —
  that window is the one place an operator finds out why nothing opened.
  The process then exits non-zero.
* **The live camera is not up yet.** It arrives with the env server, which
  the operator window then streams; the start-up screen is the log only.
* Before any of it, a one-second subprocess probe checks the GUI can open
  at all, so a missing Qt plugin costs a second rather than the whole
  checkpoint load. See [Running the GUI over SSH](#running-the-gui-over-ssh).

`--experiment.headless` skips all of this and loads the backends inline.

### The policy warm-up

The last step before the operator window is one throw-away inference. It
exists because a `torch.compile`d checkpoint — pi05 defaults to
`compile_model=true, compile_mode=max-autotune` — autotunes on its **first
real query, not at load**. Measured on the UR5e cell: the server reported
ready after 106 s and then spent a further **449 s** inside that first
inference.

Paid by the first rollout, that is a rollout in which the arm does not
move, the operator cannot tell compiling from hung, and `timeout_s` is
spent before the robot has been asked to do anything. Paid here, the first
rollout is an ordinary one.

* **The arm is never commanded for it.** One observation is read, one
  query is sent, and the actions are dropped. No `step`, no `reset`.
* **The start-up screen re-headlines to `WARMING UP THE POLICY`** for the
  duration, so the longest silence in start-up says what it is.
* **RTC state is cleared afterwards.** The warm-up chunk is a real chunk
  as far as the server is concerned, and left in the previous-chunk buffer
  it would be inpainted into the opening chunk of rollout 0.
* **A failed warm-up is not fatal** — it is logged, and the first rollout
  pays the compile as before. The operator window's banner says
  `warming up the policy` at step 0 in that case.

To skip the autotune entirely, at the cost of slower steady-state
inference:

```bash
scripts/experiments/run_cable_position_experiment.sh --baseline -- \
    --policy.compile-model off
```

`timeout_s` is measured from the rollout's **first executed step**, not
from the moment it was started, so set-up — env reset, retrieval, demo
analysis, a first inference that was not warmed — never eats the task's
budget. The record still reports the full wall-clock duration.

---

## The operator loop

| Key | Button | Does |
|---|---|---|
| `SPACE` | Start rollout | Homes the arm, then runs the policy. Only when idle. |
| `S` | Stop — success | Ends the rollout, records a success, homes. |
| `F` | Stop — failure | Ends it, homes, **then** asks which kind of failure — one or several — and, in a scene with several named objects, which one(s) it went for. |
| `C` | STOP — catastrophic | Emergency stop. Records a catastrophic failure. Asks nothing. |
| `K` | Skip condition | This staging was wrong — log it as aborted and offer it again later. |
| `Ctrl+H` | History | The rollouts collected so far, with their verdicts editable. Only between rollouts. |
| `Ctrl+E` | Settings | Change the grid, the repeats and the limits. Only between rollouts. |
| `R` | Report | Regenerate the summary and plots from what is collected so far. |
| `Esc` | Interrupt experiment | Stop, park the arm, write the report, close. |

Buttons and hotkeys are one definition
(`experiments/gui/actions.py`), so a greyed-out button always has a dead
hotkey — they cannot drift apart.

The right-hand column starts with the **wrist camera**, streamed live from
the same observation as the staging view: between rollouts it refreshes
with the preview, during one it follows the policy's own per-step images.
It is a plain picture — no grid, no ghost, nothing drawn on it — there to
show the grasp the staging camera cannot see. On an env without a wrist
camera the panel is not shown.

`F` stops the arm *first* and asks afterwards. The classification dialog is
keyboard-first (`1`–`7` toggle a mode each, Enter records). **Tick every way
it failed**: a rollout that went for the wrong object and then dropped it is
both a `wrong target` and a `dropped`, and the second observation is worth
keeping. Nothing ticked records `other`. The record keeps the whole list in
`failure_modes`, and `failure_mode` holds the first of them in taxonomy
order for anything that wants one answer per rollout.

However long the operator takes over the dialog, the answer lands: on the
pending stop if the rollout is still stepping, on the record if it has
resolved but is not yet written, or as a superseding row if it is already on
disk and the arm is homing. Rates are computed from the last row per
rollout, so nothing double-counts.

**Timeouts are failures.** `timeout_s` in the config ends the rollout and
records `TIMEOUT` on its own — nothing to press while the arm winds down.
The failure-mode panel then opens anyway, asking what the arm was doing
when the clock ran out; what you pick is recorded *alongside* `TIMEOUT`, so
the report still counts the timeout. Cancelling leaves it as a plain
timeout. `max_steps` does the same. The clock runs
from the first *executed* step, so it budgets the arm's time and not the
set-up before it — see [The policy warm-up](#the-policy-warm-up).

No display? `--experiment.headless` drives the same verdicts from stdin
with the same letters; after `f`, answer with one number or several
(`3 2`, or `3,2`).

`Ctrl+E` and `Ctrl+H` are deliberately the two actions that are not a
single keystroke. Everything else here is one key because the operator's
hand is on the keyboard and the arm is moving; those two open a modal
dialog and rewrite what is on disk, so they ask for the reach.

**While a rollout runs** the banner counts the steps taken against the
budget — `RUNNING rollout 7 - step 143/1250`. The budget is `max_steps`
from the config, or the environment's own limit for the task when the
config sets none. It is the one number that separates a policy working
slowly from a policy that has stopped moving, which from across the cell
look identical.

---

## Re-scoring a rollout

Operators misclassify: the bag was already full and the rollout counted as
a failure, or `S` landed on the rollout before the one that succeeded.
`Ctrl+H` (or the **History** button, between rollouts) lists every rollout
recorded in this experiment — condition, verdict, failure mode, steps,
time, notes — and re-scores the selected one:

| Button | Does |
|---|---|
| Mark success | Records it as a success. Any failure mode is dropped, since a success did not fail a particular way |
| Mark failure… | Opens the same `1`–`7` taxonomy the `F` key uses, pre-ticked with what was recorded, with a note field. The selection replaces the old one rather than adding to it |
| Discard (aborted) | Stops it counting toward any rate and offers its condition again — for a rollout that was mis-staged rather than failed. Asks first |

**Nothing is overwritten.** `rollouts.jsonl` is append-only, so a
correction is a new row and the original stays exactly as it was pressed
at the time. The verdict that *stands* for a rollout is its last row, and
every rate in the report is computed from those — so a rollout re-scored
twice still counts once, and the log remains the honest record of what
happened when.

The corrected row copies everything else from the original: the timing,
the step count, the staged positions, the episode path. Only the verdict
and the notes change, and the notes say what the change was
(`reclassified failure -> success`).

Re-scoring is refused while a rollout is running — the rollout thread is
about to append to the same log, and which of the two rows landed last
would decide the verdict.

---

## Changing the experiment mid-session

The grid is never right until it has been looked at. `Ctrl+E` (or the
**Settings** button, between rollouts) opens a form over the live view:

| Field | |
|---|---|
| Experiment name | The directory, and the name the experiment resumes by |
| Report title | The heading on `report.md`. Not used for resume |
| Task | The prompt handed to the policy |
| Repetitions per cell | How many times the full set of cells is swept |
| Order, seed | `systematic` or `random`, and what seeds the shuffle |
| Max steps, timeout | The per-rollout caps. Blank/zero means no cap |
| Rows, columns | The grid's shape |
| Cell size, origin, marker height | The grid's geometry, in metres, base frame |
| Cells | A subset, e.g. `A1, A2, B1`. Blank sweeps every cell |
| Region size, origin | For an `object_layout` factor: the patch of table the objects are scattered over, in metres |
| Layouts | How many scenes that factor samples |
| Min separation, edge margin | How far apart the objects are kept, and clear of the region's edge |
| Angles / variants | For orientation and appearance factors |

The dialog keeps a running count of what the current form plans — "24
rollouts planned — up to 0.8 h at the 120s timeout" — so the consequence of
a change is visible before it is applied. Applying keeps the servers up:
the policy is not reloaded, and the arm is not moved.

### What happens to results already collected

Applying is only ever destructive if it has to be, and it never decides
that for you:

| The edit | What happens |
|---|---|
| Leaves the plan alone (task, timeout, view scale, …) | Applied in place. Nothing is at risk |
| Changes the plan, nothing collected yet | The plan is rewritten in place |
| Changes the name | A new experiment — or resumes that name if it already holds the same plan |
| Changes the plan **and** rollouts exist | **Asks** |

That last case offers two ways out and a cancel:

* **Start a new experiment under a new name** — the default, pre-filled
  with a free name (`cable_position_baseline_v2`). The old results stay
  where they are and can still be resumed. This is almost always the right
  answer: it costs a name and keeps both sets.
* **Overwrite** — deletes the directory outright: records, episodes,
  videos, report. Confirmed **twice** — once for the warning, once by
  typing the experiment name back — and there is no undo. Unlike
  `--overwrite` on the command line, which renames the old directory to
  `<name>.superseded_<timestamp>`, this one really deletes.

### The edit survives closing the window

The authoritative copy of an experiment's settings is its own
`experiment.yaml`, written into the results directory. **Relaunch with the
same command and the edit is picked up** — the grid you reshaped at the
cell is the grid you resume against, with no flags to remember:

```bash
scripts/experiments/run_cable_position_experiment.sh --baseline
#   ... Ctrl+E, grid 3x4 -> 4x3, repeats 3 -> 2, close
scripts/experiments/run_cable_position_experiment.sh --baseline   # 4x3, 2 repeats
```

The operator's own YAML is deliberately *not* rewritten — it may be the
source for the other arm of the comparison, which is running against the
plan it already has. The rule on the next launch is that **the name
decides**: an experiment that already exists under `--name` resumes under
its own `experiment.yaml`, whatever the launch YAML now says, and the YAML
only ever *creates* an experiment. If the two differ, the launcher says
which fields it is not applying and carries on.

| Since the last launch | What resumes |
|---|---|
| Only the GUI settings changed | The **edited settings**. The YAML is not overriding them |
| The YAML was edited too | Still the **stored settings**, with a warning naming the fields that differ |
| The YAML was left off the command | The **stored settings**. `--name` alone is enough to resume |
| `--overwrite` was passed | The **YAML**. "Start over" means start over from the file |

(The YAML as parsed at creation is kept as `source_config_snapshot` in
`metadata.json`, for the record.)

Whatever is applied is *also* written as
`configs/experiments/<experiment name>.yaml`, next to the source, so the
edited settings can be handed to the other arm or checked in. Copying that
file (or the experiment's `experiment.yaml`) over the original is how you
make an edit apply to both arms.

**If the edit renamed the experiment**, the old directory stays exactly
where it was — so the launcher's default name still opens it. The rename is
recorded on the old experiment as `superseded_by`, the launcher says so on
resume rather than quietly collecting into the superseded half of a
comparison, and the closing summary prints the flag to use:

```
The settings were edited during this session, so resume with
    --name cable_position_baseline_v2
```

Headless (`--experiment.headless`) has the same editor behind `e`, with the
same rules and the same two confirmations, prompted at the terminal.

---

## Defining the grid

The grid is 3-D: a plane in the **robot base frame**, not a rectangle in
the image. That is what lets the same cell mean the same physical place
across sessions, and what puts a metric target position in every record.

Specifically the controller's `base` frame — the one RTDE reports poses
in and the one the env server publishes camera extrinsics in — **not**
the ROS/URDF `base_link` that hand-eye tools such as Kalib produce. The
two sit 180 degrees apart about z, so a grid written in `base_link` is
mirrored through the robot: it lands behind the arm, out of the camera's
view, and the overlay simply does not appear. See `docs/ur5.md`,
§ Camera calibration. The quick check is the arm's own numbers — on the
cell in `data/ur5_demo_depth/` the end-effector works at *negative* x, so
a grid at positive x is in the wrong frame.

```yaml
factors:
  - kind: grid_position
    name: cable_position
    grid:
      origin: [-0.30, 0.25, 0.015]   # outer corner of cell A1
      u_axis: [0.0, -1.0, 0.0]        # direction of increasing column
      v_axis: [-1.0, 0.0, 0.0]        # direction of increasing row
      cell_size_u: 0.10
      cell_size_v: 0.10
      rows: 3
      cols: 4
      marker_height: 0.015            # half the object's height
      up_hint: [0.0, 0.0, 1.0]        # which way is "off the table"
```

Cells are labelled `A1`, `B3`, … — rows are letters, columns 1-based —
because that is what an operator can read off the screen and repeat back.

Things worth knowing:

* **The axes are squared up for you.** `v_axis` is Gram-Schmidt
  orthogonalised against `u_axis` and both are normalised, so measuring
  two table edges by eye is good enough.
* **`up_hint` decides which way `marker_height` lifts.** `u × v` is a
  normal of the plane, but its sign depends on the order you wrote the
  axes in; without the hint, half the configs would sink the target
  marker into the tabletop.
* **Column numbering can come out mirrored** relative to what the
  operator sees, depending on where the camera is. Negate `u_axis` if so —
  it is a config choice, not a bug. Negating *both* axes and the origin
  is a different thing entirely: that is the `base`/`base_link` mix-up
  above, and it moves the grid rather than renumbering it.
* **`cells:`** restricts the sweep to a subset (unreachable corners,
  occluded cells) *without* renumbering the rest, so it stays comparable
  with an earlier experiment.
* **`first_col: 0`** widens a grid by one column *in front of* column 1
  after data has been collected. Move `origin` one cell back along
  `u_axis`, add one to `cols`, and the new column is `A0`, `B0`, … while
  every label already in `rollouts.jsonl` still names the same cell. This
  is how the extra cells of the positional experiment were merged into it;
  widening behind the last column needs nothing special, the labels just
  continue.

Iterate with `--experiment.calibrate`: it projects the grid onto one live
frame, writes a PNG, prints every cell centre in base coordinates, and
never commands the arm.

### The overlay needs camera calibration

Projection needs the camera's intrinsics **and** its extrinsic (the
camera-to-base pose). On the UR5e that means `--env.calibration-path` — see
[ur5.md](ur5.md#camera-calibration). Without it the GUI still runs
and still records everything; it just says the grid cannot be drawn, and
the report skips the projected map.

---

## Disambiguating the target

`configs/experiments/object_disambiguation.yaml`. Several named objects sit
on the table at once; the prompt names one of them. The scene is identical
across the objects of a layout and **only the instruction changes**, so a
difference in success rate between objects is the policy resolving — or
failing to resolve — the reference in the prompt.

```bash
scripts/experiments/run_object_disambiguation_experiment.sh --calibrate
scripts/experiments/run_object_disambiguation_experiment.sh --baseline
scripts/experiments/run_object_disambiguation_experiment.sh --guided
```

Two factors, and the plan is their cross product:

```yaml
task: "Put the {target_object} in the bag."   # a template

factors:
  - kind: object_layout
    name: object_layout
    objects: [black cable, white cable, red cable]
    layouts: 8              # 8 arrangements x 3 objects = 24 rollouts
    min_separation: 0.10    # metres between object centres
    margin: 0.06            # metres kept clear inside the region edge
    seed: 0
    region:                 # same block as `grid:`, with rows: 1, cols: 1
      origin: [-0.25, 0.20, 0.0]
      u_axis: [0.0, -1.0, 0.0]
      v_axis: [-1.0, 0.0, 0.0]
      cell_size_u: 0.40     # 40 cm across
      cell_size_v: 0.30     # 30 cm deep
      rows: 1
      cols: 1
      marker_height: 0.015

  - kind: target_object
    name: target_object
    layout: object_layout   # takes its levels from that factor's objects
```

The GUI draws the region, a labelled cross per object, and a green ring
round the one the prompt names. The operator places all of them and presses
SPACE, exactly as before.

The region is the thing that is wrong first, and `Ctrl+E` edits it without
restarting: its **size** in metres (across `u`, deep along `v`) and its
**origin** in the base frame, plus the number of layouts, the minimum
separation and the edge margin. Moving or resizing it re-samples the
layouts — a different plan — so the usual rules about results already
collected apply ([above](#what-happens-to-results-already-collected)).
Shrinking it past what the separation and margin allow is reported in the
dialog rather than raising: the summary line turns red and says so.

Things worth knowing:

* **The layouts are re-sampled, and that is the point.** With the objects
  nailed to fixed spots, "went to the named object" and "went to the same
  place as always" are the same observation — a policy that always grasps
  whatever is nearest would score 1/N and look like it was reading the
  prompt. Prefer more `layouts` to more `repeats` for the same reason: a
  repeat re-runs the same scene, a layout is a new arrangement.
* **Positions are seeded and reproducible.** Both arms of a
  baseline/guided comparison stage identical scenes. Each layout has its
  own sub-seed, so raising `layouts` leaves the earlier ones — and the
  rollouts already collected against them — untouched.
* **`min_separation` is enforced by rejection sampling.** If the objects
  cannot be placed, the config fails to load and says whether to enlarge
  the region, shrink `margin`, lower the separation, or use fewer objects.
* **`{target_object}` is the factor's name.** Rename the factor and the
  placeholder must change with it. A task with no placeholder is sent
  verbatim, which is what every other experiment does. A placeholder no
  factor supplies is logged and the prompt sent as written — mid-session, a
  literal prompt is recoverable where a crash is not.
* **Keep the wording around the placeholder to what the checkpoint was
  trained on.** It is still the prompt, and the checkpoint still only
  knows one phrasing. `prompt_name` on an object sets what the *policy* is
  told when that should differ from what the operator reads:

  ```yaml
  objects:
    - id: black_cable
      label: "black USB cable (coiled, left of the bag)"
      prompt_name: "black cable"
  ```

A different set of objects, or a different *number* of them, without
editing the YAML:

```bash
scripts/experiments/run_object_disambiguation_experiment.sh --baseline \
    --objects "black cable,white cable,red cable,blue cable" --overwrite
```

That re-samples the layouts for the new count, which is a different plan —
hence `--overwrite` if the experiment has already collected anything.

### Reading the result

The report breaks the rate down by **target object** (did the prompt work)
and by **layout** (the control — a spread here means position, not
language, is driving it). `report/target_object_rates.png` is the headline;
`report/object_layout_on_camera.png` draws every staged object on the real
camera view, scored where it was the named target, so a policy that reaches
for whatever is nearest shows up as failures in a band rather than spread
evenly.

The `wrong target` failure mode is the one to watch: it is offered in the
`F` dialog and counted per object in the report table, which is the
difference between "could not pick it up" and "picked up the wrong thing".

Every object's position is written into each rollout record —
`object_layout.black_cable`, and `target_object` for whichever was named —
so where the *distractors* were is available to the analysis too.

---

## Varying the object's appearance

`configs/experiments/pour_cup_appearance.yaml`, launched by
`scripts/experiments/run_pour_appearance_experiment.sh`, which swaps in the
pouring checkpoint. One cup is poured into another, and the cup being
poured *from* changes between conditions — other colours, heights,
materials — while the prompt stays `Pour the contents of the left cup into the right cup.`
on every rollout, naming the cups by where they stand so it is true
whichever cup is on the mark. The success rate per cup is how far the
policy generalises past the object it was shown, and the guided arm asks
whether guidance from a demo closes the gap. Note the demos were recorded
with `Pour contents of blue cup into green cup.`; set `task:` back to that
to test the training phrasing on its own.

```bash
scripts/experiments/run_pour_appearance_experiment.sh --calibrate
scripts/experiments/run_pour_appearance_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_pour \
    scripts/experiments/run_pour_appearance_experiment.sh --guided
scripts/experiments/run_pour_appearance_experiment.sh --report
```

```yaml
task: "Pour the contents of the left cup into the right cup."   # not a template
repeats: 5                                          # per cup

factors:
  - kind: appearance
    name: source_cup
    variants: [demo_cup, cup_2, cup_3]   # the demos' own cup first
    descriptions:
      demo_cup: "blue cup from the demos"
      cup_2: "second cup (edit me)"
      cup_3: "third cup (edit me)"

  - kind: grid_position                  # 1x1: a staging mark, not a sweep
    name: source_cup_position
    object: cup
    grid: {rows: 1, cols: 1, ...}
  - kind: grid_position
    name: receiving_cup_position
    object: green cup
    grid: {rows: 1, cols: 1, ...}
```

The appearance factor has no geometry: the badge on the live view says
which cup to stage (`Use the second cup source cup`), and the operator
swaps it by hand. The two single-cell grids are there so that the cups go
back to the same spot every time — the GUI draws a cross for each, the
record carries the nominal position, and `measure_positions.py` can check
the staging error afterwards, asking the detector for `cup` and `green
cup` (override per cup with `--query source_cup_position=red cup` when the
noun matters). With one cell they multiply the plan by one.

Things worth knowing:

* **Keep the demos' cup as the first variant.** It is the control every
  other cup is read against; a low rate on it says the session, not the
  cup, is the problem.
* **The variant ids are the keys.** `demo_cup`, `cup_2`… name the records
  and the report rows, so keep them once anything is collected;
  `descriptions` is only what the operator reads and can be reworded at
  any time. `Ctrl+E` edits the list mid-session; a changed list is a new
  plan, with the usual offer of a new experiment name.
* **`grouping: condition`** stages each cup once for all of its repeats,
  since the swap is the slow part. `sweep` cycles through the cups every
  repeat instead, which keeps coverage balanced if the session is cut
  short.
* **The guided arm needs pouring demos with depth.** The raw
  demonstrations the checkpoint was trained on carry neither depth nor a
  camera extrinsic, so they cannot be a retrieval source. Record one with
  `evaluate.py`, or copy a successful baseline rollout of this experiment
  (`.vla` and `.json` together) into `data/reference_dataset_pour`, encode
  it with `scripts/encode_datasets.py --source-path`, and point
  `RETRIEVAL_DIR` at the folder. See
  [ur5.md](ur5.md#guiding-from-an-earlier-real-rollout).
* **Reading it**: `report/source_cup_rates.png` is the headline — success
  per cup with Wilson bars — and `source_cup_outcomes.png` says *how* each
  cup failed, which separates "could not grasp the tall one" from "poured
  next to the green cup".

### The same experiment on the pen

`configs/experiments/pen_appearance.yaml`, launched by
`scripts/experiments/run_pen_appearance_experiment.sh`, is the pen-fetching
counterpart: the `place_the_pen_in_blue_cup_on_paper` checkpoint, a `pen`
appearance factor (`demo_pen` first, then whatever pens are on the bench)
and single-cell staging marks for the pen (`pen_position`, detector query
`pen`) and the blue cup on its paper (`cup_position`, query `blue cup`).
The prompt is the training phrase `Place the pen in blue cup on paper.`
on every rollout — it never describes the pen, so unlike the pouring
experiment there is nothing to reword, and only the object is out of
distribution. The guided arm needs an encoded pen rollout with depth in
`data/reference_dataset_pen`, made exactly as for the pour. The
segmentation defaults are left alone: on the raw pen demos they give
approach + grasp, carry + drop, and retreat. The report reads the same
way, with `report/pen_rates.png` as the headline.

```bash
scripts/experiments/run_pen_appearance_experiment.sh --calibrate
scripts/experiments/run_pen_appearance_experiment.sh --baseline
RETRIEVAL_DIR=data/reference_dataset_pen \
    scripts/experiments/run_pen_appearance_experiment.sh --guided
scripts/experiments/run_pen_appearance_experiment.sh --report
```

---

## What comes out

```
data/<run>/experiments/<name>/
  experiment.yaml     the config these results were collected under, and
                      what a relaunch resumes with after a settings edit
  plan.json           the frozen condition list — resume's source of truth
  rollouts.jsonl      one line per rollout, append-only
  metadata.json       policy setting, task id, the evaluation args, the
                      YAML this was launched from, and any rename
  reference_frame.png  the staging view the report projects onto
  observed_positions.json  where each staged object really was, measured
                      from the recordings (see below); absent until measured
  episodes/*.vla      the recordings (the existing RobodmRecorder format)
  videos/*.mp4
  report/            summary.json, report.md, plots
  evaluation.log
```

Episodes are plain `.vla` files exactly as `evaluate.py` writes them, so
`scripts/replay_episode.py` replays them unchanged.

`report/` holds:

| File | Shows |
|---|---|
| `<factor>_on_camera.png` | The success map **projected onto the real camera view** — cells with any success tinted on a green ramp (pale for one-in-three, deep for a clean sweep), cells that never succeeded red, one dot per rollout (filled = success, ring = failure). Dots sit at the **measured** object position once the experiment has been measured; on a ring around the cell centre until then |
| `<factor>_measured.png` | The measured object positions in the grid's own plane, metric, with the offset from each cell centre. Only once measured |
| `<factor>_grid_success.png` | The same rates as an abstract grid, green through red |
| `<factor>_grid_rollouts.png` | Every rollout as a dot, to spot outliers and spread |
| `failure_modes.png` | The breakdown by cause. A failure given several modes is counted under each, so the bars can add up to more than the number of failures |
| `progress.png` | Cumulative rate over the session — a drift check |
| `completion_time.png` | Every rollout's completion time, successes against failures |
| `<factor>_rates.png` | Success rate per level with Wilson bars, for non-grid factors (the target object, say) |
| `<factor>_outcomes.png` | Each condition against how its rollouts ended, as a heatmap |
| `confusion_matrix.png` | Named target against the object actually reached (disambiguation experiments only) |
| `report.md` | Per-cell table with Wilson 95% intervals, the completion times, both matrices, plus the headline |

For a disambiguation experiment `<factor>_on_camera.png` is the layout map
instead: every staged object drawn where the operator put it, scored on the
rollouts where it was the named target.

### Where the object really was

The dots on `<factor>_on_camera.png` are, by default, drawn at the centre
of the cell the operator was shown — the plan, not the experiment. A cable
never sits exactly there. To draw the dots where the object actually was,
measure the recordings after the session:

```bash
uv run python scripts/experiments/measure_positions.py \
    --name cable_position_baseline_v3 --run ur5_experiments
```

This reads each rollout's **first recorded frame** — colour, metric depth
and the calibrated camera, all of which the recorder already writes — and
runs the same perception pipeline the guided policy uses (`detect → SAM2 →
reconstruct`) on it, asking for the factor's `object:` (defaults to the
noun the factor is named after, so `cable_position` asks for "cable"). The
mask is back-projected through the depth image into the robot base frame
with the abstractor's own call, and the centroid is written to
`observed_positions.json` beside the experiment, keyed like each record's
`target_positions`. The report is rebuilt at the end, so the camera view
and the new `<factor>_measured.png` show the measured positions, and
`report.md` reports the median and worst offset from the nominal cell
centre — the staging error to check against the cell size before reading
anything into a per-cell rate.

The frame the results are drawn on is the staging view saved when the
session started, which is not always the best picture of the scene. To use
a recorded rollout's opening frame instead — one with the cable staged at
a mid-grid cell, say — pick it by level or by plan index and the report is
rebuilt on it (the first frame is kept as `reference_frame.original.png`):

```bash
uv run python scripts/experiments/set_reference_frame.py \
    --name cable_position_baseline_v3 --run ur5_experiments --level B3 --timestep 40
```

`--timestep` picks the frame within that recording: 0 is the staged scene
before the arm moves, a few dozen steps in shows the arm approaching the
object.

It needs the perception models (a GPU) and the detector's API key, exactly
as a guided run does, but no robot and no servers. Rollouts already measured
are skipped on a re-run; `--force` redoes them, `--indices` restricts to
some, and `--query cable_position=orange cable` overrides the noun when the
detector needs a different one. One annotated frame per rollout is written
to `report/measured/` to eyeball the masks.

Cell rates use **Wilson** intervals, not the normal approximation: with 3
samples a cell, the normal interval runs outside [0, 1] and collapses to
zero width on a clean sweep. `rate_spread` — the gap between the best and
worst cell — is the headline number for a distribution-shift experiment.

### Completion time

Every rollout records its wall clock — `duration_s`, from the operator
pressing start to the verdict being written — and the report has a section
for it:

```
| Rollouts                | n  | Median | Mean  | Std | Range       | p95   |
| All scored              | 17 | 50s    | 58s   | 27s | 27s – 1m53s | 1m46s |
| Successes               | 12 | 42s    | 42s   |  8s | 27s – 53s   | 52s   |
| Failures                |  5 | 1m40s  | 1m39s | 10s | 1m26s–1m53s | 1m51s |
| — of which timed out    |  3 | 1m40s  | 1m39s | 11s | 1m26s–1m53s | 1m52s |
| Warm-up, excluded above |  1 | 5m25s  | 5m25s |  —  | 5m25s–5m25s | 5m25s |
```

Split by outcome, because pooling them produces a number that means
nothing: a success time is *how long the policy took to do the task*, a
failure time is mostly how long the operator waited before calling it, and
a timeout is the cap read back. The median is reported alongside the mean
for the same reason — the distribution is usually bimodal, and the mean
describes neither mode.

**The warm-up rollout is held out.** The first inference of a session
compiles a freshly loaded checkpoint, so the first rollout to reach the
policy costs minutes rather than the ~90 s the rest do. It is flagged
`warmup: true` in `rollouts.jsonl` when the session records it, reported on
its own line, and excluded from every other figure — from the headline
mean, from the per-cell medians, from the plot, and from the GUI's estimate
of the time left. One of them in a plan of thirty-six would otherwise move
the mean further than the effect the experiment is measuring. It is *not*
excluded from success rates (it ran, and it either worked or it did not),
nor from `total_wall_clock_s`, which is the honest cost of the afternoon.

The flag goes on the first rollout that actually reached the policy, not
simply on rollout 0: one that raised on the way there warmed nothing up,
and the next one still pays. Editing the experiment from the GUI does not
restart the server, so nothing re-warms.

Per-cell times are in the factor table too (`Median time` and `Median when
successful`), and `summary.json` carries the full set —
`mean_s`, `median_s`, `std_s`, `min_s`, `max_s`, `p25_s`, `p75_s`, `p95_s`,
`total_s` — under `completion_time` and under each level's `duration`.
`completion_time.warmup` is the held-out rollouts and
`completion_time.with_warmup` the raw wall clock, for anyone who wants it.

The GUI shows the running mean and an estimate of the time left in the
**PROGRESS** panel, which is the form of the number an operator actually
wants at 4pm.

---

## The two matrices

### Outcomes by condition

A cross-tab of what was staged against how it ended: rows are the factor's
levels, columns `success` plus each failure mode that actually occurred.
Only the modes that happened get a column — seven empty ones would bury
the two carrying the result. A failure given several modes is counted once
under each, so a row can add up to more than its rollouts.

It needs no data beyond the verdict, so it is produced for every
experiment, including ones collected before any of this existed.

| target_object | success | grasp miss | wrong target |
|---|---:|---:|---:|
| black cable | 7 | 1 | |
| white cable | 2 | | 6 |
| red cable | 6 | 1 | 1 |

### Confusion matrix

For a disambiguation experiment only — it needs a named target, and a
positional experiment stages one object with nothing to confuse it with.
Rows are the object the prompt named, columns the object the policy
actually went for; the diagonal (ringed in the PNG) is the reference
resolved. The PNG is objects-by-objects only — 3x3 for three objects —
and each row carries its **task success rate** down the right-hand side.
That rate is a different question from the diagonal: the cells say which
object the arm went for, the rate says whether the task was completed on
the object named, and a row can be all-diagonal and still fail by dropping
what it picked up. Rollouts that reached no object, or whose object was
never recorded, are not cells in the PNG; they still count in the row
totals and are named under the axes when there are any. The markdown
table keeps every column, plus the task success.

This is a strictly sharper result than a success rate. "It fails on the
white cable" and "it fails on the white cable **by going to the black one
every time**" are different findings that call for different next
experiments, and only the matrix can tell them apart.

| named \ went for | black cable | white cable | red cable | *no object* | *unrecorded* |
|---|---:|---:|---:|---:|---:|
| black cable | 7 | 1 | | | |
| white cable | 6 | 2 | | | |
| red cable | 1 | | 6 | 1 | |

Two columns are not objects:

* ***no object*** — the operator watched and says it never committed to
  one. A real observation, and counted in the accuracy denominator.
* ***unrecorded*** — nobody wrote down what it reached: a rollout
  collected before the question was asked, or an object since removed
  from the config. Held out of the accuracy rather than assumed onto the
  diagonal, which would flatter the policy.

A rollout that went for **several** objects — reached for the black cable,
hesitated, then the red one — is counted once under each, the same way a
failure given several modes is, so a row can add up to more than its
rollouts. The accuracy still counts rollouts: one that touched the named
object at some point is one resolved reference, however many others it
visited first. The report says how many rollouts were counted this way.

### Where the answer comes from

Only a person watching can say which object the arm went for, so the
failure dialog asks — the `F` key's dialog, and the same one when
re-scoring from `Ctrl+H`. The objects are checkboxes (`A`, `B`, `C`… to
toggle, `0` for *no object*), **pre-ticked to the object the prompt
named**, so the commonest failure (reached the right object, fumbled it)
needs no answer at all; you only touch it when the policy went somewhere
else. Tick every object it went for, in the order it went for them: the
record keeps that order (`actual_targets`), and the first one also lands in
the older single `actual_target` column. Ticking *no object* clears the
others, and vice versa. A success is recorded as having gone for the named
object without asking, because that is what the success asserts.

Without a display the same question is a numbered list; answer `1 3` (or
`1,3`) to name several in order, `0` for none, Enter to keep the named one.

Nothing is asked for an experiment that stages one object.

---

## Resume, and not mixing results

Re-run the identical command. The plan is frozen on disk, so collection
picks up at the next rollout with no result, in the same order.

That order is, by default, every repeat of a cell back to back (`A1, A1,
A1, A2, A2, A2, …`): the object is staged once per cell and left there.
Set `grouping: sweep` in the YAML (or the *Repeat grouping* field under
`Ctrl+E`) to visit every cell once per repeat instead, which keeps the
coverage balanced if the session is cut short but costs an object move
between every rollout.

* A **scored** rollout (success or failure) is never offered again.
* An **aborted** one (`K`, or a crash) is offered again — it never really ran.
* An existing `--name` **loads the experiment's own configuration** — the
  YAML on the command line is not applied to it (a warning names any
  fields that differ) and may be left off entirely. Editing the YAML
  therefore never changes an experiment that is already collecting. To run
  the edited file, give it a new name, or start over with `--overwrite`
  (`--experiment.overwrite` when calling `collect_experiment.py`
  directly), which renames the old directory to
  `<name>.superseded_<timestamp>` rather than deleting it:

  ```bash
  scripts/experiments/run_cable_position_experiment.sh --baseline --overwrite
  ```

  `--overwrite` always starts again from the YAML, so it also discards any
  settings edited from the GUI — that is what "start over" means here.
* The other arm of a comparison can be created **from the first arm's
  directory** rather than from the YAML, with `--reference <name>`
  (`--experiment.reference-experiment` on `collect_experiment.py`; a path
  to the directory works too):

  ```bash
  scripts/experiments/run_object_disambiguation_experiment.sh --guided \
      --reference object_disambiguation_baseline
  ```

  The new experiment takes the reference's stored `experiment.yaml` —
  GUI edits and an `--objects` override included — so it stages the
  identical sequence of layouts and prompts. Positions are not written
  into the plan (each layout is re-sampled from its seed when the settings
  are loaded), so before the checkpoint loads the launcher checks that
  those settings still generate the reference's `plan.json` row for row
  and still put every object where the reference *recorded* it on each
  rollout; if either has drifted, it refuses rather than collecting under
  the old level names. `--reference` replaces the YAML and does not
  combine with `--objects`. Like the YAML, it only creates: an existing
  `--name` still resumes under its own settings, and `--overwrite`
  recreates it from the reference. The reference's directory is written
  to the new experiment's `metadata.json` as `reference_experiment`.

  While staging, the window also **ghosts the reference's own picture** of
  the condition under the live view: the first recorded frame of the
  reference rollout that staged the same assignments, blended in faintly
  (30%). The crosses say where each object's centre goes; the ghost says
  how the cable was coiled and which way the banana pointed, so the scene
  can be matched by nudging the objects until the two pictures coincide.
  `G` cycles it (faint, strong, reference only, off), and it is hidden while the
  arm moves. A condition the reference never scored has no ghost and the
  badge says so. The reference is remembered with the experiment, so a
  resume shows the ghost without `--reference` on the command line.

  This is the usual sequel to editing the grid — `--calibrate` changes the
  cells, so the plan on disk no longer describes the experiment you are
  about to run. Editing the grid from the GUI instead
  ([above](#changing-the-experiment-mid-session)) handles the same
  situation without a restart, and offers a new experiment name rather
  than assuming you want to start over.

Give each policy setting its own `--experiment.experiment-name`. Same YAML,
same plan, same conditions in the same order — which is what makes the two
arms comparable cell by cell.

---

## Later experiments

Each is a stanza in the same YAML. Nothing in the session, GUI, store or
report changes. They are all implemented and commented out in
`configs/experiments/cable_in_box_position.yaml`:

```yaml
  - kind: orientation           # arrow drawn in the anchor factor's cell
    name: cable_orientation
    anchor: cable_position
    angles_deg: [0, 45, 90, 135]

  - kind: appearance            # instruction badge; no geometry
    name: cable_appearance
    variants: [black_usb, white_hdmi, coiled_grey]

  - kind: grid_position         # a second grid, for the box
    name: box_position
    grid: {...}
```

Factor names must be unique — they key the assignment dict. Name them for
the *axis*, not the object: `cable_position` and `cable_orientation`, not
two factors both called `cable`.

Every added factor multiplies the rollout count, so add one at a time and
reconsider `repeats`. A new factor is a different plan, so give it a new
experiment name.

For a genuinely new *kind* of axis, subclass `Factor`
(`experiments/factors.py`), implement `levels` / `instruction` /
`annotate`, and `register_factor` it — configs can then name it.

---

## Safety

The arm is parked on every exit path: normal close, `Esc`, Ctrl-C in the
terminal, an exception inside a rollout, and an exception escaping the Qt
event loop. Shutdown waits for the rollout thread before stopping the
servers, so the arm is never abandoned mid-motion.

Beneath that, the env server's own guarantees still apply — joint rate
limiting, joint limits, park-on-disconnect. See
[ur5.md § Safety](ur5.md#safety). None of it substitutes for a hand on the
e-stop.

A rollout that raises is still recorded (as aborted, with the error), so
the log never quietly loses a rollout the operator watched happen.

---

## Running the GUI over SSH

`ssh -X` (or `-Y`) and run it as normal. Two things this box
needed:

* **`libxcb-cursor0`.** Qt 6.5+ needs it for the `xcb` platform
  plugin and the PyQt6 wheel does not bundle it. Without it Qt does
  not raise — it aborts:

  ```
  qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
  to load the Qt xcb platform plugin.
  ```

  ```bash
  sudo apt-get install -y libxcb-cursor0
  ```

  **No root?** The same 10 kB library, dropped into the venv:

  ```bash
  scripts/setup/vendor_qt_xcb_cursor.sh          # fetch + install
  scripts/setup/vendor_qt_xcb_cursor.sh --check  # report only
  ```

  It goes in the venv's own `PyQt6/Qt6/lib/`, which the plugin
  already searches (`RUNPATH=$ORIGIN/../../lib`) — no
  `LD_LIBRARY_PATH`, no root. Re-run it after rebuilding the venv;
  it is idempotent and a no-op when the system package is present.

* **A live `DISPLAY`.** `echo $DISPLAY` must be non-empty on the
  remote end. If it is empty, X11 forwarding did not come up.

The session probes this **before** starting any server, so a
missing GUI dependency costs a second rather than the ~3 minutes of a
checkpoint load, and it says which package to install. Nothing is
loaded and the arm is not touched when it refuses.

No display at all? `--experiment.headless` drives the same rollouts
from stdin with the same hotkey letters — you lose the live view and
the overlay, but the plan, the store and the report are identical.

---

## Rehearsing without hardware

`configs/experiments/libero_rehearsal.yaml` runs the identical code path —
same GUI, same overlay, same store, same report — against a LIBERO
tabletop with the in-process dummy policy. Use it to check a grid's
geometry, the hotkeys and the report before going near the arm.

```bash
MUJOCO_GL=egl python scripts/experiments/collect_experiment.py \
    --experiment.experiment-config configs/experiments/libero_rehearsal.yaml \
    --experiment.experiment-name rehearsal \
    --guidance.debug --guidance.fake-policy circular \
    --retrieval.no-enable --demo-analysis.no-enable --viz.no-enable \
  env:libero --env.task-suite-name libero_goal --env.perturb.no-swap \\
  policy:openpi
```

`configs/experiments/libero_disambiguation_rehearsal.yaml` does the same for
the disambiguation workflow — the staging overlay, the region, and that the
prompt really does change per rollout.

Add `--experiment.headless` on a machine with no display.

---

## Reference

| Module | Role |
|---|---|
| `experiments/grid.py` | The plane, its cells, and their world positions |
| `experiments/layout.py` | Scattering several named objects across a region |
| `experiments/factors.py` | The axes an experiment varies |
| `experiments/plan.py` | The frozen condition list: all repeats of a cell back to back by default |
| `experiments/config.py` | The YAML schema |
| `experiments/revise.py` | What editing that config means for results on disk |
| `experiments/outcomes.py` | Verdicts, failure taxonomy, record schema |
| `experiments/store.py` | On-disk layout and resume |
| `experiments/overlay.py` | Drawing the target onto a camera frame |
| `experiments/session.py` | Rollouts on a worker thread, against loaded backends |
| `experiments/report.py` | Aggregation, Wilson intervals, markdown |
| `experiments/viz.py` | Grid maps and breakdown plots |
| `experiments/gui/` | The PyQt6 operator window, and the start-up screen in front of it |
| `experiments/gui/settings.py` | The settings form and the overwrite confirmations |
| `experiments/gui/history.py` | The collected-rollouts list, and re-scoring one |
| `evaluation/backends.py` | Server/client lifecycle, shared with `evaluate.py` |
| `evaluation/config.py` | The evaluation argument tree, shared with `evaluate.py` |
