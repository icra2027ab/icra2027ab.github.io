# Real deployment — script cheat sheet

Launchers, all from the repo root, all with `--help`. Background:
[ur5.md](ur5.md).

| Script | Does |
|---|---|
| `scripts/run_ur5_rollout.sh` | Runs a policy on the arm |
| `scripts/experiments/run_cable_position_experiment.sh` | Many rollouts over a staging grid, with an operator GUI — see [real_experiments.md](real_experiments.md) |
| `scripts/run_ur5_demo.sh` | Live demo: one loaded policy, guidance flipped on and off under it from a window — see [demo.md](demo.md) |
| `scripts/debug/perception_playground.sh` | Perception only, interactive; arm never commanded |

Optional viewer for both (`rerun` is not on PATH):

```bash
.venv/bin/rerun --port 9876
```

---

## Policy rollout

```bash
scripts/run_ur5_rollout.sh                  # unguided baseline
scripts/run_ur5_rollout.sh --guided         # + retrieval, perception, guidance
scripts/run_ur5_rollout.sh --viz            # + Rerun (fails fast if no viewer)
scripts/run_ur5_rollout.sh --viz --spawn-viewer
scripts/run_ur5_rollout.sh --dry-run        # reads sensors, never moves
scripts/run_ur5_rollout.sh --simulate       # arm off, synthetic observations
scripts/run_ur5_rollout.sh --trials 3 --steps 600
scripts/run_ur5_rollout.sh -- --output.no-save-episodes   # extra evaluate.py flags
```

Overrides: `CHECKPOINT` `TASK` `RUN_NAME` `MAX_STEPS` `TRIALS` `REPLAN_STEPS`
`INSTANCE` `GPUS` `EXTERIOR_SERIAL` `WRIST_SERIAL` `UR5_ROBOT_IP`

```bash
CHECKPOINT=/path/to/pretrained_model TASK="Place the coke can in the bowl." \
    scripts/run_ur5_rollout.sh
```

Defaults: `pi05_put_all_cables_in_bag_a100`, 1250 steps, 1 trial, instance 9
(policy `8009`, env `8774`), replan every 25 steps with RTC on.

---

## Perception playground

```bash
scripts/debug/perception_playground.sh                 # cable + bag
scripts/debug/perception_playground.sh cable bag mug   # your queries
scripts/debug/perception_playground.sh --wrist         # + wrist camera (uncalibrated)
scripts/debug/perception_playground.sh --simulate      # arm off, real cameras
scripts/debug/perception_playground.sh --no-server     # attach to your own server
scripts/debug/perception_playground.sh -- --max-frames 40 --save-rrd out.rrd
```

Starts the env server and **stops it on exit**, releasing the camera. Type
while it streams:

| Input | Effect |
|---|---|
| `cable, bag` | replace queries, re-detect |
| `+ mug` | add a query, re-detect |
| `!redetect` / `!settings` | re-detect / print config |
| `!quit` / Ctrl-C | exit |

Overrides: `EXTERIOR_SERIAL` `WRIST_SERIAL` `CALIBRATION` `PORT` `UR5_ROBOT_IP`

Headless: `-- --connect None --save-rrd out.rrd`, then `.venv/bin/rerun out.rrd`.

Per frame: detect ~3.5–4 s (VLM round-trip), track ~50–66 ms (15–20 Hz).

---

## Site values

Re-check if anything moves or gets re-imaged.

| | Value | How to re-check |
|---|---|---|
| Robot IP | `192.168.1.23` | `ip neigh \| grep 00:30:d6` (UR's MAC prefix) |
| TCP offset | `0.00052 0.00109 0.22347 0 0 0` | server warns if it disagrees |
| Exterior camera | `rs:262722073432` (calibrated) | `rs-enumerate-devices -s` |
| Wrist camera | `rs:115222071166` | same |
| Checkpoint | `checkpoints/pi05_put_all_cables_in_bag_a100/pretrained_model` | |

The IP lives in `envs/ur5/config.DEFAULT_ROBOT_IP`; override with
`UR5_ROBOT_IP`, not by editing files. Confirm the arm is ready:

```bash
printf 'robotmode\nsafetystatus\nis in remote control\n' | nc "$UR5_ROBOT_IP" 29999
# want: RUNNING / NORMAL / true
```

---

## Known issues

- **The loop holds ~31 Hz, not 50** (`50/62 steps overran by 12.4 ms`), so the
  arm replays ~40 % slow. Suspected cause: two 640×480×3 frames (~1.8 MB) per
  control step over the websocket. Profile before changing anything.
- **No inference warmup.** The first replan pays a torch.compile of the RTC
  path while the arm stands still. Was implemented on the pre-merge server and
  not carried over; re-add against `serve_lerobot.py` if the stall bites.
- **Cameras can leak on a failed arm connect.** `_ensure_robot` opens the
  RealSenses before `robot.connect()`, so an arm failure leaves them held and
  the retry misreports `Device or resource busy`.

## Camera busy

```bash
lsof /dev/video*     # PIDs; `ps` alone misses holders you would not think to grep for
```

Usually a `serve_ur5_env.py` left running. The playground wrapper cleans up
after itself; a hand-started server does not.
