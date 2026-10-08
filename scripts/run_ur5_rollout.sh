#!/usr/bin/env bash
# Unguided pi0/pi0.5 rollout on the real UR5e cell.
#
# Cameras are addressed by RealSense serial (rs:<serial>) rather than
# /dev/videoN. Node numbers are handed out at enumeration and shuffle whenever
# anything is re-plugged or the machine reboots; the serial is stamped on the
# camera. Re-derive the mapping any time with `rs-enumerate-devices -s`.
#
# The rs: path also streams depth aligned to the colour frame
# (rs.align(depth -> colour)), which is what perception's reconstruct stage
# needs. It costs more per step than the plain V4L2 colour path.
#
# Usage:
#   scripts/run_ur5_rollout.sh                     # unguided baseline, Rerun on
#   scripts/run_ur5_rollout.sh --guided            # + retrieval, perception and guidance
#   RETRIEVAL_DIR=data/ur5_baseline/ur5/<task>/episodes \
#       scripts/run_ur5_rollout.sh --guided        # retrieve from a .vla folder
#   scripts/run_ur5_rollout.sh --no-viz            # skip Rerun (fastest loop)
#   scripts/run_ur5_rollout.sh --spawn-viewer      # launch a viewer for the run
#   scripts/run_ur5_rollout.sh --dry-run           # connects and reads sensors, never commands motion
#   scripts/run_ur5_rollout.sh --simulate          # arm powered off, synthetic observations
#   RETRIEVAL_DIR=... scripts/run_ur5_rollout.sh --guided --fake-policy
#                                                  # everything but the policy, no checkpoint load
#   scripts/run_ur5_rollout.sh --trials 3 --steps 600
#
# Visualisation is ON by default and connects to a viewer at $VIEWER_ADDR
# (127.0.0.1:9876) -- which is where an SSH reverse tunnel
# (`ssh -R 9876:localhost:9876 <host>`) puts a viewer running on your laptop.
# To run the viewer on this host instead, start it from the main checkout's
# venv (the `rerun` binary is not on PATH):
#     .venv/bin/rerun --port 9876
#
# The script probes that address before starting. If a viewer is there the run
# also passes --viz.require-viewer, so a tunnel that drops between the probe
# and the policy server's ~3 minute load still fails in two seconds rather than
# logging into the void. If nothing is listening it says so and runs without
# visualisation: a missing viewer should not cost you a robot run. --no-viz
# skips the probe entirely; --spawn-viewer launches a viewer for the run
# instead, which needs a display on this machine and dies with the run.
#
# --fake-policy swaps the checkpoint for an in-process fake so the ~2 minute
# policy load (and the GPU) drop out of the loop entirely. Everything upstream
# of the policy still runs for real -- cameras, perception, retrieval, demo
# analysis, alignment, guidance and the visualiser -- which is what makes it
# the way to debug those. `--fake-policy noop` (the default) holds the
# observed joints and lets the guidance gradient move them, through the UR5e
# forward kinematics, exactly as the real model's chunk would be corrected;
# `--fake-policy circular` traces a circle instead, for shaking out the
# visualiser.
#
# It also forces --dry-run: a held chunk plus a guidance correction is still a
# set of joint targets nobody has vetted, and the circular fake emits
# LIBERO-style *deltas* this arm would read as joint angles. Pass --simulate
# instead to run the same thing against synthetic observations with the arm
# powered off.
#
# Sub-trajectory segmentation is retuned for 50 Hz real demos -- see
# SUBTRAJ_SPEED_THRESHOLD in CONFIG, and `Sub-trajectory segmentation` in
# docs/ur5.md for the measurement behind the value.
#
# --guided steers the flow-matching integration with PA's __gp_*__ params. Three
# things to know: the correction lands on the model's *normalized* action tensor,
# so --guidance.factor is in normalized units for any checkpoint whose ACTION
# normalization is not IDENTITY (the server logs which at startup); the loss
# reads the chunk as UR5e joint targets through the forward kinematics
# (--policy.guidance-action-space auto -> ur5e-joints for env:ur5, see
# `How the loss reads the chunk` in docs/ur5.md -- the Euler integration the
# simulators use made every guided chunk start centimetres from the tool);
# and the first guided inference pays a torch.compile of the guided graph while
# the arm stands still. Guided runs also need a retrieved demo, so point
# --retrieval at a dataset covering this task.
#
# Visualisation is the largest per-step cost in this loop and the loop already
# runs below its 50 Hz target, so the arm replays slower with it on. That is
# the price of it being the default: pass --no-viz for timing runs.
#
# Anything in CONFIG can be overridden from the environment:
#   CHECKPOINT=checkpoints/pi05_place_coke_can_in_bowl_a100/pretrained_model \
#   TASK="Place the coke can in the bowl." scripts/run_ur5_rollout.sh
#
# Extra evaluate.py flags can be appended after `--`:
#   scripts/run_ur5_rollout.sh -- --output.no-save-episodes

set -euo pipefail

# --- CONFIG -----------------------------------------------------------------

# Address comes from envs/ur5/config.DEFAULT_ROBOT_IP (honouring UR5_ROBOT_IP)
# so there is exactly one place it is written down. Resolved after $PYTHON.
ROBOT_IP="${ROBOT_IP:-}"
TCP_OFFSET="${TCP_OFFSET:-0.00052 0.00109 0.22347 0 0 0}"

# Verified 2026-08-26 against a live capture from every connected camera.
# NB: the wrist camera is NOT the /dev/video12 device earlier docs used --
# that is serial 207222070875, a different camera.
EXTERIOR_SERIAL="${EXTERIOR_SERIAL:-262722073432}"   # scene view of the table (D435I fitted 2026-09-22)
WRIST_SERIAL="${WRIST_SERIAL:-115222071166}"         # wrist-mounted
CAMERA_RES="${CAMERA_RES:-640x480@30}"               # native training resolution

CHECKPOINT="${CHECKPOINT:-checkpoints/pi05_put_all_cables_in_bag_a100/pretrained_model}"
# The prompt handed to the policy. Must match what the checkpoint was trained
# on -- this string *is* the conditioning.
TASK="${TASK:-Put all cables in the bag.}"

INSTANCE="${INSTANCE:-9}"          # port pair: policy 8000+N, env 8765+N
GPUS="${GPUS:-0}"                  # single-GPU host; without this --instance sets CUDA_VISIBLE_DEVICES=9
RUN_NAME="${RUN_NAME:-ur5_baseline}"
MAX_STEPS="${MAX_STEPS:-1250}"     # 1250 / 50 Hz ~= 25 s of motion (~40 s at the loop's real ~31 Hz)
TRIALS="${TRIALS:-1}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"   # 25/50 Hz = 0.5 s of motion per inference
# Where --guided retrieves its demo from. A folder of .vla rollouts works
# directly once encoded (scripts/encode_datasets.py --source-path <dir>);
# without this, evaluate.py falls through to the LIBERO dataset defaults,
# which are simulator demos and not present on this host.
RETRIEVAL_DIR="${RETRIEVAL_DIR:-}"
# Sub-trajectory segmentation, in the units compute_subtrajectories() uses:
# metres per *step*, not per second. This arm runs at 50 Hz against LIBERO's
# 20 Hz, so the same physical speed is 2.5x smaller here per step, and the
# shared default (0.005 m/step = 0.25 m/s at 50 Hz) sits above almost
# everything a hand-paced real demo does -- no active run survives
# --subtraj.min-length, compute_subtrajectories falls back to one segment for
# the whole episode, and the decomposition silently does nothing. Measured on
# data/ur5_demo_depth/.../episode_000: the smoothed speed is above 0.005 for
# 4% of the episode and above 0.01 for none of it.
#   0.001 m/step = 0.05 m/s at 50 Hz -- above this demo's idle drift
#   (~0.01 m/s) and below its transit speed.
SUBTRAJ_SPEED_THRESHOLD="${SUBTRAJ_SPEED_THRESHOLD:-0.001}"
# 50 steps = 1 s at 50 Hz, the same *duration* the shared default (20 steps)
# buys at LIBERO's 20 Hz. Raise it to merge fine structure, lower it to keep
# short motions as their own sub-trajectory.
SUBTRAJ_MIN_LENGTH="${SUBTRAJ_MIN_LENGTH:-50}"
# Hand-eye calibration. Without it the server publishes no camera extrinsic,
# so the recorded depth cannot be placed in the robot frame and the episode is
# unusable as a guidance demo -- the depth would be recorded and then have no
# consumer. Set to "" to run without one.
CALIBRATION="${CALIBRATION:-configs/ur5_camera_calibration.json}"
# Which base frame that extrinsic is expressed in. Normally empty: the
# calibration file declares its own frame (configs/ur5_camera_calibration.json
# says `base_link`, the ROS/URDF convention Kalib produces), and the file is
# the right place for it -- hardcoding a frame here would silently *override*
# a different file that says otherwise. UR's two frames sit 180 degrees apart
# about z, and declaring the wrong one mirrors the scene in x and y while
# leaving the table perfectly horizontal: a plane-fit sanity check still
# passes, and only a comparison against the robot's own TCP catches it. Set
# this only to override a file you cannot edit.
CALIBRATION_BASE_FRAME="${CALIBRATION_BASE_FRAME:-}"
SDK_PATH="${SDK_PATH:-$HOME/dev/physical-ai-sdk}"
# Where a Rerun viewer is expected. One source of truth: the pre-flight probe
# and the run itself both use it.
VIEWER_ADDR="${VIEWER_ADDR:-127.0.0.1:9876}"

# --- ARGS -------------------------------------------------------------------

viz=1
guided=0
spawn_viewer=0
dry_run=0
simulate=0
fake_policy=""
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guided)         guided=1 ;;
        --viz)            viz=1 ;;
        --no-viz)         viz=0 ;;
        --spawn-viewer)   spawn_viewer=1 ;;
        --dry-run)        dry_run=1 ;;
        --simulate)       simulate=1 ;;
        # Optional kind: `--fake-policy circular`. Bare, it means noop.
        --fake-policy)    fake_policy=noop
                          case "${2:-}" in noop|circular) fake_policy="$2"; shift ;; esac ;;
        --trials)         TRIALS="$2"; shift ;;
        --steps)          MAX_STEPS="$2"; shift ;;
        --)               shift; extra=("$@"); break ;;
        # Print the header comment block, minus the shebang, as usage.
        -h|--help)        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                              "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "unknown option: $1 (pass evaluate.py flags after --)" >&2; exit 2 ;;
    esac
    shift
done

# A fake policy's actions are not joint targets for this arm, so never send
# them. --simulate is the stronger form of the same thing (no arm at all) and
# is left alone.
if [[ -n "$fake_policy" && $simulate -eq 0 ]]; then
    dry_run=1
fi

# --- INTERPRETER ------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        # A git worktree has no venv of its own -- borrow the main checkout's
        # interpreter and put this tree's sources first on the path.
        main_git="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
        PYTHON="$(dirname "$main_git")/.venv/bin/python"
    fi
fi
[[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set PYTHON=...)" >&2; exit 1; }
export PYTHONPATH="src:packages/policy_steering/src${PYTHONPATH:+:$PYTHONPATH}"

# One source of truth for the arm's address: envs/ur5/config.DEFAULT_ROBOT_IP,
# which honours $UR5_ROBOT_IP. $ROBOT_IP still wins if you set it directly.
if [[ -z "$ROBOT_IP" ]]; then
    ROBOT_IP="$("$PYTHON" -c \
        'from policy_abstraction.envs.ur5.config import DEFAULT_ROBOT_IP; print(DEFAULT_ROBOT_IP)')"
fi
[[ -n "$ROBOT_IP" ]] || { echo "could not resolve ROBOT_IP" >&2; exit 1; }

# --- PRE-FLIGHT -------------------------------------------------------------

if [[ $simulate -eq 0 ]] && command -v nc >/dev/null; then
    mode="$(printf 'robotmode\n' | nc -w 2 "$ROBOT_IP" 29999 2>/dev/null | tr -d '\r' | tail -1 || true)"
    if [[ "$mode" != *RUNNING* ]]; then
        echo "WARNING: $ROBOT_IP dashboard says '${mode:-<no reply>}', expected 'Robotmode: RUNNING'." >&2
        echo "         Check the arm is powered, braked off and in remote control." >&2
    fi
fi

# Visualisation is on by default, so check the viewer is actually there. A
# connection to a loopback address either completes or is refused immediately,
# so this costs nothing and cannot hang.
if [[ $viz -eq 1 && $spawn_viewer -eq 0 ]]; then
    if ! (exec 3<>"/dev/tcp/${VIEWER_ADDR%%:*}/${VIEWER_ADDR##*:}") 2>/dev/null; then
        echo "No Rerun viewer at $VIEWER_ADDR -- running without visualisation." >&2
        echo "  start one here:   $REPO_ROOT/.venv/bin/rerun --port ${VIEWER_ADDR##*:}" >&2
        echo "  or forward yours: ssh -R ${VIEWER_ADDR##*:}:localhost:${VIEWER_ADDR##*:} $(hostname)" >&2
        echo "  or spawn one:     scripts/run_ur5_rollout.sh --spawn-viewer" >&2
        viz=0
    fi
fi

for serial in "$EXTERIOR_SERIAL" "$WRIST_SERIAL"; do
    if command -v rs-enumerate-devices >/dev/null; then
        rs-enumerate-devices -s 2>/dev/null | grep -q "$serial" \
            || echo "WARNING: RealSense $serial not enumerated -- check the USB connection." >&2
    fi
done

# --- COMMAND ----------------------------------------------------------------

cmd=("$PYTHON" scripts/evaluate.py
     --instance "$INSTANCE"
     --iter.num-trials-per-task "$TRIALS"
     --output.run "$RUN_NAME")

if [[ -n "$fake_policy" ]]; then
    cmd+=(--guidance.debug --guidance.fake-policy "$fake_policy")
fi

# Set explicitly rather than inferred from env:ur5, for the same reason the
# policy tuning below is: it is visible in the printed command and can be
# overridden after `--`.
cmd+=(--subtraj.gripper-speed-threshold "$SUBTRAJ_SPEED_THRESHOLD"
      --subtraj.min-length "$SUBTRAJ_MIN_LENGTH")

if [[ $guided -eq 1 ]]; then
    # The UR5e publishes neither depth nor ground-truth masks the way a
    # simulator does, so the guided pipeline needs the real perception stack.
    cmd+=(--perception.mode REAL_PERCEPTION)
    [[ -n "$RETRIEVAL_DIR" ]] && cmd+=(--retrieval.dataset-dirs "$RETRIEVAL_DIR")
else
    # One switch for retrieval + demo analysis + guidance.
    cmd+=(--no-guide)
fi

if [[ $viz -eq 1 ]]; then
    # env:ur5 auto-selects the menagerie UR5e model and bolts a Robotiq
    # 2F-140 onto its flange (see Args.__post_init__), so both the arm and
    # the gripper render in 3D without naming either here.
    cmd+=(--viz.enable --viz.robot-description ur5e_mj_description)
    if [[ $spawn_viewer -eq 1 ]]; then
        cmd+=(--viz.connect None)
    else
        # The probe above already found a viewer; re-check at startup so a
        # tunnel that drops during the policy server's ~3 minute load fails in
        # two seconds rather than logging into the void.
        cmd+=(--viz.connect "$VIEWER_ADDR" --viz.require-viewer)
    fi
else
    cmd+=(--viz.no-enable)
fi

cmd+=(env:ur5
      --env.robot-ip "$ROBOT_IP"
      --env.tasks "$TASK"
      --env.max-steps "$MAX_STEPS"
      --env.tcp-offset $TCP_OFFSET
      --env.cameras "exterior_image_1_left=rs:${EXTERIOR_SERIAL}:${CAMERA_RES}"
                    "wrist_image_left=rs:${WRIST_SERIAL}:${CAMERA_RES}")

[[ -n "$CALIBRATION" ]] && cmd+=(--env.calibration-path "$CALIBRATION")
# Only when explicitly overridden -- otherwise the file's own declaration wins.
[[ -n "$CALIBRATION_BASE_FRAME" ]] \
    && cmd+=(--env.calibration-base-frame "$CALIBRATION_BASE_FRAME")

if [[ $simulate -eq 1 ]]; then
    cmd+=(--env.simulate-hardware)
elif [[ $dry_run -eq 1 ]]; then
    cmd+=(--env.dry-run)
else
    # Start from where the arm was staged rather than homing first.
    cmd+=(--env.no-home-on-reset)
fi

# Real-arm tuning, set here rather than inferred from env:ur5 so it is visible
# in the printed command and can be overridden after `--`. The arm is
# *stationary* during inference, so replanning every 5 steps (the simulator
# default) would stall it more often than it moves; RTC then blends each new
# chunk into the unexecuted tail of the last. The blend horizon must cover
# replan_steps or evaluate.py refuses to start.
if [[ -n "$fake_policy" ]]; then
    # The subcommand is still required to pick a backend, but nothing under it
    # is loaded: --guidance.debug returns before the server launch. Only
    # replan-steps is read, by the runner, to size the fake's action chunk.
    cmd+=(policy:lerobot --policy.replan-steps "$REPLAN_STEPS")
else
    cmd+=(policy:lerobot
          --policy.model-path "$CHECKPOINT"
          --policy.gpus "$GPUS"
          --policy.repo-path "$SDK_PATH"
          --policy.replan-steps "$REPLAN_STEPS"
          --policy.rtc
          --policy.rtc-execution-horizon "$REPLAN_STEPS")
fi

cmd+=("${extra[@]+"${extra[@]}"}")

printf '%q ' "${cmd[@]}"; echo
exec "${cmd[@]}"
