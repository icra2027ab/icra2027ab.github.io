#!/usr/bin/env bash
# Live guidance demo on the real UR5e cell.
#
# One policy, loaded once; one rollout at a time; and a switch that turns
# guidance on and off UNDER the running policy. The window streams both
# cameras, and while guidance is on it draws the tracked objects and the
# distance pairs on the exterior view and plots the live distances against the
# retrieved demo's. A Rerun viewer alongside shows the 3-D scene -- robot
# model, object clouds, pairs, action chunks -- in a layout with every other
# panel collapsed, for a projector.
#
# Hotkeys in the window:
#   Space   start a rollout            S    stop it and home the arm
#   G       guidance on / off          O    overlays auto / on / off
#   T       trend plot                 V    viewer layout scene / full
#   Esc     quit (parks the arm, stops the servers)
#
# Usage:
#   RETRIEVAL_DIR=data/reference_dataset_cable scripts/run_ur5_demo.sh
#                                          # guided pipeline up, switch off at start
#   RETRIEVAL_DIR=... scripts/run_ur5_demo.sh --start-guided
#                                          # switch on from the first rollout
#   scripts/run_ur5_demo.sh --unguided     # streaming-only demo, no retrieval
#   scripts/run_ur5_demo.sh --spawn-viewer # a separate native Rerun window instead of the panel
#   scripts/run_ur5_demo.sh --external-viewer  # stream to a viewer already at $VIEWER_ADDR
#   scripts/run_ur5_demo.sh --full-viewer  # the full analysis layout instead of the 3-D scene
#   scripts/run_ur5_demo.sh --no-viz       # no Rerun at all (fastest loop)
#   scripts/run_ur5_demo.sh --name day2 --max-steps 900
#   RETRIEVAL_DIR=... scripts/run_ur5_demo.sh --simulate --fake-policy circular
#                                          # rehearsal: no arm, no checkpoint, real pipeline
#   scripts/run_ur5_demo.sh --dry-run      # connects and reads sensors, never commands motion
#   scripts/run_ur5_demo.sh -- --demo.refresh-hz 10 --env.max-steps 900 --policy.gpus 1
#                                          # extra demo_ur5.py flags, routed by prefix
#
# Guided is the DEFAULT here, unlike the rollout and experiment launchers,
# because the switch needs the whole pipeline -- retrieval, demo analysis,
# real perception, guidance -- to have anything to open. That needs
# RETRIEVAL_DIR, a folder of encoded demo rollouts (encode a folder of .vla
# files with scripts/encode_datasets.py). --unguided drops all of it and the
# demo just streams the policy.
#
# Visualisation is ON by default and SERVED from the demo process: a gRPC
# server on $VIEWER_ADDR's port and the Rerun web viewer on $WEB_PORT, which
# the window embeds as its 3-D panel (V shows and hides it, L flips between
# the 3-D-only and the full layout). No viewer binary, no tunnel. Two
# alternatives: --spawn-viewer opens a separate native Rerun window for the
# run (needs a display here), and --external-viewer streams to a viewer
# already listening at $VIEWER_ADDR (your laptop's, over
# `ssh -R 9876:localhost:9876 <host>`); if nothing is there it falls back to
# the in-window viewer rather than costing a run.
#
# --fake-policy swaps the checkpoint for an in-process fake, so the ~3 minute
# policy load and the GPU drop out. Everything upstream still runs for real --
# cameras, perception, retrieval, alignment, guidance, the window and the
# viewer -- which is what makes it the rehearsal. It forces --dry-run unless
# --simulate is given: a fake's actions are not joint targets for this arm.
#
# Anything in CONFIG can be overridden from the environment:
#   CHECKPOINT=... TASK="Place the coke can in the bowl." scripts/run_ur5_demo.sh
#   RUN_NAME, INSTANCE, GPUS, REPLAN_STEPS, EXTERIOR_SERIAL, WRIST_SERIAL,
#   CALIBRATION, VIEWER_ADDR, WEB_PORT, UR5_ROBOT_IP / ROBOT_IP,
#   COMPILE_MODEL (off: no torch.compile, the default here),
#   CACHE_LIVE_OBJECTS (0: always re-query the VLM for the live objects),
#   CACHE_PAIRS (0: re-analyse the retrieved demo's pairs at every rollout),
#   TRAINING_POSITIONS (the demos' grasp positions for the heat-map overlay),
#   RECORD (1: write every rollout as a .vla episode; off by default)
#
# See docs/demo.md.

set -euo pipefail

# --- CONFIG -----------------------------------------------------------------

# Address comes from envs/ur5/config.DEFAULT_ROBOT_IP (honouring UR5_ROBOT_IP)
# so there is exactly one place it is written down. Resolved after $PYTHON.
ROBOT_IP="${ROBOT_IP:-}"
TCP_OFFSET="${TCP_OFFSET:-0.00052 0.00109 0.22347 0 0 0}"

EXTERIOR_SERIAL="${EXTERIOR_SERIAL:-262722073432}"   # scene view of the table (D435I fitted 2026-09-22)
WRIST_SERIAL="${WRIST_SERIAL:-115222071166}"         # wrist-mounted
CAMERA_RES="${CAMERA_RES:-640x480@30}"               # native training resolution

# The pairs are projected onto the live view through this. Without it the
# window still runs and cannot draw them. The checked-in file, not the raw
# hand-eye output, because that one declares which base frame it is in and
# the server refuses a file that does not. See docs/ur5.md § Camera calibration.
CALIBRATION="${CALIBRATION:-configs/ur5_camera_calibration.json}"
# Which base frame that extrinsic is expressed in. Normally empty: the
# calibration file declares its own frame, and the file is the right place for
# it -- hardcoding a frame here would silently *override* a different file that
# says otherwise. Set this only to override a file you cannot edit.
CALIBRATION_BASE_FRAME="${CALIBRATION_BASE_FRAME:-}"

CHECKPOINT="${CHECKPOINT:-checkpoints/pi05_put_all_cables_in_bag_a100/pretrained_model}"
# The prompt handed to the policy. Must match what the checkpoint was trained
# on -- this string *is* the conditioning.
TASK="${TASK:-Put all cables in the bag.}"

INSTANCE="${INSTANCE:-9}"            # port pair: policy 8000+N, env 8765+N
GPUS="${GPUS:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"   # 25/50 Hz = 0.5 s of motion per inference
# No torch.compile: the compiled paths cost minutes at start-up (and a second
# compile the first time guidance is switched on) for a faster steady state
# the demo does not need. Set COMPILE_MODEL=auto to honour the checkpoint.
COMPILE_MODEL="${COMPILE_MODEL:-off}"
# Always ask the VLM which live objects are on the table rather than replaying
# a cached answer: the demo re-stages the scene between rollouts, and a cached
# match from a different staging silently loses the target object.
CACHE_LIVE_OBJECTS="${CACHE_LIVE_OBJECTS:-0}"
# Likewise re-run the VLM's distance-pair analysis of the retrieved demo at
# every rollout instead of loading the cached answer, so each rollout's
# guidance is analysed afresh (a few seconds of VLM per rollout).
CACHE_PAIRS="${CACHE_PAIRS:-0}"
# Where the training demos grasped the object (robot base frame), from
# scripts/experiments/plot_training_on_grid.py --dataset <demos>; the window
# overlays their density on the live table (H). Missing file: no overlay.
# Episode recording is OFF for the demo: encoding two camera streams per
# step into the .vla costs time inside the control loop, and the demo shows
# rather than measures. RECORD=1 records every rollout as the experiments do.
RECORD="${RECORD:-0}"
TRAINING_POSITIONS="${TRAINING_POSITIONS:-data/training_positions/put_all_cables_in_the_bag_grasp_positions.json}"
# Sub-trajectory segmentation, retuned for this cell exactly as in
# scripts/run_ur5_rollout.sh: the threshold is metres per *step*, and at 50 Hz
# the shared default (0.005) sits above almost everything a real demo does.
# See docs/ur5.md § Sub-trajectory segmentation for the measurement.
SUBTRAJ_SPEED_THRESHOLD="${SUBTRAJ_SPEED_THRESHOLD:-0.001}"
SUBTRAJ_MIN_LENGTH="${SUBTRAJ_MIN_LENGTH:-50}"
SDK_PATH="${SDK_PATH:-$HOME/dev/physical-ai-sdk}"
RUN_NAME="${RUN_NAME:-ur5_demo}"
# Where the guided demo retrieves its demo from: a folder of encoded .vla
# rollouts, e.g. data/reference_dataset_cable. Space-separate several.
# Required unless --unguided: without it evaluate.py's defaults fall through
# to the LIBERO simulator datasets, which are not on this host.
RETRIEVAL_DIR="${RETRIEVAL_DIR:-}"
RETRIEVAL_EXAMPLE="${RETRIEVAL_EXAMPLE:-data/reference_dataset_cable}"
# Where a Rerun viewer is expected. One source of truth: the pre-flight probe
# and the run itself both use it.
VIEWER_ADDR="${VIEWER_ADDR:-127.0.0.1:9876}"   # gRPC: served by the demo, or an external viewer
WEB_PORT="${WEB_PORT:-9090}"                    # the served web viewer the window embeds

# --- ARGS -------------------------------------------------------------------

LAUNCHER="${BASH_SOURCE[0]}"

guided=1
start_guided=0
simulate=0
dry_run=0
fake_policy=""
viz=1
spawn_viewer=0
external_viewer=0
layout=scene
demo_name=""
max_steps=""
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --guided)         guided=1 ;;
        --unguided)       guided=0 ;;
        --start-guided)   start_guided=1 ;;
        --simulate)       simulate=1 ;;
        --dry-run)        dry_run=1 ;;
        # Optional kind: `--fake-policy circular`. Bare, it means noop.
        --fake-policy)    fake_policy=noop
                          case "${2:-}" in noop|circular) fake_policy="$2"; shift ;; esac ;;
        --viz)            viz=1 ;;
        --no-viz)         viz=0 ;;
        --spawn-viewer)   viz=1; spawn_viewer=1 ;;
        --external-viewer) viz=1; external_viewer=1 ;;
        --full-viewer)    viz=1; layout=full ;;
        --name)           demo_name="$2"; shift ;;
        --max-steps)      max_steps="$2"; shift ;;
        --)               shift; extra=("$@"); break ;;
        # Print the header comment block, minus the shebang, as usage.
        -h|--help)        awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                              "$LAUNCHER"; exit 0 ;;
        *) echo "unknown option: $1 (pass demo_ur5.py flags after --)" >&2; exit 2 ;;
    esac
    shift
done

# A fake policy's actions are not joint targets for this arm, so never send
# them. --simulate is the stronger form of the same thing (no arm at all) and
# is left alone.
if [[ -n "$fake_policy" && $simulate -eq 0 ]]; then
    dry_run=1
fi

# Flags after `--` are demo_ur5.py's own, and tyro accepts each only on its
# own side of the subcommands: --env.* after `env:ur5`, --policy.* after
# `policy:lerobot`, everything else before both. Appending them all at the
# end -- past both subcommands -- makes a root-level flag like
# --demo.refresh-hz look unrecognised, so sort them by prefix. A flag's
# values follow it until the next `--token`, and ride with it.
extra_root=()
extra_env=()
extra_policy=()
bucket=root
for tok in ${extra[@]+"${extra[@]}"}; do
    case "$tok" in
        --env.*)    bucket=env ;;
        --policy.*) bucket=policy ;;
        --*)        bucket=root ;;
    esac
    case "$bucket" in
        env)    extra_env+=("$tok") ;;
        policy) extra_policy+=("$tok") ;;
        *)      extra_root+=("$tok") ;;
    esac
done

# --- INTERPRETER ------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$LAUNCHER")/.." && pwd)"
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

DEMO=scripts/demo_ur5.py

# --- ROBOT ADDRESS ----------------------------------------------------------

if [[ -z "$ROBOT_IP" ]]; then
    ROBOT_IP="$("$PYTHON" -c \
        'from policy_abstraction.envs.ur5.config import DEFAULT_ROBOT_IP; print(DEFAULT_ROBOT_IP)')"
fi
[[ -n "$ROBOT_IP" ]] || { echo "could not resolve ROBOT_IP" >&2; exit 1; }

# --- PRE-FLIGHT -------------------------------------------------------------

if [[ $simulate -eq 0 ]] && command -v nc >/dev/null; then
    reply="$(printf 'robotmode\n' | nc -w 2 "$ROBOT_IP" 29999 2>/dev/null | tr -d '\r' | tail -1 || true)"
    if [[ "$reply" != *RUNNING* ]]; then
        echo "WARNING: $ROBOT_IP dashboard says '${reply:-<no reply>}', expected 'Robotmode: RUNNING'." >&2
        echo "         Check the arm is powered, braked off and in remote control." >&2
    fi
fi

# Visualisation is on by default, so check the viewer is actually there. A
# connection to a loopback address either completes or is refused immediately,
# so this costs nothing and cannot hang.
if [[ $viz -eq 1 && $external_viewer -eq 1 ]]; then
    if ! (exec 3<>"/dev/tcp/${VIEWER_ADDR%%:*}/${VIEWER_ADDR##*:}") 2>/dev/null; then
        echo "No Rerun viewer at $VIEWER_ADDR -- using the in-window viewer instead." >&2
        echo "  (start one here:   $REPO_ROOT/.venv/bin/rerun --port ${VIEWER_ADDR##*:}" >&2
        echo "   or forward yours: ssh -R ${VIEWER_ADDR##*:}:localhost:${VIEWER_ADDR##*:} $(hostname))" >&2
        external_viewer=0
    fi
fi

for serial in "$EXTERIOR_SERIAL" "$WRIST_SERIAL"; do
    if command -v rs-enumerate-devices >/dev/null; then
        rs-enumerate-devices -s 2>/dev/null | grep -q "$serial" \
            || echo "WARNING: RealSense $serial not enumerated -- check the USB connection." >&2
    fi
done

# The switch steers from a retrieved demo, so it needs somewhere to retrieve
# from. Checked here, before the servers, because the fallback is silent: the
# LIBERO defaults are not on this host and the run would only find out after
# the checkpoint load.
if [[ $guided -eq 1 ]]; then
    if [[ -z "$RETRIEVAL_DIR" ]]; then
        echo "ERROR: the guided demo needs RETRIEVAL_DIR, the folder of encoded demo" >&2
        echo "       rollouts to retrieve from, e.g." >&2
        echo "       RETRIEVAL_DIR=$RETRIEVAL_EXAMPLE $LAUNCHER" >&2
        echo "       (encode a folder of .vla files with scripts/encode_datasets.py;" >&2
        echo "       or pass --unguided for a streaming-only demo)" >&2
        exit 2
    fi
    for dir in $RETRIEVAL_DIR; do
        if [[ ! -d "$dir" ]]; then
            echo "ERROR: RETRIEVAL_DIR entry '$dir' is not a directory." >&2
            exit 2
        fi
    done
    echo "guided: retrieving demos from $RETRIEVAL_DIR" >&2
fi

if [[ ! -f "$CALIBRATION" ]]; then
    echo "WARNING: no calibration at $CALIBRATION." >&2
    echo "         The demo still runs, but the pairs cannot be drawn on the" >&2
    echo "         live view. See docs/ur5.md, Camera calibration." >&2
elif [[ -z "$CALIBRATION_BASE_FRAME" ]] && ! grep -q '"base_frame"' "$CALIBRATION"; then
    # The server refuses this file rather than guessing, and it refuses it
    # *after* the checkpoint has loaded -- so say it here, where the fix costs
    # seconds instead of three minutes.
    echo "ERROR: $CALIBRATION does not declare a base frame." >&2
    echo "       Add '\"base_frame\": \"base_link\"' to it (the ROS/URDF" >&2
    echo "       convention Kalib produces), or set CALIBRATION_BASE_FRAME." >&2
    echo "       See docs/ur5.md, Camera calibration." >&2
    exit 1
fi

# --- COMMAND ----------------------------------------------------------------

cmd=("$PYTHON" "$DEMO"
     --instance "$INSTANCE"
     --output.run "$RUN_NAME")

[[ -n "$demo_name" ]] && cmd+=(--demo.name "$demo_name")
[[ -n "$max_steps" ]] && cmd+=(--demo.max-steps "$max_steps")
[[ $start_guided -eq 1 ]] && cmd+=(--demo.guided)
[[ "$RECORD" == "0" ]] && cmd+=(--output.no-save-episodes)
if [[ -n "$TRAINING_POSITIONS" ]]; then
    if [[ -f "$TRAINING_POSITIONS" ]]; then
        cmd+=(--demo.training-positions "$TRAINING_POSITIONS")
    else
        echo "WARNING: no training positions at $TRAINING_POSITIONS; the heat map" >&2
        echo "         overlay is off. Make one with scripts/experiments/plot_training_on_grid.py." >&2
    fi
fi

if [[ -n "$fake_policy" ]]; then
    cmd+=(--guidance.debug --guidance.fake-policy "$fake_policy")
fi

if [[ $guided -eq 1 ]]; then
    # The UR5e publishes neither depth nor ground-truth masks the way a
    # simulator does, so the guided pipeline needs the real perception stack.
    cmd+=(--perception.mode REAL_PERCEPTION)
    # Unquoted on purpose: RETRIEVAL_DIR may name several folders.
    cmd+=(--retrieval.dataset-dirs $RETRIEVAL_DIR)
    if [[ "$CACHE_LIVE_OBJECTS" == "0" ]]; then
        cmd+=(--demo-analysis.no-cache-live-objects)
    fi
    if [[ "$CACHE_PAIRS" == "0" ]]; then
        cmd+=(--demo-analysis.no-cache-pairs)
    fi
    # Guidance reads the chunk as UR5e joint targets through the forward
    # kinematics (--policy.guidance-action-space auto resolves to
    # ur5e-joints for env:ur5).
else
    # One switch for retrieval + demo analysis + guidance. demo_ur5.py warns
    # that the guidance switch will have nothing to open.
    cmd+=(--no-guide)
fi

if [[ $viz -eq 1 ]]; then
    # env:ur5 auto-selects the menagerie UR5e model and bolts a Robotiq
    # 2F-140 onto its flange, so neither is named here. `scene` collapses
    # every panel but the 3-D view; V in the window flips to `full` and back.
    cmd+=(--viz.enable --viz.robot-description ur5e_mj_description
          --viz.layout "$layout")
    if [[ $spawn_viewer -eq 1 ]]; then
        # A separate native viewer window, launched for the run.
        cmd+=(--viz.no-serve --viz.connect None)
    elif [[ $external_viewer -eq 1 ]]; then
        # The probe above found a viewer; re-check at startup so a tunnel that
        # drops during the policy load fails in seconds, not silently.
        cmd+=(--viz.no-serve --viz.connect "$VIEWER_ADDR" --viz.require-viewer)
    else
        # Served from the demo process itself: the 3-D scene is a panel of the
        # window, and no viewer binary or tunnel is needed.
        cmd+=(--viz.serve --viz.serve-grpc-port "${VIEWER_ADDR##*:}"
              --viz.serve-web-port "$WEB_PORT")
    fi
else
    # Rerun logging is the biggest per-step cost, and the window already
    # shows an annotated live view of the same scene.
    cmd+=(--viz.no-enable)
fi

cmd+=(--subtraj.gripper-speed-threshold "$SUBTRAJ_SPEED_THRESHOLD"
      --subtraj.min-length "$SUBTRAJ_MIN_LENGTH")

# Root-level overrides, while the command is still root-level.
cmd+=(${extra_root[@]+"${extra_root[@]}"})

cmd+=(env:ur5
      --env.robot-ip "$ROBOT_IP"
      --env.tasks "$TASK"
      --env.tcp-offset $TCP_OFFSET
      --env.cameras "exterior_image_1_left=rs:${EXTERIOR_SERIAL}:${CAMERA_RES}"
                    "wrist_image_left=rs:${WRIST_SERIAL}:${CAMERA_RES}")

if [[ -f "$CALIBRATION" ]]; then
    cmd+=(--env.calibration-path "$CALIBRATION")
    # Only when explicitly overridden -- otherwise the file's own declaration
    # wins, exactly as in scripts/run_ur5_rollout.sh.
    [[ -n "$CALIBRATION_BASE_FRAME" ]] \
        && cmd+=(--env.calibration-base-frame "$CALIBRATION_BASE_FRAME")
fi

if [[ $simulate -eq 1 ]]; then
    cmd+=(--env.simulate-hardware)
elif [[ $dry_run -eq 1 ]]; then
    cmd+=(--env.dry-run)
fi

# The session parks the arm itself after every rollout, and the operator
# re-stages the scene while it is parked, so the env server must not home
# again on its own reset.
cmd+=(--env.no-home-on-reset)

cmd+=(${extra_env[@]+"${extra_env[@]}"})

# Real-arm tuning, spelled out rather than inferred, so it is visible in the
# printed command and can be overridden after `--`. The arm is *stationary*
# during inference, so replanning every 5 steps (the simulator default) would
# stall it more often than it moves; RTC then blends each new chunk into the
# unexecuted tail of the last -- which is also what makes a flip of the
# switch a blend rather than a jump. The blend horizon must cover
# replan_steps or the run refuses to start.
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
          --policy.compile-model "$COMPILE_MODEL"
          --policy.rtc
          --policy.rtc-execution-horizon "$REPLAN_STEPS")
fi

cmd+=(${extra_policy[@]+"${extra_policy[@]}"})

printf '%q ' "${cmd[@]}"; echo
exec "${cmd[@]}"
