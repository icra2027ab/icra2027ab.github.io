# Shared body of the real-cell experiment launchers. Sourced, never run.
#
# One physical cell, one pre-flight, one way of talking to collect_experiment.py
# — so an experiment launcher is a config path, a default prompt and a help
# comment, and the parts that are about the *hardware* live here and are fixed
# in one place when the hardware changes.
#
# A launcher sets three variables and sources this file with no arguments, so
# that the caller's "$@" is the one parsed below:
#
#     EXPERIMENT_DEFAULT_CONFIG=configs/experiments/<file>.yaml
#     EXPERIMENT_DEFAULT_NAME=<experiment name prefix>
#     EXPERIMENT_DEFAULT_TASK="<prompt the env server publishes>"
#     source "$(dirname "${BASH_SOURCE[0]}")/_ur5_experiment.sh"
#
# `--help` prints the launcher's own leading comment block, so each one
# documents its own experiment rather than this file's flags.

set -euo pipefail

for required in EXPERIMENT_DEFAULT_CONFIG EXPERIMENT_DEFAULT_NAME EXPERIMENT_DEFAULT_TASK; do
    if [[ -z "${!required:-}" ]]; then
        echo "internal error: $required must be set before sourcing $(basename "${BASH_SOURCE[0]}")" >&2
        exit 1
    fi
done

# --- CONFIG -------------------------------------------------

ROBOT_IP="${ROBOT_IP:-}"
TCP_OFFSET="${TCP_OFFSET:-0.00052 0.00109 0.22347 0 0 0}"

EXTERIOR_SERIAL="${EXTERIOR_SERIAL:-262722073432}"   # scene view of the table (D435I fitted 2026-09-22)
WRIST_SERIAL="${WRIST_SERIAL:-115222071166}"         # wrist-mounted
CAMERA_RES="${CAMERA_RES:-640x480@30}"

# The staging overlay is projected through this. Without it the GUI still
# runs and cannot draw the targets, and the report skips the projected map.
# Same default as scripts/run_ur5_rollout.sh -- the checked-in file, not the
# raw hand-eye output, because that one declares which base frame it is in and
# the server refuses a file that does not.
# See docs/ur5.md § Camera calibration.
# Sub-trajectory segmentation, retuned for this cell exactly as in
# scripts/run_ur5_rollout.sh: the threshold is metres per *step*, and at 50 Hz
# the shared default (0.005) sits above almost everything a real demo does, so
# compute_subtrajectories falls back to one segment for the whole episode.
# See docs/ur5.md § Sub-trajectory segmentation for the measurement.
SUBTRAJ_SPEED_THRESHOLD="${SUBTRAJ_SPEED_THRESHOLD:-0.001}"
SUBTRAJ_MIN_LENGTH="${SUBTRAJ_MIN_LENGTH:-50}"

CALIBRATION="${CALIBRATION:-configs/ur5_camera_calibration.json}"
# Which base frame that extrinsic is expressed in. Normally empty: the
# calibration file declares its own frame, and the file is the right place for
# it -- hardcoding a frame here would silently *override* a different file that
# says otherwise. UR's two frames sit 180 degrees apart about z, and declaring
# the wrong one mirrors the scene in x and y while leaving the table perfectly
# horizontal: a plane-fit sanity check still passes, and only a comparison
# against the robot's own TCP catches it. Set this only to override a file you
# cannot edit -- a raw Kalib export, say.
CALIBRATION_BASE_FRAME="${CALIBRATION_BASE_FRAME:-}"

CHECKPOINT="${CHECKPOINT:-checkpoints/pi05_put_all_cables_in_bag_a100/pretrained_model}"
# The prompt the env server publishes. The experiment config's own `task:` is
# what actually reaches the policy — and in a templated experiment it changes
# per rollout — so this is a fallback, not the contract.
TASK="${TASK:-$EXPERIMENT_DEFAULT_TASK}"

EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$EXPERIMENT_DEFAULT_CONFIG}"
# Where --guided retrieves its demo from: a folder of encoded .vla rollouts
# (scripts/encode_datasets.py --source-path <dir>), e.g.
# data/reference_dataset_cable. Space-separate several. Required for
# --guided: without it evaluate.py falls through to the LIBERO dataset
# defaults, which are simulator demos and not present on this host, so the
# launcher refuses rather than paying the checkpoint load and then steering
# from the wrong demo. Same variable as scripts/run_ur5_rollout.sh.
RETRIEVAL_DIR="${RETRIEVAL_DIR:-}"
# The folder the --guided error message points at. A launcher whose demos
# live elsewhere sets this before sourcing.
EXPERIMENT_RETRIEVAL_EXAMPLE="${EXPERIMENT_RETRIEVAL_EXAMPLE:-data/reference_dataset_cable}"
RUN_NAME="${RUN_NAME:-ur5_experiments}"
INSTANCE="${INSTANCE:-9}"            # port pair: policy 8000+N, env 8765+N
GPUS="${GPUS:-0}"
REPLAN_STEPS="${REPLAN_STEPS:-25}"   # 25/50 Hz = 0.5 s of motion per inference
SDK_PATH="${SDK_PATH:-$HOME/dev/physical-ai-sdk}"

# --- ARGS ---------------------------------------------------------

# The launcher, not this file — `--help` prints the experiment's own comment.
LAUNCHER="${BASH_SOURCE[${#BASH_SOURCE[@]} - 1]}"

mode=""
calibrate=0
report=0
simulate=0
dry_run=0
viz=0
spawn_viewer=0
overwrite=0
headless=0
experiment_name=""
objects=""
reference=""
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline)      mode=baseline ;;
        --guided)        mode=guided ;;
        --calibrate)     calibrate=1 ;;
        --report)        report=1 ;;
        --simulate)      simulate=1 ;;
        --dry-run)       dry_run=1 ;;
        --viz)           viz=1 ;;
        --spawn-viewer)  viz=1; spawn_viewer=1 ;;
        --overwrite)     overwrite=1 ;;
        --headless)      headless=1 ;;
        --name)          experiment_name="$2"; shift ;;
        --objects)       objects="$2"; shift ;;
        --reference)     reference="$2"; shift ;;
        --)              shift; extra=("$@"); break ;;
        -h|--help)       awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                             "$LAUNCHER"; exit 0 ;;
        *) echo "unknown option: $1 (pass collect_experiment.py flags after --)" >&2
           exit 2 ;;
    esac
    shift
done

# Flags after `--` are collect_experiment.py's own, and tyro accepts each
# only on its own side of the subcommands: --env.* after `env:ur5`,
# --policy.* after `policy:lerobot`, everything else before both. Appending
# them all at the end -- past both subcommands -- makes a root-level flag
# like --experiment.overwrite look unrecognised, so sort them by prefix.
# A flag's values follow it until the next `--token`, and ride with it.
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

if [[ $calibrate -eq 0 && $report -eq 0 && -z "$mode" ]]; then
    echo "Pick a policy setting: --baseline or --guided." >&2
    echo "(or --calibrate to check the staging targets, --report to rebuild the report)" >&2
    exit 2
fi

# Neither of these collects, so neither has results to start over. Say so
# rather than accepting the flag and quietly doing nothing with it.
if [[ $overwrite -eq 1 && ( $report -eq 1 || $calibrate -eq 1 ) ]]; then
    echo "--overwrite applies to collection; --report and --calibrate" >&2
    echo "do not write results. Drop it, or pass --baseline / --guided." >&2
    exit 2
fi

# A reference experiment brings its own settings, so the two ways of
# changing them do not combine with it.
if [[ -n "$reference" && $report -eq 1 ]]; then
    echo "--reference applies to collection; --report reads what is on disk." >&2
    exit 2
fi
if [[ -n "$reference" && -n "$objects" ]]; then
    echo "--objects re-samples the layouts; --reference reuses the ones the" >&2
    echo "named experiment collected under. Pass one or the other." >&2
    exit 2
fi

# One experiment directory per policy setting: same plan, separate
# results, each resumable on its own.
if [[ -z "$experiment_name" ]]; then
    experiment_name="${EXPERIMENT_DEFAULT_NAME}_${mode:-baseline}"
fi

# ---- INTERPRETER ------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "$LAUNCHER")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        # A git worktree has no venv of its own -- borrow the main
        # checkout's interpreter and put this tree's sources first.
        main_git="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
        PYTHON="$(dirname "$main_git")/.venv/bin/python"
    fi
fi
[[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set PYTHON=...)" >&2; exit 1; }
export PYTHONPATH="src:packages/policy_steering/src${PYTHONPATH:+:$PYTHONPATH}"

COLLECT=scripts/experiments/collect_experiment.py

# --- REPORT-ONLY: no servers, no hardware ------------------------------

if [[ $report -eq 1 ]]; then
    exec "$PYTHON" "$COLLECT" \
        --experiment.report-only \
        --experiment.experiment-name "$experiment_name" \
        --output.run "$RUN_NAME" \
        ${extra[@]+"${extra[@]}"}
fi

# --- ROBOT ADDRESS -----------------------------------------------

if [[ -z "$ROBOT_IP" ]]; then
    ROBOT_IP="$("$PYTHON" -c \
        'from policy_abstraction.envs.ur5.config import DEFAULT_ROBOT_IP; print(DEFAULT_ROBOT_IP)')"
fi
[[ -n "$ROBOT_IP" ]] || { echo "could not resolve ROBOT_IP" >&2; exit 1; }

# --- PRE-FLIGHT ---------------------------------------------------

if [[ $simulate -eq 0 ]] && command -v nc >/dev/null; then
    reply="$(printf 'robotmode\n' | nc -w 2 "$ROBOT_IP" 29999 2>/dev/null | tr -d '\r' | tail -1 || true)"
    if [[ "$reply" != *RUNNING* ]]; then
        echo "WARNING: $ROBOT_IP dashboard says '${reply:-<no reply>}', expected 'Robotmode: RUNNING'." >&2
        echo "         Check the arm is powered, braked off and in remote control." >&2
    fi
fi

for serial in "$EXTERIOR_SERIAL" "$WRIST_SERIAL"; do
    if command -v rs-enumerate-devices >/dev/null; then
        rs-enumerate-devices -s 2>/dev/null | grep -q "$serial" \
            || echo "WARNING: RealSense $serial not enumerated -- check the USB connection." >&2
    fi
done

if [[ -z "$reference" && ! -f "$EXPERIMENT_CONFIG" ]]; then
    echo "no experiment config at $EXPERIMENT_CONFIG" >&2
    exit 1
fi

# The guided arm steers from a retrieved demo, so it needs somewhere to
# retrieve from. Checked here, before the servers, because the fallback is
# silent: the LIBERO defaults are not on this host and the run would only
# find out after the checkpoint load.
if [[ "$mode" == "guided" && $calibrate -eq 0 ]]; then
    if [[ -z "$RETRIEVAL_DIR" ]]; then
        echo "ERROR: --guided needs RETRIEVAL_DIR, the folder of encoded demo" >&2
        echo "       rollouts to retrieve from, e.g." >&2
        echo "       RETRIEVAL_DIR=$EXPERIMENT_RETRIEVAL_EXAMPLE $LAUNCHER --guided" >&2
        echo "       (encode a folder of .vla files with scripts/encode_datasets.py)" >&2
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
    echo "         Rollouts still record fine, but the staging targets cannot be" >&2
    echo "         drawn on the live view. See docs/ur5.md, Camera calibration." >&2
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

# ---- COMMAND -------------------------------------------------------

cmd=("$PYTHON" "$COLLECT"
     --experiment.experiment-name "$experiment_name"
     --instance "$INSTANCE"
     --output.run "$RUN_NAME")

if [[ -n "$reference" ]]; then
    # The settings and the plan come from that experiment's directory, not
    # from the YAML; collect_experiment.py checks the two agree.
    cmd+=(--experiment.reference-experiment "$reference")
else
    cmd+=(--experiment.experiment-config "$EXPERIMENT_CONFIG")
fi

if [[ $calibrate -eq 1 ]]; then
    cmd+=(--experiment.calibrate)
else
    cmd+=(--experiment.policy-setting "$mode")
fi

if [[ -n "$objects" ]]; then
    cmd+=(--experiment.objects "$objects")
fi

if [[ $overwrite -eq 1 ]]; then
    cmd+=(--experiment.overwrite)
fi

if [[ $headless -eq 1 ]]; then
    cmd+=(--experiment.headless)
fi

if [[ "$mode" == "guided" ]]; then
    # The UR5e publishes neither depth nor ground-truth masks the way a
    # simulator does, so the guided pipeline needs the real perception stack.
    cmd+=(--perception.mode REAL_PERCEPTION)
    # Unquoted on purpose: RETRIEVAL_DIR may name several folders.
    [[ -n "$RETRIEVAL_DIR" ]] && cmd+=(--retrieval.dataset-dirs $RETRIEVAL_DIR)
    # Guidance reads the chunk as UR5e joint targets through the forward
    # kinematics (--policy.guidance-action-space auto resolves to
    # ur5e-joints for env:ur5); collect_experiment.py refuses a "guided"
    # setting whose guidance, retrieval or demo analysis is switched off.
else
    # One switch for retrieval + demo analysis + guidance.
    cmd+=(--no-guide)
fi

if [[ $viz -eq 1 ]]; then
    # env:ur5 auto-selects the menagerie UR5e model and bolts a Robotiq
    # 2F-140 onto its flange, so neither is named here.
    cmd+=(--viz.enable --viz.robot-description ur5e_mj_description)
    if [[ $spawn_viewer -eq 1 ]]; then
        cmd+=(--viz.connect None)
    else
        cmd+=(--viz.require-viewer)
    fi
else
    # Rerun logging is the biggest per-step cost, and the GUI already
    # gives the operator an annotated live view of the same scene.
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

# The session parks the arm itself between rollouts, and the operator
# stages the next scene while it is parked, so the env server must
# not home again on its own reset.
cmd+=(--env.no-home-on-reset)

cmd+=(${extra_env[@]+"${extra_env[@]}"})

# Real-arm tuning, spelled out rather than inferred, so it is visible
# in the printed command and can be overridden after `--`. The arm is
# *stationary* during inference, so replanning every 5 steps (the
# simulator default) would stall it more often than it moves; RTC then
# blends each new chunk into the unexecuted tail of the last. The
# blend horizon must cover replan_steps or the run refuses to start.
cmd+=(policy:lerobot
      --policy.model-path "$CHECKPOINT"
      --policy.gpus "$GPUS"
      --policy.repo-path "$SDK_PATH"
      --policy.replan-steps "$REPLAN_STEPS"
      --policy.rtc
      --policy.rtc-execution-horizon "$REPLAN_STEPS")

cmd+=(${extra_policy[@]+"${extra_policy[@]}"})

printf '%q ' "${cmd[@]}"; echo
exec "${cmd[@]}"
