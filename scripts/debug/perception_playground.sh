#!/usr/bin/env bash
# Interactive real-perception playground on the real UR5e cell.
#
# Starts the env server, runs the playground against it, and — this is the
# point of the wrapper — tears the server down on the way out so the RealSense
# is released. A backgrounded server left holding a camera is what makes the
# next run fail with "Device or resource busy", usually long after you've
# forgotten it is there.
#
# The arm is never commanded. The server runs with --dry-run: it reads joints,
# TCP and camera frames, and that is all. Safe to leave running while you
# rearrange the table.
#
# Usage:
#   scripts/debug/perception_playground.sh                    # cable + bag
#   scripts/debug/perception_playground.sh cable bag mug      # your own queries
#   scripts/debug/perception_playground.sh --wrist            # add the wrist camera
#   scripts/debug/perception_playground.sh --no-server        # attach to a server you started
#   scripts/debug/perception_playground.sh --simulate         # arm off, real cameras
#   scripts/debug/perception_playground.sh -- --max-frames 40 --save-rrd out.rrd
#
# Once it is up, type at the prompt: `cable, bag` replaces the query set,
# `+ mug` adds one, `!redetect` re-detects, `!settings` prints the config,
# `!quit` exits. Changing the queries resets the tracker, so the next frame
# runs full detection exactly as a replan does in deployment.
#
# Visualisation goes to a Rerun viewer on 127.0.0.1:9876, same as
# run_ur5_rollout.sh --viz. Start one from the main checkout's venv (the
# `rerun` binary is not on PATH):
#     .venv/bin/rerun --port 9876
# With nothing listening the run still works, it just logs into the void; pass
# `-- --connect None --save-rrd out.rrd` to record instead.
#
# Anything in CONFIG can be overridden from the environment:
#   EXTERIOR_SERIAL=207222070875 scripts/debug/perception_playground.sh cable

set -euo pipefail

# --- CONFIG -----------------------------------------------------------------

# Address comes from envs/ur5/config.DEFAULT_ROBOT_IP (honouring UR5_ROBOT_IP)
# so there is exactly one place it is written down. Resolved after $PYTHON.
ROBOT_IP="${ROBOT_IP:-}"

# Serials, not /dev/videoN: node numbers are handed out at enumeration and
# shuffle on re-plug or reboot. The serial is stamped on the camera.
EXTERIOR_SERIAL="${EXTERIOR_SERIAL:-262722073432}"   # scene view, the calibrated one (D435I fitted 2026-09-22)
WRIST_SERIAL="${WRIST_SERIAL:-115222071166}"         # wrist-mounted
CAMERA_RES="${CAMERA_RES:-640x480@30}"

# Only the exterior camera is hand-eye calibrated. A camera without an
# extrinsic is dropped by the perception pipeline, so adding the wrist buys
# nothing until it is calibrated too -- hence --wrist being opt-in.
CALIBRATION="${CALIBRATION:-configs/ur5_camera_calibration.json}"
CALIBRATION_FRAME="${CALIBRATION_FRAME:-base_link}"

PORT="${PORT:-8765}"
QUERIES_DEFAULT=("cable" "bag")
SDK_PATH="${SDK_PATH:-$HOME/dev/physical-ai-sdk}"
LOG_DIR="${LOG_DIR:-tmp/playground_logs}"

# --- ARGS -------------------------------------------------------------------

wrist=0
start_server=1
simulate=0
queries=()
extra=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --wrist)       wrist=1 ;;
        --no-server)   start_server=0 ;;
        --simulate)    simulate=1 ;;
        --)            shift; extra=("$@"); break ;;
        -h|--help)     awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' \
                           "${BASH_SOURCE[0]}"; exit 0 ;;
        -*)            echo "unknown option: $1 (playground flags go after --)" >&2; exit 2 ;;
        *)             queries+=("$1") ;;
    esac
    shift
done
[[ ${#queries[@]} -eq 0 ]] && queries=("${QUERIES_DEFAULT[@]}")

# --- INTERPRETERS -----------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        # A git worktree has no venv of its own -- borrow the main checkout's.
        main_git="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
        PYTHON="$(dirname "$main_git")/.venv/bin/python"
    fi
fi
[[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set PYTHON=...)" >&2; exit 1; }

# The env server needs the SDK venv (pyrealsense2, rtde, lerobot); the
# playground needs the eval venv (open_perception, rerun). Two interpreters,
# deliberately -- see docs/ur5.md.
SDK_PYTHON="${SDK_PYTHON:-$SDK_PATH/.venv/bin/python}"
if [[ $start_server -eq 1 && ! -x "$SDK_PYTHON" ]]; then
    echo "no physical-ai-sdk interpreter at $SDK_PYTHON" >&2
    echo "set SDK_PATH=... or SDK_PYTHON=..., or use --no-server" >&2
    exit 1
fi
export PYTHONPATH="src:packages/policy_steering/src${PYTHONPATH:+:$PYTHONPATH}"

# One source of truth for the arm's address: envs/ur5/config.DEFAULT_ROBOT_IP,
# which honours $UR5_ROBOT_IP. $ROBOT_IP still wins if you set it directly.
if [[ -z "$ROBOT_IP" ]]; then
    ROBOT_IP="$("$PYTHON" -c \
        'from policy_abstraction.envs.ur5.config import DEFAULT_ROBOT_IP; print(DEFAULT_ROBOT_IP)')"
fi
[[ -n "$ROBOT_IP" ]] || { echo "could not resolve ROBOT_IP" >&2; exit 1; }

# --- PRE-FLIGHT -------------------------------------------------------------

cams=("$EXTERIOR_SERIAL")
[[ $wrist -eq 1 ]] && cams+=("$WRIST_SERIAL")

if [[ $start_server -eq 1 ]]; then
    for serial in "${cams[@]}"; do
        if command -v rs-enumerate-devices >/dev/null; then
            rs-enumerate-devices -s 2>/dev/null | grep -q "$serial" \
                || echo "WARNING: RealSense $serial not enumerated -- check USB." >&2
        fi
    done

    # Name the process holding a camera rather than letting the server fail
    # with a bare "Device or resource busy" several seconds later. lsof asks
    # the kernel who owns the device node, which finds holders whose command
    # name you'd never think to grep for.
    if command -v lsof >/dev/null; then
        holders="$(lsof -t /dev/video* 2>/dev/null | sort -u || true)"
        if [[ -n "$holders" ]]; then
            echo "NOTE: these processes currently hold a video device:" >&2
            # shellcheck disable=SC2086
            ps -o pid,etime,cmd --no-headers -p $(echo "$holders" | tr '\n' ',' | sed 's/,$//') >&2 || true
            echo "      If startup fails with 'Device or resource busy', stop the one" >&2
            echo "      holding ${cams[*]} and retry." >&2
        fi
    fi
fi

# --- SERVER -----------------------------------------------------------------

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/env_server.log"
server_pid=""

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        echo
        echo "stopping env server ($server_pid) -- releasing the camera"
        kill "$server_pid" 2>/dev/null || true
        # SIGTERM lets the server close the RealSense pipeline; a hard kill
        # leaves the device wedged for the next run.
        for _ in $(seq 20); do
            kill -0 "$server_pid" 2>/dev/null || break
            sleep 0.25
        done
        kill -9 "$server_pid" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

if [[ $start_server -eq 1 ]]; then
    server_cmd=("$SDK_PYTHON" scripts/servers/serve_ur5_env.py
                --robot-ip "$ROBOT_IP"
                --tasks "perception playground"
                --primary-camera exterior_image_1_left
                --port "$PORT")

    cam_args=("exterior_image_1_left=rs:${EXTERIOR_SERIAL}:${CAMERA_RES}")
    if [[ $wrist -eq 1 ]]; then
        cam_args+=("wrist_image_left=rs:${WRIST_SERIAL}:${CAMERA_RES}")
    else
        server_cmd+=(--wrist-camera '')
    fi
    server_cmd+=(--cameras "${cam_args[@]}")

    [[ -f "$CALIBRATION" ]] \
        && server_cmd+=(--calibration-path "$CALIBRATION"
                        --calibration-base-frame "$CALIBRATION_FRAME") \
        || echo "WARNING: no calibration at $CALIBRATION -- without an extrinsic" \
                "nothing can be reconstructed." >&2

    if [[ $simulate -eq 1 ]]; then
        server_cmd+=(--simulate-hardware)
    else
        server_cmd+=(--dry-run)   # reads sensors, never commands motion
    fi

    echo "starting env server -> $SERVER_LOG"
    "${server_cmd[@]}" > "$SERVER_LOG" 2>&1 &
    server_pid=$!

    for _ in $(seq 60); do
        grep -q "listening on" "$SERVER_LOG" 2>/dev/null && break
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "env server exited during startup:" >&2
            tail -20 "$SERVER_LOG" >&2
            exit 1
        fi
        sleep 0.5
    done
    grep -q "listening on" "$SERVER_LOG" 2>/dev/null || {
        echo "env server did not come up within 30s:" >&2
        tail -20 "$SERVER_LOG" >&2
        exit 1
    }
    echo "env server ready (pid $server_pid)"
fi

# --- PLAYGROUND -------------------------------------------------------------

# Foreground, not exec: the playground reads stdin interactively, and exec
# would replace this shell and skip the cleanup trap that frees the camera.
cmd=("$PYTHON" scripts/debug/perception_playground.py
     --queries "${queries[@]}"
     --env-port "$PORT"
     "${extra[@]+"${extra[@]}"}")

printf '%q ' "${cmd[@]}"; echo
"${cmd[@]}"
