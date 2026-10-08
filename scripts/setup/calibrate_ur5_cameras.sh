#!/usr/bin/env bash
# Calibrate a UR5e camera against the robot, in one window.
#
#   scripts/setup/calibrate_ur5_cameras.sh                  # exterior camera
#   scripts/setup/calibrate_ur5_cameras.sh --camera wrist   # wrist camera
#   scripts/setup/calibrate_ur5_cameras.sh --simulate       # no hardware, no GPU
#   scripts/setup/calibrate_ur5_cameras.sh --help
#
# The tool itself is scripts/setup/calibrate_ur5_cameras.py; this wrapper only
# picks the interpreters. Capture and the GUI run in the physical-ai-sdk venv
# (pyrealsense2, ur_rtde, PyQt6); Kalib's tracker runs in its own conda env
# as a subprocess. Both locations can be overridden:
#
#   SDK_PATH      physical-ai-sdk checkout   (default ~/dev/physical-ai-sdk)
#   PYTHON        interpreter for the GUI    (default $SDK_PATH/.venv/bin/python)
#   KALIB_ROOT    Kalib checkout             (default ~/dev/Kalib)
#   KALIB_PYTHON  Kalib's interpreter        (default ~/miniconda3/envs/kalib/bin/python)
#   KALIB_DEVICE  CUDA device for tracking   (default 0)
#
# The arm is never commanded: RTDE is opened receive-only, and the operator
# moves the arm in freedrive.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SDK_PATH="${SDK_PATH:-$HOME/dev/physical-ai-sdk}"
PYTHON="${PYTHON:-$SDK_PATH/.venv/bin/python}"
export KALIB_ROOT="${KALIB_ROOT:-$HOME/dev/Kalib}"
export KALIB_PYTHON="${KALIB_PYTHON:-$HOME/miniconda3/envs/kalib/bin/python}"
export KALIB_DEVICE="${KALIB_DEVICE:-0}"

[[ -x "$PYTHON" ]] || {
    echo "no interpreter at $PYTHON (set PYTHON=... or SDK_PATH=...)" >&2
    exit 1
}

simulate=0
for a in "$@"; do [[ "$a" == "--simulate" ]] && simulate=1; done

# Fail on a missing package here, with the install line, rather than three
# screens into a traceback.
missing="$("$PYTHON" - <<'PY'
import importlib
need = {"PyQt6": "PyQt6", "numpy": "numpy", "cv2": "opencv-python", "loguru": "loguru", "tyro": "tyro"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)"
if [[ $simulate -eq 0 ]]; then
    hw_missing="$("$PYTHON" - <<'PY'
import importlib
need = {"pyrealsense2": "pyrealsense2", "rtde_receive": "ur-rtde"}
print(" ".join(pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None))
PY
)"
    missing="$missing $hw_missing"
fi
missing="$(echo "$missing" | xargs || true)"
if [[ -n "$missing" ]]; then
    echo "The interpreter $PYTHON is missing: $missing" >&2
    echo "Install with:  uv pip install --python $(dirname "$(dirname "$PYTHON")") $missing" >&2
    echo "(PyQt6 also needs libxcb-cursor0: scripts/setup/vendor_qt_xcb_cursor.sh with PYTHON=$PYTHON)" >&2
    exit 1
fi

if [[ $simulate -eq 0 ]]; then
    [[ -f "$KALIB_ROOT/easycalib_demo.py" ]] || {
        echo "Kalib checkout not found at $KALIB_ROOT (set KALIB_ROOT=...)" >&2
        exit 1
    }
    [[ -x "$KALIB_PYTHON" ]] || {
        echo "Kalib interpreter not found at $KALIB_PYTHON (set KALIB_PYTHON=...)" >&2
        exit 1
    }
fi

exec "$PYTHON" "$REPO_ROOT/scripts/setup/calibrate_ur5_cameras.py" "$@"
