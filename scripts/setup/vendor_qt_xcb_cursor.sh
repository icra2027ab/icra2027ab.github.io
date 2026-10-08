#!/usr/bin/env bash
# Put libxcb-cursor.so.0 inside the venv, for boxes without root.
#
# Qt 6.5+ needs libxcb-cursor0 to load its `xcb` platform plugin, and
# the PyQt6 wheel does not bundle it. Without it the GUI does not
# raise — Qt calls qFatal and the process aborts:
#
#   qt.qpa.plugin: From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed
#   to load the Qt xcb platform plugin.
#
# The clean fix is `sudo apt-get install -y libxcb-cursor0`. This
# script is for when you do not have root: it fetches the same 10 kB
# package and drops the one .so into the venv's Qt lib directory,
# which the plugin already searches (its RUNPATH is
# $ORIGIN/../../lib). No env var, no LD_LIBRARY_PATH, no root.
#
# Only that one library is vendored. Its four dependencies —
# libxcb-image0, libxcb-render-util0, libxcb-render0, libxcb1 — ship
# with any desktop-capable image and are checked below rather than
# assumed.
#
# Re-run after rebuilding the venv; it is idempotent.
#
#   scripts/setup/vendor_qt_xcb_cursor.sh
#   scripts/setup/vendor_qt_xcb_cursor.sh --check   # report only

set -euo pipefail

check_only=0
[[ "${1:-}" == "--check" ]] && check_only=1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
[[ -x "$PYTHON" ]] || { echo "no interpreter at $PYTHON (set PYTHON=...)" >&2; exit 1; }

QT_LIB="$("$PYTHON" -c 'import PyQt6, pathlib; print(pathlib.Path(PyQt6.__file__).parent / "Qt6" / "lib")' 2>/dev/null || true)"
if [[ -z "$QT_LIB" || ! -d "$QT_LIB" ]]; then
    echo "PyQt6 is not installed in $PYTHON, so there is no Qt to fix up." >&2
    exit 1
fi
PLUGIN="$(dirname "$QT_LIB")/plugins/platforms/libqxcb.so"

report() {
    if [[ -e "$QT_LIB/libxcb-cursor.so.0" ]]; then
        echo "vendored: $QT_LIB/libxcb-cursor.so.0"
    elif ldconfig -p 2>/dev/null | grep -q libxcb-cursor.so.0; then
        echo "system libxcb-cursor0 is installed; nothing to vendor."
    else
        echo "libxcb-cursor.so.0 is MISSING."
    fi
    if [[ -e "$PLUGIN" ]]; then
        local missing
        missing="$(ldd "$PLUGIN" 2>/dev/null | awk '/not found/ {print $1}' | sort -u | tr '\n' ' ')"
        if [[ -n "$missing" ]]; then
            echo "libqxcb.so still cannot resolve: $missing"
            return 1
        fi
        echo "libqxcb.so resolves every dependency."
    fi
    return 0
}

if [[ $check_only -eq 1 ]]; then
    report
    exit $?
fi

if [[ -e "$QT_LIB/libxcb-cursor.so.0" ]]; then
    echo "Already vendored at $QT_LIB/libxcb-cursor.so.0"
    report
    exit 0
fi

if ldconfig -p 2>/dev/null | grep -q libxcb-cursor.so.0; then
    echo "System libxcb-cursor0 is already installed; nothing to do."
    exit 0
fi

# The four libraries libxcb-cursor itself links against. They are on any
# desktop-capable image, but say so plainly rather than failing later
# inside Qt with a less obvious message.
missing_deps=()
for lib in libxcb-image.so.0 libxcb-render-util.so.0 libxcb-render.so.0 libxcb.so.1; do
    ldconfig -p 2>/dev/null | grep -q "$lib" || missing_deps+=("$lib")
done
if [[ ${#missing_deps[@]} -gt 0 ]]; then
    echo "WARNING: libxcb-cursor also needs: ${missing_deps[*]}" >&2
    echo "         Vendoring anyway, but the plugin may still not load." >&2
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "Fetching libxcb-cursor0 …"
( cd "$WORK" && apt-get download libxcb-cursor0 ) >/dev/null 2>&1 || {
    echo "apt-get download failed. Without network access, copy" >&2
    echo "libxcb-cursor.so.0 from any machine that has it into" >&2
    echo "  $QT_LIB/" >&2
    exit 1
}

deb="$(find "$WORK" -name 'libxcb-cursor0_*.deb' -print -quit)"
[[ -n "$deb" ]] || { echo "no .deb was downloaded" >&2; exit 1; }
dpkg-deb -x "$deb" "$WORK/x"

so="$(find "$WORK/x" -name 'libxcb-cursor.so.0.*' -print -quit)"
[[ -n "$so" ]] || { echo "no libxcb-cursor.so.0.* inside $deb" >&2; exit 1; }

cp "$so" "$QT_LIB/"
ln -sf "$(basename "$so")" "$QT_LIB/libxcb-cursor.so.0"
echo "Vendored $(basename "$so") into $QT_LIB"

report
