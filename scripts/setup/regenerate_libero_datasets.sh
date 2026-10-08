#!/usr/bin/env bash
# Regenerate LIBERO datasets at higher resolution (with depth + segmentation).
# Run from anywhere; cd's to repo root automatically.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# Activate the LIBERO-PRO Python 3.8 venv (created per docs/libero_pro_setup.md §3).
source third_party/LIBERO-PRO/.venv/bin/activate

export MUJOCO_GL=glx
export PYTHONPATH="$PWD/third_party/LIBERO-PRO:$PWD/src"

for suite in libero_object libero_spatial libero_goal libero_10; do
  python src/policy_abstraction/data/regenerator.py \
    --libero_task_suite "$suite" \
    --libero_raw_data_dir "datasets/$suite" \
    --libero_target_dir "datasets/${suite}_new" --y
done
