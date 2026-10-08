#!/usr/bin/env bash
# Positional distribution shift on the real UR5e cell: the cable starts
# at each cell of a grid, several times, once per policy setting.
#
# Same cell, same camera addressing and the same pre-flight as
# scripts/run_ur5_rollout.sh — this is that script with an experiment
# plan and an operator GUI on top. See docs/real_experiments.md for
# what the GUI does, and docs/ur5.md for the cell itself.
#
# Usage:
#
#   scripts/experiments/run_cable_position_experiment.sh --calibrate
#       Project the grid onto one camera frame, write a PNG, print every
#       cell centre in base coordinates, exit. The arm is never
#       commanded. Repeat until the overlay lines up with the table,
#       adjusting the `grid:` block in the experiment config
#       between runs.
#
#   scripts/experiments/run_cable_position_experiment.sh --baseline
#       Collect the unguided arm of the comparison.
#
#   scripts/experiments/run_cable_position_experiment.sh --guided
#       Collect the guided arm, over the identical plan.
#
#   scripts/experiments/run_cable_position_experiment.sh --report
#       Rebuild the report from what is on disk. No servers, no robot.
#
#   scripts/experiments/run_cable_position_experiment.sh --simulate --baseline
#       Arm powered off, synthetic observations -- exercises the
#       GUI, the store and the report without touching the robot.
#
# Interrupted? Re-run the same command. The plan is frozen on disk,
# so collection resumes at the next rollout with no result.
#
#   scripts/experiments/run_cable_position_experiment.sh --baseline --overwrite
#       Start this experiment over. Needed after editing the grid: the
#       plan on disk no longer matches the config, and resuming would
#       mix conditions. The old results are renamed aside, not deleted.
#
# The grid, the repeats and the per-rollout limits can also be changed
# from the GUI itself -- Ctrl+E, between rollouts -- without restarting
# and paying the checkpoint load again. If the edit would orphan
# rollouts already collected, it asks whether to start a new experiment
# under a new name or to delete them, rather than choosing for you.
#
# Those edits are stored with the results, so re-running the same
# command resumes under them. Editing the config file instead overrides
# them; --overwrite discards them and starts again from the file. If the
# edit renamed the experiment, the closing summary prints the --name to
# resume it with.
#
# Overridable from the environment (same names as run_ur5_rollout.sh):
#   CHECKPOINT TASK RUN_NAME EXPERIMENT_CONFIG INSTANCE GPUS RETRIEVAL_DIR
#   EXTERIOR_SERIAL WRIST_SERIAL UR5_ROBOT_IP CALIBRATION SDK_PATH
#
# --guided requires RETRIEVAL_DIR, the folder of encoded demo rollouts the
# guidance retrieves from (the launcher refuses to start without it):
#   RETRIEVAL_DIR=data/reference_dataset_cable run_cable_position_experiment.sh --guided
#
# Extra collect_experiment.py flags go after `--`:
#   scripts/experiments/run_cable_position_experiment.sh --baseline -- \
#       --experiment.headless
#
# They are routed to the right side of the tyro subcommands by prefix
# (`--env.*` after `env:ur5`, `--policy.*` after `policy:lerobot`,
# everything else before both), so they can be given in any order.

# What this experiment is; everything about the *cell* is in the shared body.
EXPERIMENT_DEFAULT_CONFIG=configs/experiments/cable_in_box_position.yaml
EXPERIMENT_DEFAULT_NAME=cable_position
# Must be the prompt the checkpoint was trained on, verbatim.
EXPERIMENT_DEFAULT_TASK="Put all cables in the bag."

source "$(dirname "${BASH_SOURCE[0]}")/_ur5_experiment.sh"
