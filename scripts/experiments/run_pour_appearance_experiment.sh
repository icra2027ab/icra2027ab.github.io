#!/usr/bin/env bash
# Appearance shift on the real UR5e cell: "pour contents of blue cup into
# green cup", poured from several different cups a few times each.
#
# The checkpoint was trained on one blue cup. Here the operator swaps in
# a different cup of the same type per condition -- other colours, heights,
# materials -- and the prompt names the cups by where they stand ("the
# left cup", "the right cup") so it stays true whichever cup is on the
# mark. The success rate per cup is how far the policy generalises past
# the object it was shown, and the guided arm asks whether guidance from
# a demo closes that gap.
#
# Same cell, same camera addressing, same pre-flight and the same
# operator GUI as run_cable_position_experiment.sh, with the pouring
# checkpoint swapped in. See docs/real_experiments.md for the GUI,
# docs/ur5.md for the cell.
#
# Usage:
#
#   scripts/experiments/run_pour_appearance_experiment.sh --calibrate
#       Project the two staging marks (source cup, receiving cup) onto a
#       camera frame, write a PNG, print both positions in base
#       coordinates, exit. The arm is never commanded. Repeat until the
#       crosses land where the cups should stand, adjusting the `grid:`
#       blocks in the experiment config between runs.
#
#   scripts/experiments/run_pour_appearance_experiment.sh --baseline
#       Collect the unguided arm of the comparison.
#
#   scripts/experiments/run_pour_appearance_experiment.sh --guided
#       Collect the guided arm, over the identical plan.
#
#   scripts/experiments/run_pour_appearance_experiment.sh --report
#       Rebuild the report from what is on disk. No servers, no robot.
#       report/source_cup_rates.png is the headline: success per cup with
#       Wilson bars; source_cup_outcomes.png says how each cup failed.
#
#   scripts/experiments/run_pour_appearance_experiment.sh --simulate --baseline
#       Arm powered off, synthetic observations -- exercises the GUI, the
#       store and the report without touching the robot.
#
# The cups live in the experiment config -- edit `variants:` and
# `descriptions:` under the `source_cup` factor, keeping the cup the demos
# were recorded with first. They can also be changed from the GUI (Ctrl+E,
# between rollouts) without restarting; a changed list is a different
# plan, so it offers a new experiment name if anything is collected.
#
# Interrupted? Re-run the same command. The plan is frozen on disk, so
# collection resumes at the next rollout with no result.
#
#   scripts/experiments/run_pour_appearance_experiment.sh --baseline --overwrite
#       Start this experiment over. Needed after editing the cups or the
#       staging marks: the plan on disk no longer matches the config, and
#       resuming would mix conditions. The old results are renamed aside,
#       not deleted.
#
# --guided needs RETRIEVAL_DIR, a folder of encoded pouring demos to
# retrieve from. The raw demos the checkpoint was trained on carry no
# depth and no camera extrinsic, so they cannot be used directly; record
# one with evaluate.py (or take a successful rollout of this experiment's
# baseline arm, .vla and .json together), put it in a folder, and encode:
#
#   mkdir -p data/reference_dataset_pour
#   cp data/ur5_experiments/experiments/pour_cup_appearance_baseline/episodes/rollout_0000_* \
#       data/reference_dataset_pour/
#   python scripts/encode_datasets.py --source-path data/reference_dataset_pour
#   RETRIEVAL_DIR=data/reference_dataset_pour \
#       scripts/experiments/run_pour_appearance_experiment.sh --guided
#
# Overridable from the environment (same names as run_ur5_rollout.sh):
#   CHECKPOINT TASK RUN_NAME EXPERIMENT_CONFIG INSTANCE GPUS RETRIEVAL_DIR
#   EXTERIOR_SERIAL WRIST_SERIAL UR5_ROBOT_IP CALIBRATION SDK_PATH
#
# Extra collect_experiment.py flags go after `--`:
#   scripts/experiments/run_pour_appearance_experiment.sh --baseline -- \
#       --experiment.headless
#
# They are routed to the right side of the tyro subcommands by prefix
# (`--env.*` after `env:ur5`, `--policy.*` after `policy:lerobot`,
# everything else before both), so they can be given in any order.

# What this experiment is; everything about the *cell* is in the shared body.
EXPERIMENT_DEFAULT_CONFIG=configs/experiments/pour_cup_appearance.yaml
EXPERIMENT_DEFAULT_NAME=pour_cup_appearance
# Only what the env server publishes; the config's own `task:` is what
# reaches the policy. Kept identical to it. The demos were recorded with
# "Pour contents of blue cup into green cup." -- set TASK and the config's
# `task:` back to that to test the training phrasing.
EXPERIMENT_DEFAULT_TASK="Pour the contents of the left cup into the right cup."
# Where --guided should be told to look when RETRIEVAL_DIR is unset.
EXPERIMENT_RETRIEVAL_EXAMPLE=data/reference_dataset_pour

# The shared body defaults CHECKPOINT to the cable checkpoint; a value set
# before sourcing wins, and the environment still overrides this one.
CHECKPOINT="${CHECKPOINT:-checkpoints/pour_contents_of_blue_cup_into_green_cup_a100/pretrained_model}"

# Sub-trajectory segmentation for the pouring demo. The shared body's
# 0.001 m/step / 50 steps was tuned on the cable demos; the pour reference
# episode (data/reference_dataset_pour, 1314 steps) moves more slowly and
# under that setting splits into just two segments, the second starting only
# at the retreat. 0.00075 / 40 gives three: approach + grasp + carry
# (0-580), the pour itself (580-1183), and set-down + retreat (1183-1314).
# It sits in the middle of the plateau where three segments come out
# (speed 0.0007-0.00075, min length 30-45), so a slightly different
# reference recording should land there too. A value set in the environment
# still wins, as for CHECKPOINT.
SUBTRAJ_SPEED_THRESHOLD="${SUBTRAJ_SPEED_THRESHOLD:-0.00075}"
SUBTRAJ_MIN_LENGTH="${SUBTRAJ_MIN_LENGTH:-40}"

source "$(dirname "${BASH_SOURCE[0]}")/_ur5_experiment.sh"
