#!/usr/bin/env bash
# Appearance shift on the real UR5e cell: "place the pen in blue cup on
# paper", fetched with several different pens a few times each.
#
# The checkpoint was trained on one pen. Here the operator swaps in a
# different pen of the same type per condition -- other colours,
# thicknesses, lengths, capped or not -- while the prompt stays the
# training phrase, which never describes the pen and so stays true
# whichever one is on the mark. The success rate per pen is how far the
# policy generalises past the object it was shown, and the guided arm asks
# whether guidance from a demo closes that gap.
#
# Same cell, same camera addressing, same pre-flight and the same
# operator GUI as run_pour_appearance_experiment.sh, with the pen
# checkpoint swapped in. See docs/real_experiments.md for the GUI,
# docs/ur5.md for the cell.
#
# Usage:
#
#   scripts/experiments/run_pen_appearance_experiment.sh --calibrate
#       Project the two staging marks (pen, blue cup on its paper) onto a
#       camera frame, write a PNG, print both positions in base
#       coordinates, exit. The arm is never commanded. Repeat until the
#       crosses land where the pen and cup should sit, adjusting the
#       `grid:` blocks in the experiment config between runs.
#
#   scripts/experiments/run_pen_appearance_experiment.sh --baseline
#       Collect the unguided arm of the comparison.
#
#   scripts/experiments/run_pen_appearance_experiment.sh --guided
#       Collect the guided arm, over the identical plan.
#
#   scripts/experiments/run_pen_appearance_experiment.sh --report
#       Rebuild the report from what is on disk. No servers, no robot.
#       report/pen_rates.png is the headline: success per pen with Wilson
#       bars; pen_outcomes.png says how each pen failed.
#
#   scripts/experiments/run_pen_appearance_experiment.sh --simulate --baseline
#       Arm powered off, synthetic observations -- exercises the GUI, the
#       store and the report without touching the robot.
#
# The pens live in the experiment config -- edit `variants:` and
# `descriptions:` under the `pen` factor, keeping the pen the demos were
# recorded with first. They can also be changed from the GUI (Ctrl+E,
# between rollouts) without restarting; a changed list is a different
# plan, so it offers a new experiment name if anything is collected.
#
# Interrupted? Re-run the same command. The plan is frozen on disk, so
# collection resumes at the next rollout with no result.
#
#   scripts/experiments/run_pen_appearance_experiment.sh --baseline --overwrite
#       Start this experiment over. Needed after editing the pens or the
#       staging marks: the plan on disk no longer matches the config, and
#       resuming would mix conditions. The old results are renamed aside,
#       not deleted.
#
# --guided needs RETRIEVAL_DIR, a folder of encoded pen demos to retrieve
# from. The raw demos the checkpoint was trained on
# (data/demonstration_data/place_the_pen_in_blue_cup_on_paper)
# carry no depth and no camera extrinsic, so they cannot be used directly;
# record one with evaluate.py (or take a successful rollout of this
# experiment's baseline arm, .vla and .json together), put it in a folder,
# and encode:
#
#   mkdir -p data/reference_dataset_pen
#   cp data/ur5_experiments/experiments/pen_appearance_baseline/episodes/rollout_0000_* \
#       data/reference_dataset_pen/
#   python scripts/encode_datasets.py --source-path data/reference_dataset_pen
#   RETRIEVAL_DIR=data/reference_dataset_pen \
#       scripts/experiments/run_pen_appearance_experiment.sh --guided
#
# Overridable from the environment (same names as run_ur5_rollout.sh):
#   CHECKPOINT TASK RUN_NAME EXPERIMENT_CONFIG INSTANCE GPUS RETRIEVAL_DIR
#   EXTERIOR_SERIAL WRIST_SERIAL UR5_ROBOT_IP CALIBRATION SDK_PATH
#   SUBTRAJ_SPEED_THRESHOLD SUBTRAJ_MIN_LENGTH
#
# Extra collect_experiment.py flags go after `--`:
#   scripts/experiments/run_pen_appearance_experiment.sh --baseline -- \
#       --experiment.headless
#
# They are routed to the right side of the tyro subcommands by prefix
# (`--env.*` after `env:ur5`, `--policy.*` after `policy:lerobot`,
# everything else before both), so they can be given in any order.

# What this experiment is; everything about the *cell* is in the shared body.
EXPERIMENT_DEFAULT_CONFIG=configs/experiments/pen_appearance.yaml
EXPERIMENT_DEFAULT_NAME=pen_appearance
# Only what the env server publishes; the config's own `task:` is what
# reaches the policy. Kept identical to it, and to the phrase the demos were
# recorded with -- the prompt never describes the pen, so there is nothing
# to reword for a different one.
EXPERIMENT_DEFAULT_TASK="Place the pen in blue cup on paper."
# Where --guided should be told to look when RETRIEVAL_DIR is unset.
EXPERIMENT_RETRIEVAL_EXAMPLE=data/reference_dataset_pen

# The shared body defaults CHECKPOINT to the cable checkpoint; a value set
# before sourcing wins, and the environment still overrides this one.
CHECKPOINT="${CHECKPOINT:-checkpoints/place_the_pen_in_blue_cup_on_paper_a100/pretrained_model}"

# Sub-trajectory segmentation: the shared body's 0.001 m/step / 50 steps
# is left alone. Swept over the raw pen demos (800-1250 steps, median speed
# 0.0009-0.0012 m/step) it gives three segments on most of them -- approach
# + grasp (0-400), carry + drop into the cup (400-790), retreat (790-end)
# -- and two or four on the rest, which is the same plateau the cable demos
# sit on. The pouring script overrides both because its demo moves more
# slowly; this one does not need to. Set SUBTRAJ_SPEED_THRESHOLD /
# SUBTRAJ_MIN_LENGTH in the environment if a recorded reference episode
# segments differently.

source "$(dirname "${BASH_SOURCE[0]}")/_ur5_experiment.sh"
