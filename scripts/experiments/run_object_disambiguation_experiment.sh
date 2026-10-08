#!/usr/bin/env bash
# Target disambiguation on the real UR5e cell: several named objects on
# the table at once, and a prompt that names one of them.
#
# The scene is held fixed across the objects of a layout and only the
# instruction changes, so a difference in success rate between objects
# is the policy resolving — or failing to resolve — the reference in
# the prompt. Positions are re-sampled per layout (at least 10 cm
# apart) so that "went to the named object" cannot be confused with
# "went to the same place as always".
#
# Same cell, same camera addressing, same pre-flight and the same
# operator GUI as run_cable_position_experiment.sh. See
# docs/real_experiments.md for the GUI, docs/ur5.md for the cell.
#
# Usage:
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --calibrate
#       Project the placement region and one layout's crosses onto a
#       camera frame, write a PNG, print every object position in base
#       coordinates and the prompt the first rollout would use, exit.
#       The arm is never commanded. Repeat until the region lands on the
#       work surface, adjusting the `region:` block in the experiment
#       config between runs.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --baseline
#       Collect the unguided arm of the comparison.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --guided
#       Collect the guided arm, over the identical plan.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --guided \
#       --reference object_disambiguation_baseline
#       The same, but taking the settings from that experiment's own
#       directory (data/<run>/experiments/<name>/experiment.yaml) instead
#       of the config file: whatever the baseline was collected under --
#       GUI edits, an --objects override, a re-sampled region -- this arm
#       stages the identical sequence of layouts and prompts. The plan
#       those settings generate is checked against the baseline's frozen
#       plan.json and its recorded positions before the checkpoint loads,
#       so a settings file edited after the fact is refused rather than
#       staged under the old level names. A path to the directory works
#       too. Only creates: if this arm already exists it resumes as usual
#       (add --overwrite to recreate it from the reference).
#
#       While staging, the reference's first recorded frame for the same
#       condition is ghosted under the live view, so the objects can be
#       nudged until the two pictures line up -- orientation and all, not
#       only the centre the cross marks. G cycles it: faint, strong,
#       reference only, off. It is remembered with the experiment, so a resume shows
#       it without repeating --reference.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --report
#       Rebuild the report from what is on disk. No servers, no robot.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --simulate --baseline
#       Arm powered off, synthetic observations -- exercises the
#       GUI, the store and the report without touching the robot.
#
# The objects, and how many of them, live in the experiment config —
# edit `objects:` there. To try a different set without editing it:
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --baseline \
#       --objects "black cable,white cable,red cable,blue cable"
#
#   That re-samples the layouts for the new count, which is a different
#   plan, so add --overwrite if this experiment has already collected.
#   Collect the other arm with --reference rather than repeating --objects,
#   and the two cannot drift apart.
#
# The prompt is a template in the config — `task: "Put the
# {target_object} in the bag."` — and `{target_object}` is filled per
# rollout from the object the plan names. Keep the wording around the
# placeholder to the phrasing the checkpoint was trained on.
#
# Interrupted? Re-run the same command. The plan is frozen on disk,
# so collection resumes at the next rollout with no result.
#
#   scripts/experiments/run_object_disambiguation_experiment.sh --baseline --overwrite
#       Start this experiment over. Needed after editing the region, the
#       object list or the layout count: the plan on disk no longer
#       matches the config, and resuming would mix conditions. The old
#       results are renamed aside, not deleted.
#
# The plan and the per-rollout limits can also be changed from the GUI
# itself -- Ctrl+E, between rollouts -- without restarting and paying
# the checkpoint load again. If the edit would orphan rollouts already
# collected, it asks whether to start a new experiment under a new name
# or to delete them, rather than choosing for you.
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
#   RETRIEVAL_DIR=data/reference_dataset_cable run_object_disambiguation_experiment.sh --guided
#
# Extra collect_experiment.py flags go after `--`:
#   scripts/experiments/run_object_disambiguation_experiment.sh --baseline -- \
#       --experiment.headless
#
# They are routed to the right side of the tyro subcommands by prefix
# (`--env.*` after `env:ur5`, `--policy.*` after `policy:lerobot`,
# everything else before both), so they can be given in any order.

# What this experiment is; everything about the *cell* is in the shared body.
EXPERIMENT_DEFAULT_CONFIG=configs/experiments/object_disambiguation.yaml
EXPERIMENT_DEFAULT_NAME=object_disambiguation
# Only what the env server publishes: the prompt the policy actually gets is
# the config's template, filled per rollout with the object being asked for.
EXPERIMENT_DEFAULT_TASK="Put the cable in the bag."

source "$(dirname "${BASH_SOURCE[0]}")/_ur5_experiment.sh"
