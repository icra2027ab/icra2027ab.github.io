# LIBERO-PRO: dataset + evaluation commands

Reference commands. Run from the repo root. Assumes you've completed
[`docs/SETUP.md`](SETUP.md) (root env + `.env`) and
[`docs/install_libero_pro.md`](install_libero_pro.md) (LIBERO-PRO fork +
asset download + sim venv).

## Download original datasets

```bash
# in the repo root: datasets/ is a symlink to your storage volume
cd datasets
wget https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip -O libero_spatial.zip
unzip -q libero_spatial.zip
wget https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip -O libero_object.zip
unzip -q libero_object.zip
wget https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip -O libero_goal.zip
unzip -q libero_goal.zip
wget https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip -O libero_100.zip
unzip -q libero_100.zip   # contains libero_10 and libero_90

rm libero_spatial.zip libero_object.zip libero_goal.zip libero_100.zip
cd ..
```

## Dataset recollection (higher resolution + depth + segmentation)

Uses the LIBERO-PRO Python 3.8 venv at `third_party/LIBERO-PRO/.venv`
(see [`libero_pro_setup.md` §3](libero_pro_setup.md#3-create-the-libero-python-environment)).
`MUJOCO_GL=glx` is set inline; everything else is picked up from `.env`.

```bash
# from the repo root
source third_party/LIBERO-PRO/.venv/bin/activate
export MUJOCO_GL=glx
export PYTHONPATH="$PWD/third_party/LIBERO-PRO:$PWD/src"

for suite in libero_object libero_spatial libero_goal libero_10; do
  python src/policy_abstraction/data/regenerator.py \
    --libero_task_suite "$suite" \
    --libero_raw_data_dir "datasets/$suite" \
    --libero_target_dir "datasets/${suite}_new" --y
done
```

(`scripts/setup/regenerate_libero_datasets.sh` runs this loop for you.)

## Extract interaction sub-trajectories

Back in the root venv:

```bash
for suite in libero_object libero_spatial libero_goal libero_10; do
  uv run scripts/ablations/extraction/extract_interactions.py \
    --input "datasets/${suite}_new/" \
    --output "datasets/interactions/${suite}_interactions.hdf5"
done
```

## Evaluation commands

```bash
uv run python scripts/evaluate.py \
  --output.video-out-path data/libero/libero_spatial \
  --env.task-suite-name libero_spatial \
  --env.perturb.swap

uv run python scripts/evaluate.py \
  --output.video-out-path data/libero/libero_goal \
  --env.task-suite-name libero_goal \
  --env.perturb.swap

uv run python scripts/evaluate.py \
  --output.video-out-path data/libero/libero_object \
  --env.task-suite-name libero_object \
  --env.perturb.swap

uv run python scripts/evaluate.py \
  --output.video-out-path data/libero/libero_10 \
  --env.task-suite-name libero_10 \
  --env.perturb.swap
```

Or a sweep:

```bash
uv run python scripts/perturbation_sweep.py \
  --task-suites libero_object libero_goal libero_10 libero_spatial \
  --perturbations task object language --output.run real_perception_HL
```
