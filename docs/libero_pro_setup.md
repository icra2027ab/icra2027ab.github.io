# LIBERO-PRO Setup Guide

Complete guide for setting up the LIBERO-PRO environment, downloading data, recollecting datasets, and running evaluations within this project.

## Prerequisites

- Linux with a display server (X11) or virtual display (`Xvfb`)
- CUDA-capable GPU with drivers installed
- `git`, `git-lfs`, `uv` (Python package manager)
- An [OpenAI API key](https://platform.openai.com/api-keys) (required for VLM-based demo analysis)

---

## 1. Clone LIBERO-PRO

```bash
cd third_party/
git clone https://github.com/Zxy-MLlab/LIBERO-PRO/
```

## 2. Download LIBERO-PRO Assets (bddl & init files)

These files define the perturbation environments used for evaluation.

```bash
# Install git-lfs if not already available
apt-get install git-lfs
git lfs install

# Download the assets from HuggingFace
git clone https://huggingface.co/datasets/zhouxueyang/LIBERO-Pro LIBERO_PRO_assets

# Move files into the LIBERO-PRO directory structure
mv LIBERO_PRO_assets/bddl_files/* third_party/LIBERO-PRO/libero/libero/bddl_files/
mv LIBERO_PRO_assets/init_files/* third_party/LIBERO-PRO/libero/libero/init_files/

# Clean up
rm -rf LIBERO_PRO_assets
```

## 3. Create the LIBERO Python Environment

LIBERO requires Python 3.8. The env server launches LIBERO via
`third_party/LIBERO-PRO/.venv/bin/python`, so the venv must live at that
path.

```bash
cd third_party/LIBERO-PRO

uv venv --python 3.8 .venv

uv pip install --python .venv/bin/python -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match
uv pip install --python .venv/bin/python -e .
uv pip install --python .venv/bin/python \
    h5py tyro websockets msgpack_numpy loguru python-dotenv
```

The env server sets `PYTHONPATH=third_party/LIBERO-PRO` so the `libero.libero`
namespace package resolves at runtime.

## 4. Configure Paths

No per-machine YAML edits are needed. The LIBERO benchmark + evaluation
configs ship inside `policy_abstraction` at
[`src/policy_abstraction/envs/libero/configs/`](../src/policy_abstraction/envs/libero/configs/)
and use `${LIBERO_PRO_PATH}/...` placeholders that the loader expands at
runtime. The LIBERO env server (`src/policy_abstraction/envs/libero/server.py`)
sets `LIBERO_PRO_PATH` and `LIBERO_CONFIG_PATH` from `PROJECT_ROOT` automatically.

---

## 5. Environment Variables

The following environment variables **must** be set before running the LIBERO-PRO evaluation:

| Variable | Purpose | Example |
|---|---|---|
| `LIBERO_CONFIG_PATH` | Points LIBERO to the PRO config directory | `<repo-root>/third_party/LIBERO-PRO` |
| `PYTHONPATH` | Allows Python to find LIBERO-PRO modules | `<repo-root>/third_party/LIBERO-PRO` |
| `DISPLAY` | X11 display for MuJoCo rendering | `:0` |
| `OPENAI_API_KEY` | OpenAI API key for VLM-based demo analysis (or use `OPENROUTER_API_KEY`) | `sk-...` |
| `OPENROUTER_API_KEY` | Route VLM calls through OpenRouter instead of OpenAI (preferred when both set) | `sk-or-...` |

Optional but recommended:

| Variable | Purpose | Example |
|---|---|---|
| `MUJOCO_GL` | MuJoCo rendering backend | `glx` (set automatically by the env server) |
| `CUDA_VISIBLE_DEVICES` | Restrict GPU visibility | `0` |

`OPENAI_API_KEY` and `DISPLAY` come from `.env` automatically (see
[`docs/SETUP.md` §3](SETUP.md#3-root-environment--env)). `LIBERO_PRO_PATH`
defaults to `<repo-root>/third_party/LIBERO-PRO` and only needs to be set if
your checkout lives elsewhere. `LIBERO_CONFIG_PATH` is set inline by the env
server.

---

## 6. Download Original LIBERO Datasets

The original LIBERO demonstration datasets (~100 GB total) are needed for retrieval and recollection.

```bash
# From the repo root
mkdir -p datasets && cd datasets

wget https://utexas.box.com/shared/static/04k94hyizn4huhbv5sz4ev9p2h1p6s7f.zip -O libero_spatial.zip
unzip -q libero_spatial.zip && rm libero_spatial.zip

wget https://utexas.box.com/shared/static/avkklgeq0e1dgzxz52x488whpu8mgspk.zip -O libero_object.zip
unzip -q libero_object.zip && rm libero_object.zip

wget https://utexas.box.com/shared/static/iv5e4dos8yy2b212pkzkpxu9wbdgjfeg.zip -O libero_goal.zip
unzip -q libero_goal.zip && rm libero_goal.zip

wget https://utexas.box.com/shared/static/cv73j8zschq8auh9npzt876fdc1akvmk.zip -O libero_100.zip
unzip -q libero_100.zip && rm libero_100.zip  # contains libero_10 and libero_90

cd ..
```

## 7. Recollect Datasets (Higher Resolution, depth and segmentation)

Recollection regenerates demonstrations at higher image resolution with consistent random seeds. This step uses the **LIBERO-PRO Python 3.8 venv** created in §3.

```bash
# from the repo root
source third_party/LIBERO-PRO/.venv/bin/activate
export MUJOCO_GL=glx
export PYTHONPATH="$PWD/third_party/LIBERO-PRO:$PWD/src"

# Recollect each suite
for suite in libero_object libero_spatial libero_goal libero_10; do
    python src/policy_abstraction/data/regenerator.py \
        --libero_task_suite $suite \
        --libero_raw_data_dir datasets/$suite \
        --libero_target_dir datasets/${suite}_new --y
done
```

`scripts/setup/regenerate_libero_datasets.sh` runs this loop for you.

## 8. Extract Interaction Sub-trajectories

After recollection, extract gripper-object interactions (uses the main `uv` environment):

```bash
for suite in libero_spatial libero_goal libero_10 libero_object; do
    uv run scripts/ablations/extraction/extract_interactions.py \
        --input datasets/${suite}_new/ \
        --output datasets/interactions/${suite}_interactions.hdf5
done
```

## 9. Encode Datasets for Retrieval

Build the embedding index used by demo retrieval:

```bash
uv run scripts/encode_datasets.py --dataset-dir datasets/
```

---

## 10. Run Evaluation

With all servers running (see the main [README](../README.md)), launch evaluation:

```bash
python scripts/evaluate.py \
    --env.task-suite-name libero_object \
    --env.perturb.swap \
    --iter.num-trials-per-task 50
```

See [libero_pro_eval_commands.md](libero_pro_eval_commands.md) for more evaluation examples.
