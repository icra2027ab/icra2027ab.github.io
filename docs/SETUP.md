# Setup guide

End-to-end walkthrough for setting `policy_abstraction` up on a fresh machine.
The aim is "follow top to bottom, no surprises."

## 0. Prerequisites

| What | Version | Notes |
|---|---|---|
| OS | Linux (Ubuntu 22.04+) | An X11 server or virtual display is required for MuJoCo rendering |
| GPU + drivers | CUDA 12.x | Required for policy + perception models |
| `uv` | latest | <https://docs.astral.sh/uv> |
| `git`, `git-lfs`, `curl`, `build-essential`, `cmake` | system pkgs | `apt install git git-lfs curl build-essential cmake` |
| ffmpeg dev headers + `pkg-config` | system pkgs | `apt install pkg-config libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev libavfilter-dev libswscale-dev libswresample-dev libpostproc-dev` (needed by `av` build in the OpenPI venv). |
| Storage | ~700 GB free | Datasets (~630 GB) + checkpoints (~35 GB) + venvs |

You need an [OpenAI API key](https://platform.openai.com/api-keys) for the
VLM-based demo analysis path.

You also need a **HuggingFace account with access to `facebook/sam3`** (gated
repo). Request access at <https://huggingface.co/facebook/sam3>, then log in
once on the machine:

```bash
uv run hf auth login   # paste a token from https://huggingface.co/settings/tokens
```

`REAL_PERCEPTION` mode loads `facebook/sam3` via `transformers.from_pretrained`,
which 401s without a logged-in token.

---

## 1. Clone the repo

The perception package `packages/open_perception` is vendored in this
release, so a plain clone is enough.

```bash
git clone <anonymized-repository-url> policy_abstraction
cd policy_abstraction
```

---

## 2. Heavy-storage symlinks

`data/`, `datasets/`, `checkpoints/` are kept on a separate storage volume
because they're large (hundreds of GB) and don't belong in your home backup.
Point `POLICY_ABSTRACTION_STORAGE` at any directory you like; the rest of the
docs use repo-relative paths and the symlinks redirect.

```bash
export POLICY_ABSTRACTION_STORAGE="$HOME/policy_abstraction_storage"
# or: export POLICY_ABSTRACTION_STORAGE=/scratch/$USER/policy_abstraction

mkdir -p "$POLICY_ABSTRACTION_STORAGE"/{data,datasets,checkpoints}
ln -s "$POLICY_ABSTRACTION_STORAGE/data"        data
ln -s "$POLICY_ABSTRACTION_STORAGE/datasets"    datasets
ln -s "$POLICY_ABSTRACTION_STORAGE/checkpoints" checkpoints
```

Verify the three symlinks resolve:

```bash
ls -L data datasets checkpoints   # should not error
```

---

## 3. Root environment + `.env`

```bash
cp .env.example .env
$EDITOR .env                  # set OPENAI_API_KEY (or OPENROUTER_API_KEY)
uv sync                       # creates .venv with all root deps
```

Set **one** VLM key: `OPENAI_API_KEY` (OpenAI) or `OPENROUTER_API_KEY` (routes
VLM calls through [OpenRouter](https://openrouter.ai/api/v1), OpenAI-wire
compatible; takes precedence when both are set — pass an OpenRouter model id
via `--vlm-model`, e.g. `openai/gpt-4o`).

Smoke test (clear any stale shell-exported vars first to make sure `.env` is
the source of truth):

```bash
unset LIBERO_PRO_PATH PYTHONPATH DATASET_BASE_PATH
uv run python -c "
from policy_abstraction import PROJECT_ROOT
import os
assert os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENROUTER_API_KEY'), \
    'set OPENAI_API_KEY or OPENROUTER_API_KEY in .env'
print('PROJECT_ROOT:', PROJECT_ROOT)
print('OK')"
```

---

## 4. Simulators

Each simulator runs in its own venv to avoid dependency conflicts with the
root environment (numpy / torch / mujoco pins differ across them).

| Simulator | Guide |
|---|---|
| LIBERO-PRO | [`docs/install_libero_pro.md`](install_libero_pro.md) + [`docs/libero_pro_setup.md`](libero_pro_setup.md) |
| Real UR5e cell | [`docs/ur5.md`](ur5.md) |

The LIBERO-PRO guide ends with a smoke test that exits 0 when the env server
can spin up successfully.

---

## 5. Policies

| Policy | Guide |
|---|---|
| OpenPI (pi0.5, simulation) | [`docs/install_openpi.md`](install_openpi.md) |
| GR00T N1.7 (simulation) | [`docs/install_groot.md`](install_groot.md) |
| LeRobot (pi0.5, real UR5e) | [`docs/install_lerobot.md`](install_lerobot.md) |

OpenPI checkpoints auto-download from `gs://openpi-assets/...` to
`~/.cache/openpi/` on first server start; no manual step is needed.

GR00T checkpoints download via `hf download` — see the linked doc for the
exact commands (they target `checkpoints/GR00T-N1.7-LIBERO/` in the repo,
which resolves to your storage volume via the symlinks).

---

## 6. Datasets

Once the simulator is installed, fetch its demonstration datasets so
retrieval-based evaluation works. The data download + recollection +
embedding pipeline is documented in
[`docs/libero_pro_setup.md`](libero_pro_setup.md).

The high-level pipeline:

```
raw demos  →  regenerator.py  →  extract_interactions.py  →  encode_datasets.py
```

All output paths are repo-relative (e.g. `datasets/libero_object_new/...`),
so files land on your storage volume via the symlinks.

---

## 7. Environment variables reference

`.env` is loaded automatically by `policy_abstraction.__init__` (override=False,
so shell-exported variables win for cluster/CI overrides). The full set:

| Variable | Required? | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes (VLM path)¹ | — | OpenAI API key for VLM-based demo analysis |
| `OPENROUTER_API_KEY` | alt. to above¹ | — | Route VLM calls through OpenRouter (OpenAI-wire compatible); preferred when both set |
| `OPENROUTER_BASE_URL` | optional | `https://openrouter.ai/api/v1` | Override the OpenRouter endpoint |
| `DISPLAY` | yes (headless GUIs) | `:0` | X11 display for MuJoCo rendering |
| `MUJOCO_GL` | sometimes | unset | `glx` for remote/headless nodes |
| `JAX_COMPILATION_CACHE_DIR` | optional | unset | Persistent JAX recompile cache (speeds up server restarts) |
| `CUDA_VISIBLE_DEVICES` | optional | unset | Restrict GPU visibility |
| `LIBERO_PRO_PATH` | optional override | `<repo>/third_party/LIBERO-PRO` | Override the LIBERO-PRO checkout location |
| `POLICY_ABSTRACTION_LEGACY_PREFIX` | optional | `/scratch/<user>/policy_abstraction/` | Used only by `scripts/debug/low-level/test_cotracker_correspondences.py` to remap paths in legacy HDF5 logs recorded on another machine |
| `COLOREDLOGS_*` | optional | sensible | Log formatting |

¹ Set exactly one of `OPENAI_API_KEY` / `OPENROUTER_API_KEY`. The VLM clients
build an OpenAI-compatible client that prefers `OPENROUTER_API_KEY` when present.

---

## 8. End-to-end smoke test

Run a 1-episode evaluation against LIBERO-PRO (once you've completed §4–§6
for that simulator):

```bash
uv run python scripts/evaluate.py \
  --env.task-suite-name libero_spatial \
  --iter.num-trials-per-task 1 \
  --iter.max-num-tasks 1
```

This launches the policy + env servers, runs one episode with guidance, and
writes a video under `data/<run>/.../videos/`.

For the real UR5e cell, use the `env:ur5 policy:lerobot` subcommands
described in [`docs/ur5.md`](ur5.md) and
[`docs/real_experiments.md`](real_experiments.md).

---

## Troubleshooting

- **Stale env vars leak in** — your shell may have `LIBERO_PRO_PATH` /
  `PYTHONPATH` exported from a previous session. Open a
  fresh shell or `unset` them; `.env` won't override shell-exported values.
- **Broken `data/`, `datasets/`, `checkpoints/` symlinks** — `POLICY_ABSTRACTION_STORAGE`
  is wrong or the target dirs don't exist. Re-run the §2 commands.
- **Simulator import errors during `evaluate.py`** — the simulator runs in
  its own venv. The error message points at which venv is missing; see §4.
