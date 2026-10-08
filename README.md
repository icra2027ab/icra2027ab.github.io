# Visual-Geometric Policy Steering from In-Context Aligned Abstractions

Code release for the paper *Visual-Geometric Policy Steering from In-Context
Aligned Abstractions* (under review). Author information and repository URLs
are withheld for double-blind review.

Test-time steering of frozen robotic manipulation policies toward
out-of-distribution (OOD) scenes — without retraining or fine-tuning.

## Overview

Pre-trained robot policies degrade when deployed in environments that differ
from their training distribution. This framework adapts such policies at
inference time by injecting a differentiable guidance signal into the
denoising process of flow-matching policies (pi0.5 via
[OpenPI](https://github.com/Physical-Intelligence/openpi) and LeRobot, and
[GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T)). The policy weights stay
frozen; only the velocity field is reshaped during denoising.

The core idea: retrieve a successful in-distribution rollout, let a
vision-language model (VLM) pick which *object-relational distances* matter
in it, align the live scene to that reference in distance space, and steer the
policy so the live distances follow the reference trends.

## Method

```
New Task
   │
   ├─► Retrieval
   │       Match by language instruction + initial observation similarity (DINOv2 + CLIP)
   │
   ├─► Abstraction
   │       A VLM selects object-relational distance pairs from the retrieved rollout
   │       Sub-trajectory segmentation by velocity thresholds
   │
   ├─► Alignment
   │       Match the live scene to the reference trajectory in distance space (DTW)
   │
   └─► Guidance injection
           Loss on deviation from the reference distance trends
           Gradient through a differentiable dynamics model (OSC integrator / UR5 FK)
           Golden-section line search on the step size with a do-no-harm guard
           Relevance gate: a relation activates when the end-effector approaches its objects
```

## Pipeline

```
retrieval → abstraction → alignment → guidance → policy action correction
```

| Module | Role |
|---|---|
| `core/retrieval` | Retrieve a relevant demo via DINOv2 / CLIP embeddings + task similarity |
| `core/abstraction` | Extract distance pairs and their reference trends from the retrieved demo |
| `core/alignment` | Align the live timeline to the demo via DTW |
| `core/guidance` | Compute the differentiable guidance gradients and line search |
| `perception/` | Open-vocabulary detection, segmentation, tracking, 3D reconstruction |
| `features/` | Modular distance / geometric feature extractors |
| `vlm/` | VLM-based distance-pair selection and demo-to-scene object correspondence |
| `policies/` | Policy-server clients (OpenPI, GR00T, LeRobot) |
| `envs/` | Environment client/server (LIBERO-PRO simulator, real UR5e cell) |

## Project Structure

```
src/policy_abstraction/
├── core/              # Pipeline phases: retrieval → abstraction → alignment → guidance
├── evaluation/        # Episode runner, benchmarks, live rollout orchestration
├── perception/        # Object detection, segmentation, 3D reconstruction, tracking
├── features/          # Modular distance/geometric feature extractors
├── vlm/               # VLM-based distance-pair selection and object correspondence
├── retrieval/         # Embedding-based demo retrieval (DINOv2 + CLIP)
├── analysis/          # Offline demo analysis (sub-trajectories, distance pairs)
├── policies/          # Policy client wrappers (OpenPI, GR00T, LeRobot)
├── envs/              # Environment client/server (LIBERO-PRO, real UR5e)
├── dynamics/          # Differentiable forward dynamics models
├── experiments/       # Operator-driven real-robot experiment tooling
├── demo/              # Live guidance demo GUI for the UR5e
├── calibration/       # Camera calibration for the real cell
├── visualization/     # Rerun 3D viewer, alignment plots, video annotation
├── data/              # Dataset loaders, trajectory recorders (.vla format)
└── utils/             # Point clouds, depth, gripper kinematics, transforms

packages/
├── policy_steering/   # Guidance functions, line search, differentiable kinematics
└── open_perception/   # Open-vocabulary multi-camera segmentation and tracking (vendored)
```

## Scripts

The main entry point is [`scripts/evaluate.py`](scripts/evaluate.py). It
orchestrates the full evaluation loop: launches policy and environment
servers, retrieves demonstrations, runs guided rollouts with perturbations,
and saves videos, episodes, and a JSON summary.

| Script | Purpose |
|---|---|
| `scripts/evaluate.py` | **Main evaluation** — run tasks (LIBERO-PRO, real UR5e) with guided policy steering |
| `scripts/perturbation_sweep.py` | Run the LIBERO-PRO perturbation matrix from a sweep config (`configs/sweeps/`) |
| `scripts/experiments/collect_experiment.py` | **Operator-driven collection** — many rollouts against one loaded policy, with a grid overlay and hotkeys (see [`docs/real_experiments.md`](docs/real_experiments.md)) |
| `scripts/experiments/run_cable_position_experiment.sh` | The positional distribution-shift sweep on the UR5e, baseline vs guided |
| `scripts/experiments/run_object_disambiguation_experiment.sh` | The object-generalization experiment on the UR5e |
| `scripts/run_ur5_demo.sh` | **Live guidance demo** — one loaded policy, guidance switched on and off from a window (see [`docs/demo.md`](docs/demo.md)) |
| `scripts/encode_datasets.py` | Encode demos with DINOv2 + CLIP embeddings (`*_embeddings.npz`) |
| `scripts/replay_episode.py` | Replay recorded `.vla` episodes in the Rerun viewer |
| `scripts/servers/serve_libero_env.py` | WebSocket server for the LIBERO-PRO simulator |
| `scripts/servers/serve_ur5_env.py` | WebSocket server for a real UR5e cell (see [`docs/ur5.md`](docs/ur5.md)) |
| `scripts/servers/serve_groot.py` | Inference server for GR00T N1.7 checkpoints |
| `scripts/servers/serve_lerobot.py` | Inference server for LeRobot pi0.5 checkpoints (real robot) |

## Getting Started

Full walkthrough: **[`docs/SETUP.md`](docs/SETUP.md)**. Quick start on a
fresh machine:

```bash
# 1. Clone (packages/open_perception is vendored in this release)
git clone <anonymized-repository-url> policy_abstraction
cd policy_abstraction

# 2. Heavy-storage symlinks (one-time per machine)
export POLICY_ABSTRACTION_STORAGE="$HOME/policy_abstraction_storage"
mkdir -p "$POLICY_ABSTRACTION_STORAGE"/{data,datasets,checkpoints}
ln -s "$POLICY_ABSTRACTION_STORAGE"/{data,datasets,checkpoints} .

# 3. Root environment
cp .env.example .env && $EDITOR .env   # set OPENAI_API_KEY (or OPENROUTER_API_KEY)
uv sync

# 4. Install the simulator + one policy (see docs/SETUP.md §4-§5)
# 5. Run evaluation
uv run python scripts/evaluate.py --env.task-suite-name libero_spatial
```

CLI arguments are managed via [tyro](https://github.com/brentyi/tyro) and grouped
into nested dataclasses (see `Args` in [scripts/evaluate.py](scripts/evaluate.py)).
Every flag lives under a group: `--policy.*`, `--env.*`, `--env.perturb.*`,
`--camera.*`, `--iter.*`, `--retrieval.*`, `--demo-analysis.*`, `--guidance.*`,
`--alignment.*`, `--subtraj.*`, `--perception.*`, `--viz.*`, `--output.*`. See
all options with:

```bash
python scripts/evaluate.py --help
```

### Key Configuration Flags

| Flag | Default | Description |
|---|---|---|
| `--env.task-suite-name` | `libero_spatial` | LIBERO task suite to evaluate |
| `--env.perturb.swap / .object / ...` | swap only | OOD perturbation types to apply |
| `--retrieval.enable` | `True` | Retrieve in-distribution demos for guidance |
| `--demo-analysis.enable` | `True` | VLM-based distance-pair selection + feature extraction |
| `--guidance.enable` | `True` | Apply differentiable guidance gradients |
| `--guidance.factor` | `1.0` | Guidance signal strength |
| `--perception.mode` | `REAL_PERCEPTION` | Live point-cloud source (`GT_POINTCLOUD`, `GT_SEG_RECONSTRUCT`, `REAL_PERCEPTION`) |
| `--alignment.mode` | `MIXED` | Temporal alignment strategy (`UNIFIED`, `PER_FEATURE`, `MIXED`) |
| `--iter.num-trials-per-task` | `1` | Episodes per task |
| `--policy.replan-steps` | `5` | Steps between policy re-planning |

### Data Preparation

Before running retrieval-based evaluation, encode your demonstration datasets:

```bash
python scripts/encode_datasets.py --dataset-dir <path-to-hdf5-demos>
```

This produces `*_embeddings.npz` files used by the retrieval index.

### Third-Party Setup

Third-party simulators and policies live in `third_party/` (cloned per
simulator/policy, each with its own venv). See:

- **[`docs/SETUP.md`](docs/SETUP.md)** — canonical end-to-end setup walkthrough
- [`docs/install_libero_pro.md`](docs/install_libero_pro.md) + [`docs/libero_pro_setup.md`](docs/libero_pro_setup.md) — LIBERO-PRO
- [`docs/install_openpi.md`](docs/install_openpi.md) — OpenPI (pi0.5) policy server
- [`docs/install_groot.md`](docs/install_groot.md) — GR00T N1.7 policy + checkpoints
- [`docs/libero_pro_eval_commands.md`](docs/libero_pro_eval_commands.md) — Example evaluation commands
- [`docs/ur5.md`](docs/ur5.md) — real UR5e arm (driver + LeRobot pi0.5 policy server)
- [`docs/install_lerobot.md`](docs/install_lerobot.md) — LeRobot (pi0.5) policy server used on the real robot
- [`docs/real_deployment.md`](docs/real_deployment.md) — bringing the real cell up
- [`docs/real_experiments.md`](docs/real_experiments.md) — operator-driven experiments: staging grid, hotkeys, resume, reports
- [`docs/demo.md`](docs/demo.md) — live guidance demo on the UR5e

## Tests

```bash
uv run pytest              # all tests
uv run pytest -m "not slow"  # skip slow tests
FORCE_CPU=1 uv run pytest  # force CPU (no CUDA)
```

Test markers: `slow`, `integration` (requires HDF5/network/simulator), `gpu` (requires CUDA).

## Key Dependencies

- **Policies:** [OpenPI](https://github.com/Physical-Intelligence/openpi) (pi0.5, simulation), [GR00T N1.7](https://github.com/NVIDIA/Isaac-GR00T) (simulation), [LeRobot](https://github.com/huggingface/lerobot) (pi0.5, real robot)
- **Simulator:** [LIBERO-PRO](https://github.com/Zxy-MLlab/LIBERO-PRO)
- **Perception:** SAM3 / SAM2 segmentation and tracking, DINOv2 / CLIP retrieval embeddings
- **Abstraction:** OpenAI-compatible VLM API (distance-pair selection, object correspondence)
- **Alignment:** DTW via [dtaidistance](https://github.com/wannesm/dtaidistance)
- **3D:** Open3D, JAX / PyTorch (differentiable guidance and kinematics)
- **Visualization:** [Rerun](https://rerun.io/)
