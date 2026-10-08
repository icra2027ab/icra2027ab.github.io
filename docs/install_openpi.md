# OpenPI (pi0) install

We use a fork of [OpenPI](https://github.com/Physical-Intelligence/openpi)
that exposes the denoising loop to the guidance hook in `packages/policy_steering`.
The fork's URL is withheld for double-blind review and will be released with the
camera-ready version. Clone under `third_party/`:

```bash
# from the repo root
cd third_party
git clone --recurse-submodules <anonymized-fork-url> openpi
cd openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync          # creates .venv (Python 3.11)
```

The fork's `pyproject.toml` references `../../packages/policy_steering` as an
editable path source, so `policy_steering` is shared between the root and
openpi venvs without copy/sync.

`policy_abstraction` only uses OpenPI as a policy server (`pi0` / `pi0.5`
flow-matching). OpenPI's own LIBERO example is not used — the LIBERO env
server runs in [`third_party/LIBERO-PRO/.venv`](install_libero_pro.md). No
extra subenv is required here.

## System dependencies

`av` (pulled in via `lerobot`) is built from sdist and needs ffmpeg dev
headers:

```bash
sudo apt install -y pkg-config \
    libavformat-dev libavcodec-dev libavdevice-dev libavutil-dev \
    libavfilter-dev libswscale-dev libswresample-dev libpostproc-dev
```

The fork pins `av<=13.1.0` in `[tool.uv] override-dependencies` to stay
compatible with ffmpeg 4 (Ubuntu 22.04).

## Policy server

Launch via the parent repo (the helper `start_openpi_policy_server` resolves
the openpi checkout from `<repo-root>/third_party/openpi`):

```bash
# from the repo root
uv run python scripts/evaluate.py --env.task-suite-name libero_spatial
```

The policy server auto-downloads checkpoints from `gs://openpi-assets/...`
into `~/.cache/openpi/` on first run (~8.7 GB). Compilation cache lives at
`~/.cache/openpi_jax_compilation_cache/`; set `JAX_COMPILATION_CACHE_DIR` in
`.env` to relocate it.
