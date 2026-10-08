# GR00T install

[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T) provides policy
backbones used by the LIBERO evaluation path in `policy_abstraction`.

Clone under `third_party/` (run from the repo root):

```bash
cd third_party
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
```

## Checkpoints

Run all `hf download` commands from the repo root so the relative
`--local-dir checkpoints/...` paths resolve via the symlink to your storage
volume (set up in [`docs/SETUP.md` §2](SETUP.md#2-heavy-storage-symlinks)).

### LIBERO checkpoints

```bash
for suite in libero_10 libero_spatial libero_object libero_goal; do
  uv run hf download nvidia/GR00T-N1.7-LIBERO \
    --include "$suite/config.json" \
              "$suite/embodiment_id.json" \
              "$suite/model-*.safetensors" \
              "$suite/model.safetensors.index.json" \
              "$suite/processor_config.json" \
              "$suite/statistics.json" \
    --local-dir checkpoints/GR00T-N1.7-LIBERO
done
```

You must first request access to the base model:
[`nvidia/Cosmos-Reason2-2B`](https://huggingface.co/nvidia/Cosmos-Reason2-2B).
