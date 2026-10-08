# LeRobot (pi0 / pi0.5) install

[HuggingFace LeRobot](https://github.com/huggingface/lerobot) provides PyTorch
implementations of the pi0 family (`pi0`, `pi05`, `pi0_fast`). The
`policy_abstraction.policies.lerobot.LerobotPolicyClient` adapter spawns the
server defined in [`scripts/servers/serve_lerobot.py`](../scripts/servers/serve_lerobot.py)
inside a lerobot-enabled venv and talks to it over ZMQ (default port
`8000 + --instance`, same as the other backends).

LeRobot pins `transformers>=5.3,<5.4` plus its own torch build, which would
collide with the root `policy_abstraction` venv — hence the standalone venv.

## Provide a lerobot venv

Any project whose `.venv` has `lerobot` installed works. In this lab that is
the `physical-ai-sdk` checkout:

```bash
# from the repo root
mkdir -p third_party
ln -s ~/dev/physical-ai-sdk third_party/physical-ai-sdk

# ZMQ server deps — not lerobot dependencies, so install them explicitly
uv pip install --python third_party/physical-ai-sdk/.venv pyzmq tyro msgpack
```

If you don't have that checkout, build a standalone venv instead. Note that it
lands at a different path, so the eval command needs `--policy.repo-path` to
match:

```bash
mkdir -p third_party/lerobot-venv && cd third_party/lerobot-venv
uv venv --python 3.12
uv pip install "lerobot[pi]>=0.5.0" "transformers>=5.3.0,<5.4.0" pyzmq tyro msgpack
cd ../..

# ...then every eval run needs:
#   policy:lerobot --policy.repo-path third_party/lerobot-venv ...
```

`start_lerobot_server` looks for the interpreter at
`<repo_path>/.venv/bin/python`, with `repo_path` defaulting to
`third_party/physical-ai-sdk`. It also puts `packages/policy_steering/src` on
the child's `PYTHONPATH` — that package (numpy + torch only) supplies the
guidance kernel, and is the sole thing this repo injects into the lerobot venv.
A hand-launched server needs the same:

```bash
PYTHONPATH=packages/policy_steering/src \
  third_party/physical-ai-sdk/.venv/bin/python scripts/servers/serve_lerobot.py ...
```

## Checkpoints

A LeRobot checkpoint is the `pretrained_model/` directory holding
`config.json`, `model.safetensors` and the `policy_{pre,post}processor*` files.
The pi0.5 runs in this project live under the `checkpoints/` symlink:

```
checkpoints/pi05_place_coke_can_in_bowl_a100/pretrained_model/
checkpoints/pi05_put_all_cables_in_bag_a100/pretrained_model/
checkpoints/pi05_place_wooden_spoon_on_plate/pretrained_model/
```

HuggingFace repo ids (e.g. `lerobot/pi0_libero_finetuned`) also work and
auto-download to `~/.cache/huggingface/`.

The first run additionally pulls the PaliGemma tokenizer
(`google/paligemma-3b-pt-224`); `pi0_fast` checkpoints also pull
`lerobot/fast-action-tokenizer`.

## Run

The server is launched automatically by `scripts/evaluate.py`:

```bash
python scripts/evaluate.py \
  --guidance.no-enable \
  env:libero \
  policy:lerobot \
    --policy.model-path checkpoints/pi05_place_coke_can_in_bowl_a100/pretrained_model
```

For a manual smoke test:

```bash
third_party/physical-ai-sdk/.venv/bin/python scripts/servers/serve_lerobot.py \
  --model-path checkpoints/pi05_place_coke_can_in_bowl_a100/pretrained_model \
  --port 5599
```

Should print `✓ Server ready — listening on 0.0.0.0:5599` once the checkpoint
loads (~10 s when the 9 GB safetensors is in the page cache; longer straight
after a reboot, when it has to come off the RAID).

## Load time

`from_pretrained` builds the policy with every module's random init before it
reads a single checkpoint weight, and lerobot does that on the CPU: for a
pi0.5 checkpoint that is ~100 s of `kaiming_uniform_`/`normal_` over ~3 B
parameters that are overwritten a moment later (the safetensors load itself is
~3 s). The server therefore constructs the policy under
`with torch.device(cfg.device):`, so the throwaway init runs on the GPU in ~2 s
and the policy's own `.to(device)` is a no-op. The loaded weights — and the
actions — are identical. The one cost is a transient construction peak of
roughly twice the model's footprint (fp32 before the bf16 cast: ~18 GB for
pi0.5, released straight after), so a 16 GB card would need the old path.

## First-inference latency (`torch.compile`)

Training configs commonly ship `compile_model: true` with
`compile_mode: max-autotune`, which lerobot applies to the sampling loop. The
autotuning happens inside the **first inference**, not at load. Measured on an
RTX 5090:

| Checkpoint | compile | First call | Warm call |
|---|---|---|---|
| `pi05_place_coke_can_in_bowl_a100` | on (config default) | ~16 s | ~73 ms |
| `pi05_place_coke_can_in_bowl_a100` | off | ~0.6 s | ~397 ms |
| `lerobot/pi0_libero_finetuned` | on (config default) | ~255 s | — |

So compilation buys a ~5x faster steady state and pays for itself after a few
dozen (pi0.5) to a few hundred (pi0) inferences — worth it for a real
evaluation, painful for a two-episode debug run. Hence a flag rather than a
hardcoded choice:

```bash
--policy.compile-model auto  # default: honour the checkpoint's config.json
--policy.compile-model off   # usable immediately; slower steady state
```

`LerobotPolicyClient`'s per-call ZMQ timeout is 15 min so the worst observed
compile can't trip it. The server prints a warning at startup when compilation
is enabled, so the long first call doesn't read as a hang.

### Keeping the compile cache across reboots

Inductor caches its autotune results and compiled kernels under
`/tmp/torchinductor_$USER`, which `systemd-tmpfiles` empties at boot — so the
first inference after a reboot pays the *full* autotune again (~6 min on an
RTX 5090 for the pi0.5 RTC + guidance graphs, versus tens of seconds when the
cache hits). `start_lerobot_server` points the child at
`~/.cache/policy_abstraction/torchinductor` instead unless
`TORCHINDUCTOR_CACHE_DIR` is already set; a hand-launched server wants the same:

```bash
TORCHINDUCTOR_CACHE_DIR=~/.cache/policy_abstraction/torchinductor \
PYTHONPATH=packages/policy_steering/src \
  third_party/physical-ai-sdk/.venv/bin/python scripts/servers/serve_lerobot.py ...
```

The cache is keyed by content hash, so an existing `/tmp/torchinductor_$USER`
can simply be copied into that directory to seed it.

What a cache hit still cannot skip is dynamo's tracing of the graph, which is
where the remaining first-call seconds go. torch 2.11's experimental
`TORCH_CACHING_PRECOMPILE=1` would cache that too, but it crashes on this
lerobot (`_PiGemmaDecoderLayerBase` is defined inside a function and cannot be
serialised), so it is not an option yet.

## Observation mapping

PA feeds every policy the same logical keys (`observation/image`,
`observation/wrist_image`, `observation/state`, `prompt`), while a LeRobot
checkpoint names its inputs after the dataset it was trained on. The server
resolves the pairing from the checkpoint's `input_features`:

| PA view | UR5e checkpoints | LIBERO checkpoints |
|---|---|---|
| `image` | `observation.images.exterior_image_1_left` | `observation.images.image` |
| `wrist_image` | `observation.images.wrist_image_left` | `observation.images.image2` |

The heuristic runs in two passes. First, a view whose name mentions `wrist`
claims a wrist-named camera feature, so feature declaration order can't hand
the wrist camera to the third-person view. Then the remaining views take
features in view-priority order, still preferring one whose name doesn't
contradict them — a lone third-person view won't land on a wrist camera while
an exterior one is free. Checkpoints with generic names (LIBERO's
`image`/`image2`) skip the first pass entirely and are matched positionally.

The resolved map is logged by both the server and the client after the first
inference — **check it whenever you point the backend at a new checkpoint**.
Pin it explicitly if the heuristic guesses wrong:

```bash
--policy.camera-map "image=observation.images.exterior_image_1_left,wrist_image=observation.images.wrist_image_left"
```

Each view and each camera feature may appear at most once; a repeat is
rejected rather than resolved silently, since the losing image would simply be
dropped and its camera padded as empty.

The state vector is passed through unchanged and its dimension must match the
checkpoint's `observation.state` feature. A mismatch (e.g. LIBERO's 8-D
`pos + axis-angle + gripper qpos` against a UR5e checkpoint's 7-D
`joints + gripper`) means the env and the checkpoint disagree about the robot,
so the server fails loudly rather than reinterpreting the numbers. Use
`--policy.state-adapter pad_or_truncate` only when you know the leading dims
genuinely line up.

## Real-Time Chunking (optional)

`--policy.rtc` enables lerobot's
[RTC](https://www.physicalintelligence.company/research/real_time_chunking):
each new chunk is inpainted to agree with the unexecuted tail of the previous
one. The server keeps that tail itself, using `--policy.replan-steps` to know
how much of the last chunk PA consumed, and passes `inference_delay=0` because
PA holds the simulator still while the policy runs. It is therefore *only*
buying chunk-boundary continuity here — the latency-hiding motivation from the
real-robot async loop doesn't apply. Off by default.

Related knobs: `--policy.rtc-execution-horizon`,
`--policy.rtc-max-guidance-weight`, `--policy.rtc-prefix-attention-schedule`
(`linear` / `exp` / `ones` / `zeros`).

Only `pi0`, `pi05` and `smolvla` read the RTC kwargs. `pi0_fast` accepts and
then ignores them — it would report as RTC-enabled while running completely
unchunked — so the server refuses `--rtc` for anything outside that list
rather than letting an A/B comparison return an artifact.

Between episodes `EpisodeRunner` calls `LerobotPolicyClient.reset()`, which
clears the server's previous-chunk buffer. Without it, episode N's action tail
would be blended into episode N+1's opening chunk. If that reset ever fails
while RTC is on, the client raises rather than continuing with a stale buffer.

## Guidance

Guided evaluation works the same way it does for the other backends — PA's
`__gp_*__` params ride in the request's `options` and steer the flow-matching
integration server-side, mirroring openpi's
[`Pi0Grounded.sample_actions`](../third_party/openpi/src/openpi/models/pi0grounded.py).
Nothing extra to pass; `--guidance.enable` (the default) is enough:

```bash
python scripts/evaluate.py \
  policy:lerobot --policy.model-path <checkpoint> \
  env:libero
```

### How the correction is injected

The GR00T backend carries a copy of its model's denoising loop
with a guidance line spliced in. This backend copies nothing. lerobot's
`sample_actions` already routes every step's velocity through
`self.rtc_processor.denoise_step(...)` whenever an `RTCConfig` is attached,
handing it the un-guided `original_denoise_step_partial` — so a
`GuidedRTCProcessor` subclass sees every step and returns a corrected `v_t`:

```
v_t = original_denoise_step_partial(x_t)         # the model's prediction
v_t = compute_guidance_velocity(..., x_t, v_t)   # PA's correction
x_t = x_t + dt * v_t                             # lerobot's Euler update
```

Consequences worth knowing:

- **One implementation covers pi0, pi0.5 and smolvla.** No per-model copy to
  keep in sync when lerobot changes.
- **Guidance and RTC compose.** The parent produces the (optionally
  RTC-inpainted) velocity and guidance is applied on top. Enabling guidance
  switches lerobot's `RTCConfig` on to expose the hook, but a previous chunk
  only steers the current one when you actually passed `--policy.rtc` —
  `info`'s `rtc_enabled` reports what you asked for, not the internal flag.
- **`pi0_fast` cannot be guided.** It has no denoising loop to hook, so the
  server refuses a guided request rather than answering it unguided. Pair it
  with `--guidance.no-enable`, or `--policy.guidance off`.

`--policy.guidance off` makes the server reject guided requests outright, which
pins an honest unguided baseline even if the rest of the pipeline is set up to
steer.

### Action space caveat

Guidance acts on the model's **internal, still-normalized** action tensor —
before the post-processor unnormalizes it. That matches what the GR00T
backend does, so `--guidance.factor` means the same thing across
backends, but it does mean the factor is in normalized units rather than
metres whenever the checkpoint's ACTION normalization isn't `IDENTITY`. The
server prints the mode at startup and reports it as `action_normalization` from
the `info` endpoint:

```
NOTE: ACTION normalization is QUANTILES, so guidance steers normalized actions
rather than metres. --guidance.factor is in those units and may need re-tuning
for this checkpoint.
```

Expect to re-tune `--guidance.factor` when moving between checkpoints with
different normalization statistics.

`--guidance.debug` works too: it zeroes the network velocity so the guidance
field can be inspected on its own, exactly as `debug_guidance` does in
`pi0grounded.py`.

## Known limitation: rollouts are not seed-reproducible

PA puts a per-inference `__rng_seed__` in the policy element, but this backend
drops it: lerobot's flow-matching sampler draws its noise from the global torch
RNG and exposes no per-call seed on `predict_action_chunk`. Two runs of the
same episode will therefore differ. The GR00T backend has the same property;
the openpi backend does not. Treat lerobot success rates as needing more
trials to separate than
openpi ones, and don't expect bit-identical replays.

## Reference

The upstream real-robot usage this integration was derived from lives in
`physical_ai_sdk/inference/rtc_inference_example.py` (UR5e async control loop
driven by `ActionQueue` + `LatencyTracker`).
