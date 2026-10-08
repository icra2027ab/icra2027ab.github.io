"""PyTorch port of ``line_search.py``.

The line search only evaluates the (already differentiable) batch_loss_fn —
no autograd is needed inside the search itself.

Golden-section search is sequential by nature: which two points the next
iteration probes depends on a comparison at the current one. Run literally,
that is ~53 tiny forward passes per denoising step, each followed by a host
sync to read the loss, and on the UR5e joint-space path (unnormalize + forward
kinematics per chunk) that was ~280 ms of a ~285 ms guided step — 97% of the
guidance cost.

:func:`golden_section_line_search` therefore evaluates *speculatively*: for a
block of ``spec_depth`` iterations it enumerates every point the search could
visit (a binary tree, ``2 + 4 + ... + 2**spec_depth`` alphas), scores them
all in one batched forward pass, and then walks the tree on the host with the
same comparisons the sequential search makes. The bracket expansion probes
all six doublings in one pass the same way. The loss function is applied to a
``(K * B, T, D)`` stack, so this needs nothing beyond the per-sample batching
``build_guidance_fns`` already provides. Six forward passes and six host
syncs per search instead of ~53, and the alphas are the ones the sequential
search returns: every probed alpha is formed with the same double arithmetic,
and the product ``alpha * direction`` is rounded to ``x_t``'s dtype before the
add exactly as the scalar expression ``x_t + alpha * direction`` rounds it.
:func:`golden_section_line_search_sequential` is kept as the reference the
tests compare against.
"""

from __future__ import annotations

import math
import os

import torch

_GR = (math.sqrt(5.0) + 1.0) / 2.0
# Doublings the bracket expansion may apply to ``max_alpha``.
_N_EXPAND = 5


def _batched_total_losses(
    batch_loss_fn,
    x_t: torch.Tensor,
    direction: torch.Tensor,
    alphas: list[float],
) -> list[float]:
    """``sum(batch_loss_fn(x_t + alpha * direction))`` for every alpha, in one pass."""
    K = len(alphas)
    B = x_t.shape[0]
    a = torch.tensor(alphas, dtype=torch.float32, device=x_t.device)
    # ``x_t + alpha * direction`` with a Python-float alpha multiplies in
    # float32 and rounds the product to the tensor dtype before adding.
    # Reproduce that rounding so the batched probes score the same points.
    step = (direction.unsqueeze(0).float() * a.view(K, *([1] * direction.ndim))).to(
        x_t.dtype
    )
    xs = (x_t.unsqueeze(0) + step).reshape(K * B, *x_t.shape[1:])
    with torch.no_grad():
        per_sample = batch_loss_fn(xs)
    return per_sample.reshape(K, B).sum(dim=1).tolist()


def golden_section_line_search(
    batch_loss_fn,
    x_t: torch.Tensor,
    direction: torch.Tensor,
    max_alpha: float = 1.0,
    n_iter: int = 20,
    probe_log: list | None = None,
    *,
    spec_depth: int = 5,
) -> torch.Tensor:
    """Golden-section line search for optimal step size, in PyTorch.

    Mirrors ``policy_steering.line_search.golden_section_line_search`` and
    returns the alpha :func:`golden_section_line_search_sequential` returns,
    using batched speculative evaluation (see the module docstring).

    Args:
        batch_loss_fn: ``(B, T, D) -> (B,)`` per-sample loss. Must accept any
            leading batch size.
        x_t: current actions, shape ``(B, T, D)``.
        direction: descent direction, shape ``(B, T, D)``.
        max_alpha: initial upper bound for the search bracket.
        n_iter: number of golden-section narrowing iterations.
        probe_log: if given, ``(alpha, loss)`` is appended for every point the
            *walk* visits, in the sequential search's order. Speculative
            points the walk never reaches are not logged.
        spec_depth: narrowing iterations scored per forward pass. The pass
            evaluates ``2 ** (spec_depth + 1) - 2`` alphas (62 at the default
            5), so raising it trades batch size for fewer passes.

    Returns:
        Scalar tensor ``alpha`` (optimal step size).
    """
    if spec_depth < 1:
        raise ValueError(f"spec_depth must be >= 1; got {spec_depth}")

    def _log(alpha: float, loss: float) -> None:
        if probe_log is not None:
            probe_log.append((float(alpha), float(loss)))

    # Bracket expansion: every doubling the sequential search could accept,
    # plus alpha = 0 for the final do-no-harm comparison, in one pass.
    # Doubling is exact in floating point, so ``max_alpha * 2**j`` is the
    # value the sequential ``b * 2.0`` chain produces.
    b0 = float(max_alpha)
    candidates = [0.0] + [b0 * float(2**j) for j in range(_N_EXPAND + 1)]
    losses = _batched_total_losses(batch_loss_fn, x_t, direction, candidates)
    loss_at_zero = losses[0]
    _log(0.0, loss_at_zero)
    expand_losses = losses[1:]
    j = 0
    _log(candidates[1], expand_losses[0])
    for _ in range(_N_EXPAND):
        # The sequential loop compares loss(2b) against loss(b) and keeps b
        # after a rejection, so it re-probes the same pair and rejects again.
        _log(candidates[2 + j], expand_losses[j + 1])
        if expand_losses[j + 1] < expand_losses[j]:
            j += 1
    a = 0.0
    b = candidates[1 + j]

    # Golden-section narrowing, ``spec_depth`` iterations per forward pass.
    done = 0
    while done < n_iter:
        depth = min(spec_depth, n_iter - done)
        # Enumerate the tree of (c, d) probes reachable in ``depth`` steps.
        # Level ``lvl`` holds ``2 ** lvl`` nodes; node ``i``'s children are
        # ``2i`` (took ``b = d``) and ``2i + 1`` (took ``a = c``).
        levels: list[list[float]] = []
        nodes = [(a, b)]
        for _ in range(depth):
            probes: list[float] = []
            children: list[tuple[float, float]] = []
            for aa, bb in nodes:
                c = bb - (bb - aa) / _GR
                d = aa + (bb - aa) / _GR
                probes.extend((c, d))
                children.append((aa, d))
                children.append((c, bb))
            levels.append(probes)
            nodes = children
        flat = [p for probes in levels for p in probes]
        flat_losses = _batched_total_losses(batch_loss_fn, x_t, direction, flat)

        # Walk the tree with the sequential search's comparisons.
        offset = 0
        node = 0
        for probes in levels:
            c, d = probes[2 * node], probes[2 * node + 1]
            loss_c = flat_losses[offset + 2 * node]
            loss_d = flat_losses[offset + 2 * node + 1]
            _log(c, loss_c)
            _log(d, loss_d)
            if loss_c < loss_d:
                b = d
                node = 2 * node
            else:
                a = c
                node = 2 * node + 1
            offset += len(probes)
        done += depth

    alpha = (a + b) / 2.0
    loss_at_alpha = _batched_total_losses(batch_loss_fn, x_t, direction, [alpha])[0]
    _log(alpha, loss_at_alpha)
    discarded = loss_at_alpha >= loss_at_zero
    if discarded:
        alpha = 0.0
    # Opt-in diagnostic: one line per line-search call so we can measure how
    # often the "do no harm" safeguard zeroes a step. Set PA_GUIDANCE_DEBUG=1.
    if os.environ.get("PA_GUIDANCE_DEBUG"):
        print(f"[gsls] alpha={alpha:.5f} discarded={int(discarded)}", flush=True)
    return torch.tensor(alpha, dtype=x_t.dtype, device=x_t.device)


def golden_section_line_search_sequential(
    batch_loss_fn,
    x_t: torch.Tensor,
    direction: torch.Tensor,
    max_alpha: float = 1.0,
    n_iter: int = 20,
    probe_log: list | None = None,
) -> torch.Tensor:
    """Reference implementation: one forward pass and one host sync per probe.

    This is the search as originally written. :func:`golden_section_line_search`
    returns the same alpha; keep this one for the equivalence tests and as the
    readable statement of the algorithm.
    """

    def _total_loss(alpha: float) -> torch.Tensor:
        with torch.no_grad():
            loss = torch.sum(batch_loss_fn(x_t + alpha * direction))
        if probe_log is not None:
            probe_log.append((float(alpha), float(loss.item())))
        return loss

    loss_at_zero = _total_loss(0.0)

    # Bracket expansion: double b up to 5 times while loss decreases.
    a = 0.0
    b = float(max_alpha)
    loss_b = _total_loss(b)
    for _ in range(_N_EXPAND):
        new_b = b * 2.0
        new_loss = _total_loss(new_b)
        if new_loss.item() < _total_loss(b).item():
            b = new_b
            loss_b = new_loss
    del loss_b

    # Golden-section narrowing.
    for _ in range(n_iter):
        c = b - (b - a) / _GR
        d = a + (b - a) / _GR
        loss_c = _total_loss(c)
        loss_d = _total_loss(d)
        if loss_c.item() < loss_d.item():
            b = d
        else:
            a = c

    alpha = (a + b) / 2.0
    discarded = _total_loss(alpha).item() >= loss_at_zero.item()
    if discarded:
        alpha = 0.0
    return torch.tensor(alpha, dtype=x_t.dtype, device=x_t.device)
