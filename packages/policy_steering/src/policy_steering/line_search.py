"""JIT-compatible golden-section line search for guidance step size."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def golden_section_line_search(batch_loss_fn, x_t, direction, max_alpha=1.0, n_iter=20):
    """JIT-compatible golden-section line search for optimal step size.

    Finds alpha in [0, max_alpha] minimizing sum(batch_loss_fn(x_t + alpha * direction)).
    Uses pure JAX control flow (lax.fori_loop) so the entire search is compilable.

    Args:
        batch_loss_fn: (B, T, D) -> (B,) per-sample loss.
        x_t: current actions, shape (B, T, D).
        direction: unit-length descent direction, shape (B, T, D).
        max_alpha: initial upper bound for the search bracket.
        n_iter: number of golden-section iterations.

    Returns:
        Scalar alpha (optimal step size).
    """
    gr = (jnp.sqrt(5.0) + 1.0) / 2.0  # golden ratio

    def _total_loss(alpha):
        return jnp.sum(batch_loss_fn(x_t + alpha * direction))

    loss_at_zero = _total_loss(0.0)

    # Bracket expansion: double b up to 5 times while loss decreases
    def _expand_body(carry, _):
        a, b, loss_b = carry
        new_b = b * 2.0
        new_loss = _total_loss(new_b)
        should_expand = new_loss < _total_loss(b)
        b = jnp.where(should_expand, new_b, b)
        loss_b = jnp.where(should_expand, new_loss, loss_b)
        return (a, b, loss_b), None

    init_loss_b = _total_loss(max_alpha)
    (a, b, _), _ = jax.lax.scan(
        _expand_body, (0.0, max_alpha, init_loss_b), None, length=5
    )

    # Golden-section narrowing
    def _gs_body(_, carry):
        a, b = carry
        c = b - (b - a) / gr
        d = a + (b - a) / gr
        loss_c = _total_loss(c)
        loss_d = _total_loss(d)
        new_a = jnp.where(loss_c < loss_d, a, c)
        new_b = jnp.where(loss_c < loss_d, d, b)
        return (new_a, new_b)

    a, b = jax.lax.fori_loop(0, n_iter, _gs_body, (a, b))
    alpha = (a + b) / 2.0

    # Only step if it actually reduces loss
    alpha = jnp.where(_total_loss(alpha) < loss_at_zero, alpha, 0.0)
    return alpha
