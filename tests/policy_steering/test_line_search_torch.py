"""The batched golden-section search returns the sequential search's alpha."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from policy_steering.line_search_torch import (
    golden_section_line_search,
    golden_section_line_search_sequential,
)


def _quadratic_loss(target: torch.Tensor):
    """Per-sample ``0.5 * ||x - target||^2`` — minimum at a known alpha."""

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.sum((x - target) ** 2, dim=(-2, -1))

    return loss_fn


def _rastrigin_like_loss(target: torch.Tensor):
    """Non-convex per-sample loss so the bracket expansion and narrowing branch."""

    def loss_fn(x: torch.Tensor) -> torch.Tensor:
        diff = x - target
        return torch.sum(diff**2 + 0.3 * torch.cos(6.0 * diff), dim=(-2, -1))

    return loss_fn


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("max_alpha", [0.05, 1.0, 4.0])
@pytest.mark.parametrize("n_iter", [20, 7, 1])
@pytest.mark.parametrize("batch", [1, 3])
def test_batched_matches_sequential(dtype, max_alpha, n_iter, batch):
    gen = torch.Generator().manual_seed(int(max_alpha * 100) + n_iter + batch)
    x = torch.randn(batch, 8, 7, generator=gen).to(dtype)
    # One target shared by every sample, as the guidance loss's parameters are.
    target = torch.randn(1, 8, 7, generator=gen).to(dtype)
    d = torch.randn(batch, 8, 7, generator=gen)
    d = (d / d.norm()).to(dtype)
    for loss_fn in (_quadratic_loss(target), _rastrigin_like_loss(target)):
        log_seq: list = []
        log_bat: list = []
        a_seq = golden_section_line_search_sequential(
            loss_fn, x, d, max_alpha=max_alpha, n_iter=n_iter, probe_log=log_seq
        )
        a_bat = golden_section_line_search(
            loss_fn, x, d, max_alpha=max_alpha, n_iter=n_iter, probe_log=log_bat
        )
        assert a_bat.dtype == a_seq.dtype
        assert float(a_bat) == float(a_seq)
        # The walk visits the sequential search's points in its order; the
        # sequential expansion re-probes ``b`` after every rejection, which the
        # batched log leaves out, so compare the sets of visited alphas.
        assert {a for a, _ in log_bat} == {a for a, _ in log_seq}


def test_finds_quadratic_minimum():
    x = torch.zeros(1, 4, 3)
    d = torch.zeros(1, 4, 3)
    d[0, 0, 0] = 1.0
    target = torch.zeros(1, 4, 3)
    target[0, 0, 0] = 0.37
    alpha = golden_section_line_search(_quadratic_loss(target), x, d, max_alpha=1.0)
    assert abs(float(alpha) - 0.37) < 1e-4


def test_bracket_expands_to_reach_far_minimum():
    x = torch.zeros(1, 4, 3)
    d = torch.zeros(1, 4, 3)
    d[0, 0, 0] = 1.0
    target = torch.zeros(1, 4, 3)
    # Beyond max_alpha. The expansion doubles while the loss keeps falling:
    # 1 -> 2 -> 4 -> 8 (loss at 16 is higher), so the bracket is [0, 8].
    target[0, 0, 0] = 7.0
    alpha = golden_section_line_search(_quadratic_loss(target), x, d, max_alpha=1.0)
    assert abs(float(alpha) - 7.0) < 1e-3


def test_do_no_harm_zeroes_ascent_direction():
    x = torch.zeros(1, 4, 3)
    d = torch.zeros(1, 4, 3)
    d[0, 0, 0] = -1.0  # walks away from the target
    target = torch.zeros(1, 4, 3)
    target[0, 0, 0] = 0.5
    alpha = golden_section_line_search(_quadratic_loss(target), x, d, max_alpha=1.0)
    assert float(alpha) == 0.0


def test_spec_depth_does_not_change_result():
    gen = torch.Generator().manual_seed(3)
    x = torch.randn(2, 6, 7, generator=gen)
    target = torch.randn(1, 6, 7, generator=gen)
    d = torch.randn(2, 6, 7, generator=gen)
    loss_fn = _rastrigin_like_loss(target)
    ref = float(golden_section_line_search_sequential(loss_fn, x, d))
    for depth in (1, 2, 3, 5, 20):
        got = float(golden_section_line_search(loss_fn, x, d, spec_depth=depth))
        assert got == ref, depth


def test_forward_pass_count():
    calls: list[int] = []

    def counting_loss(x: torch.Tensor) -> torch.Tensor:
        calls.append(x.shape[0])
        return torch.sum(x**2, dim=(-2, -1))

    x = torch.ones(1, 4, 3)
    golden_section_line_search(counting_loss, x, -x, max_alpha=1.0, n_iter=20)
    # expansion (7 alphas), four blocks of 5 iterations (62 alphas), final check
    assert calls == [7, 62, 62, 62, 62, 1]
    calls.clear()
    golden_section_line_search_sequential(counting_loss, x, -x, max_alpha=1.0)
    assert len(calls) == 1 + 1 + 2 * 5 + 2 * 20 + 1
    assert set(calls) == {1}


def test_probe_log_alphas_are_floats():
    x = torch.zeros(1, 2, 2)
    d = torch.ones(1, 2, 2)
    log: list = []
    golden_section_line_search(_quadratic_loss(x + 0.1), x, d, probe_log=log, n_iter=3)
    assert log and all(isinstance(a, float) and isinstance(v, float) for a, v in log)
    assert np.isfinite([v for _, v in log]).all()
