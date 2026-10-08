"""Fake CoTracker model for tests that exercise CoTrackerStream logic.

The real cotracker model is loaded via ``torch.hub.load``, which requires
network + git access and downloads weights. Tests use this fake instead by
injecting a loader into ``CoTrackerStream(load_fn=...)`` (or by monkey-
patching ``_load_online_cotracker``).

The fake implements only the calling convention used by
``CoTrackerStream._run_model``:

  * exposes integer ``.step`` attribute,
  * returns ``(pred_tracks, pred_vis)`` from ``__call__`` when
    ``is_first_step=False``,
  * returns ``None`` (or anything; ignored) when ``is_first_step=True``.

Tracks are deterministic: each query's track at frame ``t`` is its initial
``(x, y)`` plus ``(0.5 * t, 0.25 * t)``. Visibility is always True.
"""

from __future__ import annotations

import numpy as np
import torch


class FakeOnlineCoTrackerModel:
    """Minimal stand-in for the real cotracker3_online predictor."""

    def __init__(self, step: int = 4) -> None:
        self.step = step
        self._queries: torch.Tensor | None = None
        self._n_seen: int = 0  # frames consumed across non-first-step calls

    # The real model is itself an ``nn.Module``-ish callable; we only need
    # the side-effect of recording queries on first_step and returning
    # cumulative tracks afterwards.
    def to(self, device: str) -> "FakeOnlineCoTrackerModel":
        self._device = device
        return self

    def __call__(
        self,
        *,
        video_chunk: torch.Tensor,
        is_first_step: bool,
        queries: torch.Tensor | None = None,
        grid_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if is_first_step:
            assert queries is not None, (
                "first-step call must include queries (grid_size unused in fake)"
            )
            assert queries.dim() == 3 and queries.shape[-1] == 3, (
                f"queries must be (1, N, 3), got {tuple(queries.shape)}"
            )
            self._queries = queries.detach().clone()
            # cotracker3_online returns nothing meaningful on first_step.
            return None
        assert self._queries is not None, "step() called before is_first_step"
        assert video_chunk.dim() == 5, (
            f"video_chunk must be (1, T, 3, H, W); got {tuple(video_chunk.shape)}"
        )
        T = int(video_chunk.shape[1])
        # Cotracker online accumulates: pred_tracks for ALL frames seen so far.
        # Our fake is simpler — it pretends the cumulative track length equals
        # T (the current window). Tests only exercise the latest-snapshot
        # path, so this is sufficient.
        self._n_seen += T
        N = int(self._queries.shape[1])
        seed_xy = self._queries[0, :, 1:3].cpu().numpy()  # (N, 2)
        # tracks: at frame t, position = seed + (0.5*t, 0.25*t)
        offsets = np.stack(
            [
                np.arange(T, dtype=np.float32) * 0.5,
                np.arange(T, dtype=np.float32) * 0.25,
            ],
            axis=1,
        )  # (T, 2)
        tracks_np = seed_xy[None, :, :] + offsets[:, None, :]  # (T, N, 2)
        vis_np = np.ones((T, N), dtype=np.float32)
        return (
            torch.from_numpy(tracks_np).unsqueeze(0),  # (1, T, N, 2)
            torch.from_numpy(vis_np).unsqueeze(0),  # (1, T, N)
        )


def fake_loader_factory(step: int = 4):
    """Build a ``load_fn`` for ``CoTrackerStream`` that returns the fake."""

    def _load(model_name: str, device: str) -> FakeOnlineCoTrackerModel:
        return FakeOnlineCoTrackerModel(step=step).to(device)

    return _load
