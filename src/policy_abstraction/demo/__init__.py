"""Live demonstration of the guidance framework on the real cell.

One policy, loaded once; one long rollout at a time; and a switch that turns
guidance on and off *under the running policy*. The pieces:

* :mod:`policy_abstraction.demo.session` — :class:`DemoSession` owns the
  backends, the rollout thread, the guidance gate and the live snapshot the
  window paints from.
* :mod:`policy_abstraction.demo.overlay` — draws the perception masks and the
  distance pairs onto the exterior camera frame, in the pair colours the 3-D
  scene and the trend plot use.
* :mod:`policy_abstraction.demo.gui` — the PyQt6 window (imported on demand;
  everything else here stays importable without a display).

Launched by ``scripts/run_ur5_demo.sh``; see ``docs/demo.md``.
"""

from policy_abstraction.demo.session import (
    DemoSession,
    DemoState,
    LiveView,
    OverlayMode,
    ReferenceTrends,
    TrendSample,
)

__all__ = [
    "DemoSession",
    "DemoState",
    "LiveView",
    "OverlayMode",
    "ReferenceTrends",
    "TrendSample",
]
