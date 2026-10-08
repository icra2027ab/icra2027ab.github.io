"""
Monotonic Trajectory Alignment (MTA).

Aligns the current step counter of a live policy rollout with the timestep axis
of the pre-computed abstracted demonstration so that guidance gradients always
reference the correct phase of the reference trajectory.

Primary classes / functions (to be implemented):
  - MonotonicAligner       – online DTW / monotone-constrained alignment
"""


class MonotonicAligner:
    """Aligns live rollout step to reference demo timestep via monotone DTW."""

    def __init__(self, reference_length: int):
        self.reference_length = reference_length
        self._step = 0

    def step(self) -> int:
        """Return the aligned reference timestep for the current live step."""
        raise NotImplementedError("MonotonicAligner.step() not yet implemented.")

    def reset(self):
        self._step = 0
