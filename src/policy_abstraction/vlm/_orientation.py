"""VLM-facing image orientation helper.

The PA pipeline standardises on ``CameraSpec.image_transform="flip_vh"``
(= ``img[::-1, ::-1]``, a 180° rotation), which is what policies, the
Rerun viewer, and the projection math in ``utils/pcd.py`` expect. That
180° rotation also mirrors columns relative to the natural scene, so a
VLM that uses image-frame left/right reasoning will systematically
disagree with task wording like "put the white mug on the *left*
plate" (see ``tmp/vlm_orientation_diagnostic.py`` for the empirical
check on libero_10 task04).

Solution: convert from PA-reference to a vflip-only (scene-consistent)
view at the moment the image leaves the numpy domain and enters an
OpenAI API call. A single column-flip undoes the hflip half of
``flip_vh``, yielding the upright, non-mirrored view a human in front
of the simulator window sees.

Apply :func:`to_vlm_orientation` from each VLM client's
``_encode_image`` helper, not at the call sites that build observation
dicts — keeping the conversion right at the network boundary means
ad-hoc callers cannot bypass it.
"""

from __future__ import annotations

import numpy as np


def to_vlm_orientation(img: np.ndarray) -> np.ndarray:
    """Convert a PA-reference (180°-rotated) frame to the VLM reference.

    Concretely: column-flip the input. Stacked on the upstream
    ``flip_vh`` transform this yields a vflip-only orientation where
    image-left == scene-left.

    Args:
        img: ``(H, W, ...)`` numpy array (uint8 RGB, uint8 BGR, seg-id
             mask, etc.). Any dtype/channel arrangement is fine — only
             the spatial axes are touched.

    Returns:
        ``(H, W, ...)`` contiguous array of the same shape and dtype.
    """
    return np.ascontiguousarray(img[:, ::-1])


__all__ = ["to_vlm_orientation"]
