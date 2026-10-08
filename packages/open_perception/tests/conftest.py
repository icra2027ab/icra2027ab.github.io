"""Shared fixtures — synthetic images and stub model-backed services.

No GPU and no network access required. Stub model classes live in
`_stubs.py` so tests can import them directly.
"""

from __future__ import annotations

import os
import sys

# Make the tests directory importable as a plain package for `_stubs`.
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from open_perception.context import DetectionContext  # noqa: E402
from open_perception.encoders.reference import ReferenceBank  # noqa: E402
from open_perception.types import InferenceOptions, InferenceStats  # noqa: E402


@pytest.fixture
def synthetic_rgb() -> np.ndarray:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[4:12, 4:12] = (230, 60, 60)
    img[20:28, 20:28] = (60, 60, 230)
    return img


@pytest.fixture
def synthetic_depth() -> np.ndarray:
    return np.full((32, 32), 0.5, dtype=np.float32)


@pytest.fixture
def identity_intrinsics() -> np.ndarray:
    return np.array(
        [[20.0, 0.0, 16.0], [0.0, 20.0, 16.0], [0.0, 0.0, 1.0]], dtype=np.float32
    )


def _red_mask() -> np.ndarray:
    m = np.zeros((32, 32), dtype=bool)
    m[4:12, 4:12] = True
    return m


def _blue_mask() -> np.ndarray:
    m = np.zeros((32, 32), dtype=bool)
    m[20:28, 20:28] = True
    return m


@pytest.fixture
def red_mask() -> np.ndarray:
    return _red_mask()


@pytest.fixture
def blue_mask() -> np.ndarray:
    return _blue_mask()


@pytest.fixture
def synthetic_sam3_output() -> tuple[np.ndarray, np.ndarray]:
    red = _red_mask()
    blue = _blue_mask()
    small_enough = np.zeros((32, 32), dtype=bool)
    small_enough[14:20, 14:20] = True
    masks = np.stack([red, blue, small_enough], axis=0)
    scores = np.array([0.9, 0.8, 0.3], dtype=np.float32)
    return masks, scores


@pytest.fixture
def stub_clip_and_bank():
    from _stubs import StubClip

    mug_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    plate_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    other_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    crop_emb = np.stack([mug_vec, plate_vec, other_vec], axis=0)
    clip = StubClip(crop_emb, name_to_text={"mug": mug_vec, "plate": plate_vec})
    bank = ReferenceBank(
        embeddings={"mug": mug_vec[None, :], "plate": plate_vec[None, :]},
        space="clip",
    )
    return clip, bank


@pytest.fixture
def default_context(synthetic_rgb) -> DetectionContext:
    return DetectionContext(
        queries=["mug", "plate"],
        rgb=synthetic_rgb,
        depth=None,
        camera=None,
        options=InferenceOptions(match_threshold=0.5),
        stats=InferenceStats(),
    )
