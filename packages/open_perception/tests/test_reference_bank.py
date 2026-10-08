from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from open_perception.encoders.reference import ReferenceBank


@pytest.fixture
def fake_ref_hdf5(tmp_path: Path) -> Path:
    path = tmp_path / "refs.hdf5"
    with h5py.File(path, "w") as f:
        for name in ["mug_1", "mug_2", "plate_3"]:
            grp = f.create_group(name)
            images = grp.create_group("images")
            for k in range(2):
                sub = images.create_group(str(k))
                vec = np.random.default_rng(hash(name + str(k)) % 2**32).normal(size=16)
                vec = (vec / np.linalg.norm(vec)).astype(np.float32)
                sub.create_dataset("clip_embedding", data=vec)
                sub.create_dataset("image", data=np.zeros((4, 4, 3), dtype=np.uint8))
    return path


def test_from_hdf5_normalizes_and_stacks(fake_ref_hdf5: Path):
    bank = ReferenceBank.from_hdf5(str(fake_ref_hdf5))
    # mug_1 and mug_2 both normalize to "mug".
    assert set(bank.names) == {"mug", "plate"}
    assert bank.embeddings["mug"].shape == (4, 16)  # 2 groups * 2 refs each
    norms = np.linalg.norm(bank.embeddings["mug"], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_from_dict_encodes_via_encoder():
    class _StubEncoder:
        def __init__(self):
            self.calls = 0

        def encode_images(self, imgs):
            self.calls += 1
            return (
                np.ones((len(imgs), 4), dtype=np.float32) / 2.0
            )  # not L2-normed; the bank accepts

    enc = _StubEncoder()
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    bank = ReferenceBank.from_dict(
        {"mug": img, "plate": [img, img]}, encoder=enc, space="clip"
    )
    assert enc.calls == 2
    assert bank.embeddings["mug"].shape == (1, 4)
    assert bank.embeddings["plate"].shape == (2, 4)


def test_from_any_dispatches_on_type(fake_ref_hdf5: Path):
    bank = ReferenceBank.from_any(str(fake_ref_hdf5), encoders={}, space="clip")
    assert isinstance(bank, ReferenceBank)
    roundtrip = ReferenceBank.from_any(bank, encoders={}, space="clip")
    assert roundtrip is bank
