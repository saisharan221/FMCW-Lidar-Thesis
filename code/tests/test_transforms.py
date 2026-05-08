"""Augmentation invariants (spec §3.8)."""
from __future__ import annotations

import numpy as np

from ptv3_fmcw.data.transforms import (
    Compose,
    RandomFlip,
    RandomPointDropout,
    RandomRotateZ,
    RandomScale,
    default_train_transforms,
)


def _record(n: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    coord = rng.normal(size=(n, 3)).astype(np.float32)
    feat = np.zeros((n, 5), dtype=np.float32)
    feat[:, :3] = coord
    feat[:, 4] = rng.normal(size=n).astype(np.float32)
    return {
        "coord": coord,
        "feat": feat,
        "gt_vxy": rng.normal(size=(n, 2)).astype(np.float32),
        "gt_mask": np.ones(n, dtype=bool),
        "sem_class": np.zeros(n, dtype=np.int64),
    }


def test_rotate_preserves_norms():
    rec = _record()
    out = RandomRotateZ(rng=np.random.default_rng(0))(rec)
    assert np.allclose(np.linalg.norm(out["coord"], axis=1),
                       np.linalg.norm(rec["coord"], axis=1), atol=1e-4)
    assert np.allclose(np.linalg.norm(out["gt_vxy"], axis=1),
                       np.linalg.norm(rec["gt_vxy"], axis=1), atol=1e-4)


def test_flip_x_negates_x():
    rec = _record()
    out = RandomFlip(p_x=1.0, p_y=0.0, rng=np.random.default_rng(0))(rec)
    assert np.allclose(out["coord"][:, 0], -rec["coord"][:, 0])
    assert np.allclose(out["gt_vxy"][:, 0], -rec["gt_vxy"][:, 0])
    assert np.allclose(out["coord"][:, 1], rec["coord"][:, 1])


def test_scale_scales_coords_and_velocity():
    rec = _record()
    s = 1.05
    rng = np.random.default_rng(0)
    aug = RandomScale(low=s, high=s, rng=rng)
    out = aug(rec)
    assert np.allclose(out["coord"], rec["coord"] * s, atol=1e-5)
    assert np.allclose(out["gt_vxy"], rec["gt_vxy"] * s, atol=1e-5)


def test_dropout_drops_corresponding_rows():
    rec = _record(n=200)
    out = RandomPointDropout(low=0.3, high=0.3, rng=np.random.default_rng(0))(rec)
    assert out["coord"].shape[0] == out["gt_vxy"].shape[0] == out["gt_mask"].shape[0]
    assert out["coord"].shape[0] < rec["coord"].shape[0]


def test_default_pipeline_runs():
    rec = _record()
    out = default_train_transforms(seed=42)(rec)
    assert out["coord"].shape[1] == 3
    assert out["gt_vxy"].shape[1] == 2
