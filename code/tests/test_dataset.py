"""Smoke tests for `data/aevascenes_dataset.py`.

Skipped automatically if the in-tree dataset is not available, so the
test suite still runs in CI environments without 85 GB of data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ptv3_fmcw.data.aevascenes_dataset import (
    AevaScenesFrameDataset,
    list_sequences,
)

_DATA_ROOT = Path(__file__).resolve().parents[2] / "data" / "aevascenes_v0.2"
_HAS_DATA = _DATA_ROOT.exists() and any(_DATA_ROOT.glob("*/sequence.json"))


@pytest.mark.skipif(not _HAS_DATA, reason="AevaScenes data not available")
def test_dataset_record_schema():
    seqs = list_sequences(_DATA_ROOT)[:1]
    ds = AevaScenesFrameDataset(_DATA_ROOT, seqs)
    rec = ds[0]
    for key, dt, ndim in [
        ("coord", np.float32, 2),
        ("feat", np.float32, 2),
        ("v_radial", np.float32, 1),
        ("intensity", np.float32, 1),
        ("gt_vxy", np.float32, 2),
        ("gt_mask", np.bool_, 1),
        ("sem_class", np.int64, 1),
    ]:
        assert key in rec, key
        arr = rec[key]
        assert arr.dtype == dt, (key, arr.dtype)
        assert arr.ndim == ndim, (key, arr.ndim)
    n = rec["coord"].shape[0]
    assert rec["feat"].shape == (n, 5)
    assert rec["gt_vxy"].shape == (n, 2)
    assert rec["sequence_uuid"] == seqs[0]
    assert rec["lidar"] == "front_wide_lidar"


@pytest.mark.skipif(not _HAS_DATA, reason="AevaScenes data not available")
def test_static_points_have_zero_gt():
    """The dataset's per-point GT must be 0 for points outside any box.

    A weak-but-fast version of the dataset-level assertion.
    """
    seqs = list_sequences(_DATA_ROOT)[:1]
    ds = AevaScenesFrameDataset(_DATA_ROOT, seqs)
    rec = ds[0]
    # Static-class points (road) must have GT speed near 0.
    sem = rec["sem_class"]
    gt_speed = np.linalg.norm(rec["gt_vxy"], axis=1)
    static_mask = sem == 19  # road
    if static_mask.any():
        assert float(gt_speed[static_mask].mean()) < 0.5
