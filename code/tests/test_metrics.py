"""Unit tests for `eval/metrics.py` (spec §9.3).

The four synthetic cases from the parent PTv3 spec §6.5, plus a few
boundary cases.
"""
from __future__ import annotations

import numpy as np

from ptv3_fmcw.eval.metrics import (
    MetricAccumulator,
    decompose_radial_tangential,
    epe_breakdown_single,
    line_of_sight_xy,
)


def _coord_at(x: float, y: float, z: float = 0.0, n: int = 1) -> np.ndarray:
    return np.tile([[x, y, z]], (n, 1)).astype(np.float32)


def _v(vx: float, vy: float, n: int = 1) -> np.ndarray:
    return np.tile([[vx, vy, 0.0]], (n, 1)).astype(np.float32)


def test_pred_equals_gt_zero_error():
    coord = _coord_at(10.0, 0.0)
    gt = _v(2.0, 3.0)
    out = epe_breakdown_single(pred=gt, gt=gt, coord=coord)
    assert np.isclose(out["epe_all"], 0.0)
    assert np.isclose(out["epe_t"], 0.0)
    assert np.isclose(out["epe_r"], 0.0)


def test_zero_pred_unit_gt_norm_5():
    coord = _coord_at(10.0, 0.0)
    gt = _v(3.0, 4.0)
    pred = _v(0.0, 0.0)
    out = epe_breakdown_single(pred=pred, gt=gt, coord=coord)
    assert np.isclose(out["epe_all"], 5.0)


def test_pure_radial_match():
    """Spec §6.5 case 3: pure radial GT, pred = radial -> EPE_t=0, EPE_r=0."""
    coord = _coord_at(10.0, 0.0)
    gt = _v(1.0, 0.0)        # along +x = LOS at coord (10, 0)
    pred = _v(1.0, 0.0)
    out = epe_breakdown_single(pred=pred, gt=gt, coord=coord)
    assert np.isclose(out["epe_t"], 0.0)
    assert np.isclose(out["epe_r"], 0.0)


def test_pure_tangential_zero_pred():
    """Spec §6.5 case 4: pure tangential GT, pred=0 -> EPE_t=||gt||, EPE_r=0."""
    coord = _coord_at(10.0, 0.0)
    gt = _v(0.0, 1.0)        # perpendicular to LOS
    pred = _v(0.0, 0.0)
    out = epe_breakdown_single(pred=pred, gt=gt, coord=coord)
    assert np.isclose(out["epe_t"], 1.0)
    assert np.isclose(out["epe_r"], 0.0)


def test_decompose_origin_safe():
    """A point at the sensor origin doesn't crash decomposition."""
    coord = np.zeros((1, 3), dtype=np.float32)
    v = np.array([[1.0, 1.0, 0.0]], dtype=np.float32)
    v_r, v_t = decompose_radial_tangential(v, coord)
    assert np.isfinite(v_r).all()
    assert np.isfinite(v_t).all()


def test_decompose_z_independence():
    """Non-zero z does not affect xy decomposition (LOS is xy-only)."""
    coord_no_z = _coord_at(10.0, 0.0, 0.0)
    coord_z = _coord_at(10.0, 0.0, 5.0)
    v = _v(2.0, 3.0)
    r1, t1 = decompose_radial_tangential(v, coord_no_z)
    r2, t2 = decompose_radial_tangential(v, coord_z)
    assert np.allclose(r1, r2)
    assert np.allclose(t1, t2)


def test_los_unit_norm():
    coord = np.array([[3.0, 4.0, 1.0], [10.0, 0.0, 0.0]], dtype=np.float32)
    los = line_of_sight_xy(coord)
    norms = np.linalg.norm(los, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_pythagorean_epe_decomposition():
    """EPE_all² ≈ EPE_r² + EPE_t² for arbitrary mixed GT/pred (orthogonality)."""
    coord = _coord_at(3.0, 4.0)          # LOS = (0.6, 0.8)
    gt   = _v(5.0, 2.0)
    pred = _v(1.0, 3.0)
    out = epe_breakdown_single(pred=pred, gt=gt, coord=coord)
    assert np.isclose(out["epe_all"] ** 2, out["epe_r"] ** 2 + out["epe_t"] ** 2, atol=1e-5)


def test_per_class_breakdown():
    coord = np.array([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]], dtype=np.float32)
    gt = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=np.float32)
    pred = np.array([[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    sem = np.array([1, 9], dtype=np.int64)  # car, pedestrian
    acc = MetricAccumulator()
    acc.update(pred=pred, gt=gt, coord=coord, sem_class=sem)
    out = acc.as_dict(class_idx_to_name={1: "car", 9: "pedestrian"})
    assert np.isclose(out["per_class"]["car"]["epe_all"], 0.0)
    assert np.isclose(out["per_class"]["pedestrian"]["epe_all"], 3.0)
