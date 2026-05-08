"""Unit tests for B0-B3 (spec §8.5)."""
from __future__ import annotations

import numpy as np

from ptv3_fmcw.eval.baselines import (
    B0_Zero,
    B1_DopplerOnly,
    B2_ClassMean,
    B3_DopplerPlusClassMean,
    fit_class_mean,
)


def _record(coord: np.ndarray, v_radial: np.ndarray, sem: np.ndarray) -> dict:
    n = coord.shape[0]
    feat = np.zeros((n, 5), dtype=np.float32)
    feat[:, :3] = coord
    feat[:, 4] = v_radial
    return {
        "coord": coord.astype(np.float32),
        "feat": feat,
        "v_radial": v_radial.astype(np.float32),
        "sem_class": sem.astype(np.int64),
    }


def test_b0_returns_zeros():
    rec = _record(np.array([[1.0, 0.0, 0.0]]), np.array([5.0]), np.array([1]))
    pred = B0_Zero().predict(rec)
    assert pred.shape == (1, 2)
    assert np.allclose(pred, 0.0)


def test_b1_pure_radial_along_x():
    """Point at (10, 0, 0) with v_radial=5: B1 predicts (5, 0)."""
    rec = _record(np.array([[10.0, 0.0, 0.0]]), np.array([5.0]), np.array([1]))
    pred = B1_DopplerOnly().predict(rec)
    assert np.allclose(pred, [[5.0, 0.0]], atol=1e-6)


def test_b1_diagonal():
    """Point at (3, 4, 0) with v_radial=5: r_hat=(0.6, 0.8); pred=5*r_hat=(3,4)."""
    rec = _record(np.array([[3.0, 4.0, 0.0]]), np.array([5.0]), np.array([1]))
    pred = B1_DopplerOnly().predict(rec)
    assert np.allclose(pred, [[3.0, 4.0]], atol=1e-6)


def test_b1_zero_radial():
    rec = _record(np.array([[10.0, 0.0, 0.0]]), np.array([0.0]), np.array([1]))
    pred = B1_DopplerOnly().predict(rec)
    assert np.allclose(pred, 0.0)


def test_b2_returns_class_mean():
    means = {5: np.array([2.0, 3.0])}
    rec = _record(
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 0.0]),
        np.array([5, 0]),
    )
    pred = B2_ClassMean(class_means=means).predict(rec)
    assert np.allclose(pred[0], [2.0, 3.0])
    assert np.allclose(pred[1], [0.0, 0.0])


def test_b3_recovers_radial_when_class_mean_zero():
    means = {1: np.array([0.0, 0.0])}
    rec = _record(np.array([[10.0, 0.0, 0.0]]), np.array([5.0]), np.array([1]))
    pred = B3_DopplerPlusClassMean(class_means=means).predict(rec)
    assert np.allclose(pred, [[5.0, 0.0]], atol=1e-6)


def test_b3_recovers_class_mean_tangent_when_radial_zero():
    """v_radial=0 + class_mean (0, 1) at coord (10, 0): tangential
    component of (0, 1) in LOS=(1, 0) is (0, 1) itself."""
    means = {1: np.array([0.0, 1.0])}
    rec = _record(np.array([[10.0, 0.0, 0.0]]), np.array([0.0]), np.array([1]))
    pred = B3_DopplerPlusClassMean(class_means=means).predict(rec)
    assert np.allclose(pred, [[0.0, 1.0]], atol=1e-6)


def test_b1_negative_radial_approaching():
    """Approaching target has negative v_radial; pred should point toward sensor."""
    rec = _record(np.array([[10.0, 0.0, 0.0]]), np.array([-5.0]), np.array([1]))
    pred = B1_DopplerOnly().predict(rec)
    assert np.allclose(pred, [[-5.0, 0.0]], atol=1e-6)


def test_fit_class_mean_static_class_is_zero():
    coord = np.array([[1.0, 0.0, 0.0]])
    rec_static = {
        "coord": coord, "feat": np.zeros((1, 5), np.float32),
        "v_radial": np.zeros(1, np.float32),
        "sem_class": np.array([19]),         # road
        "gt_vxy": np.array([[0.0, 0.0]], np.float32),
    }
    rec_car = {
        "coord": coord, "feat": np.zeros((1, 5), np.float32),
        "v_radial": np.zeros(1, np.float32),
        "sem_class": np.array([1]),
        "gt_vxy": np.array([[10.0, 0.0]], np.float32),
    }
    means = fit_class_mean([rec_static, rec_car])
    assert np.allclose(means[19], [0.0, 0.0])
    assert np.allclose(means[1], [10.0, 0.0])
