"""Unit tests for `data/gt_velocity.py` (spec §6.3)."""
from __future__ import annotations

import numpy as np

from ptv3_fmcw.data.gt_velocity import (
    Box,
    per_point_gt_velocity,
    points_in_box,
    quat_to_matrix,
)


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)


def _box(
    cx=0.0, cy=0.0, cz=0.0,
    dx=2.0, dy=2.0, dz=2.0,
    vx=0.0, vy=0.0, vz=0.0,
    wx=0.0, wy=0.0, wz=0.0,
    quat=None, cls="car", cls_idx=1,
) -> Box:
    return Box(
        center=np.array([cx, cy, cz]),
        dimensions=np.array([dx, dy, dz]),
        rotation=quat if quat is not None else _IDENTITY_QUAT,
        linear_velocity=np.array([vx, vy, vz]),
        angular_velocity=np.array([wx, wy, wz]),
        cls=cls,
        cls_idx=cls_idx,
    )


def test_static_point_outside_all_boxes_gets_zero():
    pts = np.array([[100.0, 0.0, 0.0]])
    boxes = [_box(cx=0, cy=0, dx=1, dy=1, dz=1, vx=10.0)]
    gt, mask, inside = per_point_gt_velocity(pts, boxes)
    assert np.allclose(gt, 0.0)
    assert mask.all()
    assert inside[0] == -1


def test_point_in_pure_translation_box_matches_linear_velocity():
    pts = np.array([[0.5, 0.5, 0.5]])
    boxes = [_box(cx=0, cy=0, cz=0, dx=2, dy=2, dz=2, vx=3.0, vy=4.0, vz=5.0)]
    gt, _, inside = per_point_gt_velocity(pts, boxes)
    assert np.allclose(gt[0], (3.0, 4.0, 5.0))
    assert inside[0] == 0


def test_omega_term_pure_rotation():
    """Box at origin spinning with omega=(0,0,1) rad/s; point at (1,0,0)
    should have GT velocity (0, 1, 0)."""
    pts = np.array([[1.0, 0.0, 0.0]])
    boxes = [_box(dx=4, dy=4, dz=4, wz=1.0)]
    gt, _, inside = per_point_gt_velocity(pts, boxes)
    assert np.allclose(gt[0], (0.0, 1.0, 0.0), atol=1e-9)


def test_omega_term_translation_plus_rotation():
    """Combined linear + angular velocity reproduces the analytic formula."""
    p = np.array([[2.0, 0.0, 0.0]])
    boxes = [_box(dx=10, dy=10, dz=10, vx=1.0, vy=2.0, wz=3.0)]
    gt, _, _ = per_point_gt_velocity(p, boxes)
    # v_lin + omega x r = (1, 2, 0) + (0,0,3) x (2,0,0) = (1,2,0) + (0,6,0) = (1,8,0)
    assert np.allclose(gt[0], (1.0, 8.0, 0.0), atol=1e-9)


def test_smaller_box_wins_on_overlap():
    """When a point lies inside two overlapping boxes, the smaller box's
    velocity is assigned (spec §3.4)."""
    pts = np.array([[0.0, 0.0, 0.0]])
    big = _box(dx=10, dy=10, dz=10, vx=10.0, cls_idx=2)
    small = _box(dx=2, dy=2, dz=2, vx=1.0, cls_idx=1)
    gt, _, _ = per_point_gt_velocity(pts, [big, small])
    assert np.isclose(gt[0, 0], 1.0)


def test_points_in_obb_with_yaw_rotation():
    """Box yaw 90 deg around z: dimensions (4 along local x, 1 along
    local y) become (1 along world x, 4 along world y)."""
    half = np.sqrt(2) / 2
    quat = np.array([half, 0.0, 0.0, half])  # 90 deg about z
    box = _box(dx=4.0, dy=1.0, dz=1.0, quat=quat)
    p_inside = np.array([[0.0, 1.5, 0.0]])    # 1.5 m along local x via world y
    p_outside = np.array([[1.5, 0.0, 0.0]])
    assert points_in_box(p_inside, box).all()
    assert not points_in_box(p_outside, box).any()


def test_omega_term_rotation_about_y_axis():
    """Box spinning with omega=(0,1,0) rad/s; point at (0,0,1)
    should have GT velocity (1, 0, 0) via (0,1,0) x (0,0,1) = (1,0,0)."""
    pts = np.array([[0.0, 0.0, 1.0]])
    boxes = [_box(dx=4, dy=4, dz=4, wy=1.0)]
    gt, _, inside = per_point_gt_velocity(pts, boxes)
    assert np.allclose(gt[0], (1.0, 0.0, 0.0), atol=1e-9)


def test_quat_to_matrix_identity():
    R = quat_to_matrix(_IDENTITY_QUAT)
    assert np.allclose(R, np.eye(3), atol=1e-12)
