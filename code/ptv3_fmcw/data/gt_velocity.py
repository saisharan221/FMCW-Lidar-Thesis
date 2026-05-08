"""Per-point ground-truth velocity from AevaScenes 3D boxes.

Spec §3.4: for points inside box B,
    v_gt = B.linear_velocity + B.angular_velocity x (point - B.center)
For points outside any box, v_gt = 0 (spec §3.2: velocity field is
ego-motion compensated, so static scene has GT 0).

For overlapping boxes, the box with smaller volume wins (spec §3.4).

Coordinate frame is VEHICLE (rear-axle origin) throughout.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class Box:
    """A single 3D bounding box in the VEHICLE frame.

    `rotation` is a quaternion (w, x, y, z) following the AevaScenes
    `sequence.json` convention.
    """
    center: np.ndarray          # (3,) translation (m)
    dimensions: np.ndarray      # (3,) full extents along local x, y, z (m)
    rotation: np.ndarray        # (4,) quaternion (w, x, y, z)
    linear_velocity: np.ndarray   # (3,) m/s, VEHICLE frame
    angular_velocity: np.ndarray  # (3,) rad/s, VEHICLE frame
    track_id: str = ""
    cls: str = ""
    cls_idx: int = -1

    @property
    def volume(self) -> float:
        return float(np.prod(self.dimensions))


def quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quat
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def box_to_local(points: np.ndarray, box: Box) -> np.ndarray:
    """Transform points from VEHICLE frame to box-local frame.

    Box-local frame: origin at box center, axes aligned with box.
    """
    R = quat_to_matrix(box.rotation)
    return (points - box.center) @ R  # equivalent to R^T @ (p - c) per row


def points_in_box(points: np.ndarray, box: Box, eps: float = 1e-6) -> np.ndarray:
    """Boolean mask: which points lie inside the OBB.

    Edge points (within `eps` m of a face) count as inside.
    """
    local = box_to_local(points, box)
    half = 0.5 * box.dimensions + eps
    return np.all(np.abs(local) <= half, axis=1)


def _velocity_at(points: np.ndarray, box: Box) -> np.ndarray:
    """Rigid-body velocity at each point: v = v_lin + omega x (p - c).

    Computed in the VEHICLE frame (both v_lin and omega are stored in
    VEHICLE per AevaScenes convention).
    """
    r = points - box.center  # (N, 3)
    return box.linear_velocity + np.cross(box.angular_velocity, r)


def per_point_gt_velocity(
    points: np.ndarray,
    boxes: Sequence[Box],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-point GT velocity for a single frame.

    Returns
    -------
    gt_v : (N, 3) float32  — (vx, vy, vz) in VEHICLE frame
    gt_mask : (N,) bool   — True everywhere (always valid for this dataset)
    inside_box_id : (N,) int  — index of the assigned box, or -1 for background
    """
    points = np.asarray(points, dtype=np.float64)
    n = points.shape[0]
    gt_v = np.zeros((n, 3), dtype=np.float64)
    inside_box_id = np.full(n, -1, dtype=np.int64)

    if not boxes:
        return gt_v.astype(np.float32), np.ones(n, dtype=bool), inside_box_id

    # Order boxes from largest to smallest volume; smallest wins ties via overwrite.
    box_indices = sorted(range(len(boxes)), key=lambda i: -boxes[i].volume)

    for bi in box_indices:
        box = boxes[bi]
        mask = points_in_box(points, box)
        if not np.any(mask):
            continue
        gt_v[mask] = _velocity_at(points[mask], box)
        inside_box_id[mask] = bi

    return gt_v.astype(np.float32), np.ones(n, dtype=bool), inside_box_id
