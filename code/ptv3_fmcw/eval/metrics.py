"""Velocity metrics + bucketed breakdowns (spec §9, §13).

Pure-numpy. No torch import here so these can run from CPU eval scripts
on the local machine before any GPU stack is set up.

The breakdowns are accumulated frame-by-frame as running sums. Each
(metric, bucket) row stores `sum_error * count` and `count` so the
final aggregate can be expressed as a true point-weighted mean — i.e.
`mean_global = sum_all_frames(sum_err) / sum_all_frames(count)`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np


DYNAMIC_THRESHOLD: float = 0.5  # m/s, spec §5.1

# Bucket edges per spec §13. The final +inf is a sentinel for "rest".
RANGE_EDGES_M: tuple[float, ...] = (0.0, 25.0, 50.0, 75.0, np.inf)
SPEED_EDGES_MPS: tuple[float, ...] = (0.5, 5.0, 15.0, 25.0, np.inf)
RATIO_EDGES: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf)


def line_of_sight_xy(coord: np.ndarray) -> np.ndarray:
    """Per-point unit LOS vector in xy. Origin-axis points map to (0, 0)."""
    xy = coord[:, :2].astype(np.float64)
    norm = np.linalg.norm(xy, axis=1, keepdims=True)
    return np.where(norm > 1e-6, xy / np.clip(norm, 1e-6, None), 0.0)


def decompose_radial_tangential(
    v: np.ndarray, coord: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project v onto LOS-aligned (radial) and orthogonal (tangential) parts.

    Both returned arrays are 2D vectors in the xy plane.
    """
    los = line_of_sight_xy(coord)
    radial_scalar = np.einsum("nd,nd->n", v[:, :2], los)
    v_radial = radial_scalar[:, None] * los
    v_tangent = v[:, :2] - v_radial
    return v_radial, v_tangent


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-point angle (degrees) between pred and gt vectors.

    Returns NaN where either vector has near-zero magnitude (angle
    undefined). Caller is responsible for masking NaNs out before
    aggregating.
    """
    np_ = np.linalg.norm(pred[:, :2], axis=1)
    ng = np.linalg.norm(gt[:, :2], axis=1)
    valid = (np_ > 1e-6) & (ng > 1e-6)
    out = np.full(pred.shape[0], np.nan, dtype=np.float64)
    if valid.any():
        cos = (pred[valid, :2] * gt[valid, :2]).sum(axis=1) / (np_[valid] * ng[valid])
        cos = np.clip(cos, -1.0, 1.0)
        out[valid] = np.degrees(np.arccos(cos))
    return out


@dataclass
class _Acc:
    """Running accumulator: sum and count for a metric in a bucket."""
    s: float = 0.0
    n: int = 0

    def add(self, values: np.ndarray, mask: np.ndarray | None = None) -> None:
        v = values if mask is None else values[mask]
        v = v[~np.isnan(v)] if v.dtype.kind == "f" else v
        if v.size == 0:
            return
        self.s += float(v.sum())
        self.n += int(v.size)

    @property
    def mean(self) -> float:
        return self.s / self.n if self.n > 0 else float("nan")


@dataclass
class MetricAccumulator:
    """Frame-streaming accumulator for the full bucketed metric stack.

    Spec §9.2 and §13 buckets. Predictions and GT are 2D (vx, vy) m/s
    in VEHICLE frame. `coord` is (N, 3); only xy is used for LOS.
    """

    dynamic_threshold: float = DYNAMIC_THRESHOLD
    overall: dict[str, _Acc] = field(default_factory=dict)
    per_class: dict[int, dict[str, _Acc]] = field(default_factory=dict)
    per_range: dict[int, dict[str, _Acc]] = field(default_factory=dict)
    per_speed: dict[int, dict[str, _Acc]] = field(default_factory=dict)
    per_ratio: dict[int, dict[str, _Acc]] = field(default_factory=dict)
    per_scene: dict[str, dict[str, _Acc]] = field(default_factory=dict)

    def _row(self, store: dict, key) -> dict[str, _Acc]:
        if key not in store:
            store[key] = {
                "epe_all": _Acc(),
                "epe_dyn": _Acc(),
                "epe_bg": _Acc(),
                "epe_t": _Acc(),
                "epe_r": _Acc(),
                "ang_dyn": _Acc(),
                "mag_dyn": _Acc(),
                "n_pts": _Acc(),
                "n_dyn": _Acc(),
            }
        return store[key]

    @staticmethod
    def _bucket_indices(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
        """Return per-element bucket index. Right edge is exclusive."""
        idx = np.full(values.shape[0], -1, dtype=np.int64)
        for i in range(len(edges) - 1):
            mask = (values >= edges[i]) & (values < edges[i + 1])
            idx[mask] = i
        return idx

    def update(
        self,
        pred: np.ndarray,
        gt: np.ndarray,
        coord: np.ndarray,
        sem_class: np.ndarray,
        valid_mask: np.ndarray | None = None,
        scene_tag: str = "unknown",
    ) -> None:
        n = pred.shape[0]
        if valid_mask is None:
            valid_mask = np.ones(n, dtype=bool)

        diff = pred[:, :2] - gt[:, :2]
        epe_pp = np.linalg.norm(diff, axis=1)              # (N,)
        gt_speed = np.linalg.norm(gt[:, :2], axis=1)       # (N,)
        pred_speed = np.linalg.norm(pred[:, :2], axis=1)
        is_dyn = valid_mask & (gt_speed > self.dynamic_threshold)
        is_bg = valid_mask & ~(gt_speed > self.dynamic_threshold)
        v_in = valid_mask

        pred_r, pred_t = decompose_radial_tangential(pred, coord)
        gt_r, gt_t = decompose_radial_tangential(gt, coord)
        epe_t_pp = np.linalg.norm(pred_t - gt_t, axis=1)
        epe_r_pp = np.linalg.norm(pred_r - gt_r, axis=1)
        ang_pp = angular_error_deg(pred, gt)
        mag_pp = np.abs(pred_speed - gt_speed)

        ranges = np.linalg.norm(coord[:, :2], axis=1)
        gt_t_norm = np.linalg.norm(gt_t, axis=1)
        gt_r_norm = np.linalg.norm(gt_r, axis=1)
        ratio = np.where(gt_r_norm > 1e-3, gt_t_norm / np.clip(gt_r_norm, 1e-3, None), np.inf)
        ratio = np.where(gt_speed > self.dynamic_threshold, ratio, -1.0)

        range_idx = self._bucket_indices(ranges, RANGE_EDGES_M)
        speed_idx = self._bucket_indices(gt_speed, SPEED_EDGES_MPS)
        ratio_idx = self._bucket_indices(ratio, RATIO_EDGES)

        def add_to(store: dict, key) -> None:
            row = self._row(store, key)
            row["epe_all"].add(epe_pp, v_in)
            row["epe_dyn"].add(epe_pp, is_dyn)
            row["epe_bg"].add(epe_pp, is_bg)
            row["epe_t"].add(epe_t_pp, is_dyn)
            row["epe_r"].add(epe_r_pp, is_dyn)
            row["ang_dyn"].add(ang_pp, is_dyn)
            row["mag_dyn"].add(mag_pp, is_dyn)
            row["n_pts"].s += int(v_in.sum()); row["n_pts"].n += 1
            row["n_dyn"].s += int(is_dyn.sum()); row["n_dyn"].n += 1

        # Overall row.
        if not self.overall:
            self.overall = self._row({}, "_")
        row = self.overall
        row["epe_all"].add(epe_pp, v_in)
        row["epe_dyn"].add(epe_pp, is_dyn)
        row["epe_bg"].add(epe_pp, is_bg)
        row["epe_t"].add(epe_t_pp, is_dyn)
        row["epe_r"].add(epe_r_pp, is_dyn)
        row["ang_dyn"].add(ang_pp, is_dyn)
        row["mag_dyn"].add(mag_pp, is_dyn)
        row["n_pts"].s += int(v_in.sum()); row["n_pts"].n += 1
        row["n_dyn"].s += int(is_dyn.sum()); row["n_dyn"].n += 1

        # Per-bucket rows.
        for c in np.unique(sem_class):
            cls_mask = (sem_class == int(c)) & v_in
            if not cls_mask.any():
                continue
            row = self._row(self.per_class, int(c))
            row["epe_all"].add(epe_pp, cls_mask)
            row["epe_dyn"].add(epe_pp, cls_mask & is_dyn)
            row["epe_bg"].add(epe_pp, cls_mask & is_bg)
            row["epe_t"].add(epe_t_pp, cls_mask & is_dyn)
            row["epe_r"].add(epe_r_pp, cls_mask & is_dyn)
            row["ang_dyn"].add(ang_pp, cls_mask & is_dyn)
            row["mag_dyn"].add(mag_pp, cls_mask & is_dyn)
            row["n_pts"].s += int(cls_mask.sum()); row["n_pts"].n += 1
            row["n_dyn"].s += int((cls_mask & is_dyn).sum()); row["n_dyn"].n += 1

        for bucket in range(len(RANGE_EDGES_M) - 1):
            m = (range_idx == bucket) & v_in
            if not m.any():
                continue
            row = self._row(self.per_range, bucket)
            row["epe_all"].add(epe_pp, m)
            row["epe_dyn"].add(epe_pp, m & is_dyn)
            row["epe_bg"].add(epe_pp, m & is_bg)
            row["epe_t"].add(epe_t_pp, m & is_dyn)
            row["epe_r"].add(epe_r_pp, m & is_dyn)
            row["ang_dyn"].add(ang_pp, m & is_dyn)
            row["mag_dyn"].add(mag_pp, m & is_dyn)
            row["n_pts"].s += int(m.sum()); row["n_pts"].n += 1
            row["n_dyn"].s += int((m & is_dyn).sum()); row["n_dyn"].n += 1

        for bucket in range(len(SPEED_EDGES_MPS) - 1):
            m = (speed_idx == bucket) & is_dyn
            if not m.any():
                continue
            row = self._row(self.per_speed, bucket)
            row["epe_dyn"].add(epe_pp, m)
            row["epe_t"].add(epe_t_pp, m)
            row["epe_r"].add(epe_r_pp, m)
            row["ang_dyn"].add(ang_pp, m)
            row["mag_dyn"].add(mag_pp, m)
            row["n_dyn"].s += int(m.sum()); row["n_dyn"].n += 1

        for bucket in range(len(RATIO_EDGES) - 1):
            m = (ratio_idx == bucket)
            if not m.any():
                continue
            row = self._row(self.per_ratio, bucket)
            row["epe_dyn"].add(epe_pp, m)
            row["epe_t"].add(epe_t_pp, m)
            row["epe_r"].add(epe_r_pp, m)
            row["ang_dyn"].add(ang_pp, m)
            row["n_dyn"].s += int(m.sum()); row["n_dyn"].n += 1

        row = self._row(self.per_scene, scene_tag)
        row["epe_all"].add(epe_pp, v_in)
        row["epe_dyn"].add(epe_pp, is_dyn)
        row["epe_bg"].add(epe_pp, is_bg)
        row["epe_t"].add(epe_t_pp, is_dyn)
        row["epe_r"].add(epe_r_pp, is_dyn)
        row["ang_dyn"].add(ang_pp, is_dyn)
        row["mag_dyn"].add(mag_pp, is_dyn)
        row["n_pts"].s += int(v_in.sum()); row["n_pts"].n += 1
        row["n_dyn"].s += int(is_dyn.sum()); row["n_dyn"].n += 1

    @staticmethod
    def _row_to_dict(row: dict[str, _Acc]) -> dict[str, float]:
        return {
            "epe_all": row["epe_all"].mean,
            "epe_dyn": row["epe_dyn"].mean,
            "epe_bg": row["epe_bg"].mean,
            "epe_t": row["epe_t"].mean,
            "epe_r": row["epe_r"].mean,
            "ang_dyn_deg": row["ang_dyn"].mean,
            "mag_err_dyn": row["mag_dyn"].mean,
            "n_pts_total": int(row["n_pts"].s),
            "n_dyn_total": int(row["n_dyn"].s),
        }

    @staticmethod
    def _bucket_label(edges: tuple[float, ...], idx: int) -> str:
        lo, hi = edges[idx], edges[idx + 1]
        hi_str = "inf" if not np.isfinite(hi) else f"{hi:g}"
        return f"{lo:g}-{hi_str}"

    def as_dict(self, class_idx_to_name: Mapping[int, str] | None = None) -> dict:
        out: dict = {"overall": self._row_to_dict(self.overall) if self.overall else {}}
        cls_block: dict[str, dict[str, float]] = {}
        for cidx, row in sorted(self.per_class.items()):
            name = (class_idx_to_name or {}).get(cidx, f"class_{cidx}")
            cls_block[name] = self._row_to_dict(row)
        out["per_class"] = cls_block
        out["per_range_m"] = {
            self._bucket_label(RANGE_EDGES_M, i): self._row_to_dict(r)
            for i, r in sorted(self.per_range.items())
        }
        out["per_speed_mps"] = {
            self._bucket_label(SPEED_EDGES_MPS, i): self._row_to_dict(r)
            for i, r in sorted(self.per_speed.items())
        }
        out["per_tangential_ratio"] = {
            self._bucket_label(RATIO_EDGES, i): self._row_to_dict(r)
            for i, r in sorted(self.per_ratio.items())
        }
        out["per_scene"] = {
            tag: self._row_to_dict(r) for tag, r in sorted(self.per_scene.items())
        }
        return out


# Convenience for unit tests: single-frame, single-row metric report.
def epe_breakdown_single(
    pred: np.ndarray,
    gt: np.ndarray,
    coord: np.ndarray,
    sem_class: np.ndarray | None = None,
) -> dict[str, float]:
    if sem_class is None:
        sem_class = np.zeros(pred.shape[0], dtype=np.int64)
    acc = MetricAccumulator()
    acc.update(pred, gt, coord, sem_class)
    return acc._row_to_dict(acc.overall)
