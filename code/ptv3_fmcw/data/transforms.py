"""Training-time augmentations (spec §3.8).

Operate on numpy arrays inside a Pointcept-compatible record dict.
Per spec §3.8 we deliberately do not perturb `v_radial` or `intensity`
— corrupting a measured channel that the model is meant to consume is
a separate research question.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class RandomRotateZ:
    def __init__(self, rng: np.random.Generator | None = None):
        self.rng = rng or np.random.default_rng()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        theta = float(self.rng.uniform(-np.pi, np.pi))
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        out = dict(data)
        out["coord"] = data["coord"] @ R.T
        R2 = np.array([[c, -s], [s, c]], dtype=np.float32)
        out["gt_vxy"] = data["gt_vxy"] @ R2.T
        if "feat" in data and data["feat"].shape[1] >= 3:
            feat = data["feat"].copy()
            feat[:, :3] = feat[:, :3] @ R.T
            out["feat"] = feat
        return out


class RandomFlip:
    def __init__(self, p_x: float = 0.5, p_y: float = 0.5,
                 rng: np.random.Generator | None = None):
        self.p_x, self.p_y = p_x, p_y
        self.rng = rng or np.random.default_rng()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        flip_x = self.rng.random() < self.p_x
        flip_y = self.rng.random() < self.p_y
        if not (flip_x or flip_y):
            return data
        out = dict(data)
        coord = data["coord"].copy()
        gt_vxy = data["gt_vxy"].copy()
        feat = data["feat"].copy() if "feat" in data else None
        if flip_x:
            coord[:, 0] *= -1.0
            gt_vxy[:, 0] *= -1.0
            if feat is not None and feat.shape[1] >= 1:
                feat[:, 0] *= -1.0
        if flip_y:
            coord[:, 1] *= -1.0
            gt_vxy[:, 1] *= -1.0
            if feat is not None and feat.shape[1] >= 2:
                feat[:, 1] *= -1.0
        out["coord"] = coord
        out["gt_vxy"] = gt_vxy
        if feat is not None:
            out["feat"] = feat
        return out


class RandomScale:
    def __init__(self, low: float = 0.95, high: float = 1.05,
                 rng: np.random.Generator | None = None):
        self.low, self.high = low, high
        self.rng = rng or np.random.default_rng()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        s = float(self.rng.uniform(self.low, self.high))
        out = dict(data)
        out["coord"] = data["coord"] * s
        out["gt_vxy"] = data["gt_vxy"] * s
        if "feat" in data and data["feat"].shape[1] >= 3:
            feat = data["feat"].copy()
            feat[:, :3] *= s
            out["feat"] = feat
        return out


class RandomPointDropout:
    def __init__(self, low: float = 0.05, high: float = 0.15,
                 rng: np.random.Generator | None = None):
        self.low, self.high = low, high
        self.rng = rng or np.random.default_rng()

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        n = data["coord"].shape[0]
        drop_frac = float(self.rng.uniform(self.low, self.high))
        keep_mask = self.rng.random(n) >= drop_frac
        if keep_mask.all():
            return data
        out = dict(data)
        for key, val in data.items():
            if isinstance(val, np.ndarray) and val.shape[:1] == (n,):
                out[key] = val[keep_mask]
        return out


class Compose:
    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        for t in self.transforms:
            data = t(data)
        return data


def default_train_transforms(seed: int = 42) -> Compose:
    rng = np.random.default_rng(seed)
    return Compose(
        RandomRotateZ(rng=rng),
        RandomFlip(rng=rng),
        RandomScale(rng=rng),
        RandomPointDropout(rng=rng),
    )
