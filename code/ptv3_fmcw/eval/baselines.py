"""Non-ML baselines B0-B3 (spec §6.2).

These are mandatory, GPU-free, and produce floor numbers PTv3 must beat
to claim success on RQ1.

Sign convention (verified 2026-05-05 against AevaScenes v0.2, spec
changelog 2026-05-05):
    `v_radial` is positive when the point is RECEDING from the sensor.
    Therefore the LOS-aligned velocity vector reconstructed from
    Doppler is `v_radial * r_hat_xy`, where `r_hat_xy` is the unit
    vector from sensor origin to point in the xy plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol

import numpy as np

from ptv3_fmcw.eval.metrics import line_of_sight_xy


class Baseline(Protocol):
    name: str

    def predict(self, record: dict) -> np.ndarray:
        """Return (N, 2) predicted (vx, vy) for a single-frame record."""
        ...


@dataclass
class B0_Zero:
    """B0: predict (0, 0) everywhere. The trivial floor."""
    name: str = "B0_zero"

    def predict(self, record: dict) -> np.ndarray:
        n = record["coord"].shape[0]
        return np.zeros((n, 2), dtype=np.float32)


@dataclass
class B1_DopplerOnly:
    """B1: pred = v_radial * r_hat_xy. The 'no ML' baseline.

    PTv3 must beat this on EPE_t for RQ1 to claim success.
    """
    name: str = "B1_doppler_only"

    def predict(self, record: dict) -> np.ndarray:
        coord = record["coord"]
        v_radial = record["feat"][:, 4]
        los = line_of_sight_xy(coord).astype(np.float32)
        return (v_radial[:, None] * los).astype(np.float32)


@dataclass
class B2_ClassMean:
    """B2: per-class mean GT velocity, applied at test using GT class.

    Spec §6.2: B2 uses **GT class** to isolate velocity prediction from
    perception. A semseg-predicted-class variant is follow-up work.
    """
    name: str = "B2_class_mean"
    class_means: dict[int, np.ndarray] = field(default_factory=dict)
    background_class_idx: int = -1

    def predict(self, record: dict) -> np.ndarray:
        sem = record["sem_class"]
        n = sem.shape[0]
        out = np.zeros((n, 2), dtype=np.float32)
        for c, mu in self.class_means.items():
            mask = sem == c
            if mask.any():
                out[mask] = mu.astype(np.float32)
        return out


@dataclass
class B3_DopplerPlusClassMean:
    """B3: radial component from Doppler, tangential component from class mean.

    Spec §6.2: stronger non-ML baseline. PTv3 must beat this on EPE_dyn
    to claim added value over no-ML.
    """
    name: str = "B3_doppler_plus_class_mean"
    class_means: dict[int, np.ndarray] = field(default_factory=dict)

    def predict(self, record: dict) -> np.ndarray:
        coord = record["coord"]
        sem = record["sem_class"]
        v_radial = record["feat"][:, 4]
        los = line_of_sight_xy(coord)

        # Tangential component of class-mean velocity.
        n = sem.shape[0]
        tangent = np.zeros((n, 2), dtype=np.float64)
        for c, mu in self.class_means.items():
            mask = sem == c
            if not mask.any():
                continue
            radial_scalar = los[mask] @ mu[:2]
            tangent[mask] = mu[:2] - radial_scalar[:, None] * los[mask]

        radial = v_radial[:, None] * los
        return (radial + tangent).astype(np.float32)


def fit_class_mean(
    train_records: Iterable[dict],
    dynamic_threshold: float = 0.5,
    progress_every: int = 250,
    n_total: int | None = None,
) -> dict[int, np.ndarray]:
    """Compute the per-class mean GT velocity over a training set.

    All points of each class are averaged unconditionally, including
    stationary instances (e.g. parked cars). For static-by-construction
    classes (road, vegetation, building) this naturally produces a mean
    near (0, 0). For mixed classes (car, truck) the mean is diluted by
    parked instances; filtering to dynamic points only (gt_speed >
    dynamic_threshold) would give a higher, more motion-representative
    mean but is not done here.

    Prints a per-`progress_every`-frame log line so callers know the
    fit is making progress on long-running splits.

    Returns
    -------
    {class_idx: (2,) mean (vx, vy)}.
    """
    import time
    sums: dict[int, np.ndarray] = {}
    counts: dict[int, int] = {}
    t0 = time.monotonic()
    for i, rec in enumerate(train_records):
        sem = rec["sem_class"]
        gt = rec["gt_vxy"]
        for c in np.unique(sem):
            c = int(c)
            mask = sem == c
            sums[c] = sums.get(c, np.zeros(2, dtype=np.float64)) + gt[mask].sum(axis=0)
            counts[c] = counts.get(c, 0) + int(mask.sum())
        if (i + 1) % progress_every == 0:
            elapsed = time.monotonic() - t0
            rate = (i + 1) / elapsed
            tag = f"/{n_total}" if n_total else ""
            eta = f", eta {(n_total - i - 1)/rate:.0f}s" if n_total else ""
            print(
                f"  [class-mean] frame {i+1}{tag}  "
                f"({rate:.1f} fr/s, elapsed {elapsed:.0f}s{eta})",
                flush=True,
            )
    return {c: (sums[c] / max(counts[c], 1)).astype(np.float64) for c in sums}
