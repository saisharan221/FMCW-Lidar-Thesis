"""Figures F3 (tangential-dominance) and F4 (per-class EPE) — spec §11.

matplotlib only. Both functions read from the metric dict returned by
`MetricAccumulator.as_dict(...)` so they don't re-iterate the dataset.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ptv3_fmcw.data.class_names import TOP_CLASSES
from ptv3_fmcw.eval.metrics import RATIO_EDGES


_F3_BUCKETS = [
    "0-0.5", "0.5-1", "1-2", "2-5", "5-10", "10-inf",
]


def _ratio_label(idx: int) -> str:
    lo, hi = RATIO_EDGES[idx], RATIO_EDGES[idx + 1]
    hi_str = "inf" if not np.isfinite(hi) else f"{hi:g}"
    return f"{lo:g}-{hi_str}"


def generate_F3(
    results: Mapping[str, dict],
    out_path: Path,
    methods: tuple[str, ...] = ("B0_zero", "B1_doppler_only", "B3_doppler_plus_class_mean"),
) -> Path:
    """F3: EPE_t vs tangential-dominance bucket.

    Spec §11.1: shows that B1's EPE_t increases monotonically with the
    ratio bucket (B1 cannot recover any tangential component). PTv3
    will be added as a fourth curve later and the figure regenerated.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bucket_labels = _F3_BUCKETS
    x = np.arange(len(bucket_labels))

    for method in methods:
        m = results.get(method, {})
        rows = m.get("per_tangential_ratio", {})
        ys = [rows.get(label, {}).get("epe_t", float("nan")) for label in bucket_labels]
        ax.plot(x, ys, marker="o", label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(bucket_labels)
    ax.set_xlabel("|v_t| / |v_r| bucket  (dynamic points)")
    ax.set_ylabel("EPE_t  (m/s)")
    ax.set_title("F3 — Tangential EPE vs tangential-dominance ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def generate_F4(
    results: Mapping[str, dict],
    out_path: Path,
    methods: tuple[str, ...] = ("B0_zero", "B1_doppler_only", "B3_doppler_plus_class_mean"),
    classes: tuple[str, ...] = TOP_CLASSES,
) -> Path:
    """F4: per-class EPE_dyn bar chart for the top-10 classes."""
    fig, ax = plt.subplots(figsize=(11, 5))
    n_classes = len(classes)
    n_methods = len(methods)
    bar_w = 0.8 / n_methods
    x = np.arange(n_classes)

    for j, method in enumerate(methods):
        m = results.get(method, {})
        per_class = m.get("per_class", {})
        ys = [per_class.get(c, {}).get("epe_dyn", float("nan")) for c in classes]
        ax.bar(x + j * bar_w - 0.4 + bar_w / 2, ys, width=bar_w, label=method)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right")
    ax.set_ylabel("EPE_dyn  (m/s)")
    ax.set_title("F4 — Per-class EPE on dynamic points")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
