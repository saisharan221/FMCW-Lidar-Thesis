"""Verify the two silent-invalidation risks from spec §8.2.

R3 (`v_radial` sign convention): is positive Doppler approaching or
    receding? We don't take spec assumptions on faith — we infer the
    convention from data by predicting per-point v_radial from box
    `linear_velocity` (a field with a known physical interpretation)
    and comparing to the measured Doppler.

R4 (box `linear_velocity` reference frame): is it VEHICLE (ego-relative)
    or WORLD (absolute)? Since the point Doppler is ego-compensated
    (data exploration §1), a small residual between predicted and
    measured v_radial proves both fields share the same frame, which
    is necessarily VEHICLE.

This script prints a one-page report and writes
`data_exploration/stats/data_assumption_check.json` capturing the
verified conventions for the spec changelog.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ptv3_fmcw.data.aevascenes_dataset import (
    AevaScenesFrameDataset,
    list_sequences,
)
from ptv3_fmcw.data.gt_velocity import points_in_box


def _line_of_sight_xy(coord: np.ndarray) -> np.ndarray:
    xy = coord[:, :2]
    norm = np.linalg.norm(xy, axis=1, keepdims=True)
    return xy / np.clip(norm, 1e-6, None)


def analyse_frame(record: dict, boxes_meta: list, min_speed: float, min_pts: int):
    """Compare measured vs predicted v_radial for moving in-box points.

    Returns rows of (box_speed, n_pts, mean_measured, mean_pred_recede, mean_pred_approach,
                     resid_recede, resid_approach, cls).
    """
    from ptv3_fmcw.data.aevascenes_dataset import parse_box

    coord = record["coord"]
    v_radial = record["feat"][:, 4]
    los = _line_of_sight_xy(coord)

    rows = []
    for b in boxes_meta:
        box = parse_box(b)
        speed = float(np.linalg.norm(box.linear_velocity))
        if speed < min_speed:
            continue
        mask = points_in_box(coord, box)
        if int(mask.sum()) < min_pts:
            continue

        # Use the box centroid LOS as an approximation of the per-point LOS:
        # the variance within a box is small relative to the box centroid.
        v_lin_xy = box.linear_velocity[:2]
        # Measured Doppler in the box.
        measured = float(np.median(v_radial[mask]))
        # Predicted v_radial under each convention, evaluated per-point and averaged.
        per_pt_dot = (los[mask] @ v_lin_xy)
        pred_recede = float(np.mean(per_pt_dot))
        pred_approach = -pred_recede

        rows.append(
            {
                "speed": speed,
                "n_pts": int(mask.sum()),
                "measured": measured,
                "pred_recede": pred_recede,
                "pred_approach": pred_approach,
                "resid_recede": abs(measured - pred_recede),
                "resid_approach": abs(measured - pred_approach),
                "cls": b.get("class", ""),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Verify data convention assumptions.")
    p.add_argument("--data-root", default="data/aevascenes_v0.2", type=Path)
    p.add_argument(
        "--n-sequences",
        default=10,
        type=int,
        help="number of sequences to sample from (highway preferred for high-Doppler signal)",
    )
    p.add_argument("--frames-per-seq", default=10, type=int)
    p.add_argument("--min-box-speed", default=5.0, type=float)
    p.add_argument("--min-points-per-box", default=20, type=int)
    p.add_argument(
        "--output",
        default=Path("data_exploration/stats/data_assumption_check.json"),
        type=Path,
    )
    args = p.parse_args(argv)

    all_seqs = list_sequences(args.data_root)
    rng = np.random.default_rng(42)
    if len(all_seqs) > args.n_sequences:
        chosen = sorted(rng.choice(all_seqs, size=args.n_sequences, replace=False).tolist())
    else:
        chosen = all_seqs

    ds = AevaScenesFrameDataset(args.data_root, chosen)

    rows: list[dict] = []
    for uuid in chosen:
        meta = ds.metadata(uuid)
        n = len(meta["frames"])
        idxs = sorted(rng.choice(n, size=min(args.frames_per_seq, n), replace=False).tolist())
        for fi in idxs:
            rec = ds.load(uuid, fi)
            rows.extend(
                analyse_frame(
                    rec,
                    meta["frames"][fi].get("boxes", []),
                    args.min_box_speed,
                    args.min_points_per_box,
                )
            )

    if not rows:
        raise SystemExit(
            "No qualifying boxes found. Lower --min-box-speed or --min-points-per-box."
        )

    by_cls = defaultdict(list)
    for r in rows:
        by_cls[r["cls"]].append(r)

    measured = np.array([r["measured"] for r in rows])
    pred_recede = np.array([r["pred_recede"] for r in rows])
    pred_approach = np.array([r["pred_approach"] for r in rows])

    mean_resid_recede = float(np.mean(np.abs(measured - pred_recede)))
    mean_resid_approach = float(np.mean(np.abs(measured - pred_approach)))
    median_resid_recede = float(np.median(np.abs(measured - pred_recede)))
    median_resid_approach = float(np.median(np.abs(measured - pred_approach)))

    if mean_resid_recede < mean_resid_approach:
        verdict = "POSITIVE = RECEDING (point moving away from sensor)"
    else:
        verdict = "POSITIVE = APPROACHING (point moving toward sensor)"

    print()
    print("=" * 72)
    print(" v_radial sign convention check")
    print("=" * 72)
    print(f"  qualifying box samples: {len(rows)}")
    print(f"  classes: {sorted(by_cls)}")
    print(
        f"  mean |measured - v.r_hat|     = {mean_resid_recede:.3f} m/s "
        f"(predicting under 'positive = receding')"
    )
    print(
        f"  mean |measured - (-v.r_hat)|  = {mean_resid_approach:.3f} m/s "
        f"(predicting under 'positive = approaching')"
    )
    print(
        f"  median residuals: recede={median_resid_recede:.3f}, "
        f"approach={median_resid_approach:.3f}"
    )
    print(f"  -> {verdict}")
    print()
    print("  Per-class residual (recede / approach), m/s:")
    for cls in sorted(by_cls):
        sub = by_cls[cls]
        m = np.array([r["measured"] for r in sub])
        pr = np.array([r["pred_recede"] for r in sub])
        pa = np.array([r["pred_approach"] for r in sub])
        print(
            f"    {cls:>14s}  n={len(sub):4d}  "
            f"recede={float(np.mean(np.abs(m-pr))):.2f}  "
            f"approach={float(np.mean(np.abs(m-pa))):.2f}"
        )
    print()
    print("=" * 72)
    print(" Box `linear_velocity` reference frame check")
    print("=" * 72)
    if min(mean_resid_recede, mean_resid_approach) < 1.5:
        frame_verdict = (
            "VEHICLE / ego-compensated (consistent with point velocity field)"
        )
    else:
        frame_verdict = (
            "INCONSISTENT — residual exceeds 1.5 m/s, reference frame may differ"
        )
    print(f"  best residual = {min(mean_resid_recede, mean_resid_approach):.3f} m/s")
    print(f"  -> {frame_verdict}")
    print()

    output = {
        "n_box_samples": len(rows),
        "v_radial_sign_convention": {
            "verdict": verdict,
            "mean_resid_recede": mean_resid_recede,
            "mean_resid_approach": mean_resid_approach,
            "median_resid_recede": median_resid_recede,
            "median_resid_approach": median_resid_approach,
        },
        "box_linear_velocity_frame": {
            "verdict": frame_verdict,
            "best_residual": min(mean_resid_recede, mean_resid_approach),
            "residual_threshold": 1.5,
        },
        "sequences_used": chosen,
        "frames_per_seq": args.frames_per_seq,
        "min_box_speed": args.min_box_speed,
        "min_points_per_box": args.min_points_per_box,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"[verify] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
