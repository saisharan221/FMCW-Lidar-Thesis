"""Frame-streaming evaluation loop for B0-B3 (spec §10).

For each frame: predict (vx, vy) with each baseline, accumulate into a
`MetricAccumulator` per baseline, optionally save predictions for the
frozen-test lockfile. Returns the per-baseline nested metric dict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from ptv3_fmcw.data.aevascenes_dataset import AevaScenesFrameDataset
from ptv3_fmcw.data.class_names import SEMANTIC_NAMES
from ptv3_fmcw.eval.baselines import Baseline
from ptv3_fmcw.eval.metrics import MetricAccumulator


def evaluate_baselines(
    baselines: dict[str, Baseline],
    dataset: AevaScenesFrameDataset,
    save_predictions_dir: Path | None = None,
    progress: bool = True,
) -> dict[str, dict]:
    """Run all baselines through the dataset.

    Parameters
    ----------
    baselines : mapping of name -> Baseline. The order of iteration sets
        the per-frame compute order (cheap baselines first).
    dataset : AevaScenesFrameDataset (val or test split, optionally
        scene-tagged via its `scene_tag_fn`).
    save_predictions_dir : if set, dump one .npz per baseline with
        concatenated predictions and per-frame offsets.

    Returns
    -------
    {baseline_name: metric_dict_from_MetricAccumulator.as_dict(...)}.
    """
    accumulators = {name: MetricAccumulator() for name in baselines}
    pred_buffers: dict[str, list[np.ndarray]] = {name: [] for name in baselines}
    frame_keys: list[tuple[str, int]] = []

    n = len(dataset)
    for i in range(n):
        rec = dataset[i]
        coord = rec["coord"]
        sem = rec["sem_class"]
        gt2d = rec["gt_vxy"]
        gt = np.empty((coord.shape[0], 3), dtype=np.float32)
        gt[:, :2] = gt2d
        gt[:, 2] = 0.0
        scene_tag = rec.get("scene_tag", "unknown")
        frame_keys.append((rec["sequence_uuid"], int(rec["frame_idx"])))

        for name, b in baselines.items():
            pred2d = b.predict(rec)
            pred = np.empty((coord.shape[0], 3), dtype=np.float32)
            pred[:, :2] = pred2d
            pred[:, 2] = 0.0
            accumulators[name].update(pred, gt, coord, sem, valid_mask=rec["gt_mask"], scene_tag=scene_tag)
            if save_predictions_dir is not None:
                pred_buffers[name].append(pred2d.astype(np.float32))

        if progress and (i + 1) % 50 == 0:
            print(f"  frame {i+1}/{n}", flush=True)

    if save_predictions_dir is not None:
        save_predictions_dir.mkdir(parents=True, exist_ok=True)
        # Build offsets array so the concatenated predictions can be split per frame.
        for name, buffers in pred_buffers.items():
            offsets = np.cumsum([0] + [b.shape[0] for b in buffers], dtype=np.int64)
            preds = np.concatenate(buffers, axis=0) if buffers else np.zeros((0, 2), np.float32)
            np.savez_compressed(
                save_predictions_dir / f"{name}.npz",
                pred=preds,
                offsets=offsets,
                frame_keys=np.array(frame_keys, dtype=object),
            )

    return {
        name: acc.as_dict(class_idx_to_name=SEMANTIC_NAMES)
        for name, acc in accumulators.items()
    }


def b1_sign_check(
    dataset: AevaScenesFrameDataset,
    max_frames: int = 50,
    seed: int = 31,
) -> dict:
    """Spec §8.2 mitigation: print the sign convention sanity check.

    Reports:
      - mean |v_radial| on stationary points (semantic class is static).
        Expected near zero per the data exploration.
      - mean v_radial on receding cars (box motion away from sensor).
        Under "positive = receding" we expect this > 0.
      - mean v_radial on approaching cars (box motion toward sensor).
        Under "positive = receding" we expect this < 0.

    Returns the numbers as a dict so callers can also write them to
    metrics.json.
    """
    from ptv3_fmcw.data.aevascenes_dataset import parse_box
    from ptv3_fmcw.data.gt_velocity import points_in_box

    static_classes = {
        idx for idx, name in SEMANTIC_NAMES.items()
        if name in {"road", "lane_boundary", "road_marking", "sidewalk", "other_ground",
                     "building", "vegetation", "pole_trunk", "traffic_sign", "traffic_item",
                     "other_structure", "reflective_marker"}
    }

    abs_static_v: list[float] = []
    receding_v: list[float] = []
    approaching_v: list[float] = []

    # Sample frames across the dataset rather than taking the first N —
    # the first 50 of val are usually one slow city/day sequence with
    # no qualifying boxes, which silently NaN-s the receding/approaching
    # banner.
    rng = np.random.default_rng(seed)
    n_total = len(dataset)
    n = min(n_total, max_frames)
    idxs = rng.choice(n_total, size=n, replace=False).tolist() if n_total > n else range(n_total)
    for i in idxs:
        rec = dataset[i]
        coord = rec["coord"]
        v_r = rec["v_radial"]
        sem = rec["sem_class"]

        static_mask = np.isin(sem, list(static_classes))
        if static_mask.any():
            abs_static_v.append(float(np.mean(np.abs(v_r[static_mask]))))

        meta = dataset.metadata(rec["sequence_uuid"])
        boxes = meta["frames"][rec["frame_idx"]].get("boxes", [])
        for b in boxes:
            if b.get("class") not in {"car", "truck", "bus"}:
                continue
            box = parse_box(b)
            speed = float(np.linalg.norm(box.linear_velocity))
            if speed < 5.0:
                continue
            mask = points_in_box(coord, box)
            if int(mask.sum()) < 30:
                continue
            xy = coord[mask, :2]
            norm = np.linalg.norm(xy, axis=1, keepdims=True)
            r_hat = xy / np.clip(norm, 1e-6, None)
            v_lin_xy = box.linear_velocity[:2]
            box_radial = float(np.mean(r_hat @ v_lin_xy))
            measured = float(np.mean(v_r[mask]))
            if box_radial > 1.0:
                receding_v.append(measured)
            elif box_radial < -1.0:
                approaching_v.append(measured)

    return {
        "stationary_mean_abs_v_radial": float(np.mean(abs_static_v)) if abs_static_v else float("nan"),
        "receding_cars_mean_v_radial": float(np.mean(receding_v)) if receding_v else float("nan"),
        "approaching_cars_mean_v_radial": float(np.mean(approaching_v)) if approaching_v else float("nan"),
        "n_stationary_frames": len(abs_static_v),
        "n_receding_box_samples": len(receding_v),
        "n_approaching_box_samples": len(approaching_v),
    }


def print_sign_check_banner(report: dict) -> None:
    print()
    print("=" * 72)
    print(" B1 sign-check (spec §8.2)")
    print("=" * 72)
    print(f"  stationary points: mean |v_radial| = {report['stationary_mean_abs_v_radial']:.3f} m/s   (expected ~0)")
    print(f"  receding cars    : mean v_radial   = {report['receding_cars_mean_v_radial']:>+7.3f} m/s   (expected > 0 under 'positive = receding')")
    print(f"  approaching cars : mean v_radial   = {report['approaching_cars_mean_v_radial']:>+7.3f} m/s   (expected < 0 under 'positive = receding')")
    if (report["receding_cars_mean_v_radial"] < 0
        or report["approaching_cars_mean_v_radial"] > 0):
        print()
        print("  ⚠  SIGN MISMATCH — set v_radial_sign='positive_approaches' in baseline.yaml and re-run.")
    print()
