"""Evaluate a PTv3 checkpoint, with B0-B3 baselines included for context.

    python code/scripts/eval_ptv3.py \\
        --config code/configs/ptv3_5080.yaml \\
        --checkpoint checkpoints/ptv3_5080/best.pt \\
        --split val

Output structure mirrors `eval_baselines.py` so the same T1/T2/F3/F4
pipeline produces the post-PTv3 versions of those tables/figures.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code" / "Pointcept"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--data-root", default=None, type=Path)
    p.add_argument("--splits", default=None, type=Path)
    p.add_argument("--split", default="val", choices=("train", "val", "test"))
    p.add_argument("--include-baselines", action="store_true",
                   help="Also recompute B0-B3 alongside PTv3 (for T1).")
    p.add_argument("--max-eval-frames", default=None, type=int)
    p.add_argument("--output", default=None, type=Path)
    args = p.parse_args(argv)

    import torch

    from ptv3_fmcw.data.aevascenes_dataset import AevaScenesFrameDataset
    from ptv3_fmcw.eval.baselines import (
        B0_Zero, B1_DopplerOnly, B2_ClassMean, B3_DopplerPlusClassMean, fit_class_mean,
    )
    from ptv3_fmcw.eval.evaluate import evaluate_baselines
    from ptv3_fmcw.eval.metrics import MetricAccumulator
    from ptv3_fmcw.eval.tables import generate_T1, generate_T2
    from ptv3_fmcw.eval.visualize import generate_F3, generate_F4
    from ptv3_fmcw.training.config import load_config
    from ptv3_fmcw.training.trainer import collate_pointcept

    cfg = load_config(args.config)
    data_root = args.data_root or Path(cfg.data.root)
    splits_path = args.splits or Path(cfg.data.splits_file)
    splits = json.loads(splits_path.read_text())
    seqs = splits[args.split]

    if args.output is None:
        args.output = Path("frozen_test") / f"{dt.datetime.now():%Y-%m-%dT%H-%M-%S}-ptv3"
    args.output.mkdir(parents=True, exist_ok=True)

    eval_ds = AevaScenesFrameDataset(data_root, seqs, lidar=cfg.data.lidar)
    if args.max_eval_frames and len(eval_ds) > args.max_eval_frames:
        eval_ds._index = eval_ds._index[: args.max_eval_frames]

    # --- load checkpoint ----------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backbone = (cfg.model.backbone or "ptv3").lower()
    if backbone == "ptv3":
        from ptv3_fmcw.models.ptv3_velocity import (
            BackboneConfig, PTv3Velocity, PTv3VelocityConfig,
        )
        model_cfg = PTv3VelocityConfig(
            backbone=BackboneConfig(
                in_channels=cfg.model.in_channels,
                enable_flash=cfg.model.enable_flash,
            ),
            head_hidden=cfg.model.head_hidden,
            head_dropout=cfg.model.head_dropout,
        )
        model = PTv3Velocity(model_cfg).to(device).eval()
    elif backbone == "pointmlp":
        from ptv3_fmcw.models.point_mlp import PointMLPConfig, PointMLPVelocity
        model_cfg = PointMLPConfig(
            in_channels=cfg.model.in_channels,
            width=cfg.model.pointmlp_width,
            n_blocks=cfg.model.pointmlp_blocks,
            head_hidden=cfg.model.head_hidden,
            head_dropout=cfg.model.head_dropout,
        )
        model = PointMLPVelocity(model_cfg).to(device).eval()
    else:
        raise ValueError(f"unknown model.backbone: {cfg.model.backbone!r}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    print(f"[eval_ptv3] loaded checkpoint epoch={ckpt.get('epoch')}  "
          f"best_val={ckpt.get('best_val')}  device={device}")

    # --- run model + accumulate metrics --------------------------------
    from torch.utils.data import DataLoader
    from ptv3_fmcw.training.trainer import _NumpyDatasetAdapter
    val_loader = DataLoader(
        _NumpyDatasetAdapter(eval_ds), batch_size=1, shuffle=False,
        num_workers=4, collate_fn=collate_pointcept,
    )
    acc = MetricAccumulator()
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            pred = model(batch).detach().float().cpu().numpy()
            n = pred.shape[0]
            pred3 = np.zeros((n, 3), np.float32); pred3[:, :2] = pred
            gt3 = np.zeros((n, 3), np.float32); gt3[:, :2] = batch["gt_vxy"].cpu().numpy()
            coord = batch["coord"].cpu().numpy()
            sem = batch["sem_class"].cpu().numpy()
            mask = batch["gt_mask"].cpu().numpy()
            acc.update(pred3, gt3, coord, sem, valid_mask=mask)
            if (i + 1) % 50 == 0:
                print(f"  frame {i+1}/{len(val_loader)}", flush=True)

    from ptv3_fmcw.data.class_names import SEMANTIC_NAMES
    model_metrics = acc.as_dict(class_idx_to_name=SEMANTIC_NAMES)

    model_name = "PTv3_FMCW" if backbone == "ptv3" else "PointMLP_FMCW"
    results: dict[str, dict] = {model_name: model_metrics}
    if args.include_baselines:
        train_ds = AevaScenesFrameDataset(data_root, splits["train"], lidar=cfg.data.lidar)
        if args.max_eval_frames:
            train_ds._index = train_ds._index[: args.max_eval_frames]
        class_means = fit_class_mean(
            (train_ds[i] for i in range(len(train_ds))),
            n_total=len(train_ds),
        )
        baselines = {
            "B0_zero": B0_Zero(),
            "B1_doppler_only": B1_DopplerOnly(),
            "B2_class_mean": B2_ClassMean(class_means=class_means),
            "B3_doppler_plus_class_mean": B3_DopplerPlusClassMean(class_means=class_means),
        }
        baseline_results = evaluate_baselines(baselines, eval_ds)
        results.update(baseline_results)

    out_payload = {
        "split": args.split,
        "n_frames": len(eval_ds),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_best_val": ckpt.get("best_val"),
        "git_commit_at_eval": ckpt.get("git_commit"),
        "baselines": results,
    }
    (args.output / "metrics.json").write_text(json.dumps(out_payload, indent=2, default=str))
    generate_T1(results, args.output / "tables" / "T1_main_results.tex")
    generate_T2(results, args.output / "tables" / "T2_per_class_epet.tex",
                methods=("B1_doppler_only", "B3_doppler_plus_class_mean", model_name))
    generate_F3(results, args.output / "figures" / "F3_tangential_dominance.pdf",
                methods=("B0_zero", "B1_doppler_only", "B3_doppler_plus_class_mean", model_name))
    generate_F4(results, args.output / "figures" / "F4_per_class_epe.pdf",
                methods=("B0_zero", "B1_doppler_only", "B3_doppler_plus_class_mean", model_name))

    print()
    print("=" * 80)
    print(f"  PTv3 + baselines on '{args.split}' ({len(eval_ds)} frames)")
    print("=" * 80)
    hdr = f"{'method':<32s} {'EPE_dyn':>9s} {'EPE_t':>9s} {'EPE_r':>9s}"
    print(hdr); print("-" * len(hdr))
    for name, res in results.items():
        m = res["overall"]
        print(f"{name:<32s} {m['epe_dyn']:>9.3f} {m['epe_t']:>9.3f} {m['epe_r']:>9.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
