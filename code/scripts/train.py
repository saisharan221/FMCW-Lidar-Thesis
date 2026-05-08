"""Training entry point: PTv3-FMCW.

    python code/scripts/train.py --config code/configs/ptv3_pilot.yaml

Looks for the Pointcept submodule at `code/Pointcept/` and adds it
to PYTHONPATH so the model can import. Splits come from
`code/configs/splits.json`. With `data.pilot_sequences=N`, only the
first N train sequences and the first N val sequences are used —
useful for the spec §5.7 overfit-1-frame test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code"))
sys.path.insert(0, str(REPO_ROOT / "code" / "Pointcept"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--data-root", default=None, type=Path)
    p.add_argument("--splits", default=None, type=Path)
    p.add_argument("--output-dir", default=None, type=Path)
    p.add_argument("--resume-from", default=None, type=Path)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args(argv)

    from ptv3_fmcw.data.aevascenes_dataset import AevaScenesFrameDataset
    from ptv3_fmcw.data.transforms import default_train_transforms
    from ptv3_fmcw.training.config import load_config
    from ptv3_fmcw.training.trainer import train

    cfg = load_config(args.config)
    if args.no_wandb:
        cfg.wandb.enabled = False
    data_root = args.data_root or Path(cfg.data.root)
    splits_path = args.splits or Path(cfg.data.splits_file)
    output_dir = args.output_dir or Path(cfg.output_dir) / args.config.stem

    splits = json.loads(splits_path.read_text())
    train_seqs = splits["train"]
    val_seqs = splits["val"]
    if cfg.data.pilot_sequences:
        train_seqs = train_seqs[: cfg.data.pilot_sequences]
        val_seqs = val_seqs[: max(1, cfg.data.pilot_sequences // 5)]
        print(f"[train] pilot mode: {len(train_seqs)} train / {len(val_seqs)} val sequences")

    train_ds = AevaScenesFrameDataset(data_root, train_seqs, lidar=cfg.data.lidar,
                                      grid_size=cfg.data.grid_size)
    val_ds = AevaScenesFrameDataset(data_root, val_seqs, lidar=cfg.data.lidar,
                                    grid_size=cfg.data.grid_size)
    print(f"[train] train frames = {len(train_ds)}  val frames = {len(val_ds)}")

    train_transform = None if args.no_augment else default_train_transforms(seed=cfg.trainer.seed)

    summary = train(
        cfg=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_transform=train_transform,
        output_dir=output_dir,
        resume_from=args.resume_from,
    )
    print(f"[train] summary written to {output_dir / 'training_summary.json'}")
    print(f"[train] best val epe_dyn = {summary['best_val_epe_dyn']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
