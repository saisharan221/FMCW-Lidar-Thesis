"""Training loop (spec §5).

AdamW + OneCycleLR + bf16. Per-epoch val with EPE_dyn as the
checkpoint selection metric (spec §5.4). Optional W&B logging.

Pointcept's `Point` data structure expects a dict with keys
`coord`, `feat`, `grid_size`, `offset` where `offset` is the
running-sum index into the concatenated point cloud. The collator
defined here builds that offset.

Lazy-imports torch + W&B so the rest of the package stays usable on
torch-less environments.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ptv3_fmcw.data.aevascenes_dataset import AevaScenesFrameDataset
from ptv3_fmcw.eval.metrics import MetricAccumulator
from ptv3_fmcw.training.config import TrainConfig
from ptv3_fmcw.training.loss import LossConfig, velocity_loss


def collate_pointcept(records: list[dict]) -> dict:
    """Concatenate a batch of frame records into a Pointcept data dict.

    Output keys:
      coord     (sum_n_i, 3) float32
      feat      (sum_n_i, 5) float32
      offset    (B,) int64    — running sum of per-frame point counts
      grid_size float
      gt_vxy    (sum_n_i, 2) float32
      gt_mask   (sum_n_i,)   bool
      sem_class (sum_n_i,)   int64
    """
    import torch

    coord = np.concatenate([r["coord"] for r in records], axis=0)
    feat = np.concatenate([r["feat"] for r in records], axis=0)
    gt_vxy = np.concatenate([r["gt_vxy"] for r in records], axis=0)
    gt_mask = np.concatenate([r["gt_mask"] for r in records], axis=0)
    sem_class = np.concatenate([r["sem_class"] for r in records], axis=0)

    counts = np.array([r["coord"].shape[0] for r in records], dtype=np.int64)
    offset = np.cumsum(counts)
    grid_size = float(records[0].get("grid_size", 0.1))

    return {
        "coord": torch.from_numpy(coord),
        "feat": torch.from_numpy(feat),
        "offset": torch.from_numpy(offset),
        "grid_size": grid_size,
        "gt_vxy": torch.from_numpy(gt_vxy),
        "gt_mask": torch.from_numpy(gt_mask),
        "sem_class": torch.from_numpy(sem_class),
    }


class _NumpyDatasetAdapter:
    """Adapter so a torch DataLoader can iterate AevaScenesFrameDataset."""

    def __init__(
        self,
        dataset: AevaScenesFrameDataset,
        transform: Callable[[dict], dict] | None = None,
    ):
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        rec = self.dataset[idx]
        if self.transform is not None:
            rec = self.transform(rec)
        return rec


def _seed_everything(seed: int) -> None:
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    import subprocess
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return "unknown"


def train(
    cfg: TrainConfig,
    train_dataset: AevaScenesFrameDataset,
    val_dataset: AevaScenesFrameDataset,
    train_transform: Callable[[dict], dict] | None,
    output_dir: Path,
    resume_from: Path | None = None,
) -> dict:
    """Run a full training session.

    Returns a summary dict (best metric, best epoch, total wall-clock).
    """
    import torch
    from torch.utils.data import DataLoader

    _seed_everything(cfg.trainer.seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_bf16 = cfg.trainer.precision == "bf16" and device == "cuda"

    # --- model ----------------------------------------------------------
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
        model = PTv3Velocity(model_cfg).to(device)
    elif backbone == "pointmlp":
        from ptv3_fmcw.models.point_mlp import PointMLPConfig, PointMLPVelocity
        model_cfg = PointMLPConfig(
            in_channels=cfg.model.in_channels,
            width=cfg.model.pointmlp_width,
            n_blocks=cfg.model.pointmlp_blocks,
            head_hidden=cfg.model.head_hidden,
            head_dropout=cfg.model.head_dropout,
        )
        model = PointMLPVelocity(model_cfg).to(device)
    else:
        raise ValueError(f"unknown model.backbone: {cfg.model.backbone!r}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] backbone={backbone}  parameters={n_params/1e6:.2f}M  "
          f"device={device}  bf16={use_bf16}")

    # --- optim + sched -------------------------------------------------
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.optim.lr_peak,
        weight_decay=cfg.optim.weight_decay,
    )
    train_loader = DataLoader(
        _NumpyDatasetAdapter(train_dataset, train_transform),
        batch_size=cfg.trainer.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_pointcept,
        pin_memory=(device == "cuda"),
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        _NumpyDatasetAdapter(val_dataset),
        batch_size=cfg.trainer.batch_size,
        shuffle=False,
        num_workers=max(2, cfg.data.num_workers // 2),
        collate_fn=collate_pointcept,
        pin_memory=(device == "cuda"),
    )
    iters_per_epoch = math.ceil(len(train_loader) / max(cfg.trainer.grad_accum, 1))
    total_steps = iters_per_epoch * cfg.trainer.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        optim,
        max_lr=cfg.optim.lr_peak,
        total_steps=total_steps,
        pct_start=cfg.optim.pct_start,
        anneal_strategy="cos",
    )

    # --- W&B (optional) ------------------------------------------------
    wb = None
    if cfg.wandb.enabled:
        try:
            import wandb
            wb = wandb.init(
                project=cfg.wandb.project,
                name=cfg.wandb.run_name,
                tags=cfg.wandb.tags,
                config=cfg.to_dict(),
            )
        except Exception as e:
            print(f"[train] wandb disabled: {e}")
            wb = None

    loss_cfg = LossConfig(
        huber_beta=cfg.loss.huber_beta,
        w_dyn=cfg.loss.w_dyn,
        w_bg=cfg.loss.w_bg,
        dynamic_threshold=cfg.loss.dynamic_threshold,
    )

    # --- resume --------------------------------------------------------
    start_epoch = 0
    best_val = math.inf
    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optim.load_state_dict(ckpt["optimizer_state"])
        sched.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_val", math.inf)
        print(f"[train] resumed from {resume_from} at epoch {start_epoch}")

    # --- training loop -------------------------------------------------
    t_start = time.monotonic()
    best_epoch = -1
    summary = {"git_commit": _git_commit(), "config": cfg.to_dict()}

    for epoch in range(start_epoch, cfg.trainer.epochs):
        model.train()
        t_epoch = time.monotonic()
        running = {"loss": 0.0, "n": 0}
        optim.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(batch)
                loss = velocity_loss(pred, batch["gt_vxy"], batch["gt_mask"], cfg=loss_cfg)
                loss = loss / max(cfg.trainer.grad_accum, 1)
            loss.backward()

            if (step + 1) % cfg.trainer.grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optim.grad_clip)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            running["loss"] += float(loss.detach()) * cfg.trainer.grad_accum
            running["n"] += 1

            if wb is not None and (step % 20 == 0):
                wb.log({"train/loss": running["loss"] / running["n"],
                        "train/lr": float(sched.get_last_lr()[0])})

        # --- val pass ---
        if (epoch + 1) % cfg.trainer.val_every_epochs == 0:
            val_metrics = _validate(model, val_loader, device, use_bf16)
            elapsed = time.monotonic() - t_epoch
            print(
                f"[train] epoch {epoch+1}/{cfg.trainer.epochs}  "
                f"loss={running['loss']/max(running['n'],1):.4f}  "
                f"val/epe_dyn={val_metrics['epe_dyn']:.4f}  "
                f"val/epe_t={val_metrics['epe_t']:.4f}  "
                f"({elapsed:.1f}s)"
            )
            if wb is not None:
                wb.log({f"val/{k}": v for k, v in val_metrics.items()})
            current_val = val_metrics["epe_dyn"]
            if current_val < best_val:
                best_val = current_val
                best_epoch = epoch
                _save_checkpoint(output_dir / "best.pt", model, optim, sched, epoch, best_val, cfg)

        if (epoch + 1) % cfg.trainer.save_every_epochs == 0:
            _save_checkpoint(output_dir / "last.pt", model, optim, sched, epoch, best_val, cfg)

    summary.update({
        "best_val_epe_dyn": float(best_val),
        "best_epoch": best_epoch,
        "total_wall_seconds": time.monotonic() - t_start,
    })
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"[train] best val/epe_dyn = {best_val:.4f} at epoch {best_epoch+1}")
    if wb is not None:
        wb.finish()
    return summary


def _save_checkpoint(path: Path, model, optim, sched, epoch: int, best_val: float, cfg: TrainConfig):
    import torch
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optim.state_dict(),
        "scheduler_state": sched.state_dict(),
        "epoch": epoch,
        "best_val": best_val,
        "config": cfg.to_dict(),
        "git_commit": _git_commit(),
    }, path)


def _validate(model, val_loader, device: str, use_bf16: bool) -> dict:
    """Run model over val_loader, accumulate metrics, return overall row."""
    import torch
    model.eval()
    acc = MetricAccumulator()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=use_bf16):
                pred = model(batch)
            pred_np = pred.detach().float().cpu().numpy()
            n = pred_np.shape[0]
            pred_3d = np.zeros((n, 3), dtype=np.float32)
            pred_3d[:, :2] = pred_np
            gt_np = batch["gt_vxy"].detach().cpu().numpy()
            gt_3d = np.zeros((n, 3), dtype=np.float32)
            gt_3d[:, :2] = gt_np
            coord_np = batch["coord"].detach().cpu().numpy()
            sem_np = batch["sem_class"].detach().cpu().numpy()
            mask_np = batch["gt_mask"].detach().cpu().numpy()
            acc.update(pred=pred_3d, gt=gt_3d, coord=coord_np,
                       sem_class=sem_np, valid_mask=mask_np)
    overall = acc._row_to_dict(acc.overall)
    return overall
