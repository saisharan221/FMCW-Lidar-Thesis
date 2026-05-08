"""Training configuration (spec §5.2).

YAML-loaded; the same file format is consumed by `scripts/train.py`
and the trainer. Defaults match spec §5.2 (AdamW, OneCycle, bf16,
batch 4, 50 epochs).

The 5080 budget overrides (smaller batch + grad accum) live in
`code/configs/ptv3_5080.yaml` and apply on top of these defaults.
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    root: str = "data/aevascenes_v0.2"
    splits_file: str = "code/configs/splits.json"
    lidar: str = "front_wide_lidar"
    grid_size: float = 0.1
    num_workers: int = 8
    pilot_sequences: int | None = None  # if set, override splits with N pilot seqs


@dataclass
class ModelConfig:
    backbone: str = "ptv3"  # "ptv3" or "pointmlp"
    in_channels: int = 5
    head_hidden: int = 64
    head_dropout: float = 0.0
    enable_flash: bool = True
    # PointMLP-specific (ignored when backbone == "ptv3")
    pointmlp_width: int = 64
    pointmlp_blocks: int = 3


@dataclass
class OptimConfig:
    lr_peak: float = 2e-3
    weight_decay: float = 5e-3
    grad_clip: float = 1.0
    pct_start: float = 0.05
    optimizer: str = "adamw"
    scheduler: str = "onecycle"


@dataclass
class TrainerConfig:
    epochs: int = 50
    batch_size: int = 4
    grad_accum: int = 1
    precision: str = "bf16"
    seed: int = 42
    val_every_epochs: int = 1
    save_every_epochs: int = 1


@dataclass
class LossDictConfig:
    huber_beta: float = 0.5
    w_dyn: float = 1.0
    w_bg: float = 0.2
    dynamic_threshold: float = 0.5


@dataclass
class WandbConfig:
    project: str = "ptv3-fmcw"
    enabled: bool = True
    run_name: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    loss: LossDictConfig = field(default_factory=LossDictConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    output_dir: str = "checkpoints"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_env(raw: str) -> str:
    """Tiny ${oc.env:VAR, default} resolver so configs work without omegaconf."""
    pattern = re.compile(r"\$\{oc\.env:([A-Z_]+),\s*([^}]+)\}")

    def sub(m):
        return os.environ.get(m.group(1), m.group(2)).strip()
    return pattern.sub(sub, raw)


def load_config(path: str | Path) -> TrainConfig:
    raw = Path(path).read_text()
    raw = _resolve_env(raw)
    data = yaml.safe_load(raw) or {}

    cfg = TrainConfig()
    if "data" in data:
        cfg.data = DataConfig(**{**asdict(cfg.data), **data["data"]})
    if "model" in data:
        cfg.model = ModelConfig(**{**asdict(cfg.model), **data["model"]})
    if "optim" in data:
        cfg.optim = OptimConfig(**{**asdict(cfg.optim), **data["optim"]})
    if "trainer" in data:
        cfg.trainer = TrainerConfig(**{**asdict(cfg.trainer), **data["trainer"]})
    if "loss" in data:
        cfg.loss = LossDictConfig(**{**asdict(cfg.loss), **data["loss"]})
    if "wandb" in data:
        cfg.wandb = WandbConfig(**{**asdict(cfg.wandb), **data["wandb"]})
    if "output_dir" in data:
        cfg.output_dir = str(data["output_dir"])
    return cfg
