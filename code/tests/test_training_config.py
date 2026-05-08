"""Smoke-test config loading + dataclass defaults."""
from __future__ import annotations

from pathlib import Path

import pytest

from ptv3_fmcw.training.config import (
    DataConfig,
    LossDictConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    TrainerConfig,
    load_config,
)


CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"


def test_defaults_match_spec():
    cfg = TrainConfig()
    assert cfg.optim.lr_peak == 2e-3
    assert cfg.optim.weight_decay == 5e-3
    assert cfg.optim.pct_start == 0.05
    assert cfg.trainer.epochs == 50
    assert cfg.trainer.batch_size == 4
    assert cfg.trainer.precision == "bf16"
    assert cfg.loss.huber_beta == 0.5
    assert cfg.loss.w_dyn == 1.0
    assert cfg.loss.w_bg == 0.2
    assert cfg.model.in_channels == 5


@pytest.mark.parametrize("name", ["ptv3_pilot", "ptv3_full", "ptv3_5080"])
def test_yaml_configs_parse(name):
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        pytest.skip(f"{path} not present")
    cfg = load_config(path)
    assert isinstance(cfg, TrainConfig)
    assert cfg.model.in_channels == 5  # always 5 for FMCW


def test_5080_config_uses_grad_accum():
    """RTX 5080 path keeps effective batch 4 via grad accum."""
    path = CONFIGS_DIR / "ptv3_5080.yaml"
    if not path.exists():
        pytest.skip("ptv3_5080.yaml missing")
    cfg = load_config(path)
    assert cfg.trainer.batch_size * cfg.trainer.grad_accum == 4
    assert cfg.trainer.precision == "bf16"
