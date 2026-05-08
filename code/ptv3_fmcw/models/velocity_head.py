"""Velocity regression head (spec §4.4).

A 2-layer MLP that maps per-point backbone features to (vx, vy) m/s in
the VEHICLE frame. Final layer is zero-initialised so that the model
starts predicting (0, 0) — the dominant background prior. This avoids
the early-step gradient explosion typical of Kaiming-init regression
heads on imbalanced classes.

Lazy torch import: the module imports torch at construction time, not
at module load time. Files that need to introspect VelocityHead's
constructor signature (configs, tests) can still import this module
on machines without torch.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class VelocityHeadConfig:
    in_channels: int = 64       # PTv3 nuScenes-base decoder out channels
    hidden: int = 64
    dropout: float = 0.0


def build_velocity_head(cfg: VelocityHeadConfig | None = None):
    """Construct a VelocityHead. Imports torch lazily."""
    cfg = cfg or VelocityHeadConfig()
    return VelocityHead(in_channels=cfg.in_channels, hidden=cfg.hidden, dropout=cfg.dropout)


class _LazyVelocityHead:
    """Sentinel raised at attribute access if torch is unavailable."""
    def __init__(self, *_a, **_k):
        raise ImportError(
            "VelocityHead needs PyTorch installed. `pip install torch` or "
            "`conda env create -f code/environment-train.yml`."
        )


try:
    import torch
    import torch.nn as nn
    from torch.nn import functional as F  # noqa: F401  (used downstream)

    class VelocityHead(nn.Module):
        """2-layer MLP regression head with zero-init final layer."""

        def __init__(self, in_channels: int = 64, hidden: int = 64, dropout: float = 0.0):
            super().__init__()
            self.fc1 = nn.Linear(in_channels, hidden)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, 2)
            nn.init.zeros_(self.fc2.weight)
            nn.init.zeros_(self.fc2.bias)

        def forward(self, feat: "torch.Tensor") -> "torch.Tensor":
            return self.fc2(self.drop(self.act(self.fc1(feat))))

except ImportError:  # pragma: no cover -- exercised on torch-less envs
    VelocityHead = _LazyVelocityHead  # type: ignore[assignment, misc]
