"""PointMLP — a deferred architectural baseline (spec §6.2).

Plain per-point MLP plus learned global-context aggregation, no
neighbourhood transformer, no spconv, no flash-attn. Useful for two
reasons:

1. The full PTv3 stack (Pointcept + flash-attn + spconv) does not
   build cleanly on Windows + Blackwell at the time of writing, so
   PointMLP is the model we can actually train + evaluate from this
   machine. PTv3 stays the spec-aligned target on Linux + 5080.
2. The thesis spec explicitly defers a PointNet-style baseline as
   future work; this provides one without changing the eval contract.

Architecture (intentionally small):
    feat:(N, 5)  ->  Linear(5, 64) -> GELU -> LN
    -> per-point MLP block × 3 (Linear(64,128) -> GELU -> LN -> Linear(128,64) -> + residual)
    -> global mean over points (per batch frame, via offset)
    -> broadcast and concat back onto each point  (N, 64+64)
    -> Linear(128, 64) -> GELU
    -> head (fed by VelocityHead, zero-init final)

Output: (N, 2) (vx, vy) in m/s VEHICLE frame. Same contract as
PTv3Velocity so the trainer + eval CLIs work unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from ptv3_fmcw.models.velocity_head import VelocityHead


@dataclass
class PointMLPConfig:
    in_channels: int = 5
    width: int = 64
    n_blocks: int = 3
    head_hidden: int = 64
    head_dropout: float = 0.0


class PointMLPBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        hidden = width * 2
        self.fc1 = nn.Linear(width, hidden)
        self.fc2 = nn.Linear(hidden, width)
        self.ln = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc2(torch.nn.functional.gelu(self.fc1(x)))
        return self.ln(x + h)


class PointMLPVelocity(nn.Module):
    """Per-point MLP + global context. Same input/output contract as PTv3Velocity."""

    def __init__(self, cfg: PointMLPConfig | None = None):
        super().__init__()
        cfg = cfg or PointMLPConfig()
        self.cfg = cfg

        # +1 input channel for the derived log-range feature computed
        # in forward() (gives the head distance-from-sensor context).
        self.embed = nn.Sequential(
            nn.Linear(cfg.in_channels + 1, cfg.width),
            nn.GELU(),
            nn.LayerNorm(cfg.width),
        )
        self.blocks = nn.ModuleList(PointMLPBlock(cfg.width) for _ in range(cfg.n_blocks))
        self.fuse = nn.Sequential(
            nn.Linear(cfg.width * 2, cfg.width),
            nn.GELU(),
        )
        self.head = VelocityHead(in_channels=cfg.width, hidden=cfg.head_hidden,
                                 dropout=cfg.head_dropout)

    @staticmethod
    def _global_mean_per_frame(feat: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """Compute per-frame mean of feat, then broadcast to point dim.

        `offset` is the running cumulative-sum of per-frame point counts
        (Pointcept convention). Sizes:
            feat   : (N_total, C)
            offset : (B,)
        Returns (N_total, C) where each point gets its frame's mean.
        """
        out = torch.empty_like(feat)
        prev = 0
        for b in range(offset.shape[0]):
            end = int(offset[b].item())
            sl = slice(prev, end)
            mu = feat[sl].mean(dim=0, keepdim=True)
            out[sl] = mu.expand(end - prev, -1)
            prev = end
        return out

    def forward(self, data_dict: dict) -> torch.Tensor:
        feat = data_dict["feat"]
        offset = data_dict["offset"]
        # Derived log-range feature: log1p(||xyz||) keeps it bounded even
        # for nearby points; gives the head explicit distance context.
        log_range = torch.log1p(feat[:, 0:3].norm(dim=-1, keepdim=True))
        feat = torch.cat([feat, log_range], dim=-1)
        x = self.embed(feat)
        for blk in self.blocks:
            x = blk(x)
        ctx = self._global_mean_per_frame(x, offset)
        x = self.fuse(torch.cat([x, ctx], dim=-1))
        delta = self.head(x)  # (N, 2), zero-init so this starts at 0
        # Doppler residual: pred = v_radial * r̂_xy + delta. With the
        # zero-init head this exactly reproduces the B1 baseline at step
        # 0, so the model can only learn tangential improvements on top.
        coord_xy = feat[:, 0:2]
        v_radial = feat[:, 4:5]
        r_xy_norm = coord_xy.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        doppler_xy = v_radial * (coord_xy / r_xy_norm)
        return doppler_xy + delta


def build_point_mlp(cfg: PointMLPConfig | None = None) -> PointMLPVelocity:
    return PointMLPVelocity(cfg or PointMLPConfig())
