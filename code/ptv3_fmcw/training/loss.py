"""Velocity regression loss (spec §5.1).

    L = sum_p w(p) * Huber(pred_p, gt_p, beta)  /  sum_p w(p)

with `w_dyn = 1.0` for points where ||gt|| > 0.5 m/s, `w_bg = 0.2`
otherwise, and Huber β = 0.5 m/s. Spec §5.1 logs an unweighted variant
in parallel for comparison against literature.

Pure functional API; the trainer is responsible for moving tensors to
device and aggregating across grad-accum steps.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossConfig:
    huber_beta: float = 0.5
    w_dyn: float = 1.0
    w_bg: float = 0.2
    dynamic_threshold: float = 0.5  # m/s — same threshold as eval metrics


try:
    import torch
    import torch.nn.functional as F

    def velocity_loss(
        pred_vxy: "torch.Tensor",
        gt_vxy: "torch.Tensor",
        gt_mask: "torch.Tensor",
        cfg: LossConfig | None = None,
        return_components: bool = False,
    ):
        """Per-point smooth-L1 with dynamic/background weighting.

        Parameters
        ----------
        pred_vxy : (N, 2) tensor of predicted (vx, vy) in m/s
        gt_vxy   : (N, 2) tensor of ground-truth (vx, vy) in m/s
        gt_mask  : (N,) bool tensor — points to include
        cfg      : LossConfig (defaults to spec §5.1 values)

        Returns
        -------
        loss : scalar tensor (weighted mean)
        components : optional dict {loss_unweighted, loss_dyn, loss_bg, n_dyn, n_bg}
        """
        cfg = cfg or LossConfig()
        gt_speed = torch.linalg.norm(gt_vxy, dim=-1)
        is_dyn = gt_speed > cfg.dynamic_threshold

        huber = F.smooth_l1_loss(pred_vxy, gt_vxy, beta=cfg.huber_beta, reduction="none").sum(-1)
        w = torch.where(is_dyn, cfg.w_dyn, cfg.w_bg) * gt_mask.float()
        denom = w.sum().clamp_min(1.0)
        loss = (huber * w).sum() / denom

        if not return_components:
            return loss

        unweighted = (huber * gt_mask.float()).sum() / gt_mask.float().sum().clamp_min(1.0)
        dyn_mask = is_dyn & gt_mask
        bg_mask = (~is_dyn) & gt_mask
        loss_dyn = (huber[dyn_mask].mean() if dyn_mask.any() else huber.new_zeros(()))
        loss_bg = (huber[bg_mask].mean() if bg_mask.any() else huber.new_zeros(()))
        return loss, {
            "loss_unweighted": unweighted.detach(),
            "loss_dyn": loss_dyn.detach(),
            "loss_bg": loss_bg.detach(),
            "n_dyn": int(dyn_mask.sum().item()),
            "n_bg": int(bg_mask.sum().item()),
        }

except ImportError:  # pragma: no cover

    def velocity_loss(*_a, **_k):  # type: ignore[no-redef]
        raise ImportError("velocity_loss requires PyTorch.")
