"""PTv3Velocity: Pointcept PTv3 backbone + VelocityHead (spec §4.1).

The backbone is unmodified Pointcept code, instantiated with
`in_channels=5` (the only adaptation: spec §4.3). All FMCW-specific
behaviour lives in the dataset's 5-channel `feat` and the regression
head defined here.

The Pointcept submodule is expected at `code/Pointcept/`; the
training entry point puts that path on `sys.path` before importing.
This module does not import Pointcept at file-load time so it is safe
to import on machines without the submodule cloned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BackboneConfig:
    """PTv3 outdoor configuration (spec §4.2)."""

    in_channels: int = 5
    order: tuple[str, ...] = ("z", "z-trans", "hilbert", "hilbert-trans")
    stride: tuple[int, ...] = (2, 2, 2, 2)
    enc_depths: tuple[int, ...] = (2, 2, 2, 6, 2)
    enc_channels: tuple[int, ...] = (32, 64, 128, 256, 512)
    enc_num_head: tuple[int, ...] = (2, 4, 8, 16, 32)
    enc_patch_size: tuple[int, ...] = (1024,) * 5
    dec_depths: tuple[int, ...] = (2, 2, 2, 2)
    dec_channels: tuple[int, ...] = (64, 64, 128, 256)
    dec_num_head: tuple[int, ...] = (4, 4, 8, 16)
    dec_patch_size: tuple[int, ...] = (1024,) * 4
    mlp_ratio: int = 4
    qkv_bias: bool = True
    enable_rpe: bool = False
    enable_flash: bool = True
    cls_mode: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in self.__dict__.items()}


@dataclass
class PTv3VelocityConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    head_hidden: int = 64
    head_dropout: float = 0.0


def _import_ptv3():
    """Import Pointcept's PointTransformerV3 lazily."""
    try:
        from pointcept.models.point_transformer_v3.point_transformer_v3m1_base import (  # type: ignore
            PointTransformerV3,
        )
    except ImportError as e:  # pragma: no cover -- requires the submodule
        raise ImportError(
            "Pointcept's PointTransformerV3 not found on sys.path. "
            "Initialise the submodule via `git submodule update --init "
            "code/Pointcept` and ensure code/Pointcept is on PYTHONPATH "
            "(scripts/train.py does this automatically)."
        ) from e
    return PointTransformerV3


def build_ptv3_velocity(cfg: PTv3VelocityConfig | None = None):
    """Lazy constructor: imports torch + Pointcept at call time."""
    cfg = cfg or PTv3VelocityConfig()
    return PTv3Velocity(cfg)


try:
    import torch
    import torch.nn as nn

    from ptv3_fmcw.models.velocity_head import VelocityHead

    class PTv3Velocity(nn.Module):
        """Composed model. Forward signature follows Pointcept's outdoor
        semseg convention: `data_dict` carries `coord`, `feat`,
        `grid_size`, `offset`."""

        def __init__(self, cfg: PTv3VelocityConfig | None = None):
            super().__init__()
            cfg = cfg or PTv3VelocityConfig()
            self.cfg = cfg
            PointTransformerV3 = _import_ptv3()
            self.backbone = PointTransformerV3(**cfg.backbone.to_dict())
            self.head = VelocityHead(
                in_channels=cfg.backbone.dec_channels[0],
                hidden=cfg.head_hidden,
                dropout=cfg.head_dropout,
            )

        def forward(self, data_dict: dict) -> "torch.Tensor":
            point = self.backbone(data_dict)
            # Pointcept's outdoor semseg pipeline returns a `Point` object
            # exposing `.feat` of shape (N_total, dec_channels[0]).
            feat = getattr(point, "feat", point)
            return self.head(feat)

except ImportError:  # pragma: no cover

    class PTv3Velocity:  # type: ignore[no-redef]
        def __init__(self, *_a, **_k):
            raise ImportError(
                "PTv3Velocity requires PyTorch and the Pointcept submodule. "
                "See code/README.md for setup."
            )
