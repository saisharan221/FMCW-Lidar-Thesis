"""PTv3-FMCW model components.

`PTv3Velocity` composes Pointcept's unmodified PointTransformerV3
backbone with `VelocityHead` for per-point (vx, vy) regression in the
VEHICLE frame.

Lazy torch import: this package is safe to import on machines without
torch installed (the data pipeline + B0-B3 baselines are torch-free).
The torch import only happens when constructing `VelocityHead` or
`PTv3Velocity`.
"""
from ptv3_fmcw.models.velocity_head import VelocityHead, build_velocity_head
from ptv3_fmcw.models.ptv3_velocity import PTv3Velocity, build_ptv3_velocity

__all__ = [
    "PTv3Velocity",
    "VelocityHead",
    "build_ptv3_velocity",
    "build_velocity_head",
]
