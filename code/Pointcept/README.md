# Pointcept submodule

This directory is the mount point for the
[Pointcept](https://github.com/Pointcept/Pointcept) backbone library.
PTv3-FMCW depends on Pointcept as a **git submodule pinned to a specific
commit, never edited**. The submodule is intentionally not initialised
in this repo so that the baseline-algorithms tree stays usable on
machines without CUDA.

## One-time setup

```bash
# from the repo root
git submodule add https://github.com/Pointcept/Pointcept.git code/Pointcept
git -C code/Pointcept checkout v1.5.3       # or the commit you have
git submodule update --init --recursive
```

## Adapter

`code/ptv3_fmcw/models/ptv3_velocity.py` imports
`pointcept.models.point_transformer_v3.point_transformer_v3m1_base.PointTransformerV3`
with `in_channels=5` (the only adaptation). No source files inside
`code/Pointcept/` should be modified; every FMCW change lives in
`code/ptv3_fmcw/`.

## Required Python packages

The full conda env is in `../environment-train.yml`. Two packages need
post-install steps because they build CUDA kernels:

```bash
# After torch is installed:
pip install flash-attn==2.6.3 --no-build-isolation
pip install spconv-cu120
```

## Why we do not bundle a copy

Pointcept is ~10 MB of Python source plus heavy build-time CUDA
dependencies. Vendoring it into this repo would (a) bloat the diff,
(b) make commit pinning impossible, and (c) silently mask upstream
bugfixes. Submodule + pinned commit is the chosen approach.
