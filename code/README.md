# PTv3-FMCW thesis: baselines, model, training, eval

This directory contains two parts of the implementation:

1. Non-ML baselines (B0–B3) plus the shared metric stack and CLI. CPU-only.
2. Point Transformer V3 adapted for FMCW LiDAR (5-channel input, velocity head, training loop). Needs CUDA + Pointcept submodule.

The data pipeline, GT-velocity derivation, metrics, and B0–B3 are pure-numpy and run on any laptop. The PTv3 model imports torch + Pointcept lazily, so the rest of the package stays usable on machines without CUDA.

## What this gives you

- A pure-numpy data pipeline that reads AevaScenes v0.2 frames, derives
  per-point ground-truth (vx, vy) from the box `linear_velocity` /
  `angular_velocity` fields, and exposes a Pointcept-compatible record
  schema.
- The four mandatory non-ML baselines:
  - **B0** — predict (0, 0). The trivial floor.
  - **B1** — `pred = v_radial · r̂_xy` (Doppler-only).
  - **B2** — per-class mean GT velocity (uses GT class).
  - **B3** — Doppler radial component + class-mean tangential component.
- The shared metric stack: EPE_all / EPE_dyn / EPE_bg /
  EPE_t / EPE_r, angular error, magnitude error, with per-class,
  per-range, per-speed, per-tangential-dominance, and per-scene
  breakdowns.
- A B1 sign-check banner that catches the silent-invalidation risk
  on Doppler sign (verified 2026-05-05 against AevaScenes v0.2:
  positive Doppler = receding from sensor).
- Frozen-test lockfile protocol: test-split runs write a
  `LOCKED.txt` plus the git commit and config hash beside the metrics.
- LaTeX tables (T1, T2) and figures (F3, F4) ready to `\input{}` or
  `\includegraphics` from `Report.tex`.
- pytest coverage on every component (25 tests, all green locally).

## Layout

```
code/
├── environment.yml                  # CPU-only env (baselines + eval)
├── environment-train.yml            # adds torch, CUDA, Pointcept deps
├── README.md                         # this file
├── Pointcept/                        # git submodule (PTv3 backbone) — see Pointcept/README.md
├── configs/
│   ├── baseline.yaml                # eval config for B0–B3
│   ├── splits.json                  # 70/14/16 train/val/test, stratified
│   ├── ptv3_pilot.yaml              # 3-seq, 10-epoch debug run
│   ├── ptv3_full.yaml               # 50-epoch full v1 (A100 sized, batch 4)
│   └── ptv3_5080.yaml               # 50-epoch full v1 for RTX 5080 (batch 1 + grad accum 4)
├── ptv3_fmcw/
│   ├── data/
│   │   ├── aevascenes_dataset.py    # frame loader + record schema
│   │   ├── gt_velocity.py           # box → per-point (vx, vy) with ω×r
│   │   ├── class_names.py           # AevaScenes per-point class index ↔ name
│   │   └── transforms.py            # rotate-z, flip, scale, point dropout
│   ├── models/
│   │   ├── velocity_head.py         # 2-layer MLP, zero-init final│   │   └── ptv3_velocity.py         # PTv3 backbone + VelocityHead│   ├── training/
│   │   ├── config.py                # dataclass + YAML loader
│   │   ├── loss.py                  # smooth-L1 + dynamic/background weighting│   │   └── trainer.py               # AdamW + OneCycle + bf16│   └── eval/
│       ├── metrics.py               # MetricAccumulator with bucketed breakdowns
│       ├── baselines.py             # B0, B1, B2, B3
│       ├── evaluate.py              # frame-streaming loop + sign-check
│       ├── tables.py                # T1, T2 LaTeX writers
│       └── visualize.py             # F3, F4 matplotlib generators
├── scripts/
│   ├── make_splits.py               # generate configs/splits.json from EDA scene tags
│   ├── verify_data_assumptions.py   # one-shot v_radial sign + box frame check
│   ├── eval_baselines.py            # CLI: B0–B3 → metrics.json + T1/T2/F3/F4
│   ├── train.py                     # CLI: train PTv3 from a YAML config
│   └── eval_ptv3.py                 # CLI: evaluate a checkpoint, optionally with B0–B3
└── tests/                           # pytest suite, 33 cases
```

## Setup

The full conda env is described in `environment.yml`:

```bash
cd code/
conda env create -f environment.yml
conda activate ptv3-fmcw-baselines
```

Or, if you only want to run the baselines once on a machine that
already has Python ≥ 3.10:

```bash
pip install numpy pyyaml matplotlib pytest
```

The dataset is expected at `data/aevascenes_v0.2/<sequence_uuid>/`
relative to the repo root, or wherever `AEVASCENES_ROOT` points (the
config falls back to the in-tree path).

## Workflow

```bash
# from repo root
export PYTHONPATH=code

# 1. Generate the splits (one-time, deterministic; output is committed).
python code/scripts/make_splits.py \
    --data-root data/aevascenes_v0.2 \
    --output code/configs/splits.json

# 2. Verify the silent-invalidation risks against real data.
#    This is a 30-sequence sample taking ~30 s; output is captured in
#    data_exploration/stats/data_assumption_check.json.
python code/scripts/verify_data_assumptions.py

# 3. Run the test suite.
pytest code/tests/ -v

# 4. Run the val evaluation (writes frozen_test/<datetime>/).
python code/scripts/eval_baselines.py \
    --config code/configs/baseline.yaml \
    --split val

# 5. Inspect outputs.
ls frozen_test/<datetime>/
# metrics.json    sign_check.json    config.yaml    git_commit.txt
# tables/T1_main_results.tex    tables/T2_per_class_epet.tex
# figures/F3_tangential_dominance.pdf    figures/F4_per_class_epe.pdf

# 6. Final test-set run (locks the output dir).
python code/scripts/eval_baselines.py \
    --config code/configs/baseline.yaml \
    --split test
```

## Sign convention (already verified)

`scripts/verify_data_assumptions.py` was run against AevaScenes v0.2 on
2026-05-05 (50 moving-box samples drawn from 30 sequences):

| Hypothesis | mean residual `|measured − v · r̂|` |
|---|---|
| positive = **receding** | **1.18 m/s** |
| positive = approaching | 29.84 m/s |

The receding hypothesis is overwhelmingly correct. `B1` therefore uses
`pred = v_radial · r̂_xy` directly. If a future AevaScenes release
inverts the convention, set `output.v_radial_sign:
positive_approaches` in `baseline.yaml` and re-run; `eval_baselines.py`
will negate `v_radial` at load time.

The same exercise also confirms that box `linear_velocity` lives in
the **VEHICLE / ego-compensated** frame (residual 1.18 m/s, well
inside the 1.5 m/s threshold), matching the point-cloud Doppler
field — so background points carry GT (0, 0) and the formula
`gt = v_lin + ω × (p − c)` is the per-point GT for in-box points.

## Test discipline

The 16 test sequences are not to be touched until the official 3-seed
run. Even the baselines run on test only once — re-running overwrites
the lockfile and the methodology chapter must explain why.

## PTv3 training (RTX 5080)

```bash
# 1. Install GPU env + Pointcept submodule (see Pointcept/README.md).
conda env create -f code/environment-train.yml
conda activate ptv3-fmcw-train
git submodule update --init code/Pointcept
pip install flash-attn==2.6.3 --no-build-isolation
pip install spconv-cu120

# 2. Pilot run (3 sequences, 10 epochs, batch 2, ~30 min on 5080).
python code/scripts/train.py --config code/configs/ptv3_pilot.yaml --no-wandb

# 3. Full v1 run on the 5080 (batch 1 + grad accum 4 -> effective batch 4).
python code/scripts/train.py --config code/configs/ptv3_5080.yaml

# 4. Evaluate the checkpoint, including B0-B3 for the T1 row.
python code/scripts/eval_ptv3.py \
    --config code/configs/ptv3_5080.yaml \
    --checkpoint checkpoints/ptv3_5080/best.pt \
    --split val \
    --include-baselines
```

The val EPE_t baseline number to beat is **3.14 m/s** (B1 Doppler-only,
1400 frames, see `frozen_test/2026-05-06T00-28-36/`).
B3 (Doppler + class-mean tangent) gets 3.00 m/s; PTv3's added value
shows up if it beats that.
