# Context-Aware Tangential Velocity Inference for FMCW LiDAR

Bachelor's thesis, Linnaeus University (2DV50E, VT 2026)
**Students:** Suyash Mullick, Saisharan Raja
**Supervisors:** Amilcar Soares (LNU), Syargey Prasalovich (Einride)

FMCW LiDAR measures per-point radial (Doppler) velocity but cannot directly observe tangential components. This project designs and evaluates ML models that infer full 3D velocity vectors from a single FMCW LiDAR frame using geometric, semantic, and beam-geometry context.

## Repository Layout

```
Report.tex          LaTeX thesis source (uses references.bib, figures/, img/)
Report.pdf          compiled thesis
proposal/           project proposal (SPP) sources and PDF
literature/         literature review: notes, synthesis, papers.tsv, search_log.tsv
code/               implementation: data pipeline, baselines (B0-B3), PTv3-FMCW model, training, eval
data_exploration/   AevaScenes EDA scripts, stats, and figures
frozen_test/        locked test/val evaluation outputs (metrics, tables, figures)
scripts/            dataset download and extraction helpers
figures/, img/      figures used by Report.tex
```

Primary dataset: AevaScenes v0.2 (2025), 6 FMCW LiDARs, 100 sequences, 10,000 frames.

## Quick start

```bash
# 1. Conda env for baselines + eval (CPU-only).
cd code && conda env create -f environment.yml
conda activate ptv3-fmcw-baselines

# 2. Generate the train/val/test splits.
python code/scripts/make_splits.py \
    --data-root data/aevascenes_v0.2 \
    --output code/configs/splits.json

# 3. Run the baselines on the val split.
python code/scripts/eval_baselines.py \
    --config code/configs/baseline.yaml --split val
```

For PTv3 training (CUDA + Pointcept submodule) see `code/README.md`.
