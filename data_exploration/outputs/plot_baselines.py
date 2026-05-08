"""
Baseline comparison charts for FMCW LiDAR tangential velocity thesis.
Reads frozen_test metrics and produces three figures saved to this directory.
Run with: conda run -n aevascenes python data/outputs/plot_baselines.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).parent
METRICS = Path("frozen_test/2026-05-06T00-28-36/metrics.json")

with open(METRICS) as f:
    raw = f.read().replace(": NaN", ": null")
data = json.loads(raw)["baselines"]

# ── colours ──────────────────────────────────────────────────────────────────
C = {
    "B0_zero":                    "#aaaaaa",
    "B1_doppler_only":            "#4c9be8",
    "B2_class_mean":              "#e8874c",
    "B3_doppler_plus_class_mean": "#5bbf6e",
    "PointMLP_FMCW":              "#b05ec4",
}
LABELS = {
    "B0_zero":                    "B0  Zero",
    "B1_doppler_only":            "B1  Doppler-only",
    "B2_class_mean":              "B2  Class-mean",
    "B3_doppler_plus_class_mean": "B3  Dop. + class-tan",
    "PointMLP_FMCW":              "PointMLP (trained)",
}

# ── overall metrics ───────────────────────────────────────────────────────────
methods_b = list(data.keys())
epe_dyn = [data[m]["overall"]["epe_dyn"] for m in methods_b]
epe_t   = [data[m]["overall"]["epe_t"]   for m in methods_b]
epe_r   = [data[m]["overall"]["epe_r"]   for m in methods_b]

# append PointMLP best run (frozen commit 96eccca)
methods   = methods_b + ["PointMLP_FMCW"]
epe_dyn_v = epe_dyn + [3.870]
epe_t_v   = epe_t   + [3.137]
epe_r_v   = epe_r   + [1.296]

colors = [C[m] for m in methods]
xlabels = [LABELS[m] for m in methods]

# ── Figure 1: grouped bar — EPE_dyn / EPE_t / EPE_r ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(methods))
w = 0.26
bars_dyn = ax.bar(x - w, epe_dyn_v, w, label="EPE$_{dyn}$",  color=[c + "cc" for c in colors], edgecolor="white", linewidth=0.6)
bars_t   = ax.bar(x,     epe_t_v,   w, label="EPE$_t$",      color=colors,                    edgecolor="white", linewidth=0.6)
bars_r   = ax.bar(x + w, epe_r_v,   w, label="EPE$_r$",      color=[c + "66" for c in colors], edgecolor="white", linewidth=0.6)

for bar, val in zip(bars_t, epe_t_v):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
            f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel("Error (m/s)", fontsize=11)
ax.set_title("Baseline comparison — val split (1 400 frames)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.85)
ax.set_ylim(0, 11)
ax.axhline(3.00, color="#5bbf6e", linewidth=1.0, linestyle="--", alpha=0.6, label="_B3 EPE_t bar")
ax.text(len(methods) - 0.45, 3.08, "B3 EPE$_t$ bar (3.00)", color="#5bbf6e", fontsize=7.5, alpha=0.9)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig1_grouped_bar.png", dpi=160)
plt.close()
print("saved fig1_grouped_bar.png")

# ── Figure 2: EPE_t only, horizontal bar — easy to compare the key metric ───
fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(methods))
bars = ax.barh(y, epe_t_v, color=colors, edgecolor="white", linewidth=0.6, height=0.55)
for bar, val in zip(bars, epe_t_v):
    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f} m/s", va="center", fontsize=9)
ax.set_yticks(y)
ax.set_yticklabels(xlabels, fontsize=9)
ax.set_xlabel("EPE$_t$ (m/s) — lower is better", fontsize=10)
ax.set_title("Tangential error (EPE$_t$) — the metric that matters", fontsize=11, fontweight="bold")
ax.axvline(3.00, color="#5bbf6e", linewidth=1.2, linestyle="--", alpha=0.8)
ax.text(3.02, len(methods) - 0.6, "B3 bar\n3.00", color="#5bbf6e", fontsize=7.5)
ax.set_xlim(0, 14)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig2_epet_horizontal.png", dpi=160)
plt.close()
print("saved fig2_epet_horizontal.png")

# ── Figure 3: per-class EPE_t for B1 / B3 / PointMLP ────────────────────────
# classes with enough dynamic points in val
class_epe_t = {}
for cls, vals in data["B1_doppler_only"]["per_class"].items():
    ndyn = vals.get("n_dyn_total", 0) or 0
    et = vals.get("epe_t")
    if ndyn >= 500 and et is not None:
        class_epe_t[cls] = {
            "B1": et,
            "B3": data["B3_doppler_plus_class_mean"]["per_class"][cls]["epe_t"],
        }

cls_names = sorted(class_epe_t, key=lambda c: class_epe_t[c]["B1"], reverse=True)
b1_vals = [class_epe_t[c]["B1"] for c in cls_names]
b3_vals = [class_epe_t[c]["B3"] for c in cls_names]

fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(cls_names))
w = 0.35
ax.bar(x - w/2, b1_vals, w, label="B1 Doppler-only", color=C["B1_doppler_only"], alpha=0.9)
ax.bar(x + w/2, b3_vals, w, label="B3 Dop. + class-tan", color=C["B3_doppler_plus_class_mean"], alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", " ") for c in cls_names], rotation=30, ha="right", fontsize=9)
ax.set_ylabel("EPE$_t$ (m/s)", fontsize=10)
ax.set_title("Per-class tangential error — B1 vs B3", fontsize=11, fontweight="bold")
ax.legend(fontsize=9, framealpha=0.85)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "fig3_per_class_epet.png", dpi=160)
plt.close()
print("saved fig3_per_class_epet.png")

print("\nAll figures written to", OUT)
