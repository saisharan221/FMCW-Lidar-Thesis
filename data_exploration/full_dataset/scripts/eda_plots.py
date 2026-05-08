"""Generate EDA plots from precomputed stats."""
import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

ROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset")
STATS = ROOT / "stats"
PLOTS = ROOT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

LIDAR_IDS = ["front_wide_lidar", "front_narrow_lidar",
             "right_lidar", "rear_wide_lidar", "rear_narrow_lidar", "left_lidar"]
LIDAR_COLOR = {
    "front_wide_lidar": "#1f77b4",
    "front_narrow_lidar": "#ff7f0e",
    "right_lidar": "#2ca02c",
    "rear_wide_lidar": "#d62728",
    "rear_narrow_lidar": "#9467bd",
    "left_lidar": "#8c564b",
}


def load():
    seq = json.loads((STATS / "per_sequence.json").read_text())
    summary = json.loads((STATS / "summary.json").read_text())
    pcd = json.loads((STATS / "pointcloud_stats.json").read_text())
    traj = json.loads((STATS / "trajectories.json").read_text())
    return seq, summary, pcd, traj


# ───────────────────────────────────────────────────────────────────────────────
def plot_scene_distribution(summary, seq):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # scene-type pie
    sc = summary["scene_distribution_str"]
    labels, counts = zip(*sorted(sc.items(), key=lambda x: -x[1]))
    colors_pie = ["#3a86ff", "#ffb703", "#1d3557", "#fb5607"]
    axes[0].pie(counts, labels=labels, autopct="%d", colors=colors_pie,
                wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    axes[0].set_title(f"Scene type ({summary['n_sequences']} sequences)")

    # ego speed distribution per scene
    speeds_by_scene = {}
    for r in seq:
        k = f"{r['road_type']}|{r['lighting']}"
        speeds_by_scene.setdefault(k, []).append(r["ego_mean_speed_mps"])
    keys = sorted(speeds_by_scene)
    data = [speeds_by_scene[k] for k in keys]
    bp = axes[1].boxplot(data, labels=keys, patch_artist=True)
    for p, c in zip(bp["boxes"], colors_pie[:len(keys)]):
        p.set_facecolor(c); p.set_alpha(0.6)
    axes[1].set_ylabel("Mean ego speed (m/s)")
    axes[1].set_title("Ego speed by scene type")
    axes[1].grid(True, alpha=0.3)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")

    plt.tight_layout()
    plt.savefig(PLOTS / "01_scene_and_ego_speed.png", dpi=140)
    plt.close()


def plot_class_distribution(summary):
    cc = summary["class_counts"]
    items = sorted(cc.items(), key=lambda x: -x[1])
    labels, counts = zip(*items)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(items))))
    bars = ax.barh(labels, counts, color="#3a86ff")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Bounding-box count (log scale)")
    ax.set_title(f"Class distribution across all 100 sequences "
                 f"({sum(counts):,} bboxes)")
    for b, c in zip(bars, counts):
        ax.text(c, b.get_y() + b.get_height() / 2, f" {c:,}",
                va="center", fontsize=8)
    ax.grid(axis="x", alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(PLOTS / "02_class_distribution.png", dpi=140)
    plt.close()


def plot_bbox_speed(summary):
    bs = summary["bbox_speed_per_class_stats"]
    items = sorted(bs.items(), key=lambda x: -x[1]["n"])[:12]
    if not items:
        return
    labels = [k for k, _ in items]
    means = [v["mean"] for _, v in items]
    medians = [v["median"] for _, v in items]
    p95s = [v["p95"] for _, v in items]
    maxes = [v["max"] for _, v in items]
    x = np.arange(len(labels))
    w = 0.22
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - 1.5*w, medians, w, label="median", color="#1f77b4")
    ax.bar(x - 0.5*w, means,   w, label="mean",   color="#3a86ff")
    ax.bar(x + 0.5*w, p95s,    w, label="p95",    color="#ffb703")
    ax.bar(x + 1.5*w, maxes,   w, label="max",    color="#fb5607")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Speed (m/s)")
    ax.set_title("Annotated bounding-box linear-velocity speed by class (top-12 by count)")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "03_bbox_speed_per_class.png", dpi=140)
    plt.close()


def plot_trajectories(traj, seq):
    sc_color = {"city|day": "#3a86ff", "city|night": "#1d3557",
                "highway|day": "#ffb703", "highway|night": "#fb5607"}
    seq_meta = {r["uuid"]: r for r in seq}
    fig, ax = plt.subplots(figsize=(8, 8))
    for u, p in traj.items():
        p = np.asarray(p)
        m = seq_meta.get(u, {})
        k = f"{m.get('road_type','?')}|{m.get('lighting','?')}"
        ax.plot(p[:, 0], p[:, 1], color=sc_color.get(k, "gray"),
                alpha=0.7, linewidth=1)
    # legend
    for k, c in sc_color.items():
        ax.plot([], [], color=c, label=k, linewidth=2)
    ax.set_aspect("equal")
    ax.set_xlabel("Ego X (m, frame-0 ref)"); ax.set_ylabel("Ego Y (m)")
    ax.set_title("Ego trajectories — all 100 sequences (10-s clips, ≤≈100 m each)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "04_ego_trajectories.png", dpi=140)
    plt.close()


def plot_lidar_fov_polar(pcd):
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "polar"})
    ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
    az_bins = np.array(pcd["config"]["az_bins"])
    az_centers = (az_bins[:-1] + az_bins[1:]) / 2
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["az_hist"], dtype=float)
        if h.sum() > 0:
            h = h / h.max()
        ax.plot(np.deg2rad(az_centers), h, label=lid,
                color=LIDAR_COLOR[lid], linewidth=2)
    ax.set_title("Per-LiDAR azimuth coverage (sensor frame, normalized)\n"
                 "0° = sensor-forward axis", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9)
    plt.tight_layout()
    plt.savefig(PLOTS / "05_lidar_fov_polar.png", dpi=140)
    plt.close()


def plot_lidar_range(pcd):
    bins = np.array(pcd["config"]["range_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["range_hist"], dtype=float)
        h = h / h.sum() if h.sum() > 0 else h
        ax.plot(centers, h, label=lid, color=LIDAR_COLOR[lid], linewidth=1.8)
    ax.set_xlabel("Range (m)"); ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.set_title("Per-LiDAR range distribution (log scale; 5-m bins)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "06_lidar_range.png", dpi=140)
    plt.close()


def plot_lidar_velocity(pcd):
    bins = np.array(pcd["config"]["vel_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # All velocities, log-y
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["vel_hist"], dtype=float)
        h = h / h.sum() if h.sum() > 0 else h
        axes[0].plot(centers, h, label=lid, color=LIDAR_COLOR[lid], linewidth=1.5)
    axes[0].set_xlabel("Per-point Doppler velocity (m/s)")
    axes[0].set_ylabel("Density")
    axes[0].set_yscale("log"); axes[0].set_xlim(-30, 30)
    axes[0].set_title("FMCW Doppler velocity distribution per LiDAR")
    axes[0].axvline(0, color="k", linestyle="--", alpha=0.4)
    axes[0].grid(True, which="both", alpha=0.3); axes[0].legend(fontsize=9)

    # Zoom on near-zero
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["vel_hist"], dtype=float)
        h = h / h.sum() if h.sum() > 0 else h
        axes[1].plot(centers, h, label=lid, color=LIDAR_COLOR[lid], linewidth=1.5)
    axes[1].set_xlabel("Per-point Doppler velocity (m/s)")
    axes[1].set_ylabel("Density")
    axes[1].set_xlim(-3, 3)
    axes[1].set_title("Zoomed: |v| < 3 m/s (residual on stationary scene)")
    axes[1].axvline(0, color="k", linestyle="--", alpha=0.4)
    axes[1].grid(True, alpha=0.3); axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(PLOTS / "07_lidar_velocity.png", dpi=140)
    plt.close()


def plot_lidar_velocity_by_scene(pcd):
    bins = np.array(pcd["config"]["vel_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    sc_color = {"city|day": "#3a86ff", "city|night": "#1d3557",
                "highway|day": "#ffb703", "highway|night": "#fb5607"}
    for sk, info in pcd["per_scene"].items():
        h = np.array(info["vel_hist"], dtype=float)
        h = h / h.sum() if h.sum() > 0 else h
        ax.plot(centers, h, label=f"{sk} (n_frames={info['n_files']//6})",
                color=sc_color.get(sk, "gray"), linewidth=1.8)
    ax.set_xlabel("Per-point Doppler velocity (m/s)")
    ax.set_ylabel("Density")
    ax.set_yscale("log"); ax.set_xlim(-30, 30)
    ax.set_title("Doppler velocity distribution by scene type")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "08_velocity_by_scene.png", dpi=140)
    plt.close()


def plot_points_per_frame(pcd):
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [pcd["per_lidar"][lid]["points_per_frame_mean"] for lid in LIDAR_IDS]
    p95s  = [pcd["per_lidar"][lid]["points_per_frame_p95"] for lid in LIDAR_IDS]
    medians = [pcd["per_lidar"][lid]["points_per_frame_median"] for lid in LIDAR_IDS]
    x = np.arange(len(LIDAR_IDS))
    w = 0.27
    ax.bar(x - w, medians, w, label="median", color="#3a86ff")
    ax.bar(x,     means,   w, label="mean",   color="#ffb703")
    ax.bar(x + w, p95s,    w, label="p95",    color="#fb5607")
    ax.set_xticks(x); ax.set_xticklabels(LIDAR_IDS, rotation=20, ha="right")
    ax.set_ylabel("Points per frame")
    ax.set_title("Points per frame by LiDAR")
    ax.grid(axis="y", alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "09_points_per_frame.png", dpi=140)
    plt.close()


def plot_semantic_pcd(pcd):
    # aggregate semantic labels across all lidars
    agg = Counter()
    for lid in LIDAR_IDS:
        for k, v in pcd["per_lidar"][lid]["semantic_counts"].items():
            agg[k] += v
    items = sorted(agg.items(), key=lambda x: -x[1])
    if not items:
        return
    labels, counts = zip(*items)
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(items))))
    ax.barh(labels, counts, color="#06a77d")
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Per-point label count (log)")
    ax.set_title(f"Per-point semantic labels (sampled {pcd['config']['frames_per_seq']} frames/seq × 6 lidars; total {total:,} pts)")
    for i, (l, c) in enumerate(items):
        ax.text(c, i, f" {c:,}  ({100*c/total:.2f}%)", va="center", fontsize=8)
    ax.grid(axis="x", which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / "10_semantic_pointcloud.png", dpi=140)
    plt.close()


def plot_seq_overview(seq):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    # bbox per frame distribution
    bpf = [r["bbox_per_frame_mean"] for r in seq]
    axes[0, 0].hist(bpf, bins=20, color="#3a86ff", edgecolor="white")
    axes[0, 0].set_xlabel("Mean bboxes per frame"); axes[0, 0].set_ylabel("# sequences")
    axes[0, 0].set_title(f"Bboxes per frame across sequences (mean of means = {np.mean(bpf):.1f})")
    axes[0, 0].grid(alpha=0.3)

    # unique tracks per sequence
    ut = [r["n_unique_tracks"] for r in seq]
    axes[0, 1].hist(ut, bins=20, color="#06a77d", edgecolor="white")
    axes[0, 1].set_xlabel("Unique tracking IDs / sequence")
    axes[0, 1].set_ylabel("# sequences")
    axes[0, 1].set_title(f"Unique tracks per sequence (median={int(np.median(ut))})")
    axes[0, 1].grid(alpha=0.3)

    # ego speed
    sp = [r["ego_mean_speed_mps"] for r in seq]
    axes[1, 0].hist(sp, bins=25, color="#ffb703", edgecolor="white")
    axes[1, 0].axvline(np.median(sp), color="k", ls="--", label=f"median {np.median(sp):.1f} m/s")
    axes[1, 0].set_xlabel("Mean ego speed (m/s)")
    axes[1, 0].set_ylabel("# sequences")
    axes[1, 0].set_title(f"Mean ego speed per sequence")
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    # size per sequence
    sz = [r["size_mb"] for r in seq]
    axes[1, 1].hist(sz, bins=25, color="#fb5607", edgecolor="white")
    axes[1, 1].set_xlabel("Sequence on-disk size (MB)")
    axes[1, 1].set_ylabel("# sequences")
    axes[1, 1].set_title(f"Per-sequence size (mean={np.mean(sz):.0f} MB)")
    axes[1, 1].grid(alpha=0.3)

    plt.suptitle("Per-sequence statistics")
    plt.tight_layout()
    plt.savefig(PLOTS / "11_per_sequence_overview.png", dpi=140)
    plt.close()


def plot_lidar_elevation(pcd):
    bins = np.array(pcd["config"]["el_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["el_hist"], dtype=float)
        h = h / h.sum() if h.sum() > 0 else h
        ax.plot(centers, h, label=lid, color=LIDAR_COLOR[lid], linewidth=1.8)
    ax.set_xlabel("Elevation (deg, sensor frame)")
    ax.set_ylabel("Density")
    ax.axvline(0, color="k", linestyle="--", alpha=0.3)
    ax.set_title("Per-LiDAR elevation distribution")
    ax.grid(True, alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "12_lidar_elevation.png", dpi=140)
    plt.close()


def plot_lidar_reflectivity(pcd):
    bins = np.array(pcd["config"]["refl_bins"])
    centers = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(10, 5))
    for lid in LIDAR_IDS:
        h = np.array(pcd["per_lidar"][lid]["refl_hist"], dtype=float)
        if h.sum() > 0:
            h = h / h.sum()
        ax.plot(centers, h, label=lid, color=LIDAR_COLOR[lid], linewidth=1.5)
    ax.set_xlabel("Reflectivity (raw)")
    ax.set_ylabel("Density"); ax.set_yscale("log")
    ax.set_xlim(0, 4096)
    ax.set_title("Per-LiDAR reflectivity distribution")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "13_lidar_reflectivity.png", dpi=140)
    plt.close()


def main():
    seq, summary, pcd, traj = load()

    plot_scene_distribution(summary, seq)
    plot_class_distribution(summary)
    plot_bbox_speed(summary)
    plot_trajectories(traj, seq)
    plot_lidar_fov_polar(pcd)
    plot_lidar_range(pcd)
    plot_lidar_velocity(pcd)
    plot_lidar_velocity_by_scene(pcd)
    plot_points_per_frame(pcd)
    plot_semantic_pcd(pcd)
    plot_seq_overview(seq)
    plot_lidar_elevation(pcd)
    plot_lidar_reflectivity(pcd)

    print(f"Wrote plots to {PLOTS}")


if __name__ == "__main__":
    main()
