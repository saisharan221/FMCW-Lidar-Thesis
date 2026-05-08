"""Sensor-rig geometry: extract & visualize extrinsics from sequence.json metadata."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
OUT_STATS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/stats")
OUT_PLOTS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/plots")

LIDAR_COLOR = {
    "front_wide_lidar": "#1f77b4",
    "front_narrow_lidar": "#ff7f0e",
    "right_lidar": "#2ca02c",
    "rear_wide_lidar": "#d62728",
    "rear_narrow_lidar": "#9467bd",
    "left_lidar": "#8c564b",
}
CAM_COLOR = {k.replace("lidar", "camera"): v for k, v in LIDAR_COLOR.items()}


def quat_to_rot(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main():
    # Confirm extrinsics are constant across sequences (sanity)
    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])
    rigs = []
    for sd in seq_dirs[:20]:  # 20 is plenty to confirm consistency
        d = json.loads((sd / "sequence.json").read_text())
        md = d["metadata"]
        rig = {"lidars": {}, "cameras": {}}
        for lid, ext in md["vehicle_to_lidar_extrinsics"].items():
            t = ext["translation"]; q = ext["rotation"]
            rig["lidars"][lid] = {"t": [t["x"], t["y"], t["z"]],
                                   "q": [q["w"], q["x"], q["y"], q["z"]]}
        for cid, ext in md["vehicle_to_camera_extrinsics"].items():
            t = ext["translation"]; q = ext["rotation"]
            rig["cameras"][cid] = {"t": [t["x"], t["y"], t["z"]],
                                    "q": [q["w"], q["x"], q["y"], q["z"]]}
        rigs.append(rig)

    # Check consistency
    base = rigs[0]
    consistent = True
    max_dev = {"lidar": 0.0, "camera": 0.0}
    for r in rigs[1:]:
        for lid, info in r["lidars"].items():
            d = np.linalg.norm(np.array(info["t"]) - np.array(base["lidars"][lid]["t"]))
            if d > 1e-3: consistent = False
            max_dev["lidar"] = max(max_dev["lidar"], d)
        for cid, info in r["cameras"].items():
            d = np.linalg.norm(np.array(info["t"]) - np.array(base["cameras"][cid]["t"]))
            if d > 1e-3: consistent = False
            max_dev["camera"] = max(max_dev["camera"], d)
    print(f"Extrinsics consistency across 20 sequences: {consistent}")
    print(f"max translation deviation — lidar: {max_dev['lidar']:.6f} m, camera: {max_dev['camera']:.6f} m")

    # Save the canonical rig
    canonical = base
    # Add forward direction (rotation applied to [1,0,0])
    for group in ("lidars", "cameras"):
        for sid, info in canonical[group].items():
            R = quat_to_rot({"w": info["q"][0], "x": info["q"][1],
                             "y": info["q"][2], "z": info["q"][3]})
            info["forward_xy"] = (R @ np.array([1.0, 0.0, 0.0]))[:2].tolist()
            info["yaw_deg"] = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

    # Inter-sensor baselines (lidar-lidar)
    lids = list(canonical["lidars"].keys())
    baseline = {}
    for i, a in enumerate(lids):
        for b in lids[i+1:]:
            ta = np.array(canonical["lidars"][a]["t"])
            tb = np.array(canonical["lidars"][b]["t"])
            baseline[f"{a}--{b}"] = float(np.linalg.norm(ta - tb))

    out = {"consistency_max_dev_m": max_dev,
           "rig": canonical,
           "lidar_lidar_baselines_m": baseline}
    OUT_STATS.mkdir(parents=True, exist_ok=True)
    (OUT_STATS / "sensor_rig.json").write_text(json.dumps(out, indent=2))

    # ── Plot top-down rig ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    # vehicle silhouette (rough sedan ~5 m × 2 m)
    L, W = 5.0, 2.0
    veh = Rectangle((-L/2 + 1.0, -W/2), L, W, fill=True, facecolor="#e9ecef",
                    edgecolor="#495057", linewidth=1.3, zorder=0)
    ax.add_patch(veh)
    ax.text(1.0, 0, "ego", ha="center", va="center", fontsize=10, color="#495057",
            zorder=1)

    # lidars
    for lid, info in canonical["lidars"].items():
        x, y, z = info["t"]
        fwd = info["forward_xy"]
        ax.scatter(x, y, c=LIDAR_COLOR[lid], s=180, marker="o",
                   edgecolor="black", linewidth=1.0, zorder=3, label=lid)
        ax.add_patch(FancyArrowPatch((x, y), (x + fwd[0]*1.5, y + fwd[1]*1.5),
                                      arrowstyle="-|>", mutation_scale=12,
                                      color=LIDAR_COLOR[lid], zorder=2))

    # cameras (slightly different marker)
    for cid, info in canonical["cameras"].items():
        x, y, z = info["t"]
        ax.scatter(x, y, c=CAM_COLOR[cid], s=80, marker="s",
                   edgecolor="black", linewidth=0.8, alpha=0.7, zorder=3)

    ax.set_aspect("equal")
    ax.set_xlim(-3.5, 4.5); ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("Vehicle X (m, forward)")
    ax.set_ylabel("Vehicle Y (m, left)")
    ax.set_title("AevaScenes sensor rig — top-down view\n"
                 "circles = LiDAR (with forward-axis arrow), squares = camera")
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "16_sensor_rig_topdown.png", dpi=140)
    plt.close()

    # ── Side view (X vs Z) ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4.5))
    # ground line
    ax.axhline(0, color="#888", linewidth=0.8)
    for lid, info in canonical["lidars"].items():
        x, y, z = info["t"]
        ax.scatter(x, z, c=LIDAR_COLOR[lid], s=160, marker="o",
                   edgecolor="black", linewidth=0.8, zorder=3, label=lid)
    for cid, info in canonical["cameras"].items():
        x, y, z = info["t"]
        ax.scatter(x, z, c=CAM_COLOR[cid], s=70, marker="s",
                   edgecolor="black", linewidth=0.7, alpha=0.7, zorder=3)
    ax.set_xlabel("Vehicle X (m, forward)")
    ax.set_ylabel("Vehicle Z (m, up)")
    ax.set_title("Sensor heights (side view, X vs Z)")
    ax.set_xlim(-3.5, 4.5); ax.set_ylim(-0.5, 3.0)
    ax.legend(loc="best", fontsize=8, ncol=3)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "17_sensor_rig_side.png", dpi=140)
    plt.close()

    # ── Inter-LiDAR baseline matrix ───────────────────────────────────────────
    n = len(lids)
    M = np.zeros((n, n))
    for i, a in enumerate(lids):
        for j, b in enumerate(lids):
            ta = np.array(canonical["lidars"][a]["t"])
            tb = np.array(canonical["lidars"][b]["t"])
            M[i, j] = np.linalg.norm(ta - tb)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(lids, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(lids, fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i,j] < M.max()*0.6 else "black", fontsize=9)
    ax.set_title("LiDAR–LiDAR baseline (m, in vehicle frame)")
    plt.colorbar(im, ax=ax, shrink=0.7)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "18_lidar_baselines.png", dpi=140)
    plt.close()

    # console summary
    print("\n=== LiDAR positions (vehicle frame) ===")
    print(f"{'sensor':<22s} {'x':>7s} {'y':>7s} {'z':>7s} {'yaw_deg':>10s}")
    for lid, info in canonical["lidars"].items():
        t = info["t"]
        print(f"{lid:<22s} {t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f} {info['yaw_deg']:>10.2f}")
    print("\n=== Camera positions (vehicle frame) ===")
    for cid, info in canonical["cameras"].items():
        t = info["t"]
        print(f"{cid:<22s} {t[0]:>7.3f} {t[1]:>7.3f} {t[2]:>7.3f} {info['yaw_deg']:>10.2f}")
    print(f"\nLiDAR-LiDAR baseline range: [{min(baseline.values()):.3f}, {max(baseline.values()):.3f}] m")


if __name__ == "__main__":
    main()
