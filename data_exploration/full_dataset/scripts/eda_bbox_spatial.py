"""Spatial distribution of annotated bboxes (vehicle frame): range, azimuth, points-per-box."""
import json, time, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
OUT_STATS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/stats")
OUT_PLOTS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/plots")
SCENE_INFO = json.loads(Path("C:/Users/djoko/Documents/LNU/aevascenes/metadata/scene_info.json").read_text())

DYNAMIC_CLASSES = {"car", "truck", "bus", "bicycle", "bicyclist", "motorcycle",
                   "motorcyclist", "pedestrian", "trailer", "other_vehicle",
                   "vehicle_on_rails", "animal"}
LIDAR_IDS = ["front_wide_lidar", "front_narrow_lidar",
             "right_lidar", "rear_wide_lidar", "rear_narrow_lidar", "left_lidar"]
SAMPLE_FRAMES_PER_SEQ_FOR_PTS = 5  # for the per-box-points analysis (heavier)
RANDOM_SEED = 17


def quat_to_rot(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def transform_points(R, t, xyz):
    return (xyz @ R.T) + t


def main():
    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])
    rng = random.Random(RANDOM_SEED)

    # Aggregate centroids in vehicle frame, all bboxes, all sequences
    centroids_by_class = defaultdict(list)         # class -> [(x, y, z), ...]
    box_dims_by_class = defaultdict(list)          # class -> [(dx, dy, dz), ...]

    # Per-box point coverage on dynamic classes (sampled sequences/frames)
    pts_per_box_records = []  # dicts: class, scene, range_m, n_pts_total, n_pts_per_lidar

    t0 = time.time()
    for s_idx, sd in enumerate(seq_dirs):
        d = json.loads((sd / "sequence.json").read_text())
        md = d["metadata"]
        scene = SCENE_INFO.get(sd.name, {})
        scene_key = f"{scene.get('road_type','?')}|{scene.get('lighting_condition','?')}"

        # Pre-compute lidar positions in vehicle frame
        lidar_pos_v = {}
        lidar_R_v = {}
        for lid in LIDAR_IDS:
            ext = md["vehicle_to_lidar_extrinsics"][lid]
            t = ext["translation"]; q = ext["rotation"]
            R = quat_to_rot(q); tt = np.array([t["x"], t["y"], t["z"]])
            lidar_pos_v[lid] = tt
            lidar_R_v[lid] = R

        # ── all-frames pass: collect centroids + dimensions ───────────────────
        for fr in d["frames"]:
            for b in fr["boxes"]:
                cls = b.get("class", "unknown")
                p = b["pose"]["translation"]
                centroids_by_class[cls].append([p["x"], p["y"], p["z"]])
                dims = b["dimensions"]
                box_dims_by_class[cls].append([dims["x"], dims["y"], dims["z"]])

        # ── sampled pass: count points per dynamic box ────────────────────────
        sample_idx = rng.sample(range(len(d["frames"])),
                                min(SAMPLE_FRAMES_PER_SEQ_FOR_PTS, len(d["frames"])))
        for fi in sample_idx:
            fr = d["frames"][fi]
            # Load all 6 npzs once and transform to vehicle frame
            pcds_v = {}
            for lid in LIDAR_IDS:
                if lid not in fr["point_cloud"]:
                    continue
                rel = fr["point_cloud"][lid]["point_cloud_path"]
                npz = np.load(sd / rel, allow_pickle=True)
                xyz_l = npz["xyz"]
                xyz_v = transform_points(lidar_R_v[lid], lidar_pos_v[lid], xyz_l)
                pcds_v[lid] = xyz_v

            # For each dynamic box: transform points to box frame, test in/out
            for b in fr["boxes"]:
                cls = b.get("class", "unknown")
                if cls not in DYNAMIC_CLASSES:
                    continue
                p = b["pose"]["translation"]
                cx, cy, cz = p["x"], p["y"], p["z"]
                d_box = b["dimensions"]
                dx, dy, dz = d_box["x"], d_box["y"], d_box["z"]
                Rb = quat_to_rot(b["pose"]["rotation"])
                center = np.array([cx, cy, cz])
                rng_xy = float(np.linalg.norm(center[:2]))

                # Check each lidar's points
                n_per_lid = {}
                for lid, xyz_v in pcds_v.items():
                    rel = xyz_v - center
                    # rotate into box-aligned frame: x' = R^T (xyz - c)
                    rel_b = rel @ Rb  # since Rb is orthonormal, R^T == R[:, :].T but using R for rotate-back
                    # actually R rotates box-frame -> vehicle-frame; we want vehicle->box: use R.T
                    rel_b = (xyz_v - center) @ Rb
                    inside = (np.abs(rel_b[:, 0]) <= dx/2) & \
                             (np.abs(rel_b[:, 1]) <= dy/2) & \
                             (np.abs(rel_b[:, 2]) <= dz/2)
                    n_per_lid[lid] = int(inside.sum())

                pts_per_box_records.append({
                    "class": cls, "scene": scene_key,
                    "range_m": rng_xy,
                    "centroid_x": cx, "centroid_y": cy, "centroid_z": cz,
                    "n_pts_total": int(sum(n_per_lid.values())),
                    "n_pts_per_lidar": n_per_lid,
                })

        if (s_idx + 1) % 10 == 0:
            print(f"  seq {s_idx+1}/{len(seq_dirs)}  elapsed={time.time()-t0:.1f}s  "
                  f"box-records={len(pts_per_box_records)}")

    # ── Plots ────────────────────────────────────────────────────────────────
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # All-classes centroid heatmap (BEV)
    all_centroids = np.vstack([np.array(v) for v in centroids_by_class.values()])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    h, xe, ye = np.histogram2d(all_centroids[:, 0], all_centroids[:, 1],
                                bins=[np.arange(-150, 251, 5),
                                      np.arange(-100, 101, 5)])
    im = axes[0].imshow(np.log10(h.T + 1), origin="lower", aspect="auto",
                        extent=[xe[0], xe[-1], ye[0], ye[-1]],
                        cmap="magma")
    axes[0].set_xlabel("Vehicle X (m, forward)")
    axes[0].set_ylabel("Vehicle Y (m, left)")
    axes[0].set_title(f"BEV centroid heatmap — all {all_centroids.shape[0]:,} bboxes (log₁₀)")
    plt.colorbar(im, ax=axes[0], label="log₁₀(count + 1)")
    axes[0].scatter([0], [0], c="cyan", marker="x", s=100)

    # Range × class
    rng_per_class = {cls: np.linalg.norm(np.array(v)[:, :2], axis=1)
                     for cls, v in centroids_by_class.items()}
    keep = sorted(rng_per_class.keys(),
                  key=lambda c: -len(centroids_by_class[c]))[:10]
    for cls in keep:
        h, e = np.histogram(rng_per_class[cls],
                             bins=np.linspace(0, 250, 51), density=True)
        axes[1].plot((e[:-1]+e[1:])/2, h, label=f"{cls} ({len(rng_per_class[cls]):,})",
                     linewidth=1.6)
    axes[1].set_xlabel("Range from ego (m, BEV)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Bbox range distribution by class (top 10)")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "19_bbox_spatial.png", dpi=140)
    plt.close()

    # Bbox dimensions (length × width) per class — top 8
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_classes = sorted(box_dims_by_class.keys(),
                          key=lambda c: -len(box_dims_by_class[c]))[:8]
    for cls in plot_classes:
        dims = np.array(box_dims_by_class[cls])
        # subsample for plot
        if dims.shape[0] > 5000:
            dims = dims[np.random.choice(dims.shape[0], 5000, replace=False)]
        ax.scatter(dims[:, 0], dims[:, 1], s=4, alpha=0.3, label=cls)
    ax.set_xlabel("Length (x, m)"); ax.set_ylabel("Width (y, m)")
    ax.set_title("Bbox dimensions per class (sampled, top-8 classes)")
    ax.set_xlim(0, 25); ax.set_ylim(0, 5)
    ax.grid(alpha=0.3); ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "20_bbox_dimensions.png", dpi=140)
    plt.close()

    # Points-per-box analysis on dynamic classes
    if pts_per_box_records:
        rng_vals = np.array([r["range_m"] for r in pts_per_box_records])
        npts = np.array([r["n_pts_total"] for r in pts_per_box_records])
        cls_arr = np.array([r["class"] for r in pts_per_box_records])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # 2D heatmap range × points
        h, xe, ye = np.histogram2d(rng_vals, np.log10(npts + 1),
                                    bins=[np.linspace(0, 200, 41),
                                          np.linspace(0, 5, 41)])
        im = axes[0].imshow(np.log10(h.T + 1), origin="lower", aspect="auto",
                             extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap="viridis")
        axes[0].set_xlabel("Box range from ego (m)")
        axes[0].set_ylabel("log₁₀(points-in-box + 1)")
        axes[0].set_title(f"Points per dynamic bbox vs range "
                          f"(n={len(pts_per_box_records):,})")
        plt.colorbar(im, ax=axes[0])

        # Empty-box rate by range
        bins = np.linspace(0, 200, 21)
        empty_rate = []
        ge10_rate = []
        for i in range(len(bins) - 1):
            m = (rng_vals >= bins[i]) & (rng_vals < bins[i+1])
            if m.sum() == 0:
                empty_rate.append(np.nan); ge10_rate.append(np.nan); continue
            empty_rate.append((npts[m] == 0).mean())
            ge10_rate.append((npts[m] >= 10).mean())
        centers = (bins[:-1] + bins[1:]) / 2
        axes[1].plot(centers, empty_rate, "-o", color="#fb5607", label="empty (0 points)")
        axes[1].plot(centers, ge10_rate, "-o", color="#06a77d", label="≥ 10 points")
        axes[1].set_xlabel("Range from ego (m)")
        axes[1].set_ylabel("Fraction of dynamic boxes")
        axes[1].set_title("Box population vs range")
        axes[1].grid(alpha=0.3); axes[1].legend()
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "21_points_per_box.png", dpi=140)
        plt.close()

        # save summary
        per_cls = defaultdict(list)
        for r in pts_per_box_records:
            per_cls[r["class"]].append((r["range_m"], r["n_pts_total"]))
        cls_summary = {}
        for c, vs in per_cls.items():
            arr = np.array(vs)
            cls_summary[c] = {
                "n_boxes_sampled": int(len(arr)),
                "median_range_m": float(np.median(arr[:, 0])),
                "median_points": float(np.median(arr[:, 1])),
                "mean_points": float(arr[:, 1].mean()),
                "frac_empty": float((arr[:, 1] == 0).mean()),
                "frac_ge10": float((arr[:, 1] >= 10).mean()),
                "frac_ge50": float((arr[:, 1] >= 50).mean()),
            }
        out = {
            "n_records": len(pts_per_box_records),
            "by_class": cls_summary,
            "global": {
                "median_points": float(np.median(npts)),
                "mean_points": float(npts.mean()),
                "frac_empty": float((npts == 0).mean()),
                "frac_ge10": float((npts >= 10).mean()),
            },
        }
        (OUT_STATS / "bbox_spatial.json").write_text(json.dumps(out, indent=2))
        print("\n=== Points-per-box (dynamic classes, sampled) ===")
        print(f"records: {out['n_records']:,}")
        print(f"global: median={out['global']['median_points']:.0f} pts, "
              f"empty={100*out['global']['frac_empty']:.1f}%, "
              f">=10={100*out['global']['frac_ge10']:.1f}%")
        for c, s in sorted(cls_summary.items(), key=lambda x: -x[1]["n_boxes_sampled"])[:10]:
            print(f"  {c:<18s} n={s['n_boxes_sampled']:>6,}  "
                  f"med={s['median_points']:>5.0f}  "
                  f"empty={100*s['frac_empty']:>4.1f}%  "
                  f">=10={100*s['frac_ge10']:>4.1f}%  "
                  f">=50={100*s['frac_ge50']:>4.1f}%")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
