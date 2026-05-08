"""Compare measured Doppler against bbox.linear_velocity projected onto LOS.

For every annotated *dynamic* box: take points inside the box, compute the
expected radial velocity from the box's annotated velocity vector projected
onto each point's line-of-sight, and compare to the npz['velocity'].

Also performs the v_radial / v_tangential decomposition on the annotated
linear_velocity itself (independent of point density).
"""
import json, time, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
OUT_STATS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/stats")
OUT_PLOTS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/plots")
SCENE_INFO = json.loads(Path("C:/Users/djoko/Documents/LNU/aevascenes/metadata/scene_info.json").read_text())
DYNAMIC = {"car", "truck", "bus", "bicycle", "bicyclist", "motorcycle",
           "motorcyclist", "pedestrian", "trailer", "other_vehicle",
           "vehicle_on_rails"}
LIDAR_IDS = ["front_wide_lidar", "front_narrow_lidar",
             "right_lidar", "rear_wide_lidar", "rear_narrow_lidar", "left_lidar"]
FRAMES_PER_SEQ = 5
SEED = 31


def quat_to_rot(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def main():
    rng = random.Random(SEED)
    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])

    # -- Decomposition records (vt/vr from annotation only) -------------------
    # Per dynamic box (every frame, every sequence) — cheap, no npz needed.
    decomp_per_class = defaultdict(list)  # class -> list of dicts
    decomp_per_scene = defaultdict(list)  # scene -> list of dicts

    # -- Doppler-vs-projection records (sampled frames) ------------------------
    consistency_records = []   # one per (box, lidar) with >=10 points
    residuals_global = []      # (cls, residuals_array) — concatenated later

    t0 = time.time()
    for s_idx, sd in enumerate(seq_dirs):
        d = json.loads((sd / "sequence.json").read_text())
        md = d["metadata"]
        scene = SCENE_INFO.get(sd.name, {})
        scene_key = f"{scene.get('road_type','?')}|{scene.get('lighting_condition','?')}"

        # Lidar pose in vehicle frame
        lid_pos_v = {}
        lid_R_v = {}
        for lid in LIDAR_IDS:
            ext = md["vehicle_to_lidar_extrinsics"][lid]
            lid_pos_v[lid] = np.array([ext["translation"]["x"],
                                        ext["translation"]["y"],
                                        ext["translation"]["z"]])
            lid_R_v[lid] = quat_to_rot(ext["rotation"])

        # ── (1) decomposition over ALL frames ────────────────────────────────
        for fr in d["frames"]:
            for b in fr["boxes"]:
                cls = b.get("class", "unknown")
                if cls not in DYNAMIC:
                    continue
                lv = b["linear_velocity"]
                v_box = np.array([lv["x"], lv["y"], lv["z"]])
                speed = float(np.linalg.norm(v_box))
                if speed < 0.05:  # skip near-zero, ratio undefined
                    continue
                p = b["pose"]["translation"]
                center = np.array([p["x"], p["y"], p["z"]])
                # LOS from ego origin to box (use vehicle origin as a stand-in;
                # this is fine for the angular decomposition since it is
                # invariant to which sensor we pick at long range, and we'll
                # repeat the analysis per-lidar in the consistency check).
                rho = float(np.linalg.norm(center))
                if rho < 0.5:
                    continue
                los = center / rho
                v_r = float(np.dot(v_box, los))           # signed radial
                v_t_vec = v_box - v_r * los
                v_t = float(np.linalg.norm(v_t_vec))      # |tangential|
                ratio = v_t / max(speed, 1e-9)            # in [0, 1]
                # azimuth of box centre (BEV)
                az_deg = float(np.degrees(np.arctan2(center[1], center[0])))
                rec = {"speed": speed, "v_r": v_r, "v_t": v_t,
                       "ratio_t": ratio, "range_m": rho,
                       "az_deg": az_deg}
                decomp_per_class[cls].append(rec)
                decomp_per_scene[scene_key].append(rec)

        # ── (2) sampled doppler-vs-projection ────────────────────────────────
        sample_idx = rng.sample(range(len(d["frames"])),
                                min(FRAMES_PER_SEQ, len(d["frames"])))
        for fi in sample_idx:
            fr = d["frames"][fi]
            # pre-load all 6 npzs in vehicle frame
            cache = {}
            for lid in LIDAR_IDS:
                if lid not in fr["point_cloud"]:
                    continue
                rel = fr["point_cloud"][lid]["point_cloud_path"]
                npz = np.load(sd / rel, allow_pickle=True)
                xyz_l = npz["xyz"]
                xyz_v = (xyz_l @ lid_R_v[lid].T) + lid_pos_v[lid]
                cache[lid] = {"xyz_v": xyz_v,
                              "vel": npz["velocity"].reshape(-1)}

            for b in fr["boxes"]:
                cls = b.get("class", "unknown")
                if cls not in DYNAMIC:
                    continue
                lv = b["linear_velocity"]
                v_box = np.array([lv["x"], lv["y"], lv["z"]])
                if np.linalg.norm(v_box) < 0.5:
                    continue
                Rb = quat_to_rot(b["pose"]["rotation"])
                p = b["pose"]["translation"]; center = np.array([p["x"], p["y"], p["z"]])
                d_box = b["dimensions"]; dx, dy, dz = d_box["x"], d_box["y"], d_box["z"]

                for lid, info in cache.items():
                    rel = info["xyz_v"] - center
                    rel_b = rel @ Rb        # rotate vehicle->box (box frame)
                    inside = (np.abs(rel_b[:, 0]) <= dx/2) & \
                             (np.abs(rel_b[:, 1]) <= dy/2) & \
                             (np.abs(rel_b[:, 2]) <= dz/2)
                    n = int(inside.sum())
                    if n < 10:
                        continue
                    pts_v = info["xyz_v"][inside]
                    vels  = info["vel"][inside]
                    # LOS from this lidar's position
                    los_vec = pts_v - lid_pos_v[lid]
                    los_norm = np.linalg.norm(los_vec, axis=1, keepdims=True)
                    los_norm = np.maximum(los_norm, 1e-6)
                    los = los_vec / los_norm
                    # expected Doppler = projection of v_box onto LOS
                    v_exp = los @ v_box
                    resid = vels - v_exp
                    consistency_records.append({
                        "class": cls, "lidar": lid,
                        "scene": scene_key,
                        "range_m": float(np.linalg.norm(center)),
                        "n_pts": n,
                        "resid_mean": float(resid.mean()),
                        "resid_median": float(np.median(resid)),
                        "resid_std": float(resid.std()),
                        "resid_p95_abs": float(np.percentile(np.abs(resid), 95)),
                        "v_exp_mean": float(v_exp.mean()),
                        "v_meas_mean": float(vels.mean()),
                    })

        if (s_idx + 1) % 10 == 0:
            print(f"  seq {s_idx+1}/{len(seq_dirs)}  "
                  f"decomp={sum(len(v) for v in decomp_per_class.values()):,}  "
                  f"consistency={len(consistency_records):,}  "
                  f"elapsed={time.time()-t0:.1f}s")

    # ── Save raw decomp arrays ────────────────────────────────────────────────
    decomp_arr_by_class = {}
    for c, recs in decomp_per_class.items():
        if not recs: continue
        arr = np.array([(r["speed"], r["v_r"], r["v_t"], r["ratio_t"],
                          r["range_m"], r["az_deg"]) for r in recs])
        decomp_arr_by_class[c] = arr

    decomp_arr_by_scene = {}
    for s, recs in decomp_per_scene.items():
        if not recs: continue
        arr = np.array([(r["speed"], r["v_r"], r["v_t"], r["ratio_t"],
                          r["range_m"], r["az_deg"]) for r in recs])
        decomp_arr_by_scene[s] = arr

    np.savez_compressed(OUT_STATS / "tangential_radial.npz",
                        **{f"class__{c}": v for c, v in decomp_arr_by_class.items()},
                        **{f"scene__{s}": v for s, v in decomp_arr_by_scene.items()})

    # summary stats per class
    decomp_summary = {}
    for c, arr in decomp_arr_by_class.items():
        speeds = arr[:, 0]; ratios = arr[:, 3]
        decomp_summary[c] = {
            "n": int(arr.shape[0]),
            "speed_median_mps": float(np.median(speeds)),
            "speed_p95_mps":    float(np.percentile(speeds, 95)),
            "ratio_t_mean":     float(np.mean(ratios)),
            "ratio_t_median":   float(np.median(ratios)),
            "ratio_t_p95":      float(np.percentile(ratios, 95)),
            "ratio_t_above_0.5": float((ratios > 0.5).mean()),
            "ratio_t_above_0.9": float((ratios > 0.9).mean()),
        }
    decomp_scene_summary = {}
    for s, arr in decomp_arr_by_scene.items():
        ratios = arr[:, 3]
        decomp_scene_summary[s] = {
            "n": int(arr.shape[0]),
            "ratio_t_mean": float(ratios.mean()),
            "ratio_t_median": float(np.median(ratios)),
            "ratio_t_above_0.5": float((ratios > 0.5).mean()),
        }

    # consistency summary per class
    cons_summary = defaultdict(list)
    for r in consistency_records:
        cons_summary[r["class"]].append(r)
    cons_summary_out = {}
    for c, rs in cons_summary.items():
        rmeds  = np.array([r["resid_median"] for r in rs])
        rstds  = np.array([r["resid_std"] for r in rs])
        rp95   = np.array([r["resid_p95_abs"] for r in rs])
        cons_summary_out[c] = {
            "n_box_lidar_pairs": len(rs),
            "resid_median_of_medians": float(np.median(rmeds)),
            "resid_median_abs_median": float(np.median(np.abs(rmeds))),
            "resid_std_median": float(np.median(rstds)),
            "resid_p95_abs_median": float(np.median(rp95)),
            "resid_p95_abs_p95":     float(np.percentile(rp95, 95)),
        }

    out = {
        "decomp_per_class": decomp_summary,
        "decomp_per_scene": decomp_scene_summary,
        "consistency_per_class": cons_summary_out,
        "config": {"frames_per_seq": FRAMES_PER_SEQ, "seed": SEED},
    }
    (OUT_STATS / "doppler_consistency.json").write_text(json.dumps(out, indent=2))

    # ── Plots ────────────────────────────────────────────────────────────────
    OUT_PLOTS.mkdir(parents=True, exist_ok=True)

    # 1. Tangential ratio distribution: per class (top 6 by count)
    keep = sorted(decomp_summary.keys(),
                  key=lambda c: -decomp_summary[c]["n"])[:6]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.linspace(0, 1, 41)
    for c in keep:
        arr = decomp_arr_by_class[c]
        h, e = np.histogram(arr[:, 3], bins=bins, density=True)
        axes[0].plot((e[:-1]+e[1:])/2, h, label=f"{c} (n={arr.shape[0]:,})", linewidth=1.7)
    axes[0].set_xlabel("|v_tangential| / |v|  (annotation)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Tangential-velocity-ratio per dynamic class")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # By scene
    for s, arr in decomp_arr_by_scene.items():
        h, e = np.histogram(arr[:, 3], bins=bins, density=True)
        axes[1].plot((e[:-1]+e[1:])/2, h, label=f"{s} (n={arr.shape[0]:,})",
                     linewidth=1.7)
    axes[1].set_xlabel("|v_tangential| / |v|")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Tangential-velocity-ratio by scene")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "22_tangential_ratio.png", dpi=140)
    plt.close()

    # 2. v_r vs v_t scatter for cars (the core thesis figure)
    if "car" in decomp_arr_by_class:
        arr = decomp_arr_by_class["car"]
        # subsample
        N = 30000
        idx = np.random.choice(arr.shape[0], min(N, arr.shape[0]), replace=False)
        sample = arr[idx]
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(sample[:, 1], sample[:, 2], s=1, alpha=0.25, c="#3a86ff")
        # equal-speed isobars
        for sp in [5, 10, 20, 30, 40, 50]:
            theta = np.linspace(0, 2*np.pi, 200)
            ax.plot(sp*np.cos(theta), sp*np.sin(theta), color="#888",
                    linestyle="--", linewidth=0.7, alpha=0.7)
            ax.text(sp/np.sqrt(2)*1.02, sp/np.sqrt(2)*1.02, f"{sp} m/s",
                    fontsize=7, color="#666")
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)
        ax.set_xlabel("Radial velocity v_r (m/s) — what FMCW measures directly")
        ax.set_ylabel("|Tangential velocity| (m/s) — to be inferred")
        ax.set_title(f"Car-class velocity decomposition (n={sample.shape[0]:,} of {arr.shape[0]:,})")
        ax.set_xlim(-50, 50); ax.set_ylim(0, 50)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "23_vr_vt_car.png", dpi=140)
        plt.close()

    # 3. Doppler-vs-projection: residual median per class (boxplot of medians)
    if cons_summary:
        plot_classes = sorted(cons_summary.keys(),
                              key=lambda c: -len(cons_summary[c]))[:8]
        data = [[r["resid_median"] for r in cons_summary[c]] for c in plot_classes]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        bp = axes[0].boxplot(data, tick_labels=plot_classes, patch_artist=True,
                              showfliers=False)
        for p in bp["boxes"]:
            p.set_facecolor("#3a86ff"); p.set_alpha(0.6)
        axes[0].axhline(0, color="k", linestyle="--", alpha=0.4)
        axes[0].set_ylabel("Residual median (measured − projected) m/s")
        axes[0].set_title("Doppler-vs-annotation residual per box-LiDAR pair")
        axes[0].grid(alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")

        # Histogram of all residual medians (all classes combined)
        all_resid = np.array([r["resid_median"] for r in consistency_records])
        axes[1].hist(np.clip(all_resid, -10, 10), bins=80, color="#3a86ff",
                     edgecolor="white")
        axes[1].axvline(0, color="k", linestyle="--", alpha=0.4)
        axes[1].set_xlabel("Per-box residual median (clipped ±10) m/s")
        axes[1].set_ylabel("Count")
        axes[1].set_title(f"All dynamic-box residuals (n={len(consistency_records):,})\n"
                          f"median={np.median(all_resid):+.3f}, "
                          f"|median|@p50={np.median(np.abs(all_resid)):.3f}, "
                          f"|median|@p95={np.percentile(np.abs(all_resid), 95):.3f}")
        axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "24_doppler_residuals.png", dpi=140)
        plt.close()

    # console summary
    print("\n=== Tangential-velocity ratio (annotation-only decomposition) ===")
    print(f"{'class':<18s} {'n':>8s} {'speed_p95':>10s} {'ratio_med':>10s} {'>0.5':>7s} {'>0.9':>7s}")
    for c in sorted(decomp_summary, key=lambda x: -decomp_summary[x]["n"]):
        s = decomp_summary[c]
        print(f"{c:<18s} {s['n']:>8,} {s['speed_p95_mps']:>10.2f} "
              f"{s['ratio_t_median']:>10.3f} "
              f"{100*s['ratio_t_above_0.5']:>6.1f}% "
              f"{100*s['ratio_t_above_0.9']:>6.1f}%")

    print("\n=== Per-scene tangential ratio ===")
    for s, info in sorted(decomp_scene_summary.items()):
        print(f"  {s:<18s}  n={info['n']:>8,}  "
              f"ratio_med={info['ratio_t_median']:.3f}  "
              f">{50}%={100*info['ratio_t_above_0.5']:.1f}%")

    print("\n=== Doppler-vs-projection consistency (per box-lidar pair) ===")
    for c, info in sorted(cons_summary_out.items(), key=lambda x: -x[1]["n_box_lidar_pairs"])[:10]:
        print(f"  {c:<18s} n={info['n_box_lidar_pairs']:>6,}  "
              f"|median|={info['resid_median_abs_median']:>5.3f} m/s  "
              f"std_med={info['resid_std_median']:>5.3f}  "
              f"|p95|_med={info['resid_p95_abs_median']:>5.3f}")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
