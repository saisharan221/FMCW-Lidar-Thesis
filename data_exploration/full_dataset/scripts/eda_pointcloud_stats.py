"""Per-LiDAR pointcloud stats. Samples N frames per sequence × all 6 lidars."""
import json, os, time, random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
OUT_DIR = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FRAMES_PER_SEQ = 5
RANDOM_SEED = 42
LIDAR_IDS = ["front_wide_lidar", "front_narrow_lidar",
             "right_lidar", "rear_wide_lidar", "rear_narrow_lidar", "left_lidar"]

# Histogram bin definitions (also written out for the plot script)
RANGE_BINS = np.linspace(0, 400, 81)              # 5-m bins, 0..400 m
VEL_BINS = np.linspace(-30, 30, 121)              # 0.5 m/s bins
REFL_BINS = np.linspace(0, 4096, 129)             # broad sweep, log-displayed later
AZ_BINS = np.linspace(-180, 180, 73)              # 5° bins
EL_BINS = np.linspace(-30, 30, 61)                # 1° bins


def per_npz_stats(npz):
    xyz = npz["xyz"]
    vel = npz["velocity"].reshape(-1)
    refl = npz["reflectivity"].reshape(-1)
    sem = npz["semantic_labels"].reshape(-1)
    n = xyz.shape[0]
    if n == 0:
        return None

    # ranges, azimuths, elevations from sensor frame
    r = np.linalg.norm(xyz, axis=1)
    az = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    el = np.degrees(np.arctan2(xyz[:, 2], np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)))

    return {
        "n": int(n),
        "range_p50": float(np.percentile(r, 50)),
        "range_p95": float(np.percentile(r, 95)),
        "range_max": float(r.max()),
        "vel_min": float(vel.min()),
        "vel_max": float(vel.max()),
        "vel_mean": float(vel.mean()),
        "vel_std": float(vel.std()),
        "refl_p50": float(np.percentile(refl, 50)),
        "refl_p95": float(np.percentile(refl, 95)),
        "refl_max": float(refl.max()),
        "az_min": float(az.min()),
        "az_max": float(az.max()),
        "el_min": float(el.min()),
        "el_max": float(el.max()),
        # histograms
        "range_hist": np.histogram(r, RANGE_BINS)[0].tolist(),
        "vel_hist": np.histogram(vel, VEL_BINS)[0].tolist(),
        "refl_hist": np.histogram(refl, REFL_BINS)[0].tolist(),
        "az_hist": np.histogram(az, AZ_BINS)[0].tolist(),
        "el_hist": np.histogram(el, EL_BINS)[0].tolist(),
        "semantic_counts": dict(Counter(sem.tolist())),
    }


def main():
    rng = random.Random(RANDOM_SEED)
    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])
    print(f"{len(seq_dirs)} sequences, sampling {FRAMES_PER_SEQ} frames/seq × 6 lidars "
          f"= {len(seq_dirs)*FRAMES_PER_SEQ*6} npz files")

    # Aggregations per lidar
    agg = {lid: {
        "n_files": 0,
        "n_points_total": 0,
        "n_points_per_file": [],
        "range_hist": np.zeros(len(RANGE_BINS)-1, dtype=np.int64),
        "vel_hist": np.zeros(len(VEL_BINS)-1, dtype=np.int64),
        "refl_hist": np.zeros(len(REFL_BINS)-1, dtype=np.int64),
        "az_hist": np.zeros(len(AZ_BINS)-1, dtype=np.int64),
        "el_hist": np.zeros(len(EL_BINS)-1, dtype=np.int64),
        "semantic_counts": Counter(),
        "vel_min_global": float("inf"),
        "vel_max_global": float("-inf"),
        "range_max_global": 0.0,
        "az_min_global": float("inf"),
        "az_max_global": float("-inf"),
        "el_min_global": float("inf"),
        "el_max_global": float("-inf"),
    } for lid in LIDAR_IDS}

    # also per scene type (road×lighting)
    SCENE_INFO = Path("C:/Users/djoko/Documents/LNU/aevascenes/metadata/scene_info.json")
    with open(SCENE_INFO) as f:
        scene_info = json.load(f)

    per_scene = defaultdict(lambda: {
        "n_files": 0,
        "vel_hist": np.zeros(len(VEL_BINS)-1, dtype=np.int64),
        "range_hist": np.zeros(len(RANGE_BINS)-1, dtype=np.int64),
        "semantic_counts": Counter(),
    })

    t0 = time.time()
    total = 0
    for s_idx, sd in enumerate(seq_dirs):
        with open(sd / "sequence.json") as f:
            d = json.load(f)
        frames = d["frames"]
        # sample frames
        idxs = rng.sample(range(len(frames)), min(FRAMES_PER_SEQ, len(frames)))

        scene = scene_info.get(sd.name, {})
        scene_key = f"{scene.get('road_type','?')}|{scene.get('lighting_condition','?')}"

        for fi in idxs:
            fr = frames[fi]
            for lid in LIDAR_IDS:
                if lid not in fr["point_cloud"]:
                    continue
                rel = fr["point_cloud"][lid]["point_cloud_path"]
                npz_path = sd / rel
                try:
                    d_npz = np.load(npz_path, allow_pickle=True)
                    s = per_npz_stats(d_npz)
                    if s is None:
                        continue
                except Exception as e:
                    print(f"  ERR {npz_path}: {e}"); continue

                a = agg[lid]
                a["n_files"] += 1
                a["n_points_total"] += s["n"]
                a["n_points_per_file"].append(s["n"])
                a["range_hist"] += np.asarray(s["range_hist"], dtype=np.int64)
                a["vel_hist"] += np.asarray(s["vel_hist"], dtype=np.int64)
                a["refl_hist"] += np.asarray(s["refl_hist"], dtype=np.int64)
                a["az_hist"] += np.asarray(s["az_hist"], dtype=np.int64)
                a["el_hist"] += np.asarray(s["el_hist"], dtype=np.int64)
                a["semantic_counts"].update(s["semantic_counts"])
                a["vel_min_global"] = min(a["vel_min_global"], s["vel_min"])
                a["vel_max_global"] = max(a["vel_max_global"], s["vel_max"])
                a["range_max_global"] = max(a["range_max_global"], s["range_max"])
                a["az_min_global"] = min(a["az_min_global"], s["az_min"])
                a["az_max_global"] = max(a["az_max_global"], s["az_max"])
                a["el_min_global"] = min(a["el_min_global"], s["el_min"])
                a["el_max_global"] = max(a["el_max_global"], s["el_max"])

                ps = per_scene[scene_key]
                ps["n_files"] += 1
                ps["vel_hist"] += np.asarray(s["vel_hist"], dtype=np.int64)
                ps["range_hist"] += np.asarray(s["range_hist"], dtype=np.int64)
                ps["semantic_counts"].update(s["semantic_counts"])

                total += 1

        if (s_idx + 1) % 10 == 0:
            print(f"  seq {s_idx+1}/{len(seq_dirs)}  files={total}  elapsed={time.time()-t0:.1f}s")

    # serialize
    out = {
        "config": {
            "frames_per_seq": FRAMES_PER_SEQ,
            "lidars": LIDAR_IDS,
            "range_bins": RANGE_BINS.tolist(),
            "vel_bins": VEL_BINS.tolist(),
            "refl_bins": REFL_BINS.tolist(),
            "az_bins": AZ_BINS.tolist(),
            "el_bins": EL_BINS.tolist(),
        },
        "per_lidar": {},
        "per_scene": {},
    }
    for lid, a in agg.items():
        npp = np.asarray(a["n_points_per_file"])
        out["per_lidar"][lid] = {
            "n_files": int(a["n_files"]),
            "n_points_total": int(a["n_points_total"]),
            "points_per_frame_mean": float(npp.mean()) if len(npp) else 0.0,
            "points_per_frame_median": float(np.median(npp)) if len(npp) else 0.0,
            "points_per_frame_p95": float(np.percentile(npp, 95)) if len(npp) else 0.0,
            "points_per_frame_min": int(npp.min()) if len(npp) else 0,
            "points_per_frame_max": int(npp.max()) if len(npp) else 0,
            "range_max_global": a["range_max_global"],
            "vel_min_global": a["vel_min_global"],
            "vel_max_global": a["vel_max_global"],
            "az_min_global": a["az_min_global"],
            "az_max_global": a["az_max_global"],
            "el_min_global": a["el_min_global"],
            "el_max_global": a["el_max_global"],
            "range_hist": a["range_hist"].tolist(),
            "vel_hist": a["vel_hist"].tolist(),
            "refl_hist": a["refl_hist"].tolist(),
            "az_hist": a["az_hist"].tolist(),
            "el_hist": a["el_hist"].tolist(),
            "semantic_counts": dict(a["semantic_counts"]),
        }
    for sk, ps in per_scene.items():
        out["per_scene"][sk] = {
            "n_files": int(ps["n_files"]),
            "vel_hist": ps["vel_hist"].tolist(),
            "range_hist": ps["range_hist"].tolist(),
            "semantic_counts": dict(ps["semantic_counts"]),
        }

    with open(OUT_DIR / "pointcloud_stats.json", "w") as f:
        json.dump(out, f)
    print(f"\nWrote {OUT_DIR/'pointcloud_stats.json'}, scanned {total} npz files in {time.time()-t0:.1f}s")

    # Quick console summary
    print("\n=== POINTCLOUD STATS BY LIDAR ===")
    for lid in LIDAR_IDS:
        info = out["per_lidar"][lid]
        print(f"\n{lid}:")
        print(f"  files: {info['n_files']}, total pts: {info['n_points_total']:,}")
        print(f"  pts/frame: mean={info['points_per_frame_mean']:.0f} "
              f"median={info['points_per_frame_median']:.0f} "
              f"p95={info['points_per_frame_p95']:.0f}")
        print(f"  range max:    {info['range_max_global']:.1f} m")
        print(f"  velocity:     [{info['vel_min_global']:.2f}, {info['vel_max_global']:.2f}] m/s")
        print(f"  azimuth:      [{info['az_min_global']:.1f}, {info['az_max_global']:.1f}] deg")
        print(f"  elevation:    [{info['el_min_global']:.1f}, {info['el_max_global']:.1f}] deg")


if __name__ == "__main__":
    main()
