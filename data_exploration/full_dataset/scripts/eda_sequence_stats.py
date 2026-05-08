"""Per-sequence stats for AevaScenes v0.2 — only reads sequence.json (no npz)."""
import json, os, math, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
SCENE_INFO = Path("C:/Users/djoko/Documents/LNU/aevascenes/metadata/scene_info.json")
OUT_DIR = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dir_size_bytes(p: Path) -> int:
    total = 0
    for root, _, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def main():
    with open(SCENE_INFO) as f:
        scene_info = json.load(f)

    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])
    print(f"Found {len(seq_dirs)} sequence dirs")

    rows = []
    class_counter = Counter()
    scene_class_counter = defaultdict(Counter)
    trajectories = {}  # uuid -> Nx3
    timestamp_starts = {}
    bbox_speed_per_class = defaultdict(list)
    track_persistence = []  # (n_unique_ids, n_frames)

    t0 = time.time()
    for i, sd in enumerate(seq_dirs):
        sj = sd / "sequence.json"
        if not sj.exists():
            print(f"  WARN missing sequence.json {sd.name}"); continue
        with open(sj) as f:
            d = json.load(f)
        md = d["metadata"]
        frames = d["frames"]
        ts = np.array([fr["timestamp_ns"] for fr in frames], dtype=np.int64)
        dur_s = (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0.0
        dt_s = np.diff(ts) / 1e9 if len(ts) > 1 else np.array([0.0])

        # ego trajectory (relative to frame 0)
        poses = np.array([[fr["ego_pose"]["translation"]["x"],
                           fr["ego_pose"]["translation"]["y"],
                           fr["ego_pose"]["translation"]["z"]] for fr in frames])
        seg = np.linalg.norm(np.diff(poses, axis=0), axis=1)
        path_len = float(seg.sum())
        straight = float(np.linalg.norm(poses[-1] - poses[0]))
        mean_speed = path_len / dur_s if dur_s > 0 else 0.0

        trajectories[sd.name] = poses.tolist()
        timestamp_starts[sd.name] = int(ts[0])

        # bboxes
        n_boxes_total = 0
        track_ids = set()
        seq_class_counter = Counter()
        for fr in frames:
            for b in fr["boxes"]:
                n_boxes_total += 1
                track_ids.add(b["id"])
                cls = b.get("class", "unknown")
                seq_class_counter[cls] += 1
                class_counter[cls] += 1
                lv = b["linear_velocity"]
                spd = math.sqrt(lv["x"] ** 2 + lv["y"] ** 2 + lv["z"] ** 2)
                bbox_speed_per_class[cls].append(spd)
        track_persistence.append((len(track_ids), len(frames), n_boxes_total))

        # scene info
        scene = scene_info.get(sd.name, {})
        road_type = scene.get("road_type", "unknown")
        lighting = scene.get("lighting_condition", "unknown")
        scene_class_counter[(road_type, lighting)].update(seq_class_counter)

        # disk size
        size_b = dir_size_bytes(sd)

        rows.append({
            "uuid": sd.name,
            "n_frames": len(frames),
            "duration_s": float(dur_s),
            "dt_mean_s": float(dt_s.mean()),
            "dt_std_s": float(dt_s.std()),
            "n_lidars": len(md["sensors"]["lidars"]),
            "n_cameras": len(md["sensors"]["cameras"]),
            "ego_path_len_m": path_len,
            "ego_straight_m": straight,
            "ego_mean_speed_mps": mean_speed,
            "n_bboxes_total": n_boxes_total,
            "n_unique_tracks": len(track_ids),
            "bbox_per_frame_mean": n_boxes_total / max(1, len(frames)),
            "road_type": road_type,
            "lighting": lighting,
            "size_bytes": size_b,
            "size_mb": size_b / (1024 ** 2),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(seq_dirs)} processed in {time.time()-t0:.1f}s")

    # write outputs
    summary = {
        "n_sequences": len(rows),
        "total_frames": sum(r["n_frames"] for r in rows),
        "total_bboxes": sum(r["n_bboxes_total"] for r in rows),
        "total_unique_tracks": sum(r["n_unique_tracks"] for r in rows),
        "total_duration_s": sum(r["duration_s"] for r in rows),
        "total_size_bytes": sum(r["size_bytes"] for r in rows),
        "class_counts": dict(class_counter),
        "scene_distribution": Counter([(r["road_type"], r["lighting"]) for r in rows]),
        "scene_distribution_str": {f"{k[0]}|{k[1]}": v for k, v in
            Counter([(r["road_type"], r["lighting"]) for r in rows]).items()},
        "bbox_speed_per_class_stats": {
            cls: {
                "n": len(v),
                "mean": float(np.mean(v)),
                "median": float(np.median(v)),
                "p95": float(np.percentile(v, 95)),
                "max": float(np.max(v)),
            } for cls, v in bbox_speed_per_class.items() if len(v) > 0
        },
    }

    with open(OUT_DIR / "per_sequence.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({k: v for k, v in summary.items() if k != "scene_distribution"}, f, indent=2, default=str)
    with open(OUT_DIR / "trajectories.json", "w") as f:
        json.dump(trajectories, f)

    # Print quick summary
    print("\n=== SUMMARY ===")
    print(f"sequences:       {summary['n_sequences']}")
    print(f"total frames:    {summary['total_frames']:,}")
    print(f"total bboxes:    {summary['total_bboxes']:,}")
    print(f"unique tracks:   {summary['total_unique_tracks']:,}")
    print(f"total duration:  {summary['total_duration_s']:.1f} s "
          f"({summary['total_duration_s']/60:.1f} min)")
    print(f"on-disk size:    {summary['total_size_bytes']/(1024**3):.2f} GB")
    print(f"\nScene distribution:")
    for k, v in sorted(summary['scene_distribution_str'].items(), key=lambda x: -x[1]):
        print(f"  {k:35s} {v}")
    print(f"\nTop 10 classes:")
    for cls, n in class_counter.most_common(10):
        print(f"  {cls:25s} {n:,}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
