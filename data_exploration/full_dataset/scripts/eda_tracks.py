"""Track-dynamics EDA: lifetimes, speed profiles, tracks-alive timeline."""
import json, time
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


def main():
    seq_dirs = sorted([p for p in DATAROOT.iterdir() if p.is_dir()])
    t0 = time.time()

    # All tracks aggregated globally (UUID is unique across the dataset since
    # IDs are random — but we'll group per-sequence to be safe and still
    # accumulate global statistics afterwards).
    tracks_per_seq_lifetime = []   # list of length-array per seq
    tracks_per_seq_class = defaultdict(list)  # class -> length list
    tracks_per_seq_speed = defaultdict(list)  # class -> mean speed list
    tracks_alive_per_frame = []   # 100x100 array (sequence × frame)
    sequence_keys = []
    n_unique_global = set()

    for s_idx, sd in enumerate(seq_dirs):
        d = json.loads((sd / "sequence.json").read_text())
        scene = SCENE_INFO.get(sd.name, {})
        scene_key = f"{scene.get('road_type','?')}|{scene.get('lighting_condition','?')}"

        # Per-track frame counts and per-track speed samples
        track_frames = defaultdict(set)   # id -> set of frame_idx
        track_class = {}                  # id -> class
        track_speeds = defaultdict(list)  # id -> [|v|, ...]

        n_alive = []
        for fi, fr in enumerate(d["frames"]):
            ids_this = set()
            for b in fr["boxes"]:
                tid = b["id"]
                track_frames[tid].add(fi)
                track_class[tid] = b.get("class", "unknown")
                lv = b["linear_velocity"]
                track_speeds[tid].append(float(np.linalg.norm(
                    [lv["x"], lv["y"], lv["z"]])))
                ids_this.add(tid)
            n_alive.append(len(ids_this))
        tracks_alive_per_frame.append(n_alive)

        lifetimes = np.array([len(f) for f in track_frames.values()])
        tracks_per_seq_lifetime.append(lifetimes)
        sequence_keys.append(scene_key)
        for tid, frames in track_frames.items():
            cls = track_class[tid]
            tracks_per_seq_class[cls].append(len(frames))
            speeds = track_speeds[tid]
            tracks_per_seq_speed[cls].append(float(np.mean(speeds)))
            n_unique_global.add(tid)

        if (s_idx + 1) % 20 == 0:
            print(f"  seq {s_idx+1}/{len(seq_dirs)}  elapsed={time.time()-t0:.1f}s")

    # ── Global stats ─────────────────────────────────────────────────────────
    all_lifetimes = np.concatenate(tracks_per_seq_lifetime)
    print(f"\nUnique global track IDs: {len(n_unique_global):,}")
    print(f"Total track instances (sum across seqs): {len(all_lifetimes):,}")
    print(f"Lifetime: mean={all_lifetimes.mean():.1f} median={np.median(all_lifetimes):.1f} "
          f"max={all_lifetimes.max()}  >=50 frames={100*(all_lifetimes>=50).mean():.1f}%  "
          f"==100 frames={100*(all_lifetimes==100).mean():.1f}%")

    # Per-class lifetime + speed
    cls_summary = {}
    for cls, lengths in tracks_per_seq_class.items():
        l = np.array(lengths)
        s = np.array(tracks_per_seq_speed[cls])
        cls_summary[cls] = {
            "n_tracks": int(len(l)),
            "lifetime_median": float(np.median(l)),
            "lifetime_mean": float(l.mean()),
            "lifetime_p95": float(np.percentile(l, 95)),
            "frac_full_100": float((l == 100).mean()),
            "speed_median": float(np.median(s)),
            "speed_mean": float(s.mean()),
            "speed_p95": float(np.percentile(s, 95)),
        }

    out = {
        "n_global_unique": len(n_unique_global),
        "n_track_instances": int(len(all_lifetimes)),
        "lifetime_median": float(np.median(all_lifetimes)),
        "lifetime_mean": float(all_lifetimes.mean()),
        "lifetime_frac_full": float((all_lifetimes == 100).mean()),
        "by_class": cls_summary,
    }
    (OUT_STATS / "tracks.json").write_text(json.dumps(out, indent=2))

    # ── Plots ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bins = np.arange(1, 102)
    axes[0].hist(all_lifetimes, bins=bins, color="#3a86ff", edgecolor="white")
    axes[0].set_xlabel("Track lifetime (frames out of 100)")
    axes[0].set_ylabel("# tracks")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Track-lifetime distribution (n={len(all_lifetimes):,})\n"
                      f"median={np.median(all_lifetimes):.0f}, "
                      f"==100 frames: {100*(all_lifetimes==100).mean():.1f}%")
    axes[0].grid(alpha=0.3)

    # Per-class boxplot of lifetimes
    plot_classes = sorted(cls_summary.keys(),
                          key=lambda c: -cls_summary[c]["n_tracks"])[:8]
    data = [tracks_per_seq_class[c] for c in plot_classes]
    bp = axes[1].boxplot(data, tick_labels=plot_classes, showfliers=False,
                          patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor("#06a77d"); p.set_alpha(0.6)
    axes[1].set_ylabel("Lifetime (frames)")
    axes[1].set_title("Track lifetime by class (top 8)")
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")
    axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "25_track_lifetimes.png", dpi=140)
    plt.close()

    # tracks-alive over time per scene
    arr = np.array(tracks_alive_per_frame)  # [100 seqs × 100 frames]
    fig, ax = plt.subplots(figsize=(10, 5))
    sc_color = {"city|day": "#3a86ff", "city|night": "#1d3557",
                "highway|day": "#ffb703", "highway|night": "#fb5607"}
    for sk, c in sc_color.items():
        idx = [i for i, k in enumerate(sequence_keys) if k == sk]
        if not idx: continue
        sub = arr[idx]
        med = np.median(sub, axis=0)
        p25 = np.percentile(sub, 25, axis=0)
        p75 = np.percentile(sub, 75, axis=0)
        x = np.arange(100)
        ax.plot(x, med, color=c, label=sk, linewidth=2)
        ax.fill_between(x, p25, p75, color=c, alpha=0.18)
    ax.set_xlabel("Frame index (within sequence)")
    ax.set_ylabel("# tracks alive in frame")
    ax.set_title("Tracks alive vs frame (median ± IQR per scene)")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PLOTS / "26_tracks_alive.png", dpi=140)
    plt.close()

    # console summary
    print("\n=== Per-class tracks ===")
    print(f"{'class':<18s} {'n':>8s} {'life_med':>9s} {'life_p95':>9s} {'full100':>8s} "
          f"{'spd_med':>8s} {'spd_p95':>8s}")
    for c in plot_classes:
        s = cls_summary[c]
        print(f"{c:<18s} {s['n_tracks']:>8,} "
              f"{s['lifetime_median']:>9.1f} {s['lifetime_p95']:>9.1f} "
              f"{100*s['frac_full_100']:>7.1f}% "
              f"{s['speed_median']:>8.2f} {s['speed_p95']:>8.2f}")
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
