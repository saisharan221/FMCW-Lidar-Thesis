"""Sample-frame visualizations: BEV with all 6 lidars, 6-camera grid, points-on-image."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

DATAROOT = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data/aevascenes_v0.2")
PLOTS = Path("C:/Users/djoko/Documents/LNU/FMCW-Lidar-Thesis/data_exploration/full_dataset/plots")

# pick a representative city/day sequence from scene_info.json
SCENE_INFO = json.loads(Path("C:/Users/djoko/Documents/LNU/aevascenes/metadata/scene_info.json").read_text())
LIDAR_IDS = ["front_wide_lidar", "front_narrow_lidar",
             "right_lidar", "rear_wide_lidar", "rear_narrow_lidar", "left_lidar"]
CAMERA_IDS = ["front_wide_camera", "front_narrow_camera",
              "right_camera", "rear_wide_camera", "rear_narrow_camera", "left_camera"]
LIDAR_COLOR = {
    "front_wide_lidar": "#1f77b4",
    "front_narrow_lidar": "#ff7f0e",
    "right_lidar": "#2ca02c",
    "rear_wide_lidar": "#d62728",
    "rear_narrow_lidar": "#9467bd",
    "left_lidar": "#8c564b",
}


def quat_to_rot(q):
    w, x, y, z = q["w"], q["x"], q["y"], q["z"]
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def pose_to_matrix(pose):
    R = quat_to_rot(pose["rotation"])
    t = np.array([pose["translation"]["x"], pose["translation"]["y"], pose["translation"]["z"]])
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = t
    return M


def transform_points(M, xyz):
    h = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
    return (h @ M.T)[:, :3]


def find_seq(road_type, lighting):
    for u, info in SCENE_INFO.items():
        if info.get("road_type") == road_type and info.get("lighting_condition") == lighting:
            sd = DATAROOT / u
            if sd.exists():
                return sd
    return None


def bev_for_seq(sd, frame_idx, suffix, color_mode):
    with open(sd / "sequence.json") as f:
        d = json.load(f)
    md = d["metadata"]
    fr = d["frames"][frame_idx]
    fig, ax = plt.subplots(figsize=(10, 10))

    pts_all, vel_all, refl_all, lid_all = [], [], [], []
    for lid in LIDAR_IDS:
        if lid not in fr["point_cloud"]: continue
        rel = fr["point_cloud"][lid]["point_cloud_path"]
        npz = np.load(sd / rel, allow_pickle=True)
        xyz = npz["xyz"]
        # transform to vehicle frame
        M = pose_to_matrix(md["vehicle_to_lidar_extrinsics"][lid])
        xyz_v = transform_points(M, xyz)
        pts_all.append(xyz_v)
        vel_all.append(npz["velocity"].reshape(-1))
        refl_all.append(npz["reflectivity"].reshape(-1))
        lid_all.append(np.full(xyz_v.shape[0], lid))

    pts = np.concatenate(pts_all)
    vel = np.concatenate(vel_all)
    refl = np.concatenate(refl_all)

    # subsample heavily for plotting
    if pts.shape[0] > 200_000:
        idx = np.random.choice(pts.shape[0], 200_000, replace=False)
        pts, vel, refl = pts[idx], vel[idx], refl[idx]
        lid_concat = np.concatenate(lid_all)[idx]
    else:
        lid_concat = np.concatenate(lid_all)

    if color_mode == "velocity":
        # Plot stationary points in light grey as a backdrop, then dynamic on top.
        stat = np.abs(vel) < 0.5
        ax.scatter(pts[stat, 0], pts[stat, 1], c="#cccccc", s=0.6, alpha=0.5,
                   linewidths=0)
        sc = ax.scatter(pts[~stat, 0], pts[~stat, 1], c=vel[~stat], cmap="seismic",
                        vmin=-15, vmax=15, s=2.5, alpha=0.9, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7); cb.set_label("Doppler velocity (m/s)")
    elif color_mode == "lidar":
        for lid in LIDAR_IDS:
            mask = lid_concat == lid
            if mask.any():
                ax.scatter(pts[mask, 0], pts[mask, 1], c=LIDAR_COLOR[lid],
                           s=0.6, alpha=0.6, label=lid, linewidths=0)
        ax.legend(markerscale=10, loc="lower right", fontsize=9)
    else:
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=np.clip(refl, 0, 200),
                        cmap="viridis", s=0.7, alpha=0.85, linewidths=0)
        cb = fig.colorbar(sc, ax=ax, shrink=0.7); cb.set_label("Reflectivity (clipped 0–200)")

    # Draw bboxes (BEV rectangles)
    for b in fr["boxes"]:
        cx, cy = b["pose"]["translation"]["x"], b["pose"]["translation"]["y"]
        dx, dy = b["dimensions"]["x"], b["dimensions"]["y"]
        rot = quat_to_rot(b["pose"]["rotation"])
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        # corner offsets
        c = np.array([[ dx/2,  dy/2], [ dx/2, -dy/2],
                      [-dx/2, -dy/2], [-dx/2,  dy/2]])
        R2 = np.array([[math.cos(yaw), -math.sin(yaw)],
                       [math.sin(yaw),  math.cos(yaw)]])
        c = c @ R2.T + np.array([cx, cy])
        ax.plot(np.r_[c[:, 0], c[0, 0]], np.r_[c[:, 1], c[0, 1]],
                color="lime", linewidth=1, alpha=0.9)

    ax.scatter([0], [0], c="red", marker="x", s=120, label="ego")
    ax.set_xlim(-150, 250); ax.set_ylim(-100, 100)
    ax.set_aspect("equal")
    ax.set_xlabel("Vehicle X (m, forward)"); ax.set_ylabel("Vehicle Y (m, left)")
    scene = SCENE_INFO.get(sd.name, {})
    ax.set_title(f"BEV — {scene.get('road_type','?')}/{scene.get('lighting_condition','?')} "
                 f"(seq {sd.name[:8]}…, frame {frame_idx})  "
                 f"{len(fr['boxes'])} bboxes")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS / f"14_bev_{suffix}_{color_mode}.png", dpi=140)
    plt.close()


def camera_grid(sd, frame_idx, suffix):
    with open(sd / "sequence.json") as f:
        d = json.load(f)
    fr = d["frames"][frame_idx]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    layout = [("front_wide_camera",   axes[0, 0]),
              ("front_narrow_camera", axes[0, 1]),
              ("rear_wide_camera",    axes[0, 2]),
              ("left_camera",         axes[1, 0]),
              ("rear_narrow_camera",  axes[1, 1]),
              ("right_camera",        axes[1, 2])]
    for cam, ax in layout:
        if cam not in fr["image"]:
            ax.axis("off"); continue
        img = Image.open(sd / fr["image"][cam]["image_path"])
        # downsample 4× for plot
        img = img.resize((img.width // 4, img.height // 4))
        ax.imshow(np.array(img))
        ax.set_title(cam, fontsize=10)
        ax.axis("off")
    scene = SCENE_INFO.get(sd.name, {})
    plt.suptitle(f"Camera grid — {scene.get('road_type','?')}/{scene.get('lighting_condition','?')} "
                 f"(seq {sd.name[:8]}…, frame {frame_idx})", fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / f"15_cameras_{suffix}.png", dpi=120)
    plt.close()


def main():
    np.random.seed(0)
    samples = [
        ("city",    "day",   "city_day"),
        ("highway", "day",   "highway_day"),
        ("city",    "night", "city_night"),
        ("highway", "night", "highway_night"),
    ]
    for road, light, suffix in samples:
        sd = find_seq(road, light)
        if sd is None:
            print(f"  no {road}/{light} found"); continue
        print(f"  {road}/{light} -> {sd.name}")
        bev_for_seq(sd, frame_idx=50, suffix=suffix, color_mode="velocity")
        camera_grid(sd, frame_idx=50, suffix=suffix)

    # Single composite: lidar-colored BEV for one sequence
    sd = find_seq("city", "day")
    if sd:
        bev_for_seq(sd, frame_idx=50, suffix="city_day", color_mode="lidar")
        bev_for_seq(sd, frame_idx=50, suffix="city_day", color_mode="reflectivity")

    print(f"Wrote sample-frame plots to {PLOTS}")


if __name__ == "__main__":
    main()
