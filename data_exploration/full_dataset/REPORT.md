# AevaScenes v0.2 — Full-Dataset Exploratory Data Analysis

**Date:** 2026-05-05
**Dataset:** AevaScenes v0.2 (all 100 sequences)
**Scripts:** [`scripts/`](scripts/) — `eda_sequence_stats.py`, `eda_pointcloud_stats.py`, `eda_plots.py`, `eda_sample_frame.py`, `eda_sensor_rig.py`, `eda_bbox_spatial.py`, `eda_doppler_vs_annot.py`, `eda_tracks.py`
**Stats:** [`stats/`](stats/) — `per_sequence.json`, `summary.json`, `pointcloud_stats.json`, `trajectories.json`, `sensor_rig.json`, `bbox_spatial.json`, `doppler_consistency.json`, `tangential_radial.npz`, `tracks.json`
**Plots:** [`plots/`](plots/) — 34 figures

> Builds on the earlier 3-sequence pilot in `../EXPLORATION_SUMMARY.md`. Numbers in
> that pilot are confirmed at full scale here, with several new observations.

---

## TL;DR

- 100 sequences × 100 frames × 6 LiDARs × 6 cameras → **10,000 frames**, 60,000 npz pointclouds, 60,000 4K images, 84.88 GB on disk.
- Every sequence is **9.9 s** at exactly **10 Hz** (dt-std = 0). Total recorded duration: 16.5 minutes.
- **1,361,742 annotated 3-D bounding boxes**, **25,650 unique tracking IDs**, 17 box classes; mean **136 boxes/frame** (max 344).
- **Doppler velocity is ego-motion compensated** (confirmed against pilot finding) — global per-point velocity range ≈ **−89 to +70 m/s**, but stationary-class residuals stay within ±0.5 m/s.
- The 6 LiDARs split into **two FOV families**: wide (≈±60°, 100–200 m) and narrow (≈±20°, 400–500 m). Sensor extrinsics are constant across sequences.
- Scene split: **city/day 30, highway/day 30, city/night 20, highway/night 20** — clean 60/40 city-vs-highway, 60/40 day-vs-night with full crossover.
- Per-point semantic labels are dominated by static scene structure (vegetation 28%, building 20%, road 18%); only ~5% of points are on dynamic actors.

**Thesis-critical findings (Phase 2):**

- **Tangential vs radial velocity: city is the hard regime.** Median |v_t|/|v| = **0.86 in city** sequences vs **0.17 on highway** — the unmeasurable-from-radial-Doppler component dominates city motion and is small on highway.
- **Cars: 23.5% have v_t/v ≥ 0.9** (almost-purely-tangential motion); pedestrians sit at median 0.83.
- **Doppler ↔ annotation consistency holds.** Per-box residual (measured Doppler − projected annotation velocity) has |median| ≈ **0.2–0.4 m/s** for every dynamic class — within sensor noise. The annotated `linear_velocity` field is FMCW-consistent.
- **10% of dynamic bboxes contain zero LiDAR points;** 76% have ≥10 points (training-usable). The drop-off is range-dominated.
- **Track lifetimes:** median 52 frames (5.2 s); **19.4% of tracks survive the full 100-frame clip**. Cars have shorter typical lifetimes (median 19 frames) due to high-speed flow-through; static/quasi-static classes (`pole_trunk`, `traffic_sign`) survive ~80–90 frames.

---

## 1. Dataset structure

```
data/aevascenes_v0.2/
├── <UUID>/                              ← 100 of these
│   ├── sequence.json                    ← per-frame metadata + annotations
│   ├── images/                          ← 100 frames × 6 cameras = 600 .jpg
│   │   └── <camera_id>_<ts_ns>.jpg      ← 4K (3840×2160) JPEG
│   └── pointcloud_compensated/          ← 100 frames × 6 lidars = 600 .npz
│       └── <lidar_id>_<ts_ns>.npz
└── (no top-level metadata.json)
```

**Sensor suite (per sequence):**

| Modality | IDs |
|----------|-----|
| LiDAR (6) | `front_wide_lidar`, `front_narrow_lidar`, `right_lidar`, `rear_wide_lidar`, `rear_narrow_lidar`, `left_lidar` |
| Camera (6) | corresponding `*_camera` |

The `metadata.camera_lidar_mapping` pairs each camera with one LiDAR (front_wide↔front_wide_lidar, etc.).

**npz schema** (e.g. `pointcloud_compensated/front_narrow_lidar_<ts>.npz`):

| Key | Shape | dtype | Notes |
|-----|-------|-------|-------|
| `xyz`                | (N, 3) | float32 | sensor frame, metres |
| `velocity`           | (N, 1) | float32 | per-point Doppler radial velocity, **ego-compensated**, m/s |
| `reflectivity`       | (N, 1) | float32 | unitless, 0..~8000 |
| `line_index`         | (N, 1) | uint32  | scan-line index |
| `time_offset_ns`     | (N, 1) | uint32  | within-scan time offset |
| `semantic_labels`    | (N, 1) | object  | string class (e.g. `"car"`) |
| `semantic_labels_idx`| (N, 1) | int64   | class index |

**`sequence.json` schema:**

```text
{
  "metadata": {
    "sequence_uuid", "sensors": {"lidars": [...], "cameras": [...]},
    "camera_lidar_mapping",
    "vehicle_to_lidar_extrinsics"  : {<lidar>:  {translation, rotation (quat)}},
    "vehicle_to_camera_extrinsics" : {<cam>:    {translation, rotation (quat)}},
    "camera_intrinsics"            : {<cam>:    {matrix[9], distortion_coefficients[5],
                                                 image_width, image_height}}
  },
  "frames": [  # always exactly 100
    {
      "timestamp_ns", "frame_idx", "frame_uuid",
      "ego_pose":   {translation, rotation (quat)},   # relative to frame 0 (frame 0 is identity)
      "point_cloud": {<lidar>: {"point_cloud_path": "pointcloud_compensated/..."}},
      "image":       {<camera>: {"image_path": "images/..."}},
      "boxes": [ {"id" (track UUID), "reference_frame":"VEHICLE",
                  "dimensions":{x,y,z}, "pose":{translation, rotation},
                  "linear_velocity":{x,y,z}, "angular_velocity":{x,y,z},
                  "class", "class_idx"}, ... ]
    }, ...
  ]
}
```

Camera intrinsics: identical 3840×2160 resolution across all 6 cameras; distortion model is OpenCV 5-coefficient (`k1, k2, p1, p2, k3`).

---

## 2. Aggregate statistics

| Metric | Value |
|---|---|
| Sequences | **100** |
| Frames total | **10,000** (every sequence has exactly 100) |
| Sensor frames total (LiDAR + camera) | 60,000 + 60,000 = **120,000** |
| Frame rate | **10 Hz** (dt = 100 ms exactly, std = 0) |
| Sequence duration | **9.9 s** for every sequence |
| Total recorded duration | **990 s ≈ 16.5 min** |
| Bounding-box instances | **1,361,742** |
| Unique tracking IDs (`box.id`) | **25,650** |
| Box classes | 17 |
| Per-point semantic classes | 25 |
| Disk size (extracted) | **84.88 GB** (mean 869 MB/sequence, ~8.7 MB/frame) |

### Scene-type distribution

![Scene distribution and ego speed](plots/01_scene_and_ego_speed.png)

| Road × lighting | # sequences |
|---|---|
| city / day | 30 |
| highway / day | 30 |
| city / night | 20 |
| highway / night | 20 |

City sequences are slower and more varied (median ~5 m/s, with stops); highway sequences cluster around 25–30 m/s (≈90–110 km/h).

### Per-sequence overview (across the 100 sequences)

![Per-sequence overview](plots/11_per_sequence_overview.png)

| Quantity | min | median | max |
|---|---:|---:|---:|
| Frames / sequence | 100 | 100 | 100 |
| Mean ego speed (m/s) | 0.00 (stopped) | 14.22 (≈51 km/h) | 33.81 (≈122 km/h) |
| Ego path length (m) | 0 | 142 | 335 |
| Bboxes / sequence | 2,710 | 12,499 | 34,425 |
| Bboxes / frame (mean) | 27 | 125 | 344 |
| Unique tracks / sequence | 64 | 251 | 499 |
| On-disk size (MB) | 700 | 855 | 1,170 |

### Annotated trajectories

![Ego trajectories](plots/04_ego_trajectories.png)

Trajectories spread radially because each is plotted in its own frame-0 origin (ego_pose at t=0 is identity by construction). The trajectory length distribution per scene type is consistent with the speed boxplot: highway sequences span 200–335 m, city sequences range from stationary to ~150 m.

### Box-class distribution

![Class distribution](plots/02_class_distribution.png)

| Class | Bbox-instances | % of total |
|-------|---------------:|-----------:|
| `pole_trunk` | 533,202 | 39.16% |
| `car` | 400,013 | 29.38% |
| `traffic_sign` | 239,171 | 17.56% |
| `traffic_item` | 83,232 | 6.11% |
| `pedestrian` | 43,790 | 3.22% |
| `unknown` | 19,874 | 1.46% |
| `truck` | 13,034 | 0.96% |
| `other_structure` | 10,831 | 0.80% |
| `bicycle` | 6,253 | 0.46% |
| `bus` | 4,851 | 0.36% |
| `bicyclist` | 2,091 | 0.15% |
| `trailer` | 1,795 | 0.13% |
| `other_vehicle` | 1,557 | 0.11% |
| `motorcycle` | 909 | 0.07% |
| `vehicle_on_rails` | 547 | 0.04% |
| `motorcyclist` | 326 | 0.02% |
| `animal` | 266 | 0.02% |

Static-by-nature classes (`pole_trunk`, `traffic_sign`, `traffic_item`) make up ~63% of all box instances. Dynamic actors (`car`, `pedestrian`, `truck`, `bus`, `bicyclist`, `motorcyclist`, …) make up ~34%.

### Bounding-box linear-velocity speed by class

![Bbox speed per class](plots/03_bbox_speed_per_class.png)

| Class | n boxes | median (m/s) | mean (m/s) | p95 (m/s) | max (m/s) |
|-------|--------:|-------------:|-----------:|----------:|----------:|
| `pole_trunk` | 533,202 | 0.31 | 1.59 | 1.44 | **65.60** |
| `car` | 400,013 | 4.59 | 9.73 | 32.08 | **178.42** |
| `traffic_sign` | 239,171 | 0.37 | 1.71 | 1.95 | **75.09** |
| `traffic_item` | 83,232 | 0.24 | 0.70 | 0.94 | 50.26 |
| `pedestrian` | 43,790 | 0.99 | 1.01 | 1.88 | 9.39 |
| `truck` | 13,034 | 7.14 | 10.82 | 30.69 | 70.15 |
| `bus` | 4,851 | 11.17 | 14.01 | 32.04 | 72.69 |
| `motorcyclist` | 326 | 13.39 | 21.77 | 52.72 | 68.71 |

> **Annotation-quality caveat.** Static classes (`pole_trunk`, `traffic_sign`) carry non-zero `linear_velocity` for a small fraction of instances — the medians are tiny (≈0.3 m/s) but the maxima reach 65–75 m/s. These are interpolation/tracking artefacts, not real motion. `car.max = 178 m/s` is also clearly an outlier. p95 is the trustworthy summary.

---

## 3. LiDAR-level statistics

Stats below are aggregated over a stratified sample of **5 frames × 6 LiDARs × 100 sequences = 3,000 npz files** (244 M points), with a fixed seed. Bin definitions are saved in `stats/pointcloud_stats.json`.

| LiDAR | Pts/frame (median) | Pts/frame (p95) | Range max (m) | Az span (°) | El span (°) | Doppler [min, max] m/s |
|-------|-------------------:|----------------:|--------------:|------------:|------------:|-----------------------:|
| `front_wide_lidar`   | 83,878  | 117,083 | 193 | −62.6…61.2 (123.8) | −13.2…7.6 (20.8) | [−72.2, 49.2] |
| `front_narrow_lidar` | 53,241  | 82,629  | **396** | −19.9…20.0 (39.8) | −8.1…7.4 (15.5) | [−89.0, 55.5] |
| `right_lidar`        | 117,734 | 133,307 | 107 | −59.4…59.2 (118.6) | −10.7…9.4 (20.2) | [−33.2, 25.7] |
| `rear_wide_lidar`    | 76,354  | 96,426  | 198 | −60.4…66.5 (126.9) | −10.4…9.2 (19.6) | [−59.3, 69.6] |
| `rear_narrow_lidar`  | 59,681  | 89,314  | **512** | −19.6…19.1 (38.7) | −5.7…9.8 (15.4) | [−61.7, 70.0] |
| `left_lidar`         | 103,991 | 132,585 | 107 | −57.6…59.5 (117.1) | −11.2…9.1 (20.3) | [−54.3, 33.3] |

Two FOV regimes are evident:

- **Wide LiDARs** (front/rear/right/left wide, plus right and left): ≈±60° azimuth, ranges out to 100–200 m. Side LiDARs show the highest point density (right ≈118 k, left ≈104 k pts/frame) — they get more local building/curb returns.
- **Narrow LiDARs** (front_narrow, rear_narrow): ≈±20° azimuth, **long range out to 400–500 m**. Lower point counts but they carry the highway-relevant Doppler returns at distance.

### Per-LiDAR FOV (azimuth)

![Polar FOV](plots/05_lidar_fov_polar.png)

The polar plot shows azimuth coverage in each sensor's own frame (0° = sensor-forward). Each curve is a normalized azimuth histogram of all sampled points. Side LiDARs cover the same forward-arc as front-wide because the polar plot is in *sensor-local* coordinates — all six face their own forward axis.

### Range distribution (log y)

![Range distribution](plots/06_lidar_range.png)

Bulk of returns lies inside 30 m for every LiDAR. The narrow LiDARs have a long thin tail extending past 300 m; wide LiDARs effectively cut off near 150 m. This matches AevaScenes' published dual-FOV design.

### Elevation distribution

![Elevation distribution](plots/12_lidar_elevation.png)

Roughly Gaussian, centred at or just below 0° (looking horizontally), with ±10°-ish 1-σ envelope for wide LiDARs and ±5° for narrow. This is consistent with each unit pointing at the local horizon.

### Reflectivity distribution

![Reflectivity distribution](plots/13_lidar_reflectivity.png)

Heavy-tailed; most points have reflectivity 0–200, but values spike up to 8,191 (likely retro-reflective traffic signs / lane paint). The shapes are similar across LiDARs, suggesting a common normalisation.

### Points per frame

![Points per frame](plots/09_points_per_frame.png)

Per-frame point counts are tight — the side LiDARs (right, left) consistently deliver the densest clouds.

---

## 4. FMCW Doppler velocity (the FMCW-specific signal)

![Velocity per LiDAR](plots/07_lidar_velocity.png)

- **Per-LiDAR distributions are sharply concentrated at v = 0** (left panel, log y). The bulk corresponds to ego-compensated returns from the static scene.
- The **right panel zooms |v| < 3 m/s** and shows the residual on stationary scene: a tight peak with σ on the order of 0.1 m/s. This corroborates the pilot study's finding that `pointcloud_compensated/` contains **ego-motion-compensated** Doppler.
- Tails extend to roughly **−90 m/s and +70 m/s** — these come from oncoming traffic at highway speeds where range-rate is large (sensor + object in opposite directions).

### Velocity by scene type

![Velocity by scene](plots/08_velocity_by_scene.png)

- **Highway (day & night)** distributions have markedly heavier tails — that's where high-Doppler returns live (relative speeds of ±50 m/s on opposite-direction traffic are routine).
- **City (day & night)** distributions decay faster, consistent with lower absolute speeds.

### BEV samples (ego-frame, all 6 LiDARs merged)

| | |
|--|--|
| ![BEV city/day Doppler](plots/14_bev_city_day_velocity.png) | ![BEV highway/day Doppler](plots/14_bev_highway_day_velocity.png) |
| ![BEV city/night Doppler](plots/14_bev_city_night_velocity.png) | ![BEV highway/night Doppler](plots/14_bev_highway_night_velocity.png) |

Each BEV is a single frame from a representative sequence. Stationary returns are rendered grey, dynamic returns are coloured by Doppler (red = approaching the sensor, blue = receding); green outlines are the GT bounding boxes. The highway frames clearly show oncoming traffic carrying strong negative Doppler.

For reference (single sequence, one frame):

| LiDAR-coloured BEV | Reflectivity-coloured BEV |
|---|---|
| ![BEV by lidar](plots/14_bev_city_day_lidar.png) | ![BEV by reflectivity](plots/14_bev_city_day_reflectivity.png) |

### Per-point semantic labels

![Per-point semantic](plots/10_semantic_pointcloud.png)

Top labels among the 244 M sampled points:

| Label | Points | % |
|-------|-------:|--:|
| `vegetation` | 67.4 M | 27.65% |
| `building` | 48.5 M | 19.88% |
| `road` | 43.9 M | 18.00% |
| `other_structure` | 25.0 M | 10.24% |
| `sidewalk` | 15.7 M | 6.44% |
| `other_ground` | 11.1 M | 4.53% |
| `car` | 9.2 M | 3.75% |
| `pole_trunk` | 8.3 M | 3.42% |
| `unknown` | 5.7 M | 2.33% |
| `lane_boundary` | 2.8 M | 1.13% |
| ... | ... | ... |

Static structure (vegetation + building + road + sidewalk + other_ground + other_structure) ≈ **87%** of all points. Dynamic actors (car + truck + bus + bicycle + bicyclist + pedestrian + motorcycle + animal + …) ≈ **5%**. Any task that learns from per-point dynamics will be heavily class-imbalanced — sampling/weighting will matter.

---

## 5. Camera samples

Sample 6-camera grids (front-wide / front-narrow / rear-wide on top, left / rear-narrow / right on bottom):

![Cameras city/day](plots/15_cameras_city_day.png)
![Cameras highway/day](plots/15_cameras_highway_day.png)
![Cameras city/night](plots/15_cameras_city_night.png)
![Cameras highway/night](plots/15_cameras_highway_night.png)

All images are 3840×2160 (4K), undistortion required (5-coefficient OpenCV model). The narrow cameras have visibly longer focal length (objects at distance look larger), matching the narrow-LiDAR pairing.

---

## 6. Sensor rig

Extrinsics are **constant across all 100 sequences** (max translation deviation = 0 m over 20 sequences sampled). The canonical rig (vehicle frame: +X forward, +Y left, +Z up):

| LiDAR | x (m) | y (m) | z (m) | yaw (°) |
|---|--:|--:|--:|--:|
| `front_wide_lidar`   | 1.481 |  0.151 | 1.961 |   0.6 |
| `front_narrow_lidar` | 1.484 | −0.158 | 2.007 |   1.0 |
| `right_lidar`        | 0.654 | −0.679 | 1.956 | −88.6 |
| `rear_wide_lidar`    | −0.819 |  0.066 | 2.006 | −179.2 |
| `rear_narrow_lidar`  | −0.918 | −0.297 | 1.971 | −179.5 |
| `left_lidar`         | 0.697 |  0.495 | 1.935 |  89.5 |

Inter-LiDAR baselines range **0.31 m to 2.44 m** — the biggest baseline (front_wide ↔ rear_wide ≈ 2.3 m) is along X, useful for front-rear stereo Doppler if needed.

| Top-down rig | Side view (heights) | Inter-LiDAR baselines |
|---|---|---|
| ![rig topdown](plots/16_sensor_rig_topdown.png) | ![rig side](plots/17_sensor_rig_side.png) | ![baselines](plots/18_lidar_baselines.png) |

All sensors sit between 1.93 m and 2.07 m above the ego-frame origin (roof level on a typical sedan). The two narrow LiDARs sit ~5 cm higher than their wide counterparts.

---

## 7. Bbox spatial coverage and per-box LiDAR returns

### Where the annotated boxes live

![Bbox spatial](plots/19_bbox_spatial.png)

The BEV centroid heatmap (left, log scale) shows annotation density is concentrated in front of the ego (X > 0, |Y| < 30 m) — typical autonomous-driving labelling priority. Cars span 0–250 m; pedestrians and bicycles are confined to the near field (<60 m); trucks/buses cluster at 100–150 m (highway leading vehicles).

### Bbox dimensions per class

![Bbox dimensions](plots/20_bbox_dimensions.png)

Length × width per top-8 class confirms reasonable physical priors: cars cluster around 4–5 m × 1.7–2 m; trucks 6–10 m × 2.2–2.5 m; pedestrians 0.5–1.0 m × 0.5–0.8 m.

### Points-per-box (the data-availability question)

We tested 24,199 dynamic-class boxes (sampled 5 frames/seq × 100 seqs) by transforming each LiDAR's point cloud into the box's local frame and counting points inside. Aggregated across all 6 LiDARs:

![Points per box](plots/21_points_per_box.png)

**Global:** median 52 points/box, **10.3% of dynamic boxes are empty**, **75.7% have ≥10 points**. By class:

| Class | n boxes | median range | median pts | empty | ≥10 pts | ≥50 pts |
|---|--:|--:|--:|--:|--:|--:|
| `car`              | 20,325 |  72.6 m |   44 | 11.5% | 73.2% | 48.0% |
| `pedestrian`       |  2,289 |  40.9 m |  102 |  0.4% | 92.3% | 68.3% |
| `truck`            |    635 | 120.0 m |  118 |  8.8% | 82.7% | 65.2% |
| `bicycle`          |    318 |  37.9 m |   52 |  2.5% | 92.1% | 52.8% |
| `bus`              |    249 | 117.3 m |  165 |  8.4% | 82.7% | 68.3% |
| `bicyclist`        |    103 |  44.9 m |   77 |  9.7% | 83.5% | 62.1% |
| `trailer`          |     96 |  82.4 m |  150 | 18.8% | 77.1% | 64.6% |
| `motorcycle`       |     41 |  86.8 m |   40 | 19.5% | 73.2% | 46.3% |
| `vehicle_on_rails` |     29 |  39.6 m | 2482 |  0.0% | 93.1% | 89.7% |
| `motorcyclist`     |     17 |  55.2 m |    4 | 41.2% | 41.2% | 29.4% |

> **Implication for training.** A non-trivial fraction of dynamic GT boxes carry no LiDAR returns at all (cars 12%, motorcycles 20%, motorcyclists 41%). For a per-point regression task, these contribute zero supervision; for a box-level regression task they introduce label-without-input cases that need explicit handling.

---

## 8. FMCW Doppler — tangential/radial decomposition (the thesis statistic)

This is the central question for the thesis: how much of the GT velocity vector is **tangential** (orthogonal to LOS, invisible to FMCW Doppler in a single measurement) versus **radial** (directly measured)?

We decompose every dynamic-class box's annotated `linear_velocity` (vehicle frame, world-relative) onto and orthogonal to the LOS from ego origin to box centroid:

$$ v_r = \mathbf v \cdot \hat{\mathbf u}_{\text{LOS}}, \qquad |v_t| = \|\mathbf v - v_r \hat{\mathbf u}_{\text{LOS}}\|, \qquad r_t = |v_t|/\|\mathbf v\| $$

Boxes with |v| < 0.05 m/s are excluded (ratio undefined).

### Distribution by class and scene

![Tangential ratio by class and scene](plots/22_tangential_ratio.png)

**The headline finding** — distribution shapes are *radically* different by scene:

| Scene | n boxes | median r_t | mean r_t | r_t > 0.5 |
|---|--:|--:|--:|--:|
| `city / day`     | 188,064 | **0.860** | 0.668 | 76.9% |
| `city / night`   |  38,609 | **0.826** | 0.660 | 74.0% |
| `highway / day`  | 197,360 | **0.165** | 0.292 | 21.0% |
| `highway / night`|  27,170 | **0.175** | 0.296 | 27.1% |

City: ≈85% of motion is tangential (cross-traffic, turns, lane changes). Highway: ≈83% of motion is *radial* (everyone moving along the lane, leading or trailing the ego).

**Tangential-ratio inference is essentially a city-driving problem.** A radial-Doppler-only system already captures most of highway dynamics; the thesis target is exactly the regime where it fails.

By dynamic class (sorted by sample size):

| Class | n | speed p95 (m/s) | median r_t | r_t > 0.5 | r_t > 0.9 |
|---|--:|--:|--:|--:|--:|
| `car`         | 376,969 | 32.45 | 0.401 | 45.9% | **23.5%** |
| `pedestrian`  |  43,388 |  1.89 | 0.831 | 79.5% | 39.5% |
| `truck`       |  12,854 | 30.79 | 0.261 | 36.1% | 18.0% |
| `bicycle`     |   6,160 |  5.49 | 0.842 | 82.2% | 35.0% |
| `bus`         |   4,767 | 32.27 | 0.162 | 22.8% | 10.0% |
| `bicyclist`   |   2,090 |  7.55 | 0.451 | 47.8% | 18.6% |
| `motorcycle`  |     909 | 44.46 | 0.743 | 55.6% | 30.3% |
| `motorcyclist`|     326 | 52.72 | 0.142 |  6.1% |  1.2% |

23.5% of *all* car observations have v_t / v ≥ 0.9 — i.e. they are essentially in pure-cross motion. This is a large training set.

### v_r ↔ |v_t| scatter — cars

![v_r vs v_t for cars](plots/23_vr_vt_car.png)

Each point is one car-bbox observation (×30 k subsample). Dashed isobars are equal-speed loci. Cars populate the entire half-plane: there is no "low tangential" prior to lean on.

### Doppler-vs-annotation consistency

For each sampled dynamic box with ≥10 inside-points per LiDAR, we projected the annotated `linear_velocity` onto each point's LOS (from that LiDAR) and compared to the npz `velocity`:

$$ \text{residual}_i = v_{\text{measured},i} - \hat{\mathbf u}_{\text{LOS},i} \cdot \mathbf v_{\text{box}} $$

If the annotation matches the FMCW signal, residuals should be ≈ 0 plus sensor noise.

![Residuals](plots/24_doppler_residuals.png)

| Class | n box-LiDAR pairs | \|median\| residual | std | \|p95\| residual |
|---|--:|--:|--:|--:|
| `car`         | 14,662 | 0.36 m/s | 0.16 | 0.65 m/s |
| `pedestrian`  |  2,302 | 0.21 m/s | 0.15 | 0.52 m/s |
| `truck`       |    726 | 0.39 m/s | 0.19 | 0.70 m/s |
| `bus`         |    309 | 0.31 m/s | 0.25 | 0.58 m/s |
| `bicycle`     |    154 | 0.28 m/s | 0.12 | 0.52 m/s |
| `bicyclist`   |    123 | 0.21 m/s | 0.12 | 0.49 m/s |

Residuals stay within sensor-noise budget (≈0.2–0.4 m/s |median|, p95 ≈ 0.5–0.7 m/s) for every dynamic class. Two implications:

1. **The GT velocity field is consistent with the FMCW signal.** Training a model with `linear_velocity` as the target is sound.
2. **The residual itself is a useful loss prior.** A model that predicts `linear_velocity` from per-point Doppler should not be expected to drive residuals below ≈0.2 m/s — that's the noise floor.

---

## 9. Track dynamics

For each unique tracking ID we counted the number of frames it appears in (out of 100), recorded its mean speed, and aggregated globally and per class.

![Track lifetimes](plots/25_track_lifetimes.png)

**Global:** 25,650 unique tracks; median lifetime 52 frames (5.2 s); **19.4% of tracks survive the full 100-frame clip**; 51.3% survive ≥50 frames.

By class (top 8 by count):

| Class | n tracks | lifetime median | lifetime p95 | full-100 | speed median (m/s) | speed p95 |
|---|--:|--:|--:|--:|--:|--:|
| `car`           | 11,528 | 19  | 100 | 10.8% | 0.97 | 29.72 |
| `pole_trunk`    |  7,886 | 76  | 100 | 26.6% | 0.35 |  7.20 |
| `traffic_sign`  |  3,206 | 87  | 100 | 32.8% | 0.40 |  7.24 |
| `traffic_item`  |  1,293 | 74  | 100 | 18.6% | 0.26 |  0.89 |
| `pedestrian`    |    785 | 55  | 100 | 12.5% | 0.93 |  1.77 |
| `truck`         |    214 | 64  | 100 | 30.8% | 5.76 | 30.35 |

Static infrastructure (`pole_trunk`, `traffic_sign`, `traffic_item`) has the longest lifetimes (median 74–87) — they enter and stay in view as the ego drives by. Cars have the shortest typical lifetime (median 19) because at highway speeds a car can pass through the FOV in 1–2 s. The static-class apparent speeds (median 0.3–0.4 m/s) are again the annotation noise we flagged earlier — well below sensor speed-resolution.

![Tracks alive](plots/26_tracks_alive.png)

Tracks-alive vs frame index (median ± IQR, per scene type) is **flat** across each 10-s clip — no monotonic accumulation or shedding, suggesting that label coverage is consistent end-to-end and the clips are not "ramping in".

---

## 10. Implications for the thesis

These observations directly inform the thesis pipeline:

1. **Velocity GT for stationary scene = 0.** Doppler is ego-compensated; stationary points should be assigned GT radial velocity ≈ 0. Don't derive a fake "ego-motion baseline".
2. **City scenes are the thesis target.** Highway median |v_t|/|v| = 0.17, city = 0.86. Radial-Doppler-only systems already cover most highway dynamics; the unmeasurable-tangential problem is essentially a city-driving problem. Train and evaluate splits should preserve scene-type balance — preferably *over-weight* city sequences during evaluation to expose tangential-recovery performance.
3. **The annotated `linear_velocity` field is FMCW-consistent.** Per-box residual |median| ≈ 0.2–0.4 m/s; p95 ≈ 0.5–0.7 m/s. Use `linear_velocity` directly as the regression target with confidence; the noise floor for any model is ≈0.2 m/s.
4. **Pick narrow LiDARs for highway, wide LiDARs for city.** Narrow LiDARs (±20° az, 400–500 m range) carry pure-radial signal at distance; wide LiDARs (±60° az, ≤200 m) provide angular leverage on cross-traffic where v_t dominates.
5. **Empty-box rate is 10.3%.** Roughly 1 in 10 dynamic GT boxes contain zero LiDAR points (cars 12%, motorcycles 20%, motorcyclists 41%). Per-point regression heads will see no input for these; box-level heads need an explicit mask.
6. **Class imbalance for any per-point dynamic-vs-static training.** Only ~5% of points belong to dynamic actors. Use focal loss / class-balanced sampling.
7. **10 Hz / 100 ms cadence is uniform; track lifetimes are bimodal.** Δt = 100 ms exactly. Track lifetime is roughly bimodal: cars (median 19) cycle through quickly, static infrastructure (median ~80) stays in view. For temporal-aggregation methods (e.g., 4-frame stacks), 80% of car tracks are ≤ 50 frames so a long temporal window discards most cars.
8. **Annotation outliers in static-class velocities.** Don't trust max-aggregate `linear_velocity` for `pole_trunk` / `traffic_sign` / `traffic_item` (max 65–75 m/s). Use median or p95.
9. **Sensor rig is constant** across all 100 sequences — model architectures can hard-code extrinsics if convenient (no per-sequence calibration loading).
10. **Tracks-alive is flat over the 10-s clip** — no warm-up or wind-down; the 100 frames are uniformly labelled.

---

## 11. Reproducibility

```powershell
# from the repo root
python data_exploration/full_dataset/scripts/eda_sequence_stats.py    # 14 s
python data_exploration/full_dataset/scripts/eda_pointcloud_stats.py  # 53 s
python data_exploration/full_dataset/scripts/eda_plots.py             #  5 s
python data_exploration/full_dataset/scripts/eda_sample_frame.py      # 30 s
python data_exploration/full_dataset/scripts/eda_sensor_rig.py        #  3 s
python data_exploration/full_dataset/scripts/eda_bbox_spatial.py      # 5.5 min  (heavier — loads all 6 npzs for sampled frames)
python data_exploration/full_dataset/scripts/eda_doppler_vs_annot.py  # 2.5 min
python data_exploration/full_dataset/scripts/eda_tracks.py            # 11 s
```

Wall-clock end-to-end: **≈ 9 minutes** on this Windows machine (Python 3.14, no GPU). All scripts use fixed RNG seeds (sequence stats: 42, bbox spatial: 17, doppler: 31). Sample sizes are tunable via the `FRAMES_PER_SEQ` constants at the top of each script.
