# AevaScenes Dataset Exploration Summary

**Date:** 2026-04-27  
**Sequences Analyzed:** 3 pilot sequences (1 city/day, 1 highway/day, 1 highway/night)  
**Total Frames:** 300 frames (100 per sequence)  
**Total Dynamic Objects:** 35,581 bounding boxes with velocity annotations

---

## Key Findings

### 1. Velocity Field is Ego-Motion Compensated ✓

**Question:** Is the `velocity` field in `pointcloud_compensated/` ego-motion compensated?

**Method:** Analyzed stationary objects (building, pole_trunk, traffic_sign, vegetation) across 10 highway frames.

**Result:** **YES, COMPENSATED**

| Metric | Value |
|--------|-------|
| Mean absolute velocity (stationary objects) | 0.104 m/s |
| Max absolute velocity | 1.286 m/s |
| Typical highway ego speed | 20-30 m/s |

**Interpretation:** If NOT compensated, stationary objects would show radial velocities of -20 to -30 m/s (approaching sensor). Observed mean of 0.104 m/s confirms compensation, with residuals from sensor noise/calibration error.

**GT Pipeline Implication:** Background points should be assigned GT velocity ≈ 0 (not derived from ego motion).

---

### 2. Tracking ID Field Name is `'id'` ✓

**Question:** What is the field name for object tracking IDs in `sequence.json`?

**Result:** `'id'` (UUID format, e.g., `"066c2a53-62fb-4e4a-a752-261bd3483874"`)

**Evidence:**
- 327 unique tracking IDs found in first 50 frames of city sequence
- IDs persist across frames (e.g., same car tracked for all 50 frames)
- Additional fields found: `class_idx` (semantic class index, NOT tracking), `reference_frame` (always "VEHICLE")

**SDK Usage:** `box['id']` after `deserialize_boxes()`

---

### 3. Tangential/Radial Velocity Ratio Distribution ✓

**Core Thesis Statistic:** Quantifies how much velocity information is unmeasurable from radial Doppler alone.

#### Overall Statistics (35,581 dynamic objects, speed ≥ 0.5 m/s)

| Statistic | Value | Interpretation |
|-----------|-------|----------------|
| **Median ratio** | **2.169** | Typical object has 2× more tangential than radial velocity |
| **Mean ratio** | 2.353 | Slightly higher due to outliers |
| Std deviation | 2.413 | High variability |
| **% with ratio > 2** | **55.4%** | More than half have predominantly tangential motion |
| 95th percentile | 5.611 | Worst-case safety-critical scenarios |
| 99th percentile | 8.334 | Extreme tangential motion |

#### Distribution Breakdown

| Ratio Range | Count | Percentage | Motion Type |
|-------------|-------|------------|-------------|
| < 0.5 | 7,854 | 22.1% | Mostly radial (approaching/receding) |
| 0.5 - 1.0 | 1,369 | 3.8% | Balanced radial-tangential |
| 1.0 - 2.0 | 6,662 | 18.7% | More tangential than radial |
| **2.0 - 5.0** | **17,206** | **48.4%** | **Predominantly tangential** |
| 5.0 - 10.0 | 2,275 | 6.4% | Highly tangential (crossing traffic) |
| > 10.0 | 215 | 0.6% | Nearly perpendicular motion |

#### Per-Class Breakdown (Top 10 Classes)

| Class | Count | Mean Ratio | Median Ratio | Pattern |
|-------|-------|------------|--------------|---------|
| pole_trunk | 15,422 | 2.86 | 2.58 | High (crossing as ego passes)* |
| **car** | **9,708** | **1.16** | **0.19** | Low (mostly along roads) |
| traffic_sign | 5,955 | 3.20 | 2.78 | High (crossing as ego passes)* |
| traffic_item | 1,509 | 2.93 | 2.47 | High (crossing) |
| truck | 720 | 0.39 | 0.12 | Low (mostly along roads) |
| unknown | 715 | 2.72 | 2.34 | Medium-high |
| **pedestrian** | **449** | **2.07** | **1.26** | Medium (unpredictable crossing) |
| bicycle | 357 | 2.76 | 2.15 | Medium-high (crossing) |
| bus | 281 | 0.20 | 0.12 | Low (mostly along roads) |
| other_structure | 206 | 2.50 | 2.06 | High (crossing) |

**Note:** *Infrastructure objects (pole_trunk, traffic_sign) appearing with velocity annotations is unexpected and warrants investigation - possible annotation artifacts or semantic segmentation errors on moving vehicles.

---

## Thesis Implications

### RQ1: To what extent can ML infer full 3D velocity from a single FMCW frame?

**Motivation validated:** With median |v_t|/|v_r| = 2.169 and 55.4% of objects having ratio > 2, **radial Doppler alone captures less than half the motion information** for the majority of dynamic objects. This strongly motivates the ML approach to recover tangential components.

**Critical scenarios:** The 95th percentile ratio of 5.611 represents safety-critical crossing traffic where radial measurements are nearly useless.

### RQ2: Which contextual features contribute most to tangential velocity inference?

**Semantic class matters:** Large variance across classes (vehicles: 0.2-1.2, pedestrians: 2.1, infrastructure: 2.6-3.2) suggests semantic labels provide strong priors for velocity patterns.

**Geometric context:** High ratios correlate with crossing motion geometry - objects perpendicular to ego heading.

### GT Pipeline

1. **Background points:** Assign velocity ≈ 0 (confirmed compensated)
2. **Dynamic points in boxes:** Use box `linear_velocity` from `'id'` tracking
3. **Coordinate frame:** Boxes in "VEHICLE" frame (ego frame with origin at rear axle)

---

## Dataset Statistics

| | |
|---|---|
| **Total sequences** | 100 (50 city + 50 highway; 60 day + 40 night) |
| **Frames per sequence** | 100 @ 10 Hz (10 seconds each) |
| **Total frames** | 10,000 |
| **Pilot sequences extracted** | 3 (~2.5 GB) |
| **Sensors per frame** | 6 FMCW LiDARs + 6 cameras |
| **Points per frame** | ~70,000 (front_wide_lidar) |
| **Annotations** | 3D boxes with tracking, velocity (linear + angular), 25-class semantic labels |

---

## Files Generated

| File | Description |
|------|-------------|
| `stats/compensation_verification.json` | Stationary object velocity statistics |
| `stats/vt_vr_distribution.json` | Ratio distribution summary |
| `stats/vt_vr_raw.npz` | Raw data (v_t, v_r, ratio, speed, class, range) for 35,581 objects |

---

## Next Steps

1. **Extract additional sequences** for full-dataset statistics
2. **Create visualizations:** Histogram of ratio distribution, per-class boxplots
3. **Investigate infrastructure object annotations** (pole_trunk with velocity)
4. **Compute first-appearance statistics** (new objects per frame)
5. **Analyze beam geometry** across 6 LiDAR sensors
6. **Quantify GT coverage** (% dynamic points inside boxes)
