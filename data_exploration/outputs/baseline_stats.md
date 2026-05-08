# Baseline Results Summary
**Val split — 1,400 frames, 213,706 dynamic points**

## Main table

| Method | EPE_dyn | **EPE_t** | EPE_r | Δθ (°) | \|Δ\|v\|\| |
|---|---|---|---|---|---|
| B0  Zero               | 8.25 | 3.14 | 7.28 |  —   | 8.25 |
| B1  Doppler-only       | 3.87 | 3.14 | 1.29 | 45.5 | 2.08 |
| B2  Class-mean         | 9.47 | 3.00 | 8.65 | 86.0 | 6.46 |
| B3  Dop. + class-tan   | 3.70 | 3.00 | 1.29 | 45.9 | 1.97 |
| PointMLP (trained)     | 3.87 | **3.14** | 1.30 | — | — |

**PTv3 bar to beat: EPE_t < 3.00 m/s**

## Key observations

- B1 and B0 have identical EPE_t (3.14). Doppler fixes radial but contributes nothing tangential.
- B3's class-mean tangent only buys 0.14 m/s improvement over B0/B1 on EPE_t.
- B2 destroys radial accuracy (EPE_r 8.65 > B0's 7.28) — class mean alone is worse than doing nothing.
- Angular error ~45° for both B1 and B3: direction is essentially unpredictable without learning.
- PointMLP (trained with Doppler-residual prior): learned to match B1 on EPE_r but EPE_t = 3.137, identical to B1. Zero tangential learning from a per-point MLP architecture.

## Per-class EPE_t (B1 / B3, classes with ≥500 dynamic points in val)

| Class | B1 EPE_t | B3 EPE_t | Note |
|---|---|---|---|
| bus           | 11.71 | 11.96 | Hardest; large fast objects |
| car           |  9.95 |  9.94 | High individual variation (turning vs. straight) |
| unknown       |  3.88 |  3.88 | — |
| vehicle_on_rails |  3.61 |  3.61 | — |
| trailer       |  3.34 |  3.34 | — |
| truck         |  1.73 |  1.84 | B3 slightly *worse* — class-mean overshoots for mostly-straight trucks |
| traffic_sign  |  1.24 |  1.24 | Quasi-static |
| pole_trunk    |  1.15 |  1.15 | Quasi-static |
| pedestrian    |   —   |   —   | 0 dynamic points in val split (speed below 0.5 m/s threshold) |
| bicycle       |   —   |   —   | 0 dynamic points in val split |

## Per-scene EPE_t (B0, as difficulty baseline)

| Scene | EPE_t | EPE_dyn | n_dyn |
|---|---|---|---|
| city/day      | 2.27 |  3.31 |  33,075 |
| city/night    | 0.63 |  0.75 |     887 |
| highway/day   | 2.71 |  6.99 | 156,391 |
| highway/night | 7.35 | 23.93 |  23,353 |

Highway/night is the hardest subpopulation — high absolute speeds dominate EPE despite low tangential ratio.

## Per-speed EPE_t (B0)

| Speed bucket | EPE_t | n_dyn |
|---|---|---|
| 0.5–5 m/s   | 0.66 |  68,066 |
| 5–15 m/s    | 2.88 | 112,693 |
| 15–25 m/s   | 7.75 |  26,683 |
| 25+ m/s     | 15.07 |   6,264 |

## Tangential ratio breakdown (B0 EPE_t by |v_t|/|v| bucket)

| Ratio bucket | EPE_t | Note |
|---|---|---|
| 0–0.5 (radially dominated) | 2.64 | Component is small, error is small |
| 0.5–1 (mixed)              | 3.15 | Hardest in absolute terms |
| 1–2 (tangentially dominated) | **5.63** | Peak difficulty |
| 2–5 (slow crossing)        | 0.70 | Slow objects, tiny absolute error |
| 5–10                       | 0.65 | |
| 10+                        | 0.83 | |

## Dataset stats (AevaScenes v0.2)

| Stat | Value |
|---|---|
| Sequences | 100 |
| Frames | 10,000 |
| Duration | 16.5 min (9.9s each @ 10 Hz) |
| Total bboxes | 1,361,742 |
| Unique tracks | 25,650 |
| Scene split | 30 city/day, 30 hwy/day, 20 city/night, 20 hwy/night |
| City median \|v_t\|/\|v\| | **0.86** |
| Highway median \|v_t\|/\|v\| | **0.17** |
| Doppler sign convention | positive = receding (verified) |
| box linear_velocity frame | VEHICLE / ego-compensated (verified) |
| Stationary residual | 0.082 m/s (near-zero, good compensation) |

## Figures

- `fig1_grouped_bar.png` — EPE_dyn / EPE_t / EPE_r grouped bar for all methods
- `fig2_epet_horizontal.png` — EPE_t horizontal bar, emphasises the key metric
- `fig3_per_class_epet.png` — per-class EPE_t, B1 vs B3
