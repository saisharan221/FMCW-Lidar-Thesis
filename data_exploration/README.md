# AevaScenes Dataset Analysis

**Exploration Date:** 2026-04-27
**Purpose:** Extract thesis-relevant insights from AevaScenes dataset

## Quick Summary

- 3 pilot sequences extracted (city/day, highway/day, highway/night), 2.5 GB
- 2 open questions resolved (compensation status, tracking ID field)
- Core thesis statistic computed: |v_t|/|v_r| distribution for 35,581 dynamic objects

## Key Findings

### 1. Velocity is Ego-Motion Compensated
- Stationary objects show mean velocity of **0.104 m/s** (near zero)
- Background GT should be **velocity ≈ 0**

### 2. Tracking ID Field is `'id'`
- UUID format: `"066c2a53-62fb-4e4a-a752-261bd3483874"`
- Persists across frames for object tracking

### 3. Tangential/Radial Velocity Ratio
**CRITICAL THESIS INSIGHT:** 
- **Median ratio = 2.169** (tangential is 2× radial)
- **55.4% have ratio > 2** (predominantly tangential motion)
- **Implication:** Radial Doppler captures <50% of velocity information for majority of objects

## Files Generated

```
data/aevascenes/
├── EXPLORATION_SUMMARY.md          ← Comprehensive findings document
├── README.md                        ← This file
├── extracted/                       ← 3 pilot sequences (2.5 GB)
│   ├── 3f8b6a3e-3b79-4635-8bf4-818ba8d4eaf8/  (city/day)
│   ├── ab87b214-a867-4e43-8d74-a2123966ed3d/  (highway/day)
│   └── 2dc3a21e-57ab-4941-98d1-a110c2cde428/  (highway/night)
├── scripts/
│   ├── verify_compensation.py       ← Compensation verification
│   ├── find_tracking_id.py          ← Tracking ID identification
│   ├── compute_vt_vr_ratio.py       ← Velocity ratio computation
│   └── plot_vt_vr_histogram.py      ← Visualization
├── stats/
│   ├── compensation_verification.json   ← Stationary object velocities
│   ├── vt_vr_distribution.json          ← Ratio distribution summary
│   └── vt_vr_raw.npz                    ← Raw data (35,581 objects)
└── plots/
    ├── vt_vr_histogram.png              ← Ratio distribution histogram
    └── vt_vr_by_class.png               ← Per-class boxplots
```

## Next Steps

1. Review `EXPLORATION_SUMMARY.md` for detailed findings
2. View plots in `plots/` directory
3. Extract additional sequences if needed (97 remaining)
4. Implement GT velocity derivation pipeline
5. Design ML model architecture informed by these statistics

## Usage

All scripts use conda environment `aevascenes`:
```bash
conda run -n aevascenes python3 scripts/script_name.py
```
