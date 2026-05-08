# Student Project Proposal

## Context-Aware Tangential Velocity Inference for FMCW LiDAR
*Inferring Full 3D Velocity Vectors from Radial Doppler Measurements Using Machine Learning*

**Authors:** Suyash Mullick, Saisharan Raja
**Supervisor:** Amilcar Soares
**Semester:** VT 2026
**Course:** 2DV50E
**Subject:** Computer Science / Media Technology

---

## Contents
1. [Administrative Details](#1-administrative-details)
    1.1 [Supervision Status](#11-supervision-status)
    1.2 [Co-operative Partner](#12-co-operative-partner)
2. [Preliminary Thesis Title](#2-preliminary-thesis-title)
3. [Elevator Pitch](#3-elevator-pitch)
4. [Steps/Milestones/Actions](#4-stepsmilestonesactions)
5. [Risks](#5-risks)
6. [Background and Motivations](#6-background-and-motivations)
7. [Related Work](#7-related-work)
8. [Knowledge Gap/Challenge/Problem](#8-knowledge-gapchallengeproblem)
9. [Knowledge Contribution/Action](#9-knowledge-contributionaction)
10. [Empirical Evidence/Evaluation](#10-empirical-evidenceevaluation)
11. [References](#references)

---

## 1 Administrative Details
**Name of students:** Suyash Mullick, Saisharan Raja

### 1.1 Supervision Status
Agreed with Amilcar Soares but not yet started.

### 1.2 Co-operative Partner
Working with **Einride**, a freight technology company developing autonomous electric transport solutions. A contract for the thesis collaboration was signed in January 2026. Einride provides domain expertise in autonomous driving perception and potential access to FMCW LiDAR sensor data relevant to autonomous driving scenarios.

## 2 Preliminary Thesis Title
Context-Aware Tangential Velocity Inference for FMCW LiDAR: Inferring Full 3D Velocity Vectors from Radial Doppler Measurements Using Machine Learning

## 3 Elevator Pitch
**Background:** FMCW LiDAR is a next-generation sensor for autonomous driving that measures both range and per-point radial velocity through the Doppler effect, providing richer scene information than conventional Time-of-Flight LiDAR [1, 2]. 

**Challenge:** However, FMCW LiDAR can only capture the velocity component along the laser beam direction; tangential components perpendicular to the beam are physically unmeasurable. This means objects moving sideways relative to the sensor, such as crossing traffic at intersections, may appear nearly stationary in raw data despite travelling at high speed, creating a critical safety limitation [2]. Existing approaches to resolve full velocity require multi-frame tracking, which introduces latency and fails for newly appearing objects [7]. 

**Action:** We intend to design and evaluate a machine learning model that infers full 3D velocity vectors from a single FMCW LiDAR frame by leveraging geometric context, object orientation cues, and learned semantic priors. 

**Evaluation:** We will evaluate the model using standard velocity estimation metrics (endpoint error, angular error) on public autonomous driving datasets with ground-truth velocity annotations, with specific focus on objects exhibiting high tangential-to-radial velocity ratios.

## 4 Steps/Milestones/Actions
1. **Literature review and theoretical foundation** – Conduct a focused literature survey on FMCW LiDAR velocity measurement, scene flow estimation, Doppler-aided perception, and point cloud deep learning to establish the theoretical background and confirm the knowledge gap.
2. **Dataset preparation and preprocessing pipeline** – Use the AevaScenes dataset, a real-world FMCW LiDAR dataset with per-point Doppler velocity annotations, to which we have confirmed access. Implement data loading, coordinate transformations, object-level ground-truth velocity extraction, and train/validation/test splits stratified by scenario type.
3. **Feature engineering and analysis** – Design and extract contextual feature representations including geometric neighbourhood structure, object-level orientation cues, semantic class labels, and beam geometry vectors. Analyze their statistical relationship with tangential velocity.
4. **Model architecture design and implementation** – Design a neural network architecture that takes a single FMCW LiDAR point cloud frame as input and predicts the object-level full 3D velocity vector. Implement training and inference pipelines.
5. **Training, hyperparameter tuning, and ablation study** – Train the model on the prepared dataset. Conduct ablation experiments to determine which contextual features (RQ2) contribute most to tangential velocity inference accuracy.
6. **Baseline comparison and evaluation** – Implement or adapt multi-frame scene flow baselines for comparison (RQ3). Evaluate all methods using endpoint error, angular error, and per-category metrics, focusing on high-tangential-motion objects (e.g., crossing vehicles, perpendicular pedestrians).
7. **Analysis, writing, and thesis completion** – Analyze results with respect to all three research questions, discuss findings in the context of related work, and complete the thesis report.

## 5 Risks
1. **Limited availability of FMCW LiDAR datasets with ground-truth velocity annotations.** Public datasets with per-point Doppler velocity are still relatively scarce compared to conventional LiDAR datasets. *Mitigation:* We have identified recent benchmarks that include Aeva sensor data with Doppler annotations. If needed, we can simulate FMCW measurements by projecting ground-truth velocity onto beam directions from existing datasets (e.g., nuScenes with annotated bounding box velocities).
2. **Tangential velocity may be fundamentally unrecoverable from single-frame data alone.** It is possible that the geometric and semantic context in a single frame does not carry enough information to infer tangential motion reliably for all scenarios. *Mitigation:* We treat this as a research finding rather than a failure. If single-frame inference proves limited, we will characterize under which conditions it succeeds and fails, which is itself a valuable contribution (RQ1).
3. **Computational complexity of point cloud models.** Training deep learning models on large-scale 3D point clouds can be computationally expensive. *Mitigation:* We will use efficient point cloud processing architectures and, if necessary, subsample or voxelize the point clouds. LNU computing resources and potential access to Einride’s infrastructure will support this.
4. **Difficulty in fair comparison with multi-frame baselines.** Multi-frame methods have access to strictly more information than single-frame approaches, making direct comparison inherently asymmetric. *Mitigation:* We will clearly frame this comparison (RQ3) as measuring the gap rather than claiming superiority, and focus on scenarios where single-frame methods have a structural advantage (first-frame detection, newly appearing objects).

## 6 Background and Motivations
Autonomous driving stacks rely on 3D LiDAR for perception, yet conventional Time-of-Flight systems capture only geometry, providing no direct measurement of motion [1]. Frequency-Modulated Continuous-Wave (FMCW) LiDAR resolves this by encoding per-point radial velocity via the Doppler effect [2]: for a target with true velocity **v** along beam direction **r̂**, the measured component is *v_r* = **v** · **r̂**. This is, however, a fundamental physical constraint, tangential components perpendicular to **r̂** produce no frequency shift and are completely invisible to the sensor [2]. A vehicle crossing an intersection perpendicularly thus registers as stationary, a safety-critical failure for collision avoidance.

This thesis sits within the CS research area of **machine learning for autonomous perception**, targeting researchers in autonomous driving, LiDAR signal processing, and 3D computer vision. It is motivated scientifically by the open question of whether learned contextual priors can compensate for physically unmeasurable quantities, and societally by the need to detect tangential motion for collision avoidance.

## 7 Related Work
*Scene flow* methods, FlowNet3D [3], PointPWC-Net [4], and FLOT [5], demonstrate that 3D motion is learnable from point clouds, but all require at least two temporally separated frames and ignore per-point Doppler measurements. *Object-level velocity estimation* (CenterPoint [6], MotionNet [7]) estimates velocity at the bounding-box level but similarly depends on multi-frame association. *Doppler-aided perception* (Najibi et al. [8]) incorporates radial velocity as an auxiliary detection feature but does not attempt to recover missing tangential components; this direction is active, with publications at CVPR and ECCV through 2022–2024.

| Method | Doppler Input | Single-Frame | Full Velocity | Granularity |
| :--- | :---: | :---: | :---: | :---: |
| FlowNet3D [3] | No | No | Yes (3D) | Point-level |
| PointPWC-Net [4] | No | No | Yes (3D) | Point-level |
| FLOT [5] | No | No | Yes (3D) | Point-level |
| CenterPoint [6] | No | No | Partial (2D) | Object-level |
| MotionNet [7] | No | No | Partial (2D) | Object-level |
| Najibi et al. [8] | Yes | No | No | Object-level |
| Multi-receiver radar [9] | Yes | Yes | Yes (3D) | Point-level |
| **Ours** | **Yes** | **Yes** | **Yes (3D)** | **Object-level** |

**Table 7.1:** Comparison of closest related approaches. "Full Velocity" denotes whether the method recovers the complete velocity vector: Yes (3D) means full three-dimensional recovery, Partial (2D) means recovery in the bird's-eye-view plane only. Multi-receiver radar achieves single-frame full velocity recovery but requires specialised hardware configurations unavailable in single-sensor LiDAR setups.

Our work is positioned differently from all of the above: we investigate whether a single FMCW frame, with its radial velocities and spatial context, contains sufficient information to infer the full 3D velocity vector without any temporal input or additional hardware.

## 8 Knowledge Gap/Challenge/Problem
Radial-to-full velocity inference from a single Doppler-annotated frame remains unexplored: scene flow methods require multiple frames and ignore Doppler; object-level estimators rely on multi-frame association; Doppler-aided approaches use radial velocity as a detection feature rather than a reconstruction target. Classical radar techniques triangulate velocity using spatially separated receivers [9], but require hardware unavailable in single-sensor LiDAR setups.

From a CS perspective, this gap asks whether learned geometric, semantic, and beam-geometry representations can compensate for a fundamental physical measurement limitation, advancing knowledge of what is implicitly encoded in a single LiDAR frame.

## 9 Knowledge Contribution/Action
Following a **design science** methodology, we will design and implement a neural network that takes a single FMCW LiDAR frame (per-point coordinates, radial velocity, intensity) and predicts the object-level full 3D velocity vector. The contribution is organized around three research questions:

1. **RQ1:** To what extent can a machine learning model infer the full 3D velocity vector **v** = [*v_x*, *v_y*, *v_z*]^T  from a single FMCW LiDAR frame containing per-point radial velocity, spatial coordinates, and intensity?
2. **RQ2:** Which contextual features (geometric neighbourhood structure, object-level orientation, semantic class, or beam geometry) contribute most significantly to accurate tangential velocity inference?
3. **RQ3:** How does the accuracy of single-frame context-aware velocity inference compare to multi-frame scene flow baselines, particularly for objects with predominantly tangential motion (e.g., crossing traffic)?

## 10 Empirical Evidence/Evaluation
We follow an **experimental methodology** [10], training and evaluating on FMCW LiDAR data with ground-truth velocity annotations. Primary metrics are:
* **Endpoint Error (EPE):** Euclidean distance between predicted and ground-truth velocity vectors.
* **Angular Error:** Angle between predicted and ground-truth directions, independent of speed.
* **Tangential Error:** Error on velocity components perpendicular to the beam, directly measuring performance on the unmeasured quantity.

RQ2 is addressed through ablation experiments over contextual feature subsets; RQ3 through comparison with multi-frame scene flow baselines stratified by tangential-to-radial velocity ratio. **Reliability** is ensured via train/validation/test splits and multiple random-seed runs with reported mean and standard deviation. **Validity** is supported through standard public benchmarks; the threat that results may not generalise beyond the evaluation sensor configuration will be discussed explicitly. **Ethical considerations** are minimal: the work uses only publicly available datasets and involves no human subjects.

## References
[1] J. Yin, J. Shen, C. Guan, D. Zhou, and R. Yang, "LiDAR-Based Online 3D Video Object Detection with Graph-Based Message Passing and Spatiotemporal Transformer Attention," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 11495–11504, 2020. https://arxiv.org/abs/2004.01389

[2] S. Royo and M. Ballesta-Garcia, "An Overview of Lidar Imaging Systems for Autonomous Vehicles," *Applied Sciences*, vol. 9, no. 19, p. 4093, 2019. https://doi.org/10.3390/app9194093

[3] X. Liu, C. R. Qi, and L. J. Guibas, "FlowNet3D: Learning Scene Flow in 3D Point Clouds," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 529–537, 2019. https://arxiv.org/abs/1806.01411

[4] W. Wu, Z. Y. Wang, Z. Li, W. Liu, and F. Li, "PointPWC-Net: Cost Volume on Point Clouds for (Self-)Supervised Scene Flow Estimation," in *Proc. European Conf. Computer Vision (ECCV)*, pp. 88–107, 2020. https://arxiv.org/abs/1911.12408

[5] G. Puy, A. Boulch, and R. Marlet, "FLOT: Scene Flow on Point Clouds Guided by Optimal Transport," in *Proc. European Conf. Computer Vision (ECCV)*, pp. 527–544, 2020. https://arxiv.org/abs/2007.11142

[6] T. Yin, X. Zhou, and P. Krähenbühl, "Center-Based 3D Object Detection and Tracking," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 11784–11793, 2021. https://arxiv.org/abs/2006.11275

[7] P. Wu, S. Chen, and D. N. Metaxas, "MotionNet: Joint Perception and Motion Prediction for Autonomous Driving Based on Bird’s Eye View Maps," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, pp. 11385–11395, 2020. https://arxiv.org/abs/2003.06754

[8] M. Najibi, J. Ji, Y. Zhou, C. R. Qi, X. Yan, S. Ettinger, and D. Anguelov, "Motion Inspired Unsupervised Perception and Prediction in Autonomous Driving," in *Proc. European Conf. Computer Vision (ECCV)*, pp. 424–443, 2022. https://arxiv.org/abs/2210.08061

[9] S. Patole, M. Torlak, D. Wang, and M. Ali, "Automotive Radars: A Review of Signal Processing Techniques," *IEEE Signal Processing Magazine*, vol. 34, no. 2, pp. 22–35, 2017. https://doi.org/10.1109/MSP.2016.2628914

[10] M. Shaw, "What makes good research in software engineering?" *International Journal on Software Tools for Technology Transfer*, vol. 4, no. 1, pp. 1–7, 2002. http://www.cs.cmu.edu/~Compose/ftp/shaw-fin-etaps.pdf
