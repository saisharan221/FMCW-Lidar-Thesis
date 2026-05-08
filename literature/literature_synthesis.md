# Literature Synthesis

## Overview

This synthesis organizes 48 papers collected for the thesis "Context-Aware Tangential Velocity Inference for FMCW LiDAR" into thematic strands, traces how they connect to the three research questions, and identifies the knowledge gap the thesis addresses.

---

## 1. Point Cloud Deep Learning Foundations

**Papers:** PointNet [qi2017pointnet], PointNet++ [qi2017pointnetpp], MinkowskiNet [choy2019minkowski], SPVNAS [tang2020spvnas], Point Transformer V2 [wu2022pointtransformerv2], Point Transformer V3 [wu2024pointtransformerv3], VoxelNet [zhou2018voxelnet], PointPillars [lang2019pointpillars], KPConv [thomas2019kpconv]

These nine papers establish the architectural building blocks for processing 3D point clouds with neural networks. Three distinct paradigms have emerged:

- **Point-based methods** (PointNet, PointNet++, KPConv): Operate directly on raw points. PointNet introduced permutation-invariant processing via shared MLPs and max-pooling; PointNet++ added hierarchical local feature learning through set abstraction. KPConv placed learnable kernel points in Euclidean space, enabling deformable convolutions that adapt to local geometry. These methods preserve per-point resolution but scale poorly to large outdoor scenes.

- **Voxel/pillar-based methods** (VoxelNet, PointPillars, MinkowskiNet, SPVNAS): Discretize the point cloud into voxels or pillars for structured convolution. VoxelNet pioneered end-to-end voxel-based detection; PointPillars replaced 3D convolutions with 2D convolutions over vertical pillars, achieving 62 Hz inference. MinkowskiNet generalized sparse convolutions to high-dimensional sparse tensors, and SPVNAS combined sparse voxel and point-based branches with neural architecture search.

- **Transformer-based methods** (Point Transformer V2/V3): Apply attention mechanisms to point clouds. V3 replaced precise neighbor search with serialized attention, scaling receptive fields from 16 to 1024 points while achieving 3x speedup over V2.

**Relevance to RQ2 (feature contributions):** The choice of backbone directly determines which contextual features the model can capture. Point-based methods excel at fine-grained local geometry (relevant for inferring surface orientation and motion direction). Sparse convolution methods handle large-scale outdoor scans efficiently. Transformer architectures capture long-range context (object-level relationships). Our architecture will likely combine efficient sparse encoding (MinkowskiNet or pillar-based) with local geometric feature extraction (PointNet++ style set abstraction) to capture both object-level context and per-point geometric cues.

**Key insight:** No existing architecture was designed for velocity prediction from Doppler-annotated point clouds. All treat the point cloud as a static geometric input. Extending these architectures to incorporate per-point radial velocity as a fourth input dimension (x, y, z, v_r) is a necessary design contribution.

---

## 2. Scene Flow Estimation: The Multi-Frame Paradigm

**Papers:** NSFP [li2021nsfp], FastNSF [li2023fastnsfp], DELFlow [peng2023delflow], DeFlow [zhang2024deflow], Flow4D [cho2025flow4d], DeltaFlow [zhang2025deltaflow], Scene Flow Survey [li2023sceneflow_survey]

Scene flow methods estimate dense 3D motion fields between consecutive point cloud frames. The field has evolved through three generations:

1. **Optimization-based** (NSFP, FastNSF): Use neural networks as implicit priors optimized per-scene at test time. NSFP fits a coordinate-based network to minimize Chamfer distance between warped source and target clouds. FastNSF replaced the Chamfer loss with a distance-transform loss, achieving 30x speedup. These methods are robust to distribution shift but remain slower than feed-forward networks.

2. **Supervised feed-forward** (DELFlow, DeFlow, Flow4D, DeltaFlow): Train neural networks to predict flow in a single forward pass. DeFlow introduced GRU-based iterative refinement with point-voxel-point feature transitions. Flow4D fused temporal frames via 4D voxel networks, winning the 2024 Argoverse 2 Scene Flow Challenge with 45.9% improvement over prior art. DeltaFlow (NeurIPS 2025 Spotlight) introduced delta-based temporal encoding, achieving 22% lower error and 2x faster inference than Flow4D.

3. **Self-supervised** (covered in Section 3 below).

**Relevance to RQ3 (comparison with multi-frame baselines):** These methods represent the strongest baselines for full velocity estimation. Flow4D and DeltaFlow are the current state-of-the-art supervised methods. However, all require at least two temporally separated frames, creating a fundamental structural limitation: they cannot estimate velocity for newly appearing objects (first detection), objects visible for only one frame, or in scenarios where latency from multi-frame accumulation is unacceptable.

**Key insight:** Scene flow methods solve a related but distinct problem. They estimate displacement between frames (which yields velocity when divided by time), while our method directly predicts velocity from a single frame using Doppler measurements. The comparison in RQ3 should be framed as measuring the information cost of dropping temporal context, not as a direct competition.

---

## 3. Self-Supervised Scene Flow

**Papers:** Just Go with the Flow [mittal2020justgo], SeFlow [zhang2024seflow], VoteFlow [voteflow2025], TeFlow [zhang2026teflow], DoGFlow [khoche2026dogflow]

Self-supervised scene flow methods train without ground-truth annotations, using geometric consistency losses instead. The progression shows increasingly sophisticated self-supervision signals:

- **Mittal et al. (2020)** pioneered self-supervised scene flow using nearest-neighbor and cycle-consistency losses, matching supervised performance on nuScenes.
- **SeFlow (ECCV 2024)** classified points as static vs. dynamic and designed targeted losses for each category, improving handling of mixed-motion scenes.
- **VoteFlow (CVPR 2025)** enforced local rigidity through a differentiable voting mechanism, exploiting the prior that rigid objects have uniform velocity fields.
- **TeFlow (CVPR 2026)** used temporal ensembling to aggregate consistent motion cues across multiple frames, achieving performance on par with optimization-based methods while being 150x faster.
- **DoGFlow (RA-L 2026)** introduced cross-modal Doppler guidance, using 4D radar Doppler velocity to generate pseudo-labels for LiDAR scene flow. This achieved 90% of supervised performance with only 10% ground truth data.

**Relevance to all three RQs:**
- **RQ1:** DoGFlow demonstrates that Doppler velocity can effectively guide scene flow estimation, validating the information content of radial velocity measurements.
- **RQ2:** VoteFlow's rigidity prior connects to our work: rigid objects have uniform velocity, which is an exploitable geometric prior for tangential inference.
- **RQ3:** TeFlow is the strongest self-supervised multi-frame baseline. DoGFlow is the most relevant cross-modal baseline since it uses Doppler velocity as supervision.

**Key insight:** The trajectory from geometric self-supervision (Mittal 2020) to Doppler-guided self-supervision (DoGFlow 2026) shows the field moving toward incorporating velocity measurements. Our thesis extends this direction by making Doppler the primary input rather than a supervision signal, and by operating from a single frame.

---

## 4. Doppler-Aided Perception and Odometry

**Papers:** DICP [hexsel2022dicp], Picking Up Speed [wu2023pickingupspeed], Need for Speed [yoon2023needforspeed], Dynamic-ICP [dynamicicp2025], DopplerPTNet [dopplerptnet2024]

This strand represents work that directly uses FMCW LiDAR Doppler measurements. The evolution moves from ego-motion estimation toward scene understanding:

- **DICP (RSS 2022)** extended ICP with a Doppler velocity residual, enabling robust registration in feature-scarce environments. Demonstrated on real Aeva FMCW LiDAR data and contributed to the Open3D library.
- **Picking Up Speed (RA-L 2023)** introduced continuous-time Doppler LiDAR odometry using Gaussian process regression, handling motion distortion in scanning LiDAR.
- **Need for Speed (IROS 2023)** formulated 6-DOF ego-velocity estimation from Doppler as a linear problem, achieving 5.6ms per frame. Crucially, their **observability analysis proved that angular velocity is unobservable from a single FMCW LiDAR**, establishing a theoretical limit on what radial-only measurements can determine.
- **Dynamic-ICP (arXiv 2025)** extended DICP for dynamic scenes, clustering dynamic objects and reconstructing object-wise translational velocities from ego-compensated radial measurements using a constant-velocity model. Evaluated on AevaScenes.
- **DopplerPTNet (arXiv 2024)** extended PointRCNN for FMCW LiDAR 4D point clouds with Doppler-aware attention, treating (x, y, z, v) as 4D input for object detection.

**Relevance to RQ1 (feasibility of single-frame inference):**
- Need for Speed's observability analysis is foundational: it confirms that radial Doppler alone is geometrically insufficient for full velocity determination. This motivates using ML to supply the missing information through learned priors.
- Dynamic-ICP reconstructs object-level velocity from radial Doppler using a constant-velocity geometric model. This serves as a non-learning baseline; our ML approach should improve on this by learning scene-level priors.
- DopplerPTNet validates that Doppler velocity is a learnable feature, but uses it for detection rather than velocity recovery.

**Key insight:** All five papers use Doppler velocity from FMCW LiDAR, but none attempt per-point full velocity recovery via learning. The ego-motion papers (DICP, Picking Up Speed, Need for Speed) solve a well-posed linear problem. Dynamic-ICP extends to objects but uses rigid-body geometric constraints rather than learned representations. DopplerPTNet uses Doppler for detection. The gap between using Doppler for geometric estimation (these papers) and using Doppler for full velocity recovery (our thesis) remains open.

---

## 5. Velocity Estimation and Recovery

**Papers:** CaRLi-V [zhou2025carliv], POD [pod2025], Shifrin et al. [shifrin2025tangential], Khurana et al. [khurana2023pointcloud_forecast], Doppler Clustering [ding2022doppler_clustering]

This is the most directly relevant strand, containing papers that address radial-to-full velocity recovery or use velocity information for prediction:

- **Doppler Clustering (ITSC 2022)** proposed a single-scan algorithm for detecting and clustering moving points by Doppler similarity, then estimating object-level velocity from the cluster. This is the simplest classical baseline for our problem.
- **POD (arXiv 2025)** used single-frame FMCW LiDAR Doppler to generate virtual future point clouds via ray casting for "predictive object detection." Uses ego-velocity compensation from ground point clustering. Evaluated on a private dataset.
- **Shifrin et al. (EUSIPCO 2025)** performed a theoretical identifiability analysis for tangential velocity from radar using Cramer-Rao bounds. Showed that tangential velocity is identifiable under near-field conditions where target migrations in range/Doppler are resolvable. This classical signal processing analysis complements our ML approach.
- **CaRLi-V (arXiv 2025)** achieved point-wise 3D velocity estimation through camera-radar-LiDAR fusion, creating a radar "velocity cube" and combining radial velocity with optical flow and depth in a closed-form solution. Requires three sensors.
- **Khurana et al. (CVPR 2023)** formulated point cloud forecasting as 4D occupancy forecasting, implicitly capturing motion through self-supervised differentiable rendering. Captures velocity information without explicit velocity prediction.

**Relevance to RQ1:**
- Doppler Clustering and Dynamic-ICP demonstrate that radial Doppler alone carries useful velocity information even without ML.
- POD validates that single-frame Doppler is useful for understanding object motion, though it uses velocity for prediction rather than recovery.
- Shifrin's identifiability analysis establishes that tangential velocity can theoretically be recovered under certain geometric conditions, providing a signal-processing foundation for our ML approach.
- CaRLi-V achieves full velocity recovery but requires three sensors, highlighting what is possible and setting up the question of whether ML can achieve comparable results from a single sensor.

**Key insight:** No existing paper uses machine learning to predict full 3D velocity from a single FMCW LiDAR frame. Classical methods (Doppler Clustering, Dynamic-ICP) recover only radial-consistent velocity without tangential components. Multi-sensor approaches (CaRLi-V) require additional hardware. Prediction approaches (POD) use velocity but do not estimate it. Theoretical analysis (Shifrin) shows tangential is recoverable under specific conditions. Our thesis fills the intersection of all these directions: ML-based, single-sensor, single-frame, per-point full velocity recovery.

---

## 6. Radar Scene Flow and Cross-Modal Methods

**Papers:** CMFlow/Hidden Gems [ding2023cmflow], milliFlow [gao2024milliflow], SGE-Flow [sgeflow2026], RaLiFlow [raliflow2025], RadarPillars [radarpillars2024], RadarMOSEVE [li2024radarmoseve]

Radar-based methods are analogous to our FMCW LiDAR setting because 4D radar also measures per-point Doppler velocity with the same radial-only constraint:

- **CMFlow (CVPR 2023 Highlight)** pioneered cross-modal supervision for radar scene flow, using LiDAR flow as a teacher signal. This established that Doppler-annotated point clouds contain sufficient information for motion estimation when combined with appropriate supervision.
- **milliFlow (ECCV 2024)** tailored scene flow estimation for mmWave radar, incorporating Doppler as an input feature. Applied to human motion sensing rather than driving.
- **SGE-Flow (Sensors 2026)** introduced an inter-frame flow module using a Transformer to compensate for missing tangential velocity in radar data. This explicitly addresses the tangential velocity gap through learned temporal features.
- **RaLiFlow (AAAI 2025)** achieved 70.5% improvement over radar-only baselines by fusing radar Doppler with LiDAR geometry for scene flow, using dynamic-aware bidirectional cross-modal attention.
- **RadarPillars (ICRA 2024)** decomposed radial velocity into Cartesian (x, y) components for better feature alignment in pillar-based detection. This velocity decomposition technique is directly transferable.
- **RadarMOSEVE (AAAI 2024)** jointly performed moving object segmentation and ego-velocity estimation using a Doppler-aware transformer with a novel Doppler loss function.

**Relevance to RQ2 (feature contributions):**
- RadarPillars' velocity decomposition into Cartesian components is a feature engineering technique we should adopt.
- RadarMOSEVE's Doppler loss function and ego-velocity compensation are directly relevant.
- SGE-Flow's approach to compensating for tangential velocity absence validates the problem's importance.

**Key insight:** The radar literature has extensively explored Doppler velocity as an input feature, but these methods face challenges (extreme sparsity, lower angular resolution) that do not apply to FMCW LiDAR. FMCW LiDAR provides orders of magnitude denser point clouds with precise geometry plus Doppler, potentially making single-frame velocity inference more tractable. Techniques from radar (velocity decomposition, Doppler-aware attention, ego-velocity compensation) transfer to our setting.

---

## 7. Datasets

**Papers:** AevaScenes [aeva2025aevascenes], View-of-Delft [palffy2022vod], ZOD [alibeigi2023zod], Argoverse 2 [wilson2023argoverse2], Waymo [sun2020waymo], FlyingThings3D [mayer2016flyingthings3d], HeLiPR [jung2024helipr]

Dataset availability is a critical factor for this thesis:

| Dataset | Sensor | Doppler | Velocity GT | Scale | Suitability |
|---------|--------|---------|-------------|-------|-------------|
| AevaScenes | FMCW LiDAR | Per-point | Tracking-derived | Medium | **Primary candidate** |
| Argoverse 2 | ToF LiDAR | No (can simulate) | Scene flow labels | Large | Simulated Doppler training |
| Waymo | ToF LiDAR | No (can simulate) | Box-level velocity | Very large | Simulated Doppler training |
| HeLiPR | FMCW LiDAR | Per-point | Limited | Medium | Supplementary evaluation |
| View-of-Delft | ToF LiDAR + 4D Radar | Radar only | Box-level | Medium | Cross-modal comparison |
| ZOD | ToF LiDAR | No | Limited | Large | European driving scenarios |
| FlyingThings3D | Synthetic | No | Dense flow | Large | Pretraining |

**Dataset strategy:**
- **AevaScenes** is the primary candidate: it provides the exact data modality (FMCW LiDAR with per-point Doppler). Need to verify that full 3D velocity ground truth is available (not just radial).
- **Argoverse 2** can generate synthetic Doppler by projecting scene flow ground truth onto beam directions. Its large scale and established benchmarks make it valuable for training and comparison.
- **Waymo** provides object-level velocity annotations that can generate point-level pseudo-labels within bounding boxes.
- **HeLiPR** offers additional real FMCW LiDAR data for evaluation diversity.

**Key concern:** If AevaScenes does not provide full 3D velocity ground truth (only radial Doppler), we need to derive it from tracking annotations or fall back to the Argoverse 2 simulation strategy described in the SPP.

---

## 8. The Knowledge Gap

Mapping the literature against our three research questions reveals a clear gap:

### What exists:
- **Multi-frame scene flow** methods (Flow4D, DeltaFlow, TeFlow) achieve accurate full velocity estimation but require 2+ frames and ignore Doppler
- **Doppler-aided odometry** methods (DICP, Picking Up Speed, Need for Speed) use single-frame Doppler but estimate only ego-motion, not per-object velocity
- **Classical velocity recovery** (Doppler Clustering, Dynamic-ICP) extracts object-level velocity from single-frame Doppler but uses geometric constraints, not learned representations, and cannot recover tangential components
- **Doppler-aided detection** (POD, DopplerPTNet) uses Doppler as a feature for detection but does not attempt velocity recovery
- **Multi-sensor fusion** (CaRLi-V, RaLiFlow) achieves full velocity but requires multiple sensors
- **Theoretical analysis** (Shifrin) shows tangential recovery is possible under specific conditions but does not provide a learning-based solution
- **Cross-modal Doppler guidance** (DoGFlow) uses radar Doppler to supervise LiDAR scene flow but still requires two frames and two sensors

### What does not exist:
**A machine learning method that infers per-point full 3D velocity vectors (including tangential components) from a single FMCW LiDAR frame using learned contextual priors.**

This gap sits at the intersection of:
1. Point cloud deep learning (architectures exist, velocity prediction does not)
2. Scene flow estimation (velocity estimation exists, single-frame does not)
3. Doppler perception (FMCW usage exists, tangential recovery does not)
4. Velocity recovery (the problem is defined, a learning-based solution is not)

---

## 9. Connections to Research Questions

### RQ1: Can ML infer full 3D velocity from a single FMCW LiDAR frame?

Supporting evidence that this is feasible:
- DoGFlow shows Doppler velocity contains enough signal to guide scene flow to 90% of supervised accuracy
- Dynamic-ICP recovers object-level velocity from single-frame Doppler using simple geometric constraints
- POD demonstrates single-frame Doppler enables future state prediction
- Shifrin's identifiability analysis shows tangential recovery is theoretically possible

Challenges:
- Need for Speed proves that angular velocity is unobservable from single FMCW LiDAR (some information is fundamentally missing)
- The tangential component is orthogonal to the measurement; recovery requires exploiting contextual priors not present in the raw measurements

### RQ2: Which contextual features contribute most?

The literature suggests four categories of features, each supported by different papers:

1. **Geometric neighborhood structure** (PointNet++, KPConv): Local surface geometry encodes object shape and orientation, which constrains plausible motion directions
2. **Object-level orientation cues** (VoteFlow rigidity prior, Dynamic-ICP clustering): Rigid objects have uniform velocity fields; knowing an object's extent and orientation constrains its velocity
3. **Semantic class** (SeFlow static/dynamic classification): Object class strongly predicts motion patterns (vehicles move along roads, pedestrians on sidewalks)
4. **Beam geometry vectors** (Need for Speed observability analysis, RadarPillars velocity decomposition): The relationship between beam direction and velocity determines how much information Doppler captures; decomposing radial velocity into directional components aids learning

### RQ3: Single-frame vs. multi-frame comparison

The comparison framework should include:

| Method | Type | Frames | Doppler | Level |
|--------|------|--------|---------|-------|
| **Ours** | ML, single-frame | 1 | Yes | Per-point |
| Flow4D | Supervised | 2+ | No | Per-point |
| DeltaFlow | Supervised multi-frame | 2+ | No | Per-point |
| TeFlow | Self-supervised | 2+ | No | Per-point |
| FastNSF | Optimization | 2 | No | Per-point |
| Dynamic-ICP | Geometric | 1 | Yes | Object-level |
| Doppler Clustering | Classical | 1 | Yes | Object-level |

The comparison should focus on:
- Overall accuracy (EPE, angular error)
- Performance on high-tangential-motion objects (crossing traffic, perpendicular pedestrians)
- First-frame detection scenarios (where multi-frame methods have no prior frame)
- Computational cost and latency

---

## 10. Architecture Design Implications

The literature points toward a specific architecture direction:

1. **Input representation:** 4D point cloud (x, y, z, v_r) with beam direction vectors, following DopplerPTNet and RadarPillars
2. **Backbone:** Sparse convolution (MinkowskiNet) or pillar-based (PointPillars) for efficiency, combined with local feature extraction (PointNet++ set abstraction) for geometric context
3. **Velocity decomposition:** Decompose radial velocity into Cartesian components following RadarPillars
4. **Ego-velocity compensation:** Use ground point Doppler clustering following POD and Need for Speed
5. **Output:** Per-point 3D velocity vectors (v_x, v_y, v_z)
6. **Training supervision:** Full 3D velocity from AevaScenes tracking or synthetic Doppler from Argoverse 2 scene flow labels
7. **Loss function:** EPE loss with potential Doppler consistency term (predicted velocity projected onto beam direction should match measured radial velocity)

---

## 11. Summary Statistics

- **Total papers:** 48
- **Year range:** 2016--2026
- **Categories:** Point cloud DL (9), Scene flow (8), Self-supervised (6), Doppler perception (5), Velocity estimation (5), Datasets (7), Radar (5), Sensor fusion (3), FMCW LiDAR (1)
- **High relevance:** 35 papers
- **Medium relevance:** 13 papers
- **Top venues:** CVPR (12), ECCV (4), NeurIPS (4), ICCV (4), ICRA (3), RA-L (3), IROS (2), RSS (1), AAAI (2)
- **Most relevant research groups:** KTH-RPL (DeFlow, SeFlow, DeltaFlow, TeFlow, DoGFlow), Aeva/associated (DICP, Picking Up Speed, Need for Speed, AevaScenes), TU Delft (CMFlow, VoD, Doppler Clustering)
