# New Algorithms for Context-Aware Tangential Velocity Inference (2024-2026 Scan)

This document catalogs algorithms that are NOT yet in the 48-paper literature collection but are directly relevant to the thesis "Context-Aware Tangential Velocity Inference for FMCW LiDAR." Each entry explains the algorithmic contribution, the mechanism of relevance to the thesis, and concrete ways the technique can be incorporated into the architecture, training scheme, or evaluation.

The thesis goal, restated for grounding: predict per-point full 3D velocity from a single FMCW LiDAR sweep (x, y, z, v_r) by learning contextual priors that supply the missing tangential component. The space of useful algorithms therefore divides into (1) backbones and pretraining, (2) generative / probabilistic regression heads, (3) rigidity and equivariance priors, (4) cross-modal supervision, and (5) implicit motion field representations.

---

## 1. Diffusion-based Scene Flow Heads

### 1.1 DifFlow3D (Liu et al., CVPR 2024; extended TPAMI 2025)
- arXiv: 2311.17456. Code: github.com/IRMVLab/DifFlow3D.
- Iterative diffusion refinement on top of a coarse correlation-based scene flow estimator. The diffusion step is conditioned on three flow-related features so that generation diversity is restrained while still providing per-point uncertainty.
- Achieves millimeter-level EPE on KITTI scene flow for the first time. Authors explicitly position the diffusion module as plug-and-play.
- Relevance: the regression-from-Doppler problem is fundamentally underdetermined in the tangential direction, so the posterior over tangential velocity is multi-modal. A diffusion head conditioned on (geometric features, beam direction, radial velocity) is a natural way to capture that multi-modality and emit calibrated uncertainty per point. This directly addresses RQ1's identifiability concern from Need-for-Speed and Shifrin.

### 1.2 DiffSF (Zhang et al., NeurIPS 2024 Spotlight)
- arXiv: 2403.05327. Code: github.com/ZhangYushan3/DiffSF.
- Combines a transformer scene-flow backbone with a denoising diffusion process whose forward chain perturbs the ground-truth flow vectors and whose reverse chain is conditioned on the source/target clouds.
- Estimates uncertainty via the inherent randomness of the reverse process; reports SOTA EPE on FlyingThings3D and KITTI.
- Relevance: provides a reference for designing a single-frame variant where the conditioning is replaced by (point features, beam direction, measured radial velocity) and the supervision is the residual tangential velocity. This is a stronger probabilistic baseline than deterministic regression.

---

## 2. State Space Model (Mamba) Backbones

### 2.1 MambaFlow (Ma et al., 2025)
- arXiv: 2502.16907.
- Replaces the attention-based decoder in scene flow networks with a flow-guided Mamba decoder. Linear complexity in the number of points and SOTA on Argoverse 2 with real-time inference.
- Relevance: the thesis must scan dense outdoor sweeps (>100k points). Quadratic-attention transformers are expensive; Mamba decoders preserve global context cheaply, which is critical when contextual priors must reach across the scene to disambiguate tangential velocity (e.g. matching a rigid object's points against the road plane).

### 2.2 FlowMamba (2412.17366)
- Iterative Unit based on SSMs (ISU) that propagates global motion information first and then aggregates local cues. Designed for scene flow.
- Relevance: gives a recipe for "global-first, local-second" message passing that fits the rigidity prior used in RQ2 (object-level coherence first, per-point refinement second).

### 2.3 PointMamba (NeurIPS 2024) and Voxel Mamba (NeurIPS 2024)
- PointMamba serializes per-point features for SSM consumption. Voxel Mamba serializes the entire voxel grid into a single sequence with linear complexity, alleviating proximity loss.
- Relevance: candidate backbones for ablation against MinkowskiNet and PTv3. Important because the thesis's RQ2 explicitly studies which contextual feature classes contribute most to tangential recovery; an SSM backbone changes which long-range cues are reachable.

### 2.4 Mamba3D (ACMMM 2024)
- Adds a local norm pooling block to compensate for the SSM's weakness on local geometry.
- Relevance: relevant if the ablation finds that local geometric structure (KPConv-style) matters more than long-range context for the tangential prediction.

---

## 3. SE(3)-Equivariant and Rigidity-Aware Methods

### 3.1 Multi-body SE(3) Equivariance for Unsupervised Rigid Segmentation and Motion Estimation
- ICLR 2024 (OpenReview ID 9lygTqLdWn).
- Decomposes a scene flow field into per-rigid-body SE(3) transformations using point-level invariant features for segmentation masks and SE(3)-equivariant features for motion. Joint optimization couples segmentation and motion.
- Relevance: the strongest principled framework for the rigidity prior already informally used by VoteFlow. It allows the network to commit to a small set of rigid motions per object cluster, which collapses the tangential ambiguity to a 6-DoF parameter set per cluster instead of 3D per point. Highly attractive as a structural prior layer between the backbone and the per-point velocity head.

### 3.2 Equivariant Flow Matching for Point Cloud Assembly (Wang et al., 2025)
- arXiv: 2505.21539.
- Flow-matching solver respecting SE(3) equivariance for assembling a point cloud from pieces.
- Relevance: the underlying flow-matching machinery (continuous-time velocity field that transports noise to data) is closely related to predicting a per-point velocity field. Flow matching is also a strong alternative to diffusion that gives straight-line probability paths and thus faster inference, useful for real-time autonomous driving.

---

## 4. Cross-Modal and Doppler-Aware Methods Not Yet in the Collection

### 4.1 TARS: Traffic-Aware Radar Scene Flow Estimation (Wu et al., ICCV 2025)
- arXiv: 2503.10210.
- Constructs a Traffic Vector Field in feature space, jointly trains object detection and scene flow, and enforces traffic-level rigid motion. Reports +23 percent on a proprietary dataset and +15 percent on View-of-Delft.
- Relevance: SGE-Flow already addresses missing tangential velocity by inter-frame learning; TARS shows how to enforce rigid motion at the traffic level rather than just the local-rigidity level of VoteFlow. The TVF idea adapts cleanly to single-frame FMCW LiDAR, where the same rigid-body decomposition is what supplies the missing tangential component.

### 4.2 MoRAL: Motion-aware Multi-Frame 4D Radar and LiDAR Fusion (2025)
- arXiv: 2505.09422.
- Compensates dynamic-object misalignment between frames using a Moving-Object-Segmentation-based radar motion encoder.
- Relevance: although multi-frame, the MOS-based encoder is portable: applying it on the single FMCW frame and conditioning the velocity head on the resulting motion mask is a candidate auxiliary supervision signal that should improve tangential recovery for fast-moving objects.

### 4.3 RadarMOSEVE-style joint MOS + ego-velocity (already in collection but worth pairing with MoRAL)
- Pairing this with MoRAL's encoder design gives a clean ablation dimension: motion-mask conditioning vs. no motion mask.

---

## 5. Pretraining and Self-Supervised Representations

### 5.1 Sonata: Self-Supervised Learning of Reliable Point Representations (Wu et al., CVPR 2025 Highlight)
- arXiv: 2503.16429. Code: github.com/facebookresearch/sonata.
- Self-distillation pretraining of Point Transformer V3 over 140k scenes; tripled linear probing accuracy on ScanNet and nearly doubled performance with 1 percent of fine-tuning data.
- Relevance: the thesis is data-bound, especially if AevaScenes lacks dense full-3D velocity ground truth. Initializing the geometric backbone with Sonata weights is a low-risk way to substantially reduce the supervision needed for the tangential component to emerge during fine-tuning. Directly supports the RQ2 analysis because Sonata explicitly studies which features are useful at the linear-probing layer.

### 5.2 Concerto / Utonia (Pointcept successors of Sonata, 2025)
- Latest extensions of the Pointcept self-supervised line, listed on github.com/pointcept/pointcept.
- Relevance: candidates for backbone initialization. Worth tracking even if not used in the thesis itself.

---

## 6. Implicit Motion Fields and 4D Gaussian Representations

### 6.1 SplatFlow: Self-Supervised Dynamic Gaussian Splatting in Neural Motion Flow Field (Sun et al., CVPR 2025)
- arXiv: 2411.15482.
- Models temporal motion of LiDAR points and Gaussians as a continuous neural motion flow field, learned without 3D bounding boxes.
- Relevance: the implicit motion-flow field idea provides a regularizer for the per-point velocity head. Training a sibling INR head that reconstructs a smooth velocity field over the local neighborhood and matching it to the per-point predictions is a single-frame analogue of SplatFlow's NMFF.

### 6.2 4D Implicit Neural Representation for Dynamic LiDAR Mapping (Mersch et al., 2024)
- arXiv: 2405.03388.
- Models the 4D space-time occupancy of dynamic LiDAR scenes with a single INR.
- Relevance: an INR over (x, y, z, t) is a natural place to embed a velocity head, since the gradient of the field with respect to t is precisely the velocity. Even in single-frame inference, the INR can be trained on multi-frame data and queried with t=0 to recover a velocity field.

### 6.3 CoDa-4DGS (Song et al., ICCV 2025)
- Adds context and deformation awareness to 4DGS for autonomous driving.
- Relevance: provides design ideas for incorporating semantic context into a Gaussian-based scene representation.

---

## 7. Mapping to the Thesis Research Questions

| RQ | Algorithm | Why it is relevant |
|----|-----------|-------------------|
| RQ1 (single-frame feasibility) | DifFlow3D, DiffSF, Equivariant Flow Matching | Probabilistic heads quantify the inherent multi-modality of tangential prediction and emit calibrated uncertainty, directly addressing the identifiability bound from Need-for-Speed |
| RQ2 (which features matter) | Sonata, Multi-body SE(3) Equivariance, MambaFlow, PointMamba | Each isolates a distinct feature axis: pretrained geometric features (Sonata), object-level rigid grouping (multi-body SE(3)), long-range context (Mamba). Direct ablation candidates |
| RQ3 (vs. multi-frame baselines) | TARS, MoRAL, SplatFlow | New 2025 SOTA baselines that should be added to the comparison table. TARS in particular is the strongest current radar-side analogue and uses the same rigidity insight at a larger spatial scale |

---

## 8. Recommended Concrete Additions to the Architecture

Based on the scan, four additions stand out as low-risk and high-value:

1. Replace the deterministic per-point velocity regressor with a DifFlow3D-style diffusion head (or a flow-matching head following the Equivariant Flow Matching design). Gain: per-point uncertainty plus multi-modal posterior over tangential direction.
2. Insert a multi-body SE(3) equivariance layer between the backbone and the velocity head. Gain: enforces that points belonging to the same rigid body share a consistent tangential component, transferring the rigidity prior of VoteFlow up to traffic-scale rigidity as in TARS.
3. Initialize the backbone with Sonata weights. Gain: stronger geometric features without additional labeled data, particularly relevant if AevaScenes ground truth is sparse.
4. Add an INR sibling head trained on (x, y, z, t) over multi-frame data, following Mersch et al. and SplatFlow's NMFF. Gain: a self-consistency regularizer for the per-point velocity prediction at inference time t=0.

---

## 9. Recommended Additions to the Baseline Table

The thesis's RQ3 comparison table should add (with citations):

| Method | Type | Frames | Doppler | Notes |
|--------|------|--------|---------|-------|
| TARS (Wu et al., ICCV 2025) | Supervised | 2 | Radar Doppler | Strongest radar-side rigidity-prior baseline |
| MoRAL (2025) | Supervised | Multi-frame | Radar+LiDAR | Motion-aware fusion baseline |
| MambaFlow (2025) | Supervised | 2+ | No | New SOTA on Argoverse 2 with linear cost |
| DifFlow3D (CVPR 2024 / TPAMI 2025) | Supervised | 2 | No | Uncertainty-aware diffusion baseline |
| DiffSF (NeurIPS 2024) | Supervised | 2 | No | Probabilistic transformer baseline |

---

## 10. Source Index

Diffusion / flow-matching:
- DifFlow3D: https://arxiv.org/abs/2311.17456
- DiffSF: https://arxiv.org/abs/2403.05327
- Equivariant Flow Matching: https://arxiv.org/abs/2505.21539

State Space Models:
- MambaFlow: https://arxiv.org/abs/2502.16907
- FlowMamba: https://arxiv.org/abs/2412.17366
- PointMamba: https://arxiv.org/abs/2403.00762
- Voxel Mamba: https://openreview.net/forum?id=gHYhVSCtDH
- Mamba3D: https://arxiv.org/abs/2404.14966

Equivariance:
- Multi-body SE(3) equivariance: https://openreview.net/forum?id=9lygTqLdWn

Cross-modal radar/LiDAR:
- TARS: https://arxiv.org/abs/2503.10210
- MoRAL: https://arxiv.org/abs/2505.09422

Pretraining:
- Sonata: https://arxiv.org/abs/2503.16429
- Pointcept (Concerto / Utonia): https://github.com/pointcept/pointcept

Implicit / Gaussian motion fields:
- SplatFlow: https://arxiv.org/abs/2411.15482
- 4D INR for Dynamic LiDAR: https://arxiv.org/abs/2405.03388
- CoDa-4DGS: ICCV 2025 (https://github.com/Chenwei-Liang/CoDa-4DGS)
