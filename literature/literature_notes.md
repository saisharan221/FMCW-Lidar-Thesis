# Literature Notes

Structured notes for the FMCW LiDAR thesis literature review.

---

## [qi2017pointnet] PointNet (2017)

**Full title:** PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
**Authors:** Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas
**Venue:** CVPR 2017
**Category:** point_cloud_dl

**Summary:** Pioneering architecture that directly consumes raw point clouds without voxelization. Uses shared MLPs and a symmetric max-pooling function to achieve permutation invariance, providing a unified framework for classification, part segmentation, and semantic segmentation.

**Key contributions:**
- First deep network to operate directly on unordered point sets
- Theoretical analysis showing the network learns to approximate a general continuous set function
- Spatial transformer networks for input and feature alignment

**Relevance to our thesis:**
- Foundational architecture for any point cloud processing model we build
- Our velocity inference network will likely build on PointNet-style feature extraction for per-point predictions

**Limitations / gaps this paper leaves open:**
- Does not capture local geometric structure (addressed by PointNet++)
- No velocity or motion estimation capability

**Key references to follow up:**
- PointNet++ (Qi et al., 2017)

---

## [qi2017pointnetpp] PointNet++ (2017)

**Full title:** PointNet++: Deep Hierarchical Feature Learning on Point Sets in a Metric Space
**Authors:** Charles R. Qi, Li Yi, Hao Su, Leonidas J. Guibas
**Venue:** NeurIPS 2017
**Category:** point_cloud_dl

**Summary:** Extends PointNet with hierarchical feature learning through set abstraction layers that group points at increasing scales and apply PointNet within local regions. Handles non-uniform point density with multi-scale grouping.

**Key contributions:**
- Hierarchical point set feature learning with set abstraction layers
- Multi-scale grouping (MSG) and multi-resolution grouping (MRG) for density-robust features
- Significant improvements over PointNet on fine-grained tasks

**Relevance to our thesis:**
- Local geometric neighborhood features are critical for inferring tangential velocity from spatial context (RQ2)
- PointNet++ set abstraction can serve as our backbone or a baseline feature extractor

**Limitations / gaps this paper leaves open:**
- Relatively slow due to ball query grouping at each level
- No motion or velocity estimation

**Key references to follow up:**
- None new (already tracked)

---

## [choy2019minkowski] Minkowski ConvNets (2019)

**Full title:** 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks
**Authors:** Christopher Choy, JunYoung Gwak, Silvio Savarese
**Venue:** CVPR 2019
**Category:** point_cloud_dl

**Summary:** Introduces generalized sparse convolutions on high-dimensional sparse tensors, enabling efficient 3D and 4D convolutions on point clouds. The Minkowski Engine provides GPU-accelerated auto-differentiation for sparse tensors.

**Key contributions:**
- Generalized sparse convolutions that encompass all discrete convolutions
- Open-source MinkowskiEngine library for sparse tensor operations
- State-of-the-art on ScanNet and other 3D benchmarks

**Relevance to our thesis:**
- Sparse convolutions are a leading architecture choice for large-scale point cloud processing
- Could be used as backbone for our velocity prediction model, especially for efficiency on large outdoor scans

**Limitations / gaps this paper leaves open:**
- Voxelization loses sub-voxel precision
- No motion or velocity estimation in the original work

**Key references to follow up:**
- SPVNAS (Tang et al., 2020) for point-voxel hybrid approach

---

## [tang2020spvnas] SPVNAS (2020)

**Full title:** Searching Efficient 3D Architectures with Sparse Point-Voxel Convolution
**Authors:** Haotian Tang, Zhijian Liu, Shengyu Zhao, Yujun Lin, Ji Lin, Hanrui Wang, Song Han
**Venue:** ECCV 2020
**Category:** point_cloud_dl

**Summary:** Proposes Sparse Point-Voxel Convolution (SPVConv) that combines sparse voxel-based convolution with a high-resolution point-based branch to preserve fine details. Uses neural architecture search (NAS) to find optimal configurations.

**Key contributions:**
- SPVConv module combining sparse convolution and point-based features
- NAS for 3D architectures achieving 3x speedup over MinkowskiNet with higher accuracy
- First place on SemanticKITTI leaderboard at time of publication

**Relevance to our thesis:**
- Point-voxel hybrid approach is relevant if we need both efficiency and fine-grained per-point features
- Could inform our architecture design choices

**Limitations / gaps this paper leaves open:**
- Focused on segmentation, not motion estimation

**Key references to follow up:**
- None new

---

## [wu2022pointtransformerv2] Point Transformer V2 (2022)

**Full title:** Point Transformer V2: Grouped Vector Attention and Partition-based Pooling
**Authors:** Xiaoyang Wu, Yixing Lao, Li Jiang, Xihui Liu, Hengshuang Zhao
**Venue:** NeurIPS 2022
**Category:** point_cloud_dl

**Summary:** Improves upon the original Point Transformer with grouped vector attention and partition-based pooling, achieving stronger performance on indoor and outdoor 3D understanding benchmarks.

**Key contributions:**
- Grouped vector attention mechanism for more expressive feature learning
- Partition-based pooling replacing FPS+kNN for efficiency
- State-of-the-art on multiple 3D segmentation benchmarks

**Relevance to our thesis:**
- Transformer-based point cloud architectures are a potential backbone choice
- Attention mechanisms may help the model learn contextual relationships between points for velocity inference

**Limitations / gaps this paper leaves open:**
- Trade-off between accuracy and efficiency addressed in V3

**Key references to follow up:**
- Point Transformer V3 (Wu et al., 2024)

---

## [wu2024pointtransformerv3] Point Transformer V3 (2024)

**Full title:** Point Transformer V3: Simpler, Faster, Stronger
**Authors:** Xiaoyang Wu, Li Jiang, Peng-Shuai Wang, Zhijian Liu, Xihui Liu, Yu Qiao, Wanli Ouyang, Tong He, Hengshuang Zhao
**Venue:** CVPR 2024 (Oral)
**Category:** point_cloud_dl

**Summary:** Replaces precise neighbor search with serialized neighbor mapping, scaling the receptive field from 16 to 1024 points while achieving 3x speedup and 10x memory reduction over V2. Demonstrates that scale matters more than intricate attention design.

**Key contributions:**
- Serialized attention replacing KNN-based attention
- Massive efficiency gains enabling larger receptive fields
- Strong performance across indoor/outdoor segmentation and detection

**Relevance to our thesis:**
- State-of-the-art point cloud backbone; could serve as our feature extractor
- Large receptive fields are valuable for capturing object-level context needed for velocity inference

**Limitations / gaps this paper leaves open:**
- Not designed for motion estimation tasks

**Key references to follow up:**
- Sonata and Concerto (follow-up works from same group)

---

## [li2021nsfp] Neural Scene Flow Prior (2021)

**Full title:** Neural Scene Flow Prior
**Authors:** Xueqian Li, Jhony Kaesemodel Pontes, Simon Lucey
**Venue:** NeurIPS 2021 (Spotlight)
**Category:** scene_flow

**Summary:** Uses a coordinate-based neural network as an implicit regularizer for scene flow estimation at test time, without any training data. Optimizes per-scene by fitting a network to minimize Chamfer distance between warped source and target point clouds.

**Key contributions:**
- Training-free scene flow estimation using neural network as implicit prior
- Robust to out-of-distribution data since no learned data priors
- Strong performance on both synthetic and real-world data

**Relevance to our thesis:**
- Important baseline for scene flow estimation (multi-frame approach for RQ3 comparison)
- Demonstrates that neural priors can regularize motion estimation in 3D

**Limitations / gaps this paper leaves open:**
- Requires two frames (cannot do single-frame inference)
- Very slow (up to 100x slower than learning-based methods)

**Key references to follow up:**
- FastNSF (Li et al., 2023)

---

## [li2023fastnsfp] Fast Neural Scene Flow (2023)

**Full title:** Fast Neural Scene Flow
**Authors:** Xueqian Li, Jianqiao Zheng, Francesco Ferroni, Jhony Kaesemodel Pontes, Simon Lucey
**Venue:** ICCV 2023
**Category:** scene_flow

**Summary:** Accelerates NSFP by replacing Chamfer distance loss with a distance transform-based loss, achieving up to 30x speedup while maintaining accuracy. Makes runtime optimization competitive with learning-based methods.

**Key contributions:**
- Distance transform as efficient, correspondence-free loss function
- 30x speedup over NSFP, reaching real-time on 8k points
- Analysis showing architectural speedups have minimal effect (bottleneck is loss function)

**Relevance to our thesis:**
- More practical multi-frame baseline for RQ3 comparison
- Distance transform loss could be useful for our training objectives

**Limitations / gaps this paper leaves open:**
- Still requires two frames

**Key references to follow up:**
- None new

---

## [peng2023delflow] DELFlow (2023)

**Full title:** DELFlow: Dense Efficient Learning of Scene Flow for Large-Scale Point Clouds
**Authors:** Chensheng Peng, Guangming Wang, Xian Wan Lo, Xinrui Wu, Chenfeng Xu, Masayoshi Tomizuka, Wei Zhan, Hesheng Wang
**Venue:** ICCV 2023
**Category:** scene_flow

**Summary:** Proposes a dense and efficient scene flow estimation method for large-scale outdoor point clouds, addressing scalability challenges in autonomous driving scenarios.

**Key contributions:**
- Efficient scene flow computation for large-scale (100k+) point clouds
- Dense flow estimation rather than sparse sampling

**Relevance to our thesis:**
- Addresses scalability for real-world LiDAR point clouds
- Potential multi-frame baseline

**Limitations / gaps this paper leaves open:**
- Multi-frame requirement

**Key references to follow up:**
- None new

---

## [zhang2024deflow] DeFlow (2024)

**Full title:** DeFlow: Decoder of Scene Flow Network in Autonomous Driving
**Authors:** Qingwen Zhang, Yi Yang, Heng Fang, Ruoyu Geng, Patric Jensfelt
**Venue:** ICRA 2024
**Category:** scene_flow

**Summary:** Enhances scene flow estimation with an efficient point-voxel-point decoder using GRU-based iterative refinement. Transitions from voxel features back to point-level features for fine-grained flow prediction.

**Key contributions:**
- GRU-based iterative refinement for scene flow decoding
- Point-voxel-point feature transition architecture
- Strong results on Argoverse 2 and Waymo

**Relevance to our thesis:**
- Modern architecture for scene flow; informs our model design choices
- Serves as supervised multi-frame baseline for RQ3

**Limitations / gaps this paper leaves open:**
- Requires two frames, does not use Doppler velocity

**Key references to follow up:**
- SeFlow (Zhang et al., 2024) builds on this

---

## [zhang2024seflow] SeFlow (2024)

**Full title:** SeFlow: A Self-Supervised Scene Flow Method in Autonomous Driving
**Authors:** Qingwen Zhang, Yi Yang, Peizheng Li, Olov Andersson, Patric Jensfelt
**Venue:** ECCV 2024
**Category:** self_supervised

**Summary:** Self-supervised scene flow method that classifies static vs. dynamic points and designs targeted loss functions for each. Builds on DeFlow architecture with dynamic/static decomposition.

**Key contributions:**
- Dynamic/static point classification integrated into self-supervised scene flow
- Targeted objective functions for different motion patterns
- State-of-the-art self-supervised results on Argoverse 2 and Waymo

**Relevance to our thesis:**
- Self-supervised multi-frame baseline for RQ3
- Static/dynamic decomposition is relevant since our model also needs to distinguish moving objects

**Limitations / gaps this paper leaves open:**
- Requires multiple frames, no Doppler information used

**Key references to follow up:**
- VoteFlow (2025), TeFlow (2026)

---

## [cho2025flow4d] Flow4D (2025)

**Full title:** Flow4D: Leveraging 4D Voxel Network for LiDAR Scene Flow Estimation
**Authors:** Jaeyeul Cho et al.
**Venue:** IEEE Robotics and Automation Letters (RA-L) 2025
**Category:** scene_flow

**Summary:** Fuses multiple point cloud frames after 3D voxel encoding into a 4D voxel network for spatio-temporal feature extraction. Won 1st place in the 2024 Argoverse 2 Scene Flow Challenge with 45.9% improvement over prior art.

**Key contributions:**
- 4D voxel network for explicit spatio-temporal feature extraction
- Real-time scene flow estimation
- Winner of Argoverse 2 Scene Flow Challenge 2024

**Relevance to our thesis:**
- State-of-the-art supervised multi-frame baseline for RQ3 comparison
- 4D voxel approach could inform how we encode temporal + Doppler dimensions

**Limitations / gaps this paper leaves open:**
- Multi-frame input required

**Key references to follow up:**
- None new

---

## [ding2025voteflow] VoteFlow (2025)

**Full title:** VoteFlow: Enforcing Local Rigidity in Self-Supervised Scene Flow
**Authors:** Ding et al.
**Venue:** CVPR 2025
**Category:** self_supervised

**Summary:** Enforces local rigidity constraints in self-supervised scene flow through a differentiable voting mechanism. Operates on pillars and learns representative features for translation voting.

**Key contributions:**
- Discretized voting space for rigid body motion estimation
- Local rigidity constraint as self-supervised signal
- Pillar-based efficient processing

**Relevance to our thesis:**
- Rigidity assumptions connect to our work: rigid objects have uniform velocity, which is a prior we can exploit
- Latest self-supervised baseline for RQ3

**Limitations / gaps this paper leaves open:**
- Still multi-frame

**Key references to follow up:**
- TeFlow (2026)

---

## [li2023sceneflow_survey] Scene Flow Survey (2023)

**Full title:** Deep Learning for Scene Flow Estimation on Point Clouds: A Survey and Prospective Trends
**Authors:** Zheming Li et al.
**Venue:** Computer Graphics Forum 2023
**Category:** scene_flow

**Summary:** Comprehensive survey of deep learning methods for 3D scene flow estimation on point clouds, covering supervised, self-supervised, and runtime optimization approaches.

**Key contributions:**
- Taxonomy of scene flow methods (supervised, self-supervised, optimization-based)
- Benchmark comparison across methods and datasets
- Discussion of future directions

**Relevance to our thesis:**
- Essential background for our related work section on scene flow
- Helps position our single-frame approach against the multi-frame paradigm

**Limitations / gaps this paper leaves open:**
- Does not cover Doppler-based velocity estimation
- Published 2023, missing most recent methods (Flow4D, VoteFlow, etc.)

**Key references to follow up:**
- Individual methods cited within

---

## [palffy2022vod] View-of-Delft Dataset (2022)

**Full title:** Multi-Class Road User Detection with 3+1D Radar in the View-of-Delft Dataset
**Authors:** Andras Palffy, Ewoud Pool, Saber Babaians, Julian F.P. Kooij, Dariu M. Gavrila
**Venue:** IEEE Robotics and Automation Letters (RA-L) 2022
**Category:** datasets

**Summary:** Introduces the View-of-Delft (VoD) dataset with 8600+ synchronized frames of 64-layer LiDAR, stereo camera, and 3+1D radar (range, azimuth, elevation, Doppler) data. Contains 123,000+ 3D bounding box annotations.

**Key contributions:**
- First large-scale dataset combining high-end LiDAR and 3+1D radar with Doppler
- Multi-class detection benchmark (pedestrians, cyclists, cars)
- Analysis of radar vs. LiDAR detection performance

**Relevance to our thesis:**
- Contains Doppler velocity from radar, could be used for training/evaluation if radar-to-LiDAR analogies hold
- Benchmark for detection with velocity information

**Limitations / gaps this paper leaves open:**
- Doppler comes from radar, not FMCW LiDAR
- No per-point LiDAR velocity annotations

**Key references to follow up:**
- RadarPillars (2024) uses this dataset

---

## [alibeigi2023zod] Zenseact Open Dataset (2023)

**Full title:** Zenseact Open Dataset: A Large-Scale Multi-Modal Dataset for Autonomous Driving
**Authors:** Mina Alibeigi et al.
**Venue:** ICCV 2023
**Category:** datasets

**Summary:** Large multi-modal dataset collected across 14 European countries over 2 years with full sensor suite. Only AD dataset under CC BY-SA 4.0 license allowing commercial use.

**Key contributions:**
- Large-scale, diverse geographic coverage
- Permissive open license
- Full sensor suite including LiDAR

**Relevance to our thesis:**
- Potential training dataset if Doppler annotations are available or can be simulated
- European driving scenarios relevant to Einride's domain

**Limitations / gaps this paper leaves open:**
- Uses conventional ToF LiDAR, no native Doppler velocity

**Key references to follow up:**
- Check if ZOD has been extended with velocity annotations

---

## [aeva2025aevascenes] AevaScenes (2025)

**Full title:** AevaScenes: The First Open-Access FMCW 4D LiDAR and Camera Dataset for Autonomous Vehicle Research
**Authors:** Aeva Technologies
**Venue:** Dataset release 2025
**Category:** datasets

**Summary:** Industry-first open dataset with synchronized FMCW 4D LiDAR and camera data including per-point Doppler velocity, semantic segmentation, tracking, and lane line annotations. Covers up to 400m range.

**Key contributions:**
- First open FMCW LiDAR dataset with per-point velocity
- Ultra-long range annotations (up to 400m)
- Rich annotation types (detection, segmentation, lanes)

**Relevance to our thesis:**
- Primary candidate dataset for our experiments; provides the exact data modality we need (FMCW LiDAR with per-point Doppler velocity)
- Ground-truth velocity annotations for training and evaluation

**Limitations / gaps this paper leaves open:**
- Released late 2025, may have limited community benchmarks so far
- Need to verify availability of full 3D ground-truth velocity vectors (not just radial)

**Key references to follow up:**
- Check for papers already using AevaScenes

---

## [zhou2025carliv] CaRLi-V (2025)

**Full title:** CaRLi-V: Camera-RADAR-LiDAR Point-Wise 3D Velocity Estimation
**Authors:** Zhou et al.
**Venue:** arXiv 2025
**Category:** velocity_estimation

**Summary:** Multi-sensor fusion pipeline for point-wise 3D velocity estimation. Creates a "velocity cube" from raw radar measurements, combines with optical flow for tangential velocity and LiDAR for range, producing dense 3D velocity estimates via closed-form solution.

**Key contributions:**
- Novel radar "velocity cube" representation for dense radial velocity encoding
- Closed-form fusion of radial velocity, optical flow, and depth for full 3D velocity
- Open-source ROS2 package

**Relevance to our thesis:**
- Directly addresses full velocity recovery from partial (radial) measurements, our core problem
- Uses multi-sensor fusion approach vs. our single-sensor ML approach; good comparison point
- Closed-form solution provides an analytical baseline

**Limitations / gaps this paper leaves open:**
- Requires camera + radar + LiDAR (three sensors), not single-sensor
- Not ML-based; relies on classical geometric fusion

**Key references to follow up:**
- RaLiFlow (2025) for radar-LiDAR scene flow

---

## [chae2024lidar4dradar] LiDAR-4D Radar Fusion (2024)

**Full title:** Towards Robust 3D Object Detection with LiDAR and 4D Radar Fusion in Various Weather Conditions
**Authors:** Yeongha Chae et al.
**Venue:** CVPR 2024
**Category:** sensor_fusion

**Summary:** Proposes 3D-LRF module for fusing LiDAR and 4D radar based on their 3D spatial relationship, leveraging complementary strengths (LiDAR precision vs. radar weather robustness and Doppler velocity).

**Key contributions:**
- 3D spatial relationship-based fusion of LiDAR and 4D radar
- Weather-robust 3D object detection
- Demonstrates benefit of Doppler velocity from radar in detection

**Relevance to our thesis:**
- Shows value of Doppler velocity as a feature for 3D perception
- Fusion approach differs from our single-FMCW-sensor approach

**Limitations / gaps this paper leaves open:**
- Object-level detection, not per-point velocity estimation

**Key references to follow up:**
- L4DR (2024) for LiDAR-4DRadar fusion

---

## [gao2024milliflow] milliFlow (2024)

**Full title:** milliFlow: Scene Flow Estimation on mmWave Radar Point Cloud for Human Motion Sensing
**Authors:** Gao et al.
**Venue:** ECCV 2024
**Category:** radar

**Summary:** Scene flow estimation specifically designed for mmWave radar point clouds, applied to human motion sensing. Addresses unique challenges of radar data (sparsity, noise, Doppler velocity).

**Key contributions:**
- First scene flow method tailored for mmWave radar point clouds
- Leverages Doppler velocity as input feature for flow estimation
- Human motion sensing application

**Relevance to our thesis:**
- Uses Doppler velocity as input for scene flow, directly analogous to our FMCW LiDAR setting
- Methodological insights on incorporating radial velocity into flow networks

**Limitations / gaps this paper leaves open:**
- Focused on radar (much sparser than LiDAR)
- Human motion domain, not autonomous driving

**Key references to follow up:**
- RaLiFlow (2025) for radar-LiDAR scene flow fusion

---

## [zhou2022deepradar_survey] Deep Radar Perception Survey (2022)

**Full title:** Towards Deep Radar Perception for Autonomous Driving: Datasets, Methods, and Challenges
**Authors:** Yongjia Zhou et al.
**Venue:** Sensors 2022
**Category:** radar

**Summary:** Comprehensive survey of deep learning methods for radar perception in autonomous driving, covering detection, segmentation, tracking, and velocity estimation across various radar representations.

**Key contributions:**
- Taxonomy of radar perception methods and representations
- Coverage of velocity estimation approaches using Doppler
- Dataset overview including radar-specific benchmarks

**Relevance to our thesis:**
- Background on how Doppler velocity is used in radar perception
- Radar velocity disambiguation techniques may transfer to FMCW LiDAR

**Limitations / gaps this paper leaves open:**
- Radar-focused, not directly FMCW LiDAR
- Published 2022, missing recent developments

**Key references to follow up:**
- 4D mmWave Radar survey (2023)

---

## [hexsel2022dicp] DICP: Doppler ICP (2022)

**Full title:** DICP: Doppler Iterative Closest Point Algorithm
**Authors:** Bruno Hexsel, Heethesh Vhavle, Yi Chen
**Venue:** RSS 2022
**Category:** doppler_perception

**Summary:** Extends ICP with a Doppler velocity objective for FMCW LiDAR point cloud registration. Jointly optimizes geometric alignment and Doppler velocity compatibility, enabling robust registration in feature-denied environments (tunnels, hallways).

**Key contributions:**
- Novel Doppler velocity residual term for ICP
- Demonstrated on real FMCW LiDAR (Aeva) data
- Significant improvement in registration accuracy in feature-scarce scenes
- Contributed to Open3D library

**Relevance to our thesis:**
- Directly uses FMCW LiDAR Doppler measurements, same sensor modality as our work
- Shows how radial velocity can constrain geometric estimation -- analogous reasoning to our velocity inference
- From Aeva, relevant to our dataset choice

**Limitations / gaps this paper leaves open:**
- Focused on ego-motion estimation, not per-point velocity prediction for dynamic objects
- Does not attempt to recover tangential velocity

**Key references to follow up:**
- Wu et al. (2023) STEAM-DICP continuous-time extension

---

## [wu2023pickingupspeed] Picking Up Speed (2023)

**Full title:** Picking Up Speed: Continuous-Time Lidar-Only Odometry Using Doppler Velocity Measurements
**Authors:** Yuchen Wu, David J. Yoon, Keenan Burnett, Soeren Kammel, Yi Chen, Heethesh Vhavle, Timothy D. Barfoot
**Venue:** IEEE RA-L 2023
**Category:** doppler_perception

**Summary:** First continuous-time FMCW LiDAR odometry using Gaussian process regression and Doppler velocity. Compensates for motion distortion and leverages per-point radial velocity for trajectory estimation.

**Key contributions:**
- First continuous-time framework for Doppler LiDAR odometry
- Gaussian process regression for smooth trajectory estimation
- Handles motion distortion inherent in scanning LiDAR

**Relevance to our thesis:**
- Demonstrates successful use of per-point Doppler velocity from FMCW LiDAR for motion estimation
- Focuses on ego-motion; our work extends the idea to object-level velocity

**Limitations / gaps this paper leaves open:**
- Estimates ego-motion only, not dynamic object velocities
- Does not recover tangential components

**Key references to follow up:**
- Need for Speed (Yoon et al., 2023) for correspondence-free variant

---

## [yoon2023needforspeed] Need for Speed (2023)

**Full title:** Need for Speed: Fast Correspondence-Free Lidar-Inertial Odometry Using Doppler Velocity
**Authors:** David J. Yoon, Keenan Burnett, Jingxing Qian, Yuchen Wu, Timothy D. Barfoot
**Venue:** IROS 2023
**Category:** doppler_perception

**Summary:** Formulates 6-DOF velocity estimation from FMCW LiDAR as a linear problem using per-point Doppler measurements. Processes frames in ~5.6ms. Shows that angular velocity is unobservable from a single FMCW LiDAR and uses a gyroscope to resolve it.

**Key contributions:**
- Linear velocity estimation from Doppler measurements (no correspondence needed)
- Observability analysis: single FMCW LiDAR cannot observe angular velocity
- Real-time performance (5.64ms per frame)

**Relevance to our thesis:**
- Observability analysis is directly relevant: confirms that radial-only Doppler constrains but does not fully determine ego-motion
- Analogous reasoning applies to per-object velocity: radial velocity alone is insufficient for full 3D velocity
- Motivates our ML approach to recover what is geometrically unobservable

**Limitations / gaps this paper leaves open:**
- Ego-motion only, not per-object velocity
- Requires IMU for angular velocity

**Key references to follow up:**
- None new

---

## [ding2023cmflow] Hidden Gems / CMFlow (2023)

**Full title:** Hidden Gems: 4D Radar Scene Flow Learning Using Cross-Modal Supervision
**Authors:** Fangqiang Ding, Andras Palffy, Dariu M. Gavrila, Chris Xiaoxuan Lu
**Venue:** CVPR 2023 (Highlight)
**Category:** radar

**Summary:** First cross-modal supervision approach for 4D radar scene flow. Uses LiDAR scene flow as supervision signal for training radar scene flow networks, exploiting co-located sensor redundancy.

**Key contributions:**
- Cross-modal learning: LiDAR supervises radar scene flow
- Multi-task architecture with motion segmentation and ego-motion estimation
- State-of-the-art 4D radar scene flow results

**Relevance to our thesis:**
- Cross-modal supervision concept could apply to our setting (e.g., using multi-frame flow as supervision for single-frame Doppler-based prediction)
- Uses Doppler velocity from radar as input feature

**Limitations / gaps this paper leaves open:**
- Radar-specific, not FMCW LiDAR
- Requires LiDAR as supervision source

**Key references to follow up:**
- RaLiFlow (2025) extends this to full radar-LiDAR fusion

---

## [raliflow2025] RaLiFlow (2025)

**Full title:** RaLiFlow: Scene Flow Estimation with 4D Radar and LiDAR Point Clouds
**Authors:** Various
**Venue:** AAAI 2025
**Category:** sensor_fusion

**Summary:** First joint scene flow framework for 4D radar and LiDAR fusion. Proposes Dynamic-aware Bidirectional Cross-modal Fusion (DBCF) module and constructs a radar-LiDAR scene flow dataset. Achieves 70.5% improvement over radar-only baseline.

**Key contributions:**
- First radar-LiDAR scene flow fusion framework
- Novel DBCF module integrating dynamic radar cues into cross-attention
- Custom loss functions mitigating unreliable radar data

**Relevance to our thesis:**
- Shows value of combining Doppler velocity (radar) with geometric features (LiDAR) for motion estimation
- Analogous to our FMCW LiDAR setting where both are in one sensor

**Limitations / gaps this paper leaves open:**
- Requires two sensors (radar + LiDAR)
- Multi-frame approach

**Key references to follow up:**
- None new

---

## [zhang2025deltaflow] DeltaFlow (NeurIPS 2025)

**Full title:** DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
**Authors:** Qingwen Zhang et al.
**Venue:** NeurIPS 2025 (Spotlight)
**Category:** scene_flow

**Summary:** Lightweight scene flow framework capturing motion cues via a delta scheme with minimal overhead regardless of frame count. Achieves 22% lower error and 2x faster inference than prior multi-frame methods, with strong cross-domain generalization.

**Key contributions:**
- Delta-based temporal feature extraction (efficient multi-frame)
- Cross-domain generalization ability
- State-of-the-art multi-frame supervised scene flow

**Relevance to our thesis:**
- Strong multi-frame baseline for RQ3 comparison
- Efficient multi-frame processing could inform our approach if we extend to multi-frame

**Limitations / gaps this paper leaves open:**
- Multi-frame requirement
- No Doppler velocity usage

**Key references to follow up:**
- None new

---

## [zhang2026teflow] TeFlow (CVPR 2026)

**Full title:** TeFlow: Enabling Multi-frame Supervision for Self-Supervised Feed-forward Scene Flow Estimation
**Authors:** Qingwen Zhang, Chenhan Jiang, Xiaomeng Zhu, Yunqi Miao, Yushan Zhang, Olov Andersson, Patric Jensfelt
**Venue:** CVPR 2026
**Category:** self_supervised

**Summary:** Uses temporal ensembling to aggregate consistent motion cues across multiple frames as reliable self-supervision. Achieves 33% improvement on Argoverse 2 and nuScenes, on par with optimization-based methods but 150x faster.

**Key contributions:**
- Temporal ensembling strategy for robust self-supervised signals
- Handles occlusion-induced correspondence failures
- State-of-the-art self-supervised feed-forward scene flow

**Relevance to our thesis:**
- Most recent and strongest self-supervised baseline for RQ3
- Temporal consistency idea could inform how we validate single-frame predictions

**Limitations / gaps this paper leaves open:**
- Multi-frame input required
- No Doppler information used

**Key references to follow up:**
- None new

---

## [pod2025] POD: Predictive Object Detection (2025)

**Full title:** POD: Predictive Object Detection with Single-Frame FMCW LiDAR Point Cloud
**Authors:** Various
**Venue:** arXiv 2025
**Category:** velocity_estimation

**Summary:** Extends 3D object detection to predictive detection using single-frame FMCW LiDAR. Uses per-point Doppler velocity to generate virtual future point clouds via ray casting, then encodes current+virtual frames with a sparse 4D encoder. Compensates ego velocity using ground point clustering.

**Key contributions:**
- Novel "predictive object detection" task for single-frame FMCW LiDAR
- Ray casting mechanism to generate virtual future frames from Doppler velocity
- Ego-velocity compensation via ground point clustering
- Demonstrates that Doppler velocity improves detection accuracy as a feature

**Relevance to our thesis:**
- **EXTREMELY RELEVANT** -- directly addresses single-frame FMCW LiDAR with Doppler velocity
- Uses velocity for prediction rather than recovery, but the feature engineering and ego-velocity compensation methods transfer directly
- Validates that single-frame Doppler information is useful for understanding object motion
- Potential comparison method

**Limitations / gaps this paper leaves open:**
- Does not attempt full 3D velocity recovery (tangential components)
- Uses velocity for bounding box prediction, not per-point velocity estimation
- Evaluated on private in-house dataset only

**Key references to follow up:**
- Check their ego-velocity compensation approach for our pipeline

---

## [dynamicicp2025] Dynamic-ICP (2025)

**Full title:** Dynamic-ICP: Doppler-Aware Iterative Closest Point Registration for Dynamic Scenes
**Authors:** Various
**Venue:** arXiv 2025
**Category:** doppler_perception

**Summary:** Extends DICP for dynamic scenes by clustering dynamic objects and reconstructing object-wise translational velocities from ego-compensated radial measurements. Uses constant-velocity prediction for dynamic points and a rotation-only Doppler residual.

**Key contributions:**
- Object-wise translational velocity reconstruction from per-point Doppler
- Dynamic point clustering and constant-velocity prediction
- Evaluated on AevaScenes dataset (our target dataset)
- No external sensors needed

**Relevance to our thesis:**
- Reconstructs object-level velocities from radial Doppler -- closely related to our per-point velocity inference goal
- Uses constant-velocity model as geometric prior; our ML approach should improve on this
- Evaluated on AevaScenes, demonstrating the dataset's suitability

**Limitations / gaps this paper leaves open:**
- Assumes constant velocity (no learned priors)
- Object-level, not per-point velocity
- Does not recover tangential components via learning

**Key references to follow up:**
- HeRCULES and HeLiPR datasets

---

## [zhou2018voxelnet] VoxelNet (2018)

**Full title:** VoxelNet: End-to-End Learning for Point Cloud Based 3D Object Detection
**Authors:** Yin Zhou, Oncel Tuzel
**Venue:** CVPR 2018
**Category:** point_cloud_dl

**Summary:** Pioneering end-to-end 3D detection architecture that voxelizes point clouds, applies voxel feature encoding (VFE) layers, and connects to a 3D convolutional middle layer and RPN for detection.

**Key contributions:**
- First end-to-end 3D detection from raw point clouds
- Voxel feature encoding (VFE) layer
- 3D convolution + 2D RPN detection pipeline

**Relevance to our thesis:**
- Foundational voxel-based architecture for 3D detection
- Voxelization approach relevant to our model design choices

**Limitations / gaps this paper leaves open:**
- Computationally expensive 3D convolutions
- No velocity estimation

**Key references to follow up:**
- PointPillars (Lang et al., 2019) for faster variant

---

## [lang2019pointpillars] PointPillars (2019)

**Full title:** PointPillars: Fast Encoders for Object Detection from Point Clouds
**Authors:** Alex H. Lang, Sourabh Vora, Holger Caesar, Lubing Zhou, Jiong Yang, Oscar Beijbom
**Venue:** CVPR 2019
**Category:** point_cloud_dl

**Summary:** Uses PointNet to encode point clouds organized in vertical pillars, converting to 2D pseudo-images for efficient 2D convolution. Achieves real-time 3D detection at 62 Hz.

**Key contributions:**
- Pillar-based point cloud encoding (no hand-tuned vertical binning)
- 2D convolution backbone (highly efficient)
- Real-time 3D object detection

**Relevance to our thesis:**
- Fast pillar-based encoding could be adapted for velocity prediction
- Demonstrates efficient point cloud processing for real-time applications

**Limitations / gaps this paper leaves open:**
- No velocity estimation
- Loses vertical resolution via pillar aggregation

**Key references to follow up:**
- PillarNet (ECCV 2022) for improved pillar-based methods

---

## [mittal2020justgo] Just Go with the Flow (2020)

**Full title:** Just Go with the Flow: Self-Supervised Scene Flow Estimation
**Authors:** Himangi Mittal, Brian Okorn, David Held
**Venue:** CVPR 2020
**Category:** self_supervised

**Summary:** Pioneering self-supervised scene flow method using nearest-neighbor and cycle-consistency losses. Matches supervised performance without annotations on nuScenes and KITTI.

**Key contributions:**
- First self-supervised scene flow method matching supervised accuracy
- Nearest-neighbor and cycle-consistency loss functions
- Enables training on large unlabeled datasets

**Relevance to our thesis:**
- Foundational self-supervised scene flow work
- Self-supervised losses could be adapted for our single-frame velocity prediction training

**Limitations / gaps this paper leaves open:**
- Requires two frames
- No Doppler velocity usage

**Key references to follow up:**
- SeFlow, VoteFlow, TeFlow (later self-supervised methods)

---

## [wilson2023argoverse2] Argoverse 2 (2023)

**Full title:** Argoverse 2: Next Generation Datasets for Self-Driving Perception and Forecasting
**Authors:** Benjamin Wilson et al.
**Venue:** NeurIPS 2023
**Category:** datasets

**Summary:** Large-scale autonomous driving dataset with 1000 annotated sensor sequences and 20,000 unlabeled LiDAR sequences. Includes 3D scene flow benchmark with piecewise-rigid flow labels.

**Key contributions:**
- 6 million LiDAR frames across 20,000 scenarios
- 3D scene flow benchmark with tracking-derived labels
- Multi-task benchmarks (detection, forecasting, scene flow)

**Relevance to our thesis:**
- Primary scene flow benchmark used by recent methods (DeFlow, SeFlow, Flow4D, etc.)
- Can simulate FMCW measurements by projecting ground-truth velocity onto beam directions
- Large unlabeled set enables self-supervised pretraining

**Limitations / gaps this paper leaves open:**
- ToF LiDAR (no native Doppler velocity)
- Need to synthesize Doppler measurements for our use case

**Key references to follow up:**
- AV2 Scene Flow Challenge results

---

## [sun2020waymo] Waymo Open Dataset (2020)

**Full title:** Scalability in Perception for Autonomous Driving: Waymo Open Dataset
**Authors:** Pei Sun et al.
**Venue:** CVPR 2020
**Category:** datasets

**Summary:** Large-scale autonomous driving dataset with high-resolution LiDAR and camera data, detailed 3D bounding box annotations including velocity vectors, and diverse driving scenarios.

**Key contributions:**
- Large-scale, high-quality annotations with velocity
- Diverse geographic and weather conditions
- Benchmark for detection, tracking, and motion prediction

**Relevance to our thesis:**
- Bounding box velocity annotations can serve as ground truth
- Can simulate per-point Doppler by projecting box velocities onto beam directions
- Widely used benchmark for comparison

**Limitations / gaps this paper leaves open:**
- ToF LiDAR, no native Doppler
- Velocity at object level, not per-point

**Key references to follow up:**
- WOMD-LiDAR (2024) for motion forecasting benchmark

---

## [shifrin2025tangential] Tangential Velocity Estimation (2025)

**Full title:** Tangential Velocity Estimation Using Near-Field Automotive Radar Model
**Authors:** Shifrin, Tabrikian, Bilik
**Venue:** EUSIPCO 2025
**Category:** velocity_estimation

**Summary:** Performs identifiability analysis for tangential velocity estimation from radar using Cramer-Rao bound and ambiguity function. Introduces a near-field radar model that exploits target migrations in range, radial velocity, and Doppler to estimate tangential velocity from a single sensor measurement.

**Key contributions:**
- Theoretical analysis: when is tangential velocity identifiable from radial-only measurements?
- Near-field model exploiting migration effects for tangential recovery
- Maximum likelihood algorithm for tangential velocity estimation

**Relevance to our thesis:**
- **DIRECTLY RELEVANT** -- addresses the same fundamental problem (recovering tangential from radial velocity)
- Uses classical signal processing (Cramer-Rao); our ML approach offers a complementary path
- Radar-based, but the physics of the identifiability analysis transfers to FMCW LiDAR

**Limitations / gaps this paper leaves open:**
- Radar-specific signal processing, not learning-based
- Single-point estimation, not scene-level
- Requires specific near-field conditions

**Key references to follow up:**
- Bilik's other publications on radar velocity

---

## [pod2025] POD: Predictive Object Detection (2025) -- already noted above

---

## [sgeflow2026] SGE-Flow (2026)

**Full title:** SGE-Flow: 4D mmWave Radar 3D Object Detection via Spatiotemporal Geometric Enhancement and Inter-Frame Flow
**Authors:** Various
**Venue:** Sensors 2026
**Category:** radar

**Summary:** Radar 3D detector with three modules: velocity displacement compensation (VDC), distribution-aware density (DAD), and inter-frame flow (IFF). The IFF module uses a Transformer to infer latent motion from pillar occupancy changes, compensating for missing tangential velocity.

**Key contributions:**
- Transformer-based inter-frame flow to compensate for absent tangential velocity
- Velocity displacement compensation aligns accumulated points
- Plug-and-play modules for existing detectors

**Relevance to our thesis:**
- IFF module explicitly addresses tangential velocity absence via learned temporal features
- Validates that tangential velocity compensation improves detection
- Our approach would do this from single-frame Doppler instead of multi-frame occupancy

**Limitations / gaps this paper leaves open:**
- Multi-frame approach (temporal occupancy changes)
- Radar-specific

**Key references to follow up:**
- None new

---

## [khurana2023pointcloud_forecast] Point Cloud Forecasting (2023)

**Full title:** Point Cloud Forecasting as a Proxy for 4D Occupancy Forecasting
**Authors:** Tarasha Khurana, Peiyun Hu, David Held, Deva Ramanan
**Venue:** CVPR 2023
**Category:** velocity_estimation

**Summary:** Formulates point cloud forecasting as 4D occupancy forecasting, implicitly capturing ego-motion, sensor intrinsics, and object shape/motion from unannotated LiDAR sequences. Uses differentiable rendering for self-supervised training.

**Key contributions:**
- Self-supervised 4D occupancy from unannotated LiDAR
- Implicit motion and velocity capture
- Differentiable rendering supervision

**Relevance to our thesis:**
- Self-supervised approach to motion learning from LiDAR could inform our training methodology
- Occupancy-based motion representation is an alternative to per-point velocity

**Limitations / gaps this paper leaves open:**
- No explicit velocity prediction
- Requires sequence of frames

**Key references to follow up:**
- None new

---

## [khoche2026dogflow] DoGFlow (2026)

**Full title:** DoGFlow: Self-Supervised LiDAR Scene Flow via Cross-Modal Doppler Guidance
**Authors:** Ajinkya Khoche, Qingwen Zhang, Yixi Cai, Sina Sharif Mansouri, Patric Jensfelt
**Venue:** IEEE RA-L 2026
**Category:** self_supervised

**Summary:** Self-supervised framework that uses 4D radar Doppler velocity to generate motion pseudo-labels for LiDAR scene flow, without manual annotations. Uses dynamic-aware association and ambiguity-resolved propagation to transfer radar Doppler cues to LiDAR domain.

**Key contributions:**
- Cross-modal Doppler guidance: radar Doppler supervises LiDAR scene flow
- Achieves 90% of supervised performance with only 10% ground truth data
- 18.4% improvement over FastNSF (best prior self-supervised method)
- Cluster-level velocity estimation from per-point Doppler via linear constraints

**Relevance to our thesis:**
- **HIGHLY RELEVANT** -- directly bridges Doppler velocity and scene flow estimation
- The cluster-level velocity reconstruction from radial Doppler is closely related to our problem
- Cross-modal supervision concept (Doppler -> full flow) is analogous to our single-sensor approach
- From KTH-RPL group (same as DeFlow, SeFlow, TeFlow) -- indicates active research direction

**Limitations / gaps this paper leaves open:**
- Uses radar Doppler (separate sensor), not FMCW LiDAR Doppler
- Still multi-frame (two LiDAR frames as input)
- Radar-to-LiDAR transfer has noise from sensor misalignment

**Key references to follow up:**
- MAN TruckScenes dataset

---

## [dopplerptnet2024] DopplerPTNet (2024)

**Full title:** DopplerPTNet: Object Detection Network with Doppler Velocity Information for FMCW LiDAR Point Cloud
**Authors:** Various
**Venue:** arXiv 2024
**Category:** doppler_perception

**Summary:** Extends PointRCNN for FMCW LiDAR 4D point clouds with a Doppler-aware attention-based feature extraction backbone. Treats spatial (x,y,z) and Doppler velocity (v) as a 4D input.

**Key contributions:**
- Attention-based feature extraction for combined spatial + Doppler features
- 4D Euclidean distance farthest point sampling
- Demonstrates that Doppler velocity information improves detection accuracy

**Relevance to our thesis:**
- Directly processes FMCW LiDAR with Doppler as 4D input -- same sensor modality
- Attention mechanism for Doppler features informs our architecture design
- Shows detection improvement from Doppler, suggesting velocity information is learnable

**Limitations / gaps this paper leaves open:**
- Object detection task, not velocity estimation
- Does not attempt full velocity recovery

**Key references to follow up:**
- None new

---

## [jung2024helipr] HeLiPR Dataset (2024)

**Full title:** HeLiPR: Heterogeneous LiDAR Dataset for inter-LiDAR Place Recognition under Spatiotemporal Variations
**Authors:** Minwoo Jung, Wooseong Yang, Dongjae Lee, Hyeonjae Gil, Giseop Kim, Ayoung Kim
**Venue:** International Journal of Robotics Research 2024
**Category:** datasets

**Summary:** First heterogeneous LiDAR dataset with multiple LiDAR types including FMCW LiDAR with Doppler velocity. Covers diverse environments from urban to highway over multiple sessions.

**Key contributions:**
- Multiple LiDAR types including FMCW with Doppler velocity
- Diverse environments and temporal variations
- Supports cross-LiDAR comparison research

**Relevance to our thesis:**
- Contains FMCW LiDAR data with Doppler velocity
- Could serve as additional evaluation dataset
- Enables comparison between FMCW and conventional LiDAR

**Limitations / gaps this paper leaves open:**
- Focused on place recognition, not velocity estimation
- Limited velocity annotations beyond raw Doppler

**Key references to follow up:**
- HeRCULES dataset for complementary radar+FMCW data

---

## [thomas2019kpconv] KPConv (2019)

**Full title:** KPConv: Flexible and Deformable Convolution for Point Clouds
**Authors:** Hugues Thomas, Charles R. Qi, Jean-Emmanuel Deschaud, Beatriz Marcotegui, Francois Goulette, Leonidas J. Guibas
**Venue:** ICCV 2019
**Category:** point_cloud_dl

**Summary:** Introduces kernel point convolution that operates directly on point clouds with convolution weights located in Euclidean space by learnable kernel points. Supports both rigid and deformable variants that adapt to local geometry.

**Key contributions:**
- Point convolution with learnable kernel point positions
- Deformable variant adapts to local geometry
- Strong outdoor and indoor segmentation results

**Relevance to our thesis:**
- Alternative point cloud backbone to PointNet++/sparse convolutions
- Deformable kernels may capture geometric structure useful for velocity inference
- Used as backbone in several scene flow methods

**Limitations / gaps this paper leaves open:**
- Slower than sparse convolution methods for large-scale outdoor scenes
- No motion estimation

**Key references to follow up:**
- None new

---

## [ding2022doppler_clustering] Doppler Velocity-Based Clustering (2022)

**Full title:** Doppler Velocity-Based Algorithm for Clustering and Velocity Estimation of Moving Objects
**Authors:** Fangqiang Ding et al.
**Venue:** ITSC 2022
**Category:** velocity_estimation

**Summary:** Proposes a single-scan, real-time algorithm for motion state detection and velocity estimation using FMCW LiDAR Doppler measurements. Clusters moving points based on Doppler velocity similarity and reconstructs object-level velocity.

**Key contributions:**
- Single-scan moving object detection using Doppler velocity
- Doppler-based clustering for dynamic objects
- Object-level velocity estimation from per-point radial Doppler

**Relevance to our thesis:**
- **DIRECTLY RELEVANT** -- single-scan velocity estimation from FMCW LiDAR Doppler
- Classical algorithmic baseline for our ML approach
- Demonstrates that meaningful velocity information can be extracted from single-frame Doppler

**Limitations / gaps this paper leaves open:**
- Classical clustering, not learning-based
- Cannot recover tangential velocity (only estimates velocity consistent with radial measurements)
- Object-level, not per-point

**Key references to follow up:**
- Dynamic-ICP extends this concept

---

## [gu2022fmcw_tracking] FMCW LiDAR Tracking (2022)

**Full title:** Learning Moving-Object Tracking with FMCW LiDAR
**Authors:** Yi Gu et al.
**Venue:** IROS 2022
**Category:** fmcw_lidar

**Summary:** Learning-based moving-object tracking using FMCW LiDAR with per-point Doppler velocity. Uses contrastive learning to improve instance-level tracking by pulling together features from the same instance in embedding space.

**Key contributions:**
- First learning-based method for FMCW LiDAR moving-object tracking
- Contrastive learning framework for instance feature learning
- Semi-automatic label generation using Doppler velocity

**Relevance to our thesis:**
- Demonstrates ML on FMCW LiDAR point clouds with Doppler
- Contrastive learning for instance features could inform our approach
- Shows Doppler velocity enables semi-automatic annotation

**Limitations / gaps this paper leaves open:**
- Tracking task, not velocity estimation
- Multi-frame approach

**Key references to follow up:**
- None new

---

## [radarpillars2024] RadarPillars (2024)

**Full title:** RadarPillars: Efficient Object Detection from 4D Radar Point Clouds
**Authors:** Various
**Venue:** ICRA 2024
**Category:** radar

**Summary:** Pillar-based 4D radar detection network that decomposes radial velocity into x/y components and introduces PillarAttention treating non-empty pillars as tokens. Achieves state-of-the-art radar-only detection at 1.99 GFLOPs.

**Key contributions:**
- Radial velocity decomposition into Cartesian components for better feature alignment
- PillarAttention: self-attention on non-empty pillars only
- State-of-the-art radar-only detection on View-of-Delft

**Relevance to our thesis:**
- Radial velocity decomposition is directly transferable: decomposing Doppler into directional components could benefit our model
- PillarAttention mechanism for sparse point cloud processing
- Uses VoD dataset with Doppler velocity

**Limitations / gaps this paper leaves open:**
- Detection only, no velocity estimation
- Radar-specific sparsity handling

**Key references to follow up:**
- None new

---

## [li2024radarmoseve] RadarMOSEVE (2024)

**Full title:** RadarMOSEVE: A Spatial-Temporal Transformer Network for Radar-Only Moving Object Segmentation and Ego-Velocity Estimation
**Authors:** Li et al.
**Venue:** AAAI 2024
**Category:** radar

**Summary:** Jointly performs moving object segmentation (MOS) and ego-velocity estimation (EVE) from radar point clouds using spatial-temporal transformer with Doppler velocity features. Uses novel Doppler loss and compensates radial velocity using ego-velocity.

**Key contributions:**
- Joint MOS + EVE from radar-only data
- Doppler-aware transformer architecture
- Novel Doppler loss function
- Ego-velocity compensation from Doppler measurements

**Relevance to our thesis:**
- Doppler-aware transformer architecture directly relevant to our model design
- Ego-velocity compensation methodology transfers to FMCW LiDAR
- Joint motion segmentation + velocity estimation is related to our per-point velocity task

**Limitations / gaps this paper leaves open:**
- Radar, not LiDAR
- Estimates ego-velocity, not per-object full velocity

**Key references to follow up:**
- None new
