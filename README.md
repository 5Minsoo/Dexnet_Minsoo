# GQ-CNN Based Grasp Planning for Metallic Objects Using Sim-to-Real Depth Synthesis with SAPIEN

Depth-Noise-Robust 5-DOF Grasp Planning Using GQ-CNN for Metallic Raw Materials with Low-Cost Depth Cameras

KITECH Autonomous Manufacturing Process Research Division

> 📄 Minsoo Oh, Jaehak Lee*, "*Depth-Noise-Robust 5-DOF Robotic Grasping of Metallic Objects Using Synthetic Depth Data*", Journal of the Korean Society of Manufacturing Technology Engineers (KSMTE), accepted, 2026.


## Overview

A 5-DOF grasp framework based on [GQ-CNN (Dex-Net 2.0)](https://arxiv.org/abs/1703.09312) for grasping metallic raw materials in manufacturing.

- **Problem**: 6-DOF grasping is computationally heavy; 4-DOF is lightweight but top-down only. Metallic surfaces cause severe depth noise on low-cost cameras (RealSense D435i), creating a large sim-to-real gap.
- **Tilt-based 5-DOF**: discrete tilts about the grasp axis extend 4-DOF to 5-DOF — wrench space invariance lets existing labels be reused without relabeling.
- **Noise-realistic synthetic data**: training data generated in [SAPIEN](https://github.com/haosulab/SAPIEN) with an active stereo sensor model + Landau et al. IR noise model, reproducing real metallic reflection noise.
<p align="center">
  <img width="45%" alt="RealSense depth image" src="https://github.com/user-attachments/assets/64070c2b-cfd6-4799-b388-552a3ad95f36" />
  <img width="45%" alt="Simulation comparison" src="https://github.com/user-attachments/assets/ceab5755-5424-44ba-bb08-7e05bbc83caf" />
</p>
<p align="center"><em>RealSense Depth Image (Left) / Simulation Depth Image (Right)</em></p>

## Method

- Use a 2D image-based lightweight model (GQ-CNN), with the robot tilting to capture and execute grasps at different approach angles
- Generate a synthetic dataset (~8M depth patch–label pairs from 750 GraspFactory meshes) in SAPIEN, reflecting real sensor noise on metallic surfaces via an active stereo sensor model and the Landau et al. IR noise model
- Sample antipodal grasp candidates g(x, y, z, θ, φ), augment tilt angles within ±90° based on wrench space invariance, and filter by approach angle (φ ≤ 30°) and collision constraints

<img width="12231" height="4782" alt="Image" src="https://github.com/user-attachments/assets/d069705f-5070-41d1-9fa8-499fb5fcdb42" />
<p align="center"><em>Dataset Generation</em></p>
<br>
<img width="11699" height="3664" alt="Image" src="https://github.com/user-attachments/assets/6cc2727d-42ea-4df0-9c5d-5879d8929e37" />
<p align="center"><em>Realtime Inference</em></p>


## Hardware

- **Robot**: Doosan M1013
- **Depth Camera**: Intel RealSense D435i (hand-eye configuration)
- **Inference Device**: RTX 3070 Laptop GPU (real-time inference)
- **Target Objects**: Metallic raw materials / pre-machining workpieces


## Results

Real-world experiments were conducted on 10 unseen objects — 6 metallic (CNC1–CNC4, Flange, Stamp) and 4 non-metallic (T-pipe, Nipper, Stripper, Housing) — none of which were included in the training dataset.

- **Overall grasp success rate: 90.4%** (470 / 520 trials)
- **Metallic: 89.2%** (264/296) vs. **Non-metallic: 92.0%** (206/224) — the network trained purely on noise-modeled synthetic data achieved nearly equal performance on highly reflective metallic objects
- **By tilt angle**: 89.6% (0°), 90.9% (15°), 91.7% (30°)
- **Wall-adjacent objects (30° condition)**: Flange 90%, T-pipe 95%, Housing 90% — these placements admit no collision-free vertical grasp, and were reliably grasped only through the proposed tilt-based 5-DOF extension
- **Inference time: ~0.3 s** per grasp attempt, enabling real-time operation

<div align="center">
  
| Tilt (°) | CNC1 | CNC2 | CNC3 | CNC4 | Flange | Stamp | T-pipe | Nipper | Stripper | Housing |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 87.0 | 95.7 | 87.0 | 87.0 | 82.6 | 87.0 | 95.7 | 91.3 | 95.7 | 87.0 |
| 15 | 87.0 | 91.3 | 91.3 | 82.6 | 95.7 | 95.7 | 91.3 | 95.7 | 95.7 | 82.6 |
| 30* | – | – | – | – | 90.0 | – | 95.0 | – | – | 90.0 |

<sub>*30° trials were conducted only for small objects placed against a bin wall.</sub>

</div>


## Future Work

The current model selects from discretized tilt angles explicitly. Future work will extend the framework with multi-task learning so that a single network can infer the optimal approach angle and collision probability.


## Acknowledgments

This work was supported by the Korea Institute of Industrial Technology (KITECH), Autonomous Manufacturing Process Research Division, and funded by the Ministry of Trade, Industry and Energy (MOTIE) Materials & Components Technology Development Program (No. 20026387).
