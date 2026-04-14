# GQ-CNN Based Grasp Planning for Metallic Objects Using Sim-to-Real Depth Synthesis with SAPIEN (Work in Progress)

Discrete Tilt-based Grasp Planning Using GQ-CNN for CNC Grasping  with Low-Cost Depth Cameras

KITECH Autonomous Manufacturing Process Research Division | Korean Society for Precision Engineering (KSPE) (한국정밀공학회)

---

## Overview

This project proposes a grasp framework using [GQ-CNN (Dex-Net 2.0)](https://arxiv.org/abs/1703.09312) for automating raw material grasping in manufacturing processes.

Current research mainly focuses on either 6DOF grasp using Point Cloud or 4DOF grasp using 2D images. 6DOF methods are heavy in model size and require high computational resources. 4DOF methods are lightweight but limited to top-down approaches only, making it fundamentally impossible to grasp objects attached to bin walls or in configurations where vertical access is blocked.

I propose a multi-view hand-eye grasp that preserves the lightweight computation of 4DOF while introducing discrete tilt angles via a hand-eye camera configuration. This enables grasping from different angles when objects are blocked by walls or otherwise difficult to reach from above.

In real manufacturing environments, raw materials are metallic with high reflectivity, causing severe depth noise on low-cost cameras like the Intel RealSense D435i. The conventional inpainting-based inference used in GQ-CNN suffers from a large sim-to-real gap in these conditions. To address this, I generate training data exclusively using [SAPIEN](https://github.com/haosulab/SAPIEN)-based synthetic images that reflect real camera noise characteristics, minimizing the sim-to-real gap.

This also helps in situations where highly reflective or thin objects cannot be recognized from above, by enabling grasping from alternative angles.

<p align="center">
  <img width="45%" alt="RealSense depth image" src="https://github.com/user-attachments/assets/64070c2b-cfd6-4799-b388-552a3ad95f36" />
  <img width="45%" alt="Simulation comparison" src="https://github.com/user-attachments/assets/ceab5755-5424-44ba-bb08-7e05bbc83caf" />
</p>
<p align="center"><em>RealSense Depth Image (Left) / Simulation Depth Image (Right)</em></p>

---

## Problem Statement

- 6DOF Grasp: Heavy model, high computation cost
- 4DOF Grasp: Top-down only, lacks degrees of freedom to handle occlusion
- Raw Material Grasping : Metallic objects cause severe noise on low-cost depth cameras (RealSense)

---

## Method

- Use a 2D image-based lightweight model (GQ-CNN), with the robot tilting to capture depth images at different angles
- Generate synthetic training images in SAPIEN that reflect real camera noise on metallic surfaces
- Validate sim-to-real gap through testing on a real robot

<p align="center">
  <img width="400" alt="Force closure antipodal sampling 1" src="https://github.com/user-attachments/assets/6c339cc2-a00c-49e5-9e7e-e0e5041918a4" />
  <img width="400" alt="Force closure antipodal sampling 2" src="https://github.com/user-attachments/assets/d695dc9e-62ad-4df5-b33c-49dd85408056" />
</p>
<p align="center"><em>Force Closure & Antipodal Grasp Sampling</em></p>

<p align="center">
  <img width="400" alt="Model evaluation" src="https://github.com/user-attachments/assets/d2d42ed6-70ba-4581-ac04-04997a438ba4" />
</p>
<p align="center"><em>Model Evaluation</em></p>

---

## Contributions

1. SAPIEN-based synthetic data pipeline for metallic reflective depth simulation
2. Sim-to-real grasp planning for low-cost depth cameras on metallic objects 
3. Multi-view hand-eye grasp with discrete tilt for occlusion handling in bin environments

---

## Hardware

- Robot: Doosan M1013
- Depth Camera: Intel RealSense D435i (hand-eye configuration)
- Inference Device: Microsoft Surface Pro 9 (real-time inference)
- Target Objects: Metallic raw materials / pre-machining workpieces

---

## Current Status

Experiments are currently in progress. Since the number of object classes is small, per-class success rate weighting is being tested during model training.

I expect that this approach can grasp raw materials and pre-machining parts while avoiding obstacles such as bins, without occlusion constraints, while maintaining low-cost depth cameras and lightweight computation.

Quantitative results will be updated after experiments are completed.

---

## Acknowledgments

This work was supported by the Korea Institute of Industrial Technology (KITECH), Autonomous Manufacturing Process Research Division.


