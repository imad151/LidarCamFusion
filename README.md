#  Lidar-Camera Fusion for Screw Detection in Isaac Sim

This project implements a full simulation pipeline using **Isaac Sim 4.5** for **3D object scanning** and **screw detection** via **lidar-camera fusion** on a UR10 robot with a wrist-mounted sensor suite.

###  Objective

Detect screw locations on a 3D object by:

* Sweeping over the object with a robot arm
* Fusing RGB (camera) and depth (LiDAR) data
* Generating a colored point cloud
* Identifying screw positions in 3D space using YOLO + projection

---

##  Modules Breakdown

### 1. `world_init`

> Sets up the Isaac Sim scene with:

* UR10 robot arm
* Robotiq 2F-140 gripper
* Table + object environment

### 2. `rmpflow_controller`

> Uses Isaac's **RMPflow** to control the UR10 using:

* Joint-space PID control
* Smooth trajectory execution for scanning

### 3. `top_cam_processing`

> Overhead static camera:

* Detects object bounding boxes
* Helps localize the target area before scanning

### 4. `eef_cam_processing` & `lidar_scanner`

> Sensors mounted on the robot’s end-effector:

* Collect RGB and LiDAR data while sweeping
* Captures rich 3D surface and screw detail

### 5. `GeneratePointCloud`

> Fuses all the data to:

* Create a **colored point cloud**
* Run YOLO on RGB frames to detect screws
* Use LiDAR depth to project those detections into **3D screw coordinates**
* Save final point cloud and mesh

---

##  Pipeline Overview

```mermaid
flowchart LR
    A[World Init] --> B[RMPflow Movement]
    B --> C[Top Cam → Bounding Box]
    C --> D[EEF Camera + LiDAR Scanning]
    D --> E[Color + Depth Fusion]
    E --> F[YOLO → 2D Detections]
    F --> G[LiDAR Projection → 3D Screws]
    G --> H[Save Colored Pointcloud + Mesh]
```

---

##  Dependencies

* [Isaac Sim 4.5](https://developer.nvidia.com/isaac-sim)
* Python 3.10
* OpenCV
* NumPy
* YOLOv5 or YOLOv8 (for 2D screw detection)
* Open3D
---
