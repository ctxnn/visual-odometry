# Stereo Visual Odometry and 3D Mapping

A 100% software-based Visual Odometry system using classical computer vision methods.

This project implements:
- **Monocular Visual Odometry** - Single camera motion estimation
- **Stereo Visual Odometry** - Dual camera depth estimation
- **3D Point Cloud Generation** - Dense environment mapping
- **Camera Trajectory Tracking** - Real-time path reconstruction

---

## Features

✅ Pure Classical Computer Vision (No ML/Deep Learning)
✅ Harris Corner + ORB Feature Detection
✅ KLT Optical Flow Tracking
✅ Essential Matrix + RANSAC Motion Estimation
✅ SGBM Stereo Depth Computation
✅ 3D Point Cloud Export (PLY Format)
✅ Trajectory Visualization

---

## Project Structure

```
project/
├── __init__.py              # Package initialization
├── main.py                  # Main entry point
├── dataset_loader.py        # Load images (monocular/stereo)
├── calibration.py           # Parse camera calibration
├── feature_detection.py    # Harris corners + ORB descriptors
├── stereo_depth.py         # SGBM disparity & depth computation
├── optical_flow.py         # KLT feature tracking
├── motion_estimation.py    # Essential matrix + pose recovery
├── trajectory.py           # Camera trajectory accumulation
├── monocular_vo.py        # Monocular VO implementation
├── visualization.py        # Plotting & export functions
├── MATHEMATICS.md          # Math behind visual odometry
├── CODE_WORKING.md         # Code explanation
├── APPLICATIONS.md         # Real-world applications
└── requirements.txt        # Dependencies
```

---

## Installation

1. **Install dependencies:**
```bash
cd project
pip install -r requirements.txt
```

2. **Prepare your dataset:**
   - **Monocular**: Folder with sequential images (PNG/JPG)
   - **Stereo**: KITTI Odometry format with `image_0/` and `image_1/` folders

---

## Quick Start

### Monocular Mode (Single Camera)
```bash
python main.py --mode mono --folder "../kitti dataset/train/img" --start 0 --end 100
```

### Stereo Mode (KITTI Odometry)
```bash
python main.py --mode stereo --sequence "../sequences/00" --start 0 --end 100
```

---

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--mode` | 'mono' or 'stereo' | mono |
| `--folder` | Image folder (mono mode) | kitti dataset/train/img |
| `--sequence` | Sequence folder (stereo mode) | sequences/00 |
| `--start` | Starting frame | 0 |
| `--end` | Ending frame | All |
| `--stride` | Frame step size | 1 |
| `--focal-length` | Focal length in pixels | Auto |
| `--scale` | Scale factor | 1.0 |
| `--num-disparities` | Disparity levels (stereo) | 128 |
| `--max-features` | Max features to track | 500 |
| `--output-dir` | Output directory | Input folder |
| `--no-visualize` | Skip plotting | False |

---

## Output Files

The system generates 4 output files:

| File | Description | Format |
|------|-------------|--------|
| `trajectory.txt` | Camera X, Y, Z positions | CSV |
| `motion_vectors.txt` | Rotation matrix + translation per frame | CSV |
| `trajectory_with_motion.csv` | Combined position & motion | CSV |
| `point_cloud.ply` | 3D point cloud | PLY |

### Sample Output

**trajectory.txt:**
```
x,y,z
0.000,0.000,0.000
0.038,0.048,0.998
-0.840,0.010,1.477
```

**motion_vectors.txt:**
```
frame,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz
0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0
1,0.99,-0.01,0.0,0.01,0.99,0.0,0.0,0.0,1.0,0.5,0.1,2.3
```

**point_cloud.ply** (Open in MeshLab/CloudCompare/Blender):
```
ply
format ascii 1.0
element vertex 15000
property float x
property float y
property float z
end_header
-7.903055 -11.098827 30.230267
...
```

---

## Methods Used

| Component | Method |
|-----------|--------|
| Feature Detection | Harris Corners + ORB |
| Feature Matching | Brute Force (Hamming) + Lowe's Ratio Test |
| Stereo Matching | Semi-Global Block Matching (SGBM) |
| Depth Estimation | Z = f × B / d |
| Motion Estimation | Essential Matrix + RANSAC |
| Pose Recovery | SVD Decomposition |
| Feature Tracking | KLT Optical Flow |

---

## Documentation

- **MATHEMATICS.md** - Detailed math formulas and derivations
- **CODE_WORKING.md** - Module-by-module code explanation
- **APPLICATIONS.md** - Real-world use cases and career opportunities

---

## Requirements

- Python 3.8+
- opencv-python >= 4.8.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- scipy >= 1.10.0

---

## Real-World Applications

- 🏎️ **Autonomous Vehicles** - Self-driving car navigation
- 🤖 **Robotics** - Robot localization and mapping
- 🥽 **AR/VR** - Head tracking for augmented reality
- 🚁 **Drones** - GPS-denied navigation
- 🏗️ **3D Reconstruction** - Structure from Motion

---

## Limitations

- **Monocular Scale**: No absolute scale (use stereo for real scale)
- **Motion Blur**: Fast movement causes tracking loss
- **Low Texture**: Works poorly on blank walls
- **Drift**: Errors accumulate over long sequences

---

## License

For educational purposes.

---

## Credits

Built with classical computer vision techniques from:
- Hartley & Zisserman, "Multiple View Geometry"
- Nistér, "An efficient solution to the five-point relative pose problem"
- Lucas & Kanade, "An iterative image registration technique"
