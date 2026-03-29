# Stereo Visual Odometry and 3D Mapping

A 100% software-based Stereo Visual Odometry and 3D Mapping system using the KITTI dataset.

This project implements classical computer vision methods for:
- Stereo depth estimation
- Feature detection and tracking
- Camera motion estimation
- 3D point cloud generation

## Project Structure

```
project/
├── __init__.py
├── dataset_loader.py    # Load KITTI stereo images
├── calibration.py       # Parse calib.txt
├── feature_detection.py # Harris corner + ORB features
├── stereo_depth.py      # SGBM disparity & depth
├── optical_flow.py      # KLT feature tracking
├── motion_estimation.py # Essential matrix + pose
├── trajectory.py        # Camera trajectory
├── visualization.py     # Plotting results
├── main.py              # Entry point
└── requirements.txt     # Dependencies
```

## Installation

1. Install dependencies:
```bash
pip install -r project/requirements.txt
```

2. Download KITTI Odometry dataset:
   - Download from: https://www.cvlibs.net/datasets/kitti/eval_odometry.php
   - Extract to `sequences/` folder

## Usage

Run visual odometry:
```bash
cd project
python main.py --sequence ../sequences/00 --start 0 --end 100
```

Arguments:
- `--sequence`: Path to KITTI sequence folder
- `--start`: Starting frame (default: 0)
- `--end`: Ending frame (default: all)
- `--stride`: Frame step (default: 1)
- `--num-disparities`: Disparity levels (default: 128)
- `--max-features`: Max features to track (default: 500)
- `--no-visualize`: Disable visualization

## Methods Used

| Component | Method |
|-----------|--------|
| Feature Detection | Harris Corners + ORB |
| Feature Matching | Brute Force (Hamming) |
| Stereo Matching | Semi-Global Block Matching (SGBM) |
| Depth | Z = f * B / d |
| Motion | Essential Matrix + RANSAC |
| Pose | SVD Decomposition |

## Output

- Camera trajectory (X, Y, Z positions)
- 3D point cloud
- Trajectory plots

## Requirements

- Python 3.8+
- OpenCV 4.8+
- NumPy 1.24+
- Matplotlib 3.7+
- SciPy 1.10+

## License

For educational purposes.
