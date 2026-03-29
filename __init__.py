"""
Stereo Visual Odometry and 3D Mapping System
==============================================
A classical computer vision project using KITTI dataset.

Modules:
- dataset_loader: Load KITTI stereo images
- calibration: Parse calibration files
- feature_detection: Harris corner detection
- stereo_depth: Disparity and depth computation
- optical_flow: KLT feature tracking
- motion_estimation: Camera pose estimation
- trajectory: Camera trajectory computation
- visualization: Plotting results
"""

__version__ = "1.0.0"
