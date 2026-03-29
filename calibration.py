"""
Calibration Module
==================
Parses KITTI calibration files and provides camera parameters.

KITTI Calibration File Format (calib.txt):
    P0: 7x4 projection matrix for left camera (cam0)
    P1: 7x4 projection matrix for right camera (cam1)
    P2: 7x4 projection matrix for cam2
    P3: 7x4 projection matrix for cam3
    Tr: 4x4 transformation matrix (velodyne to camera)
    
For stereo, we primarily use P0 and P1:
    P0 = [fx  0  cx  tx]
         [ 0 fy  cy  ty]
         [ 0  0   1   0]
    
    P1 = [fx  0  cx  tx']    <- tx' = tx - fx * baseline
         [0  fy  cy  ty]
         [0   0   1   0]
"""

import os
import numpy as np
from typing import Tuple, Optional


class CameraCalibration:
    """
    Stores and provides access to camera calibration parameters.
    """
    
    def __init__(self, P0: np.ndarray, P1: np.ndarray, 
                 image_width: int, image_height: int):
        """
        Initialize with projection matrices.
        
        Args:
            P0: 3x4 projection matrix for left camera
            P1: 3x4 projection matrix for right camera
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.P0 = P0
        self.P1 = P1
        self.image_width = image_width
        self.image_height = image_height
        
        # Extract intrinsic parameters from P0
        self.fx = P0[0, 0]
        self.fy = P0[1, 1]
        self.cx = P0[0, 2]
        self.cy = P0[1, 2]
        
        # Baseline from P1 - P0 translation
        # P0[0, 3] = -fx * baseline (typically 0 for P0)
        # P1[0, 3] = -fx * baseline + fx * baseline = 0? 
        # Actually in KITTI: P1[0,3] = P0[0,3] - fx * baseline
        # Since P0[0,3] = 0, baseline = -P1[0,3] / fx
        self.baseline = -P1[0, 3] / self.fx
        
        # Camera matrix (3x3 intrinsic)
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])
        
    def get_focal_length(self) -> Tuple[float, float]:
        """Return focal lengths (fx, fy)."""
        return self.fx, self.fy
    
    def get_principal_point(self) -> Tuple[float, float]:
        """Return principal point (cx, cy)."""
        return self.cx, self.cy
    
    def get_baseline(self) -> float:
        """Return stereo baseline in meters."""
        return self.baseline
    
    def get_camera_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix."""
        return self.K.copy()
    
    def __repr__(self):
        return (f"CameraCalibration(fx={self.fx:.2f}, fy={self.fy:.2f}, "
                f"cx={self.cx:.2f}, cy={self.cy:.2f}, baseline={self.baseline:.4f}m)")


def parse_calibration(calib_path: str) -> CameraCalibration:
    """
    Parse KITTI calibration file.
    
    Args:
        calib_path: Path to calib.txt file
        
    Returns:
        CameraCalibration object with parameters
        
    Example calib.txt format:
        P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00
            0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
            0.000000000000e+00 0.000000000000e+01 0.000000000000e+00 1.000000000000e+00
        P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02
            0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00
            0.000000000000e+00 0.000000000000e+01 0.000000000000e+00 1.000000000000e+00
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")
    
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    
    # Parse P0 and P1 matrices
    P0 = None
    P1 = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Parse matrix rows (each P* line spans 4 rows in the file)
        if line.startswith('P0:'):
            P0 = _parse_projection_matrix(line)
            # Next 2 lines are part of the same matrix
            continue
        elif line.startswith('P1:'):
            P1 = _parse_projection_matrix(line)
            continue
        elif P0 is not None and P0.shape[0] < 3:
            # Continue reading P0
            P0 = np.vstack([P0, _parse_projection_row(line)])
        elif P1 is not None and P1.shape[0] < 3:
            # Continue reading P1
            P1 = np.vstack([P1, _parse_projection_row(line)])
    
    if P0 is None or P1 is None:
        raise ValueError("Failed to parse P0 and P1 matrices from calibration file")
    
    # Get image dimensions from calibration (usually 1242x375 for KITTI)
    # If not explicitly stored, use default or extract from file
    image_width, image_height = 1242, 375  # Default KITTI resolution
    
    return CameraCalibration(P0, P1, image_width, image_height)


def _parse_projection_matrix(line: str) -> np.ndarray:
    """
    Parse a projection matrix from a calibration line.
    
    Args:
        line: Line containing 'Px:' prefix and first row
        
    Returns:
        3x4 numpy array
    """
    # Remove 'Px:' prefix and parse numbers
    parts = line.split(':')[1].strip().split()
    values = [float(x) for x in parts]
    
    # First row
    matrix = np.array(values).reshape(1, -1)
    return matrix


def _parse_projection_row(line: str) -> np.ndarray:
    """Parse additional row of projection matrix."""
    values = [float(x) for x in line.strip().split()]
    return np.array(values).reshape(1, -1)


def get_stereo_parameters(calib: CameraCalibration) -> dict:
    """
    Get stereo vision parameters as a dictionary.
    
    Args:
        calib: CameraCalibration object
        
    Returns:
        Dictionary with fx, fy, cx, cy, baseline
    """
    return {
        'fx': calib.fx,
        'fy': calib.fy,
        'cx': calib.cx,
        'cy': calib.cy,
        'baseline': calib.baseline,
        'K': calib.K
    }


def compute_depth_from_disparity(disparity: np.ndarray, 
                                   fx: float, 
                                   baseline: float) -> np.ndarray:
    """
    Compute depth map from disparity map using the formula:
        depth = (focal_length * baseline) / disparity
    
    Args:
        disparity: Disparity map (pixels)
        fx: Focal length in pixels
        baseline: Stereo baseline in meters
        
    Returns:
        Depth map in meters (same shape as disparity)
    """
    # Avoid division by zero
    disparity_safe = np.maximum(disparity, 1e-6)
    
    depth = (fx * baseline) / disparity_safe
    
    # Set invalid disparities to zero depth
    depth[disparity <= 0] = 0
    
    return depth
