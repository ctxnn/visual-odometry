"""
Stereo Depth Module
==================
Computes disparity and depth from stereo image pairs.

Stereo Vision Basics:
- Two cameras separated by baseline B
- Corresponding points have horizontal offset (disparity d)
- Depth Z = f * B / d

Methods:
- Block Matching (BM): Fast, good quality
- Semi-Global Block Matching (SGBM): Higher quality, slower
- SGBM with confidence: Best quality with confidence estimates
"""

import numpy as np
import cv2
from typing import Tuple, Optional


class StereoDepthEstimator:
    """
    Stereo depth estimator using Block Matching or SGBM.
    """
    
    def __init__(self, 
                 method: str = 'sgbm',
                 min_disparity: int = 0,
                 num_disparities: int = 128,
                 block_size: int = 5,
                 P1: Optional[int] = None,
                 P2: Optional[int] = None,
                 disp12_max_diff: int = 1,
                 prefilter_cap: int = 31,
                 uniqueness_ratio: float = 10.0,
                 speckle_window_size: int = 100,
                 speckle_range: int = 32):
        """
        Initialize stereo depth estimator.
        
        Args:
            method: 'bm' for Block Matching, 'sgbm' for SGBM
            min_disparity: Minimum disparity
            num_disparities: Number of disparity levels (must be divisible by 16)
            block_size: Block size for matching
            P1: First parameter for SGBM smoothness
            P2: Second parameter for SGBM smoothness
            disp12_max_diff: Maximum allowed difference in disparity check
            prefilter_cap: Pre-filter cap
            uniqueness_ratio: Uniqueness ratio for matching
            speckle_window_size: Speckle window size
            speckle_range: Speckle range
        """
        
        # Compute P1 and P2 if not provided
        if P1 is None:
            P1 = 8 * block_size**2
        if P2 is None:
            P2 = 32 * block_size**2
        
        self.method = method
        self.min_disparity = min_disparity
        self.num_disparities = num_disparities
        self.block_size = block_size
        
        if method == 'sgbm':
            # Semi-Global Block Matching
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=min_disparity,
                numDisparities=num_disparities,
                blockSize=block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=disp12_max_diff,
                preFilterCap=prefilter_cap,
                uniquenessRatio=int(uniqueness_ratio),
                speckleWindowSize=speckle_window_size,
                speckleRange=speckle_range,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        elif method == 'bm':
            # Block Matching (faster but lower quality)
            self.stereo = cv2.StereoBM_create(
                numDisparities=num_disparities,
                blockSize=block_size
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def compute_disparity(self, 
                          left_image: np.ndarray, 
                          right_image: np.ndarray) -> np.ndarray:
        """
        Compute disparity map from stereo pair.
        
        Args:
            left_image: Left grayscale image
            right_image: Right grayscale image
            
        Returns:
            Disparity map (float32, in pixels)
        """
        # Ensure images are grayscale and uint8
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        if len(right_image.shape) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        if left_image.dtype != np.uint8:
            left_image = left_image.astype(np.uint8)
        if right_image.dtype != np.uint8:
            right_image = right_image.astype(np.uint8)
        
        # Compute disparity
        disparity = self.stereo.compute(left_image, right_image)
        
        # Convert to float32 (OpenCV returns int16 for SGBM)
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    def compute_depth(self, 
                       disparity: np.ndarray, 
                       focal_length: float, 
                       baseline: float) -> np.ndarray:
        """
        Compute depth map from disparity using formula: Z = f * B / d
        
        Args:
            disparity: Disparity map in pixels
            focal_length: Focal length in pixels
            baseline: Stereo baseline in meters
            
        Returns:
            Depth map in meters
        """
        # Avoid division by zero
        disparity_safe = np.maximum(disparity, 0.1)
        
        # Depth = focal_length * baseline / disparity
        depth = (focal_length * baseline) / disparity_safe
        
        # Set invalid disparities (negative or near-zero) to zero depth
        depth[disparity <= 0] = 0
        depth[depth > 1000] = 0  # Clip very far points
        
        return depth
    
    def compute_point_cloud(self, 
                            disparity: np.ndarray,
                            focal_length: float,
                            baseline: float,
                            Q: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D point cloud from disparity.
        
        Args:
            disparity: Disparity map
            focal_length: Focal length in pixels
            baseline: Stereo baseline in meters
            Q: Reprojection matrix (if None, uses basic triangulation)
            
        Returns:
            Tuple of (points_3d, colors)
                points_3d: N x 3 array of 3D points
                colors: N x 3 array of RGB colors
        """
        height, width = disparity.shape
        
        # Create meshgrid for pixel coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Compute depth
        depth = self.compute_depth(disparity, focal_length, baseline)
        
        # Filter valid points
        valid_mask = (depth > 0) & (depth < 100)
        
        # Get valid coordinates
        x_valid = x[valid_mask].astype(np.float32)
        y_valid = y[valid_mask].astype(np.float32)
        z_valid = depth[valid_mask].astype(np.float32)
        
        # Triangulate 3D points
        # X = (x - cx) * Z / fx
        # Y = (y - cy) * Z / fy
        cx = width / 2  # Assume principal point at center
        cy = height / 2
        
        # For KITTI, baseline is already accounted in disparity
        # Use simplified triangulation
        fx = focal_length
        fy = focal_length
        
        X = (x_valid - cx) * z_valid / fx
        Y = (y_valid - cy) * z_valid / fy
        
        points_3d = np.column_stack([X, Y, z_valid])
        
        # Return empty arrays if no valid points
        if len(points_3d) == 0:
            return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
        
        return points_3d, np.array([])


def compute_disparity_sgbm(left_image: np.ndarray, 
                             right_image: np.ndarray,
                             min_disparity: int = 0,
                             num_disparities: int = 128) -> np.ndarray:
    """
    Convenience function to compute disparity using SGBM.
    
    Args:
        left_image: Left grayscale image
        right_image: Right grayscale image  
        min_disparity: Minimum disparity
        num_disparities: Number of disparity levels
        
    Returns:
        Disparity map
    """
    estimator = StereoDepthEstimator(
        method='sgbm',
        min_disparity=min_disparity,
        num_disparities=num_disparities
    )
    return estimator.compute_disparity(left_image, right_image)


def compute_depth_map(left_image: np.ndarray,
                       right_image: np.ndarray,
                       focal_length: float,
                       baseline: float,
                       min_disparity: int = 0,
                       num_disparities: int = 128) -> np.ndarray:
    """
    Complete pipeline: compute disparity and convert to depth.
    
    Args:
        left_image: Left grayscale image
        right_image: Right grayscale image
        focal_length: Focal length in pixels
        baseline: Stereo baseline in meters
        
    Returns:
        Depth map in meters
    """
    estimator = StereoDepthEstimator(
        method='sgbm',
        min_disparity=min_disparity,
        num_disparities=num_disparities
    )
    
    # Compute disparity
    disparity = estimator.compute_disparity(left_image, right_image)
    
    # Convert to depth
    depth = estimator.compute_depth(disparity, focal_length, baseline)
    
    return depth


def visualize_disparity(disparity: np.ndarray, 
                         colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Visualize disparity map with colormap.
    
    Args:
        disparity: Disparity map
        colormap: OpenCV colormap
        
    Returns:
        Colorized disparity image
    """
    # Normalize to 0-255 range
    disp_min = disparity.min()
    disp_max = disparity.max()
    
    if disp_max > disp_min:
        disparity_norm = ((disparity - disp_min) / (disp_max - disp_min) * 255).astype(np.uint8)
    else:
        disparity_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Apply colormap
    disparity_color = cv2.applyColorMap(disparity_norm, colormap)
    
    return disparity_color


def visualize_depth(depth: np.ndarray,
                    colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    Visualize depth map with colormap.
    
    Args:
        depth: Depth map in meters
        colormap: OpenCV colormap
        
    Returns:
        Colorized depth image
    """
    # Filter valid depth
    depth_valid = depth.copy()
    depth_valid[depth_valid > 50] = 50  # Clip to 50m
    depth_valid[depth_valid <= 0] = 0
    
    # Normalize
    if depth_valid.max() > 0:
        depth_norm = (depth_valid / depth_valid.max() * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_valid, dtype=np.uint8)
    
    # Apply colormap
    depth_color = cv2.applyColorMap(depth_norm, colormap)
    
    return depth_color
