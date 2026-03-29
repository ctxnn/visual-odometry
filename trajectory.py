"""
Trajectory Module
=================
Computes and manages camera trajectory from motion estimates.

Camera Pose:
- Represented as rotation matrix R and translation vector t
- Transformation T = [R | t] (3x4 matrix)
- Accumulates relative motions to get absolute pose

Coordinate Systems:
- World frame: First camera position
- Camera frame: Moving camera
- P_world = T_cw * P_camera
"""

import numpy as np
from typing import List, Tuple, Optional


class Trajectory:
    """
    Manages camera trajectory by accumulating relative poses.
    """
    
    def __init__(self):
        """Initialize empty trajectory."""
        self.poses = []  # List of 4x4 transformation matrices
        self.positions = []  # List of camera positions (x, y, z)
        self.rotations = []  # List of rotation matrices
        
    def add_pose(self, rotation: np.ndarray, translation: np.ndarray):
        """
        Add a new camera pose.
        
        Args:
            rotation: 3x3 rotation matrix
            translation: 1x3 or 3x1 translation vector
        """
        if translation.shape != (3,):
            translation = translation.flatten()
            
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = translation
        
        self.poses.append(T)
        self.rotations.append(rotation)
        
        # Compute camera position in world frame
        # P_world = -R^T * t (for camera pose in world)
        position = -rotation.T @ translation
        self.positions.append(position)
        
    def get_pose(self, index: int) -> np.ndarray:
        """
        Get transformation matrix at specified index.
        
        Args:
            index: Pose index
            
        Returns:
            4x4 transformation matrix
        """
        if index < 0:
            index = len(self.poses) + index
        return self.poses[index]
    
    def get_position(self, index: int = -1) -> np.ndarray:
        """
        Get camera position at index.
        
        Args:
            index: Position index (-1 for latest)
            
        Returns:
            3D position (x, y, z)
        """
        if index < 0:
            index = len(self.positions) + index
        return self.positions[index]
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get complete trajectory as N x 3 array.
        
        Returns:
            Array of positions (N x 3)
        """
        return np.array(self.positions)
    
    def get_num_poses(self) -> int:
        """Get number of poses in trajectory."""
        return len(self.poses)
    
    def compute_total_distance(self) -> float:
        """
        Compute total distance traveled.
        
        Returns:
            Total distance in meters
        """
        if len(self.positions) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            diff = self.positions[i] - self.positions[i-1]
            total_distance += np.linalg.norm(diff)
            
        return total_distance
    
    def save_trajectory(self, filepath: str):
        """
        Save trajectory to file.
        
        Args:
            filepath: Output file path
        """
        trajectory = self.get_trajectory()
        np.savetxt(filepath, trajectory, delimiter=',', 
                   header='x,y,z', comments='')
        
    def load_trajectory(self, filepath: str):
        """
        Load trajectory from file.
        
        Args:
            filepath: Input file path
        """
        trajectory = np.loadtxt(filepath, delimiter=',', skiprows=1)
        
        for pos in trajectory:
            # Initialize with identity rotations
            self.add_pose(np.eye(3), -pos)
            
    def __len__(self):
        return len(self.poses)


class VisualOdometry:
    """
    Visual odometry system that tracks camera motion.
    """
    
    def __init__(self, camera_matrix: np.ndarray, baseline: float):
        """
        Initialize visual odometry.
        
        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            baseline: Stereo baseline in meters
        """
        self.camera_matrix = camera_matrix
        self.baseline = baseline
        
        # Trajectory
        self.trajectory = Trajectory()
        
        # Previous data
        self.prev_points = None
        self.prev_image = None
        self.prev_depth = None
        
        # First frame flag
        self.is_first_frame = True
        
        # Current pose
        self.current_pose = np.eye(4)
        
    def process_frame(self,
                      image: np.ndarray,
                      depth: np.ndarray,
                      current_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a new frame and estimate motion.
        
        Args:
            image: Current grayscale image
            depth: Depth map
            current_points: Tracked feature points in current image
            
        Returns:
            Tuple of (rotation, translation, num_inliers)
        """
        from motion_estimation import MotionEstimator
        
        if self.is_first_frame:
            # Initialize with first frame
            self.prev_image = image
            self.prev_depth = depth
            self.prev_points = current_points
            
            # Add initial pose (identity)
            self.trajectory.add_pose(np.eye(3), np.zeros(3))
            self.is_first_frame = False
            
            return np.eye(3), np.zeros(3), len(current_points)
        
        # Motion estimation
        motion_estimator = MotionEstimator()
        
        # Get depth for previous points
        if self.prev_depth is not None and len(self.prev_points) > 0:
            # Get depth values for points
            depths = []
            valid_points = []
            for pt in self.prev_points:
                x, y = int(pt[0][0]), int(pt[0][1])
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                    d = depth[y, x]
                    if d > 0:
                        depths.append(d)
                        valid_points.append(pt)
            
            if len(valid_points) > 5:
                depths = np.array(depths)
                valid_points = np.array(valid_points).reshape(-1, 2)
                
                # Estimate motion using 3D-2D
                R, t, mask, num_inliers = motion_estimator.estimate_motion(
                    valid_points, current_points, self.camera_matrix, depths
                )
            else:
                # Fallback to 2D-2D
                R, t, mask, num_inliers = motion_estimator.estimate_motion(
                    self.prev_points.reshape(-1, 2), 
                    current_points, 
                    self.camera_matrix
                )
        else:
            # 2D-2D motion estimation
            R, t, mask, num_inliers = motion_estimator.estimate_motion(
                self.prev_points.reshape(-1, 2) if self.prev_points is not None else np.array([]),
                current_points, 
                self.camera_matrix
            )
        
        # Update current pose
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        self.current_pose = self.current_pose @ np.linalg.inv(T)
        
        # Add to trajectory
        self.trajectory.add_pose(
            self.current_pose[:3, :3],
            self.current_pose[:3, 3]
        )
        
        # Update previous data
        self.prev_image = image
        self.prev_depth = depth
        self.prev_points = current_points
        
        return R, t, num_inliers
    
    def get_trajectory(self) -> np.ndarray:
        """Get camera trajectory."""
        return self.trajectory.get_trajectory()
    
    def reset(self):
        """Reset visual odometry state."""
        self.trajectory = Trajectory()
        self.prev_points = None
        self.prev_image = None
        self.prev_depth = None
        self.is_first_frame = True
        self.current_pose = np.eye(4)


def compute_absolute_scale(translation: np.ndarray, 
                            baseline: float,
                            depth: float) -> float:
    """
    Compute absolute scale for monocular visual odometry.
    
    Args:
        translation: Relative translation vector
        baseline: Stereo baseline (for scale)
        depth: Average depth of tracked features
        
    Returns:
        Absolute scale factor
    """
    # For stereo, scale is known from baseline
    # Z = f * B / d
    # Scale = baseline
    return baseline


def align_trajectory(trajectory: np.ndarray, 
                      ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Align trajectory using Umeyama algorithm.
    
    Args:
        trajectory: Estimated trajectory
        ground_truth: Ground truth trajectory (optional)
        
    Returns:
        Aligned trajectory
    """
    if ground_truth is None:
        return trajectory
        
    # Simple alignment (translation only)
    offset = ground_truth[0] - trajectory[0]
    aligned = trajectory + offset
    
    return aligned
