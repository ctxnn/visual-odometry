"""
Motion Estimation Module
========================
Estimates camera motion from feature correspondences.

Methods:
- Essential Matrix: 5-point algorithm + RANSAC
- Fundamental Matrix: 8-point algorithm (for uncalibrated)
- RANSAC: Random Sample Consensus for robust estimation
- SVD Decomposition: Recover rotation and translation

Camera Motion Model:
- Relative pose between consecutive frames
- R, t: Rotation matrix and translation vector
- E = [t]_x * R (Essential matrix)
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List


class MotionEstimator:
    """
    Estimates camera motion from feature correspondences.
    """
    
    def __init__(self,
                 confidence: float = 0.999,
                 ransac_threshold: float = 3.0,
                 max_iterations: int = 2000):
        """
        Initialize motion estimator.
        
        Args:
            confidence: RANSAC confidence level
            ransac_threshold: RANSAC inlier threshold (pixels)
            max_iterations: Maximum RANSAC iterations
        """
        self.confidence = confidence
        self.ransac_threshold = ransac_threshold
        self.max_iterations = max_iterations
        
    def estimate_essential_matrix(self,
                                   points1: np.ndarray,
                                   points2: np.ndarray,
                                   camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate essential matrix using RANSAC.
        
        Args:
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            camera_matrix: Camera intrinsic matrix (3 x 3)
            
        Returns:
            Tuple of (essential_matrix, inlier_mask)
        """
        # Compute fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(
            points1,
            points2,
            cv2.FM_RANSAC,
            self.ransac_threshold,
            self.confidence,
            self.max_iterations
        )
        
        if F is None:
            return np.eye(3), np.zeros(len(points1)).astype(bool)
        
        # Convert Fundamental to Essential matrix
        # E = K2^T * F * K1
        E = camera_matrix.T @ F @ camera_matrix
        
        # Enforce rank-2 constraint (singular value decomposition)
        U, S, Vt = np.linalg.svd(E)
        S = np.diag([(S[0] + S[1]) / 2, (S[0] + S[1]) / 2, 0])
        E = U @ S @ Vt
        
        return E, mask.ravel().astype(bool)
    
    def recover_pose(self,
                     essential_matrix: np.ndarray,
                     points1: np.ndarray,
                     points2: np.ndarray,
                     camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover relative pose from essential matrix.
        
        Args:
            essential_matrix: 3x3 essential matrix
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            Tuple of (rotation, translation, mask)
                rotation: 3x3 rotation matrix
                translation: 1x3 translation vector
                mask: Inlier mask
        """
        # Recover pose using singular value decomposition
        # Returns rotation, translation, and mask
        points1 = points1.reshape(-1, 1, 2)
        points2 = points2.reshape(-1, 1, 2)
        
        # Use OpenCV's recoverPose
        ret_val, R, t, mask = cv2.recoverPose(
            essential_matrix,
            points1,
            points2,
            camera_matrix
        )
        
        if ret_val == 0:
            # Fallback to identity if recovery fails
            return np.eye(3), np.zeros(3), np.zeros(len(points1)).astype(int)
        
        return R, t.flatten(), mask.ravel().astype(bool)
    
    def estimate_motion(self,
                        points1: np.ndarray,
                        points2: np.ndarray,
                        camera_matrix: np.ndarray,
                        depth1: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        Complete motion estimation pipeline.
        
        Args:
            points1: Points in first image (N x 2)
            points2: Points in second image (N x 2)
            camera_matrix: Camera intrinsic matrix
            depth1: Depth values for points (optional, for 3D-2D motion estimation)
            
        Returns:
            Tuple of (rotation, translation, inlier_mask, num_inliers)
        """
        if len(points1) < 5 or len(points2) < 5:
            return np.eye(3), np.zeros(3), np.zeros(len(points1)).astype(bool), 0
        
        # If we have depth, use 3D-2D motion estimation
        if depth1 is not None:
            return self._estimate_motion_3d_2d(points1, points2, depth1, camera_matrix)
        else:
            # Use 2D-2D motion estimation (Essential matrix)
            return self._estimate_motion_2d_2d(points1, points2, camera_matrix)
    
    def _estimate_motion_2d_2d(self,
                                 points1: np.ndarray,
                                 points2: np.ndarray,
                                 camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        2D-2D motion estimation using Essential matrix.
        """
        # Find essential matrix
        E, inlier_mask = self.estimate_essential_matrix(
            points1, points2, camera_matrix
        )
        
        # Recover pose
        R, t, pose_mask = self.recover_pose(
            E, points1, points2, camera_matrix
        )
        
        # Combine masks
        if len(inlier_mask) == len(pose_mask):
            combined_mask = inlier_mask & pose_mask
        else:
            combined_mask = inlier_mask
            
        num_inliers = np.sum(combined_mask)
        
        return R, t, combined_mask, num_inliers
    
    def _estimate_motion_3d_2d(self,
                                 points1: np.ndarray,
                                 points2: np.ndarray,
                                 depth1: np.ndarray,
                                 camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """
        3D-2D motion estimation using PnP (Perspective-n-Point).
        """
        # Get 3D points from depth
        points3d = self._backproject_points(points1, depth1, camera_matrix)
        
        # Filter out points with invalid depth
        valid_mask = (depth1 > 0) & (depth1 < 100)
        points3d_valid = points3d[valid_mask]
        points2_valid = points2[valid_mask]
        
        if len(points3d_valid) < 4:
            return np.eye(3), np.zeros(3), valid_mask, 0
        
        # Solve PnP with RANSAC
        success, R, t, inliers = cv2.solvePnPRansac(
            points3d_valid,
            points2_valid,
            camera_matrix,
            distCoeffs=None,
            iterationsCount=self.max_iterations,
            reprojectionError=self.ransac_threshold,
            confidence=self.confidence
        )
        
        if not success:
            return np.eye(3), np.zeros(3), valid_mask, 0
        
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(R)
        
        return R, t.flatten(), valid_mask, len(inliers)
    
    def _backproject_points(self,
                             points: np.ndarray,
                             depth: np.ndarray,
                             camera_matrix: np.ndarray) -> np.ndarray:
        """
        Backproject 2D points to 3D using depth.
        
        Args:
            points: 2D points (N x 2)
            depth: Depth values (N)
            camera_matrix: Camera intrinsic matrix
            
        Returns:
            3D points (N x 3)
        """
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        
        points3d = np.zeros((len(points), 3))
        
        for i, (x, y) in enumerate(points):
            Z = depth[i]
            if Z > 0:
                points3d[i, 0] = (x - cx) * Z / fx
                points3d[i, 1] = (y - cy) * Z / fy
                points3d[i, 2] = Z
                
        return points3d


def estimate_essential_matrix(points1: np.ndarray,
                              points2: np.ndarray,
                              camera_matrix: np.ndarray,
                              threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to estimate essential matrix.
    
    Args:
        points1: Points in first image
        points2: Points in second image
        camera_matrix: Camera intrinsic matrix
        threshold: RANSAC threshold
        
    Returns:
        Tuple of (essential_matrix, inlier_mask)
    """
    estimator = MotionEstimator(ransac_threshold=threshold)
    return estimator.estimate_essential_matrix(points1, points2, camera_matrix)


def decompose_essential_matrix(E: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decompose essential matrix into rotation and translation.
    
    The essential matrix has 4 possible decompositions.
    We return the one with most points in front of camera.
    
    Args:
        E: 3x3 essential matrix
        
    Returns:
        Tuple of (rotation, translation)
    """
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation matrix
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    
    # Translation
    t = U[:, 2]
    
    # Rotation matrices
    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]], dtype=np.float64)
    
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Ensure proper rotation
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        
    return R1, t


def validate_rotation_matrix(R: np.ndarray, tolerance: float = 1e-6) -> bool:
    """
    Validate that a matrix is a proper rotation matrix.
    
    Args:
        R: 3x3 matrix
        tolerance: Numerical tolerance
        
    Returns:
        True if valid rotation matrix
    """
    if R.shape != (3, 3):
        return False
    
    # Check orthogonality: R^T * R = I
    should_be_identity = R.T @ R
    if not np.allclose(should_be_identity, np.eye(3), atol=tolerance):
        return False
    
    # Check determinant = 1
    if not np.isclose(np.linalg.det(R), 1.0, atol=tolerance):
        return False
    
    return True


def motion_from_essential(E: np.ndarray,
                         points1: np.ndarray,
                         points2: np.ndarray,
                         camera_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract motion from essential matrix.
    
    Args:
        E: Essential matrix
        points1: Points in first image
        points2: Points in second image
        camera_matrix: Camera matrix
        
    Returns:
        Tuple of (rotation, translation)
    """
    R, t = decompose_essential_matrix(E)
    
    # Use OpenCV for disambiguation
    # Create 4 possible camera poses
    R1, R2, t1, t2 = decompose_essential_matrix(E)
    
    # Return first valid solution
    if validate_rotation_matrix(R1):
        return R1, t1
    else:
        return R2, t2
