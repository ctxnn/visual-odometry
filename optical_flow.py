"""
Optical Flow Module
====================
Tracks features across frames using KLT (Kanade-Lucas-Tomasi) optical flow.

KLT Optical Flow:
- Sparse method tracking corner features
- Assumes brightness constancy and small motion
- Uses iterative Lucas-Kanade method with pyramids

Features:
- Tracks from previous frame to current
- Filters out bad matches
- Handles feature initialization and pruning
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class OpticalFlowTracker:
    """
    KLT optical flow tracker for feature tracking.
    """
    
    def __init__(self,
                 win_size: Tuple[int, int] = (21, 21),
                 max_levels: int = 3,
                 criteria: int = cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                 max_iter: int = 30,
                 epsilon: float = 0.01,
                 min_distance: float = 5.0):
        """
        Initialize KLT optical flow tracker.
        
        Args:
            win_size: Window size for tracking
            max_levels: Number of pyramid levels
            criteria: Termination criteria
            max_iter: Maximum iterations
            epsilon: Epsilon for convergence
            min_distance: Minimum distance for valid track
        """
        self.win_size = win_size
        self.max_levels = max_levels
        self.criteria = (criteria, max_iter, epsilon)
        self.min_distance = min_distance
        
    def track_features(self,
                       prev_image: np.ndarray,
                       curr_image: np.ndarray,
                       prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Track features from previous to current frame.
        
        Args:
            prev_image: Previous grayscale image
            curr_image: Current grayscale image
            prev_points: Feature points in previous image (N x 1 x 2)
            
        Returns:
            Tuple of:
                curr_points: Tracked points in current image
                status: Status array (1=tracked, 0=not tracked)
                error: Tracking error for each point
        """
        if len(prev_points) == 0:
            return np.array([]).reshape(0, 1, 2).astype(np.float32), \
                   np.array([]), \
                   np.array([])
        
        # Convert images to uint8 if needed
        if prev_image.dtype != np.uint8:
            prev_image = (prev_image * 255).astype(np.uint8) if prev_image.max() <= 1.0 else prev_image.astype(np.uint8)
        if curr_image.dtype != np.uint8:
            curr_image = (curr_image * 255).astype(np.uint8) if curr_image.max() <= 1.0 else curr_image.astype(np.uint8)
        
        # Calculate optical flow using Lucas-Kanade method with pyramids
        curr_points, status, error = cv2.calcOpticalFlowPyrLK(
            prev_image, 
            curr_image,
            prev_points,
            None,
            winSize=self.win_size,
            maxLevel=self.max_levels,
            criteria=self.criteria,
            minEigThreshold=0.001
        )
        
        # Filter out points with bad status
        if curr_points is not None and status is not None:
            # Only keep points where status == 1
            good_mask = status.flatten() == 1
            
            curr_points = curr_points[good_mask]
            status = status[good_mask]
            error = error[good_mask] if error is not None else np.array([])
            
        return curr_points, status, error
    
    def filter_by_displacement(self,
                                prev_points: np.ndarray,
                                curr_points: np.ndarray,
                                max_displacement: float = 50.0) -> np.ndarray:
        """
        Filter tracks by maximum displacement.
        
        Args:
            prev_points: Previous points
            curr_points: Current points
            max_displacement: Maximum allowed displacement
            
        Returns:
            Boolean mask of valid tracks
        """
        if len(prev_points) == 0 or len(curr_points) == 0:
            return np.array([], dtype=bool)
        
        # Compute displacement
        dx = curr_points[:, 0, 0] - prev_points[:, 0, 0]
        dy = curr_points[:, 0, 1] - prev_points[:, 0, 1]
        displacement = np.sqrt(dx**2 + dy**2)
        
        return displacement < max_displacement
    
    def filter_by_error(self, 
                        error: np.ndarray,
                        max_error: float = 10.0) -> np.ndarray:
        """
        Filter tracks by tracking error.
        
        Args:
            error: Tracking error
            max_error: Maximum allowed error
            
        Returns:
            Boolean mask of valid tracks
        """
        if len(error) == 0:
            return np.array([], dtype=bool)
            
        return error < max_error


class FeatureTracker:
    """
    High-level feature tracking with initialization and management.
    """
    
    def __init__(self,
                 max_features: int = 500,
                 quality_level: float = 0.01,
                 min_distance: int = 10,
                 block_size: int = 7):
        """
        Initialize feature tracker.
        
        Args:
            max_features: Maximum number of features to track
            quality_level: Quality level for Shi-Tomasi corners
            min_distance: Minimum distance between features
            block_size: Block size for corner detection
        """
        self.max_features = max_features
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        
        # Optical flow tracker
        self.flow_tracker = OpticalFlowTracker()
        
        # Current tracked points
        self.prev_points = None
        self.prev_image = None
        
    def detect_features(self, image: np.ndarray) -> np.ndarray:
        """
        Detect new features using Shi-Tomasi corner detector.
        
        Args:
            image: Grayscale image
            
        Returns:
            Array of detected feature points
        """
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Shi-Tomasi corner detection
        corners = cv2.goodFeaturesToTrack(
            image,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size
        )
        
        return corners if corners is not None else np.array([]).reshape(0, 1, 2).astype(np.float32)
    
    def track(self,
              image: np.ndarray,
              detect_new: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Track features in new image.
        
        Args:
            image: Current grayscale image
            detect_new: Whether to detect new features if tracking fails
            
        Returns:
            Tuple of (tracked_points, tracked_indices)
        """
        if self.prev_image is None or self.prev_points is None:
            # First frame - just detect features
            self.prev_points = self.detect_features(image)
            self.prev_image = image
            return self.prev_points, np.arange(len(self.prev_points))
        
        # Track existing features
        curr_points, status, error = self.flow_tracker.track_features(
            self.prev_image, image, self.prev_points
        )
        
        # Get indices of successfully tracked points
        if len(curr_points) > 0:
            tracked_indices = np.where(status.flatten() == 1)[0]
        else:
            tracked_indices = np.array([], dtype=int)
        
        # Detect new features if needed
        if detect_new and len(curr_points) < self.max_features:
            new_points = self.detect_features(image)
            
            # Add new points that are far from existing tracked points
            if len(curr_points) > 0:
                new_points_filtered = []
                for new_pt in new_points:
                    x, y = new_pt[0]
                    min_dist = np.min(np.sqrt(
                        (curr_points[:, 0, 0] - x)**2 + 
                        (curr_points[:, 0, 1] - y)**2
                    ))
                    if min_dist > self.min_distance:
                        new_points_filtered.append(new_pt)
                
                if new_points_filtered:
                    new_points = np.array(new_points_filtered)
                else:
                    new_points = np.array([]).reshape(0, 1, 2).astype(np.float32)
            
            # Combine tracked and new points
            if len(new_points) > 0:
                curr_points = np.vstack([curr_points, new_points])
        
        # Update state
        self.prev_points = curr_points
        self.prev_image = image
        
        return curr_points, tracked_indices
    
    def reset(self):
        """Reset tracker state."""
        self.prev_points = None
        self.prev_image = None


def compute_optical_flow(prev_image: np.ndarray,
                          curr_image: np.ndarray,
                          prev_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for optical flow computation.
    
    Args:
        prev_image: Previous frame
        curr_image: Current frame
        prev_points: Points to track
        
    Returns:
        Tuple of (next_points, status)
    """
    tracker = OpticalFlowTracker()
    next_points, status, error = tracker.track_features(prev_image, curr_image, prev_points)
    return next_points, status


def draw_optical_flow(image: np.ndarray,
                       prev_points: np.ndarray,
                       curr_points: np.ndarray,
                       status: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Visualize optical flow by drawing motion vectors.
    
    Args:
        image: Input image (grayscale or color)
        prev_points: Previous points
        curr_points: Current points
        status: Status array (optional)
        
    Returns:
        Image with drawn flow vectors
    """
    # Convert to color if grayscale
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    if len(prev_points) == 0 or len(curr_points) == 0:
        return vis_image
    
    # Use status if provided
    if status is not None:
        valid_indices = np.where(status.flatten() == 1)[0]
    else:
        valid_indices = range(len(prev_points))
    
    # Draw motion vectors
    for i in valid_indices:
        pt1 = (int(prev_points[i][0][0]), int(prev_points[i][0][1]))
        pt2 = (int(curr_points[i][0][0]), int(curr_points[i][0][1]))
        
        # Color based on direction
        color = (0, 255, 0)  # Green
        
        # Draw line
        cv2.line(vis_image, pt1, pt2, color, 2)
        
        # Draw circle at end point
        cv2.circle(vis_image, pt2, 3, color, -1)
    
    return vis_image
