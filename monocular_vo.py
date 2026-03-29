"""
Monocular Visual Odometry Module
================================
Monocular visual odometry using 2D-2D feature matching.

Since we don't have stereo depth, we use:
1. Feature detection and matching between frames
2. Essential matrix estimation
3. Relative pose recovery
4. Scale estimation (assumed constant or from other sources)

Note: Monocular VO has scale ambiguity - the trajectory will be
in arbitrary units. For real scale, you need:
- Stereo vision
- GPS/IMU data
- Known object sizes
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class MonocularVisualOdometry:
    """
    Monocular visual odometry system.
    """
    
    def __init__(self, 
                 focal_length: float = 700.0,
                 principal_point: Tuple[float, float] = (640, 360),
                 scale_factor: float = 1.0):
        """
        Initialize monocular visual odometry.
        
        Args:
            focal_length: Focal length in pixels (approximate for KITTI)
            principal_point: Principal point (cx, cy)
            scale_factor: Scale factor for translation
        """
        self.focal_length = focal_length
        self.cx, self.cy = principal_point
        self.scale_factor = scale_factor
        
        # Camera matrix
        self.K = np.array([
            [focal_length, 0, self.cx],
            [0, focal_length, self.cy],
            [0, 0, 1]
        ])
        
        # Trajectory storage
        self.poses = []
        self.positions = []
        self.rotations = []  # List of rotation matrices
        self.translations = []  # List of translation vectors
        
        # Previous frame data
        self.prev_image = None
        self.prev_points = None
        self.prev_descriptors = None
        
        # Initialize first pose
        self.rotations.append(np.eye(3))
        self.translations.append(np.zeros(3))
        
        # Feature detector
        self.feature_detector = cv2.ORB_create(nfeatures=1000)
        
        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        
        # Initialize first pose
        self._initialize()
        
    def _initialize(self):
        """Initialize first pose as origin."""
        self.poses.append(np.eye(4))
        self.positions.append(np.zeros(3))
        
    def detect_features(self, image: np.ndarray) -> Tuple:
        """
        Detect features in image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints = self.feature_detector.detect(image, None)
        keypoints, descriptors = self.feature_detector.compute(image, keypoints)
        return keypoints, descriptors
    
    def match_features(self, 
                       desc1: np.ndarray, 
                       desc2: np.ndarray) -> List:
        """
        Match features between two images.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []
        
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def estimate_motion(self,
                        matches: List,
                        keypoints1: List,
                        keypoints2: List) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Estimate camera motion from feature matches.
        
        Args:
            matches: List of good matches
            keypoints1: Keypoints from first image
            keypoints2: Keypoints from second image
            
        Returns:
            Tuple of (rotation, translation, num_inliers)
        """
        if len(matches) < 10:
            return np.eye(3), np.zeros(3), 0
        
        # Extract matched points
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
        
        # Compute essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1, pts2,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=3.0
        )
        
        if E is None:
            return np.eye(3), np.zeros(3), 0
        
        # Recover pose
        success, R, t, mask = cv2.recoverPose(
            E, pts1, pts2, self.K
        )
        
        if not success:
            return np.eye(3), np.zeros(3), 0
        
        # Count inliers
        num_inliers = np.sum(mask > 0)
        
        # Apply scale factor to translation
        t = t * self.scale_factor
        
        return R, t.flatten(), num_inliers
    
    def process_frame(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Process a new frame and estimate motion.
        
        Args:
            image: Current grayscale image
            
        Returns:
            Tuple of (rotation, translation, num_inliers)
        """
        # Detect features in current frame
        keypoints, descriptors = self.detect_features(image)
        
        if self.prev_image is None:
            # First frame
            self.prev_image = image
            self.prev_points = keypoints
            self.prev_descriptors = descriptors
            return np.eye(3), np.zeros(3), len(keypoints)
        
        # Match features with previous frame
        matches = self.match_features(self.prev_descriptors, descriptors)
        
        # Estimate motion
        R, t, num_inliers = self.estimate_motion(matches, self.prev_points, keypoints)
        
        # Update trajectory
        self._update_trajectory(R, t)
        
        # Store current frame data
        self.prev_image = image
        self.prev_points = keypoints
        self.prev_descriptors = descriptors
        
        return R, t, num_inliers
    
    def _update_trajectory(self, R: np.ndarray, t: np.ndarray):
        """
        Update trajectory with new pose.
        
        Args:
            R: Rotation matrix
            t: Translation vector
        """
        if len(self.positions) == 0:
            self.positions.append(np.zeros(3))
            self.poses.append(np.eye(4))
            return
        
        # Get previous position
        prev_pos = self.positions[-1]
        
        # Compute new position
        # For small motions: new_pos = prev_pos + t
        new_pos = prev_pos + t
        
        # Create pose matrix
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = new_pos
        
        self.positions.append(new_pos)
        self.poses.append(pose)
        
        # Store rotation and translation
        self.rotations.append(R)
        self.translations.append(t)
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get camera trajectory.
        
        Returns:
            N x 3 array of camera positions
        """
        return np.array(self.positions)
    
    def get_rotations(self) -> List[np.ndarray]:
        """
        Get list of rotation matrices.
        
        Returns:
            List of 3x3 rotation matrices
        """
        return self.rotations
    
    def get_translations(self) -> List[np.ndarray]:
        """
        Get list of translation vectors.
        
        Returns:
            List of 3-element translation vectors
        """
        return self.translations
    
    def get_motion_data(self) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Get complete motion data.
        
        Returns:
            Tuple of (trajectory, rotations, translations)
        """
        return np.array(self.positions), self.rotations, self.translations
    
    def get_num_poses(self) -> int:
        """Get number of poses."""
        return len(self.poses)
    
    def compute_total_distance(self) -> float:
        """
        Compute total distance traveled.
        
        Returns:
            Total distance
        """
        if len(self.positions) < 2:
            return 0.0
            
        total_distance = 0.0
        for i in range(1, len(self.positions)):
            diff = self.positions[i] - self.positions[i-1]
            total_distance += np.linalg.norm(diff)
            
        return total_distance
    
    def reset(self):
        """Reset visual odometry state."""
        self.poses = []
        self.positions = []
        self.prev_image = None
        self.prev_points = None
        self.prev_descriptors = None
        self._initialize()


def run_monocular_vo(image_folder: str,
                      start_frame: int = 0,
                      end_frame: Optional[int] = None,
                      focal_length: float = 700.0) -> np.ndarray:
    """
    Run monocular visual odometry on image sequence.
    
    Args:
        image_folder: Path to folder containing images
        start_frame: Starting frame index
        end_frame: Ending frame index (None = all frames)
        focal_length: Focal length in pixels
        
    Returns:
        Camera trajectory (N x 3)
    """
    from dataset_loader import MonocularDatasetLoader
    
    # Load dataset
    dataset = MonocularDatasetLoader(image_folder)
    
    if end_frame is None:
        end_frame = len(dataset)
    
    # Initialize VO
    vo = MonocularVisualOdometry(focal_length=focal_length)
    
    # Process frames
    for idx in range(start_frame, end_frame):
        image = dataset.load_image(idx)
        
        R, t, inliers = vo.process_frame(image)
        
        if idx % 10 == 0:
            print(f"Frame {idx}: inliers={inliers}")
    
    return vo.get_trajectory()
