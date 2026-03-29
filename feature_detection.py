"""
Feature Detection Module
========================
Detects and matches features using Harris corners and ORB descriptors.

Harris Corner Detection:
- Finds corners by analyzing intensity changes in local neighborhoods
- Computes the Harris response R = det(M) - k * trace(M)^2
- Corners have high R values

ORB (Oriented FAST and Rotated BRIEF):
- FAST corner detection with orientation
- BRIEF descriptor with rotation invariance
- More efficient than SIFT/SURF
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class FeatureDetector:
    """
    Feature detector using Harris corners with optional ORB descriptors.
    """
    
    def __init__(self, 
                 harris_block_size: int = 2,
                 harris_ksize: int = 3,
                 harris_k: float = 0.04,
                 harris_threshold: float = 0.01,
                 max_corners: int = 500):
        """
        Initialize Harris corner detector.
        
        Args:
            harris_block_size: Neighborhood size for corner detection
            harris_ksize: Aperture parameter for Sobel derivatives
            harris_k: Harris detector free parameter (typically 0.04-0.06)
            harris_threshold: Threshold for corner detection (0.01-0.1)
            max_corners: Maximum number of corners to detect
        """
        self.block_size = harris_block_size
        self.ksize = harris_ksize
        self.k = harris_k
        self.threshold = harris_threshold
        self.max_corners = max_corners
        
        # ORB detector for additional robustness
        self.orb = cv2.ORB_create(nfeatures=max_corners)
        
    def detect_harris(self, image: np.ndarray) -> np.ndarray:
        """
        Detect Harris corners in an image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Array of corner points (N x 1 x 2) in format for OpenCV
        """
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
        
        # Compute Harris corners
        harris = cv2.cornerHarris(image, 
                                   blockSize=self.block_size,
                                   ksize=self.ksize,
                                   k=self.k)
        
        # Dilate for better corner localization
        harris = cv2.dilate(harris, None)
        
        # Threshold for corner selection
        threshold = self.threshold * harris.max()
        
        # Find corner points
        corners = np.where(harris > threshold)
        
        # Convert to corner points format (N x 1 x 2)
        points = []
        for y, x in zip(corners[0], corners[1]):
            points.append([[x, y]])
            
        if not points:
            return np.array([]).reshape(0, 1, 2).astype(np.float32)
            
        points = np.array(points, dtype=np.float32)
        
        # Apply non-maximum suppression
        points = self._non_maximum_suppression(image, points)
        
        # Limit to max corners
        if len(points) > self.max_corners:
            points = points[:self.max_corners]
            
        return points
    
    def _non_maximum_suppression(self, image: np.ndarray, 
                                   corners: np.ndarray, 
                                   min_distance: int = 10) -> np.ndarray:
        """
        Apply non-maximum suppression to keep well-separated corners.
        
        Args:
            image: Original image
            corners: Detected corner points
            min_distance: Minimum distance between corners
            
        Returns:
            Filtered corner points
        """
        if len(corners) == 0:
            return corners
            
        # Compute Harris response for each corner
        harris = cv2.cornerHarris(image, 
                                   blockSize=self.block_size,
                                   ksize=self.ksize,
                                   k=self.k)
        
        # Sort by Harris response
        indices = []
        for i, corner in enumerate(corners):
            x, y = int(corner[0][0]), int(corner[0][1])
            indices.append((i, harris[y, x]))
            
        indices.sort(key=lambda x: x[1], reverse=True)
        
        # Keep corners that are far enough apart
        filtered = []
        for idx, _ in indices:
            corner = corners[idx]
            x, y = corner[0][0], corner[0][1]
            
            # Check distance to already selected corners
            too_close = False
            for selected in filtered:
                sx, sy = selected[0][0], selected[0][1]
                if np.sqrt((x - sx)**2 + (y - sy)**2) < min_distance:
                    too_close = True
                    break
                    
            if not too_close:
                filtered.append(corner)
                
        return np.array(filtered, dtype=np.float32) if filtered else corners
    
    def detect_orb(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect ORB keypoints and compute descriptors.
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        keypoints = self.orb.detect(image, None)
        keypoints, descriptors = self.orb.compute(image, keypoints)
        
        if descriptors is None:
            return keypoints, np.array([])
            
        return keypoints, descriptors
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect Harris corners and compute ORB descriptors.
        
        Args:
            image: Grayscale image
            
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Detect Harris corners
        harris_corners = self.detect_harris(image)
        
        # Convert to KeyPoint objects for ORB descriptor computation
        keypoints = [cv2.KeyPoint(x, y, 1) for y, x in zip(
            harris_corners[:, 0, 1].astype(int), 
            harris_corners[:, 0, 0].astype(int)
        )]
        
        # Compute ORB descriptors
        keypoints, descriptors = self.orb.compute(image, keypoints)
        
        if descriptors is None:
            return keypoints, np.array([])
            
        return keypoints, descriptors


class FeatureMatcher:
    """
    Feature matcher using Brute Force or FLANN-based matching.
    """
    
    def __init__(self, matcher_type: str = 'bf', distance_threshold: float = 0.7):
        """
        Initialize feature matcher.
        
        Args:
            matcher_type: 'bf' for Brute Force, 'flann' for FLANN-based
            distance_threshold: Ratio test threshold for BF matcher
        """
        self.matcher_type = matcher_type
        self.distance_threshold = distance_threshold
        
        if matcher_type == 'bf':
            # Brute Force matcher with Hamming distance (for ORB)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # FLANN-based matcher
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH,
                               table_number=6,
                               key_size=12,
                               multi_probe_level=1)
            search_params = dict(checks=50)
            self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, 
              descriptors1: np.ndarray, 
              descriptors2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match descriptors between two images.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            
        Returns:
            List of DMatch objects (best matches)
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
            
        # KNN match (k=2 for ratio test)
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test (Lowe's ratio test)
        good_matches = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.distance_threshold * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def match_with_ratio_test(self,
                                descriptors1: np.ndarray,
                                descriptors2: np.ndarray,
                                ratio: float = 0.75) -> List[cv2.DMatch]:
        """
        Match with Lowe's ratio test for better filtering.
        
        Args:
            descriptors1: Descriptors from first image
            descriptors2: Descriptors from second image
            ratio: Ratio threshold (typically 0.75)
            
        Returns:
            List of good matches after ratio test
        """
        if len(descriptors1) == 0 or len(descriptors2) == 0:
            return []
            
        matches = self.matcher.knnMatch(descriptors1, descriptors2, k=2)
        
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def match_stereo(self, 
                     descriptors_left: np.ndarray, 
                     descriptors_right: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between stereo pair (left to right).
        
        Args:
            descriptors_left: Descriptors from left image
            descriptors_right: Descriptors from right image
            
        Returns:
            List of matches
        """
        return self.match(descriptors_left, descriptors_right)


def detect_and_match_features(image1: np.ndarray, 
                               image2: np.ndarray,
                               detector: Optional[FeatureDetector] = None,
                               matcher: Optional[FeatureMatcher] = None) -> Tuple:
    """
    Complete feature detection and matching pipeline.
    
    Args:
        image1: First grayscale image
        image2: Second grayscale image
        detector: Feature detector (creates default if None)
        matcher: Feature matcher (creates default if None)
        
    Returns:
        Tuple of (keypoints1, keypoints2, matches)
    """
    if detector is None:
        detector = FeatureDetector()
    if matcher is None:
        matcher = FeatureMatcher()
    
    # Detect and compute features
    kp1, desc1 = detector.detect_and_compute(image1)
    kp2, desc2 = detector.detect_and_compute(image2)
    
    # Match features
    matches = matcher.match(desc1, desc2)
    
    return kp1, kp2, matches


def draw_matches(image1: np.ndarray, 
                  image2: np.ndarray,
                  keypoints1: List[cv2.KeyPoint],
                  keypoints2: List[cv2.KeyPoint],
                  matches: List[cv2.DMatch],
                  max_matches: int = 50) -> np.ndarray:
    """
    Draw feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of matches
        max_matches: Maximum number of matches to draw
        
    Returns:
        Image with drawn matches
    """
    # Convert grayscale to color if needed
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # Limit matches
    matches = matches[:max_matches]
    
    # Draw matches
    result = cv2.drawMatches(image1, keypoints1, image2, keypoints2, 
                             matches, None,
                             matchColor=(0, 255, 0),
                             singlePointColor=(255, 0, 0),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return result
