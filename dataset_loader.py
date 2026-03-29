"""
Dataset Loader Module
=====================
Loads image sequences from KITTI dataset (single view or stereo).

For Monocular (Single View) Dataset:
    dataset_folder/
        img/           (single camera images)
        
For Stereo Dataset:
    sequence_folder/
        image_0/       (left camera)
        image_1/       (right camera)
        calib.txt
"""

import os
import numpy as np
import cv2
from typing import Tuple, List, Optional


class MonocularDatasetLoader:
    """
    Loads monocular image sequence.
    Works with single-view datasets like KITTI detection/segmentation.
    """
    
    def __init__(self, image_folder: str):
        """
        Initialize the dataset loader.
        
        Args:
            image_folder: Path to folder containing images
        """
        self.image_folder = image_folder
        self._validate_path()
        self.image_files = self._get_sorted_images()
        self.num_frames = len(self.image_files)
        
    def _validate_path(self):
        """Check that required directory exists."""
        if not os.path.exists(self.image_folder):
            raise FileNotFoundError(f"Image folder not found: {self.image_folder}")
            
    def _get_sorted_images(self) -> List[str]:
        """Get sorted list of image files."""
        all_files = os.listdir(self.image_folder)
        image_files = [f for f in all_files if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # Sort by numeric value in filename
        def get_number(fname):
            # Extract number from filename like "000071.png"
            num_str = ''.join(filter(str.isdigit, fname))
            return int(num_str) if num_str else 0
            
        image_files.sort(key=get_number)
        return image_files
    
    def load_image(self, frame_idx: int) -> np.ndarray:
        """
        Load a single image.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Grayscale image as numpy array
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range [0, {self.num_frames}]")
        
        img_path = os.path.join(self.image_folder, self.image_files[frame_idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
            
        return img
    
    def load_image_color(self, frame_idx: int) -> np.ndarray:
        """
        Load a single color image.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Color image as numpy array (BGR format)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            raise IndexError(f"Frame index {frame_idx} out of range")
        
        img_path = os.path.join(self.image_folder, self.image_files[frame_idx])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
            
        return img
    
    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of images in the dataset."""
        img = self.load_image(0)
        return img.shape[::-1]  # (width, height)
    
    def __len__(self) -> int:
        """Return number of frames in sequence."""
        return self.num_frames
    
    def get_filename(self, frame_idx: int) -> str:
        """Get filename for a frame."""
        return self.image_files[frame_idx]


class KITTI_StereoDatasetLoader:
    """
    Loads stereo image pairs from KITTI odometry sequences.
    """
    
    def __init__(self, sequence_path: str):
        """
        Initialize the dataset loader.
        
        Args:
            sequence_path: Path to KITTI sequence folder (e.g., 'sequences/00')
        """
        self.sequence_path = sequence_path
        self.left_dir = os.path.join(sequence_path, 'image_0')
        self.right_dir = os.path.join(sequence_path, 'image_1')
        
        self._validate_paths()
        self.num_frames = self._get_num_frames()
        
    def _validate_paths(self):
        """Check that required directories exist."""
        if not os.path.exists(self.left_dir):
            raise FileNotFoundError(f"Left image folder not found: {self.left_dir}")
        if not os.path.exists(self.right_dir):
            raise FileNotFoundError(f"Right image folder not found: {self.right_dir}")
            
    def _get_num_frames(self) -> int:
        """Count number of frames in the sequence."""
        left_images = sorted(os.listdir(self.left_dir))
        return len([f for f in left_images if f.endswith('.png')])
    
    def load_stereo_pair(self, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a stereo image pair (left and right).
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            Tuple of (left_image, right_image) as grayscale numpy arrays
        """
        filename = f"{frame_idx:06d}.png"
        
        left_path = os.path.join(self.left_dir, filename)
        right_path = os.path.join(self.right_dir, filename)
        
        left_img = cv2.imread(left_path, cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_GRAYSCALE)
        
        if left_img is None:
            raise IOError(f"Failed to load left image: {left_path}")
        if right_img is None:
            raise IOError(f"Failed to load right image: {right_path}")
            
        return left_img, right_img
    
    def load_image(self, frame_idx: int, camera: int = 0) -> np.ndarray:
        """
        Load a single image from specified camera.
        
        Args:
            frame_idx: Frame index (0-based)
            camera: Camera index (0 = left, 1 = right)
            
        Returns:
            Grayscale image as numpy array
        """
        if camera == 0:
            img_dir = self.left_dir
        elif camera == 1:
            img_dir = self.right_dir
        else:
            raise ValueError(f"Invalid camera index: {camera}")
            
        filename = f"{frame_idx:06d}.png"
        img_path = os.path.join(img_dir, filename)
        
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Failed to load image: {img_path}")
            
        return img
    
    def get_image_shape(self) -> Tuple[int, int]:
        """Get the shape of images in the dataset."""
        left_img, _ = self.load_stereo_pair(0)
        return left_img.shape[::-1]
    
    def __len__(self) -> int:
        """Return number of frames in sequence."""
        return self.num_frames


def load_image_sequence(image_folder: str, 
                        start_frame: int = 0, 
                        end_frame: Optional[int] = None) -> List[np.ndarray]:
    """
    Load multiple images from a sequence.
    
    Args:
        image_folder: Path to image folder
        start_frame: Starting frame index
        end_frame: Ending frame index (None = all frames)
        
    Returns:
        List of grayscale images
    """
    loader = MonocularDatasetLoader(image_folder)
    
    if end_frame is None:
        end_frame = len(loader)
        
    frames = []
    for idx in range(start_frame, end_frame):
        frames.append(loader.load_image(idx))
        
    return frames
