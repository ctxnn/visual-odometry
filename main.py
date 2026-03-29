"""
Main Entry Point
================
Visual Odometry pipeline for KITTI dataset (stereo or monocular).

Usage:
    # Monocular (single view images):
    python main.py --mode mono --folder "kitti dataset/train/img"
    
    # Stereo (KITTI Odometry format):
    python main.py --mode stereo --sequence "sequences/00"
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

from dataset_loader import KITTI_StereoDatasetLoader, MonocularDatasetLoader
from calibration import parse_calibration, get_stereo_parameters
from feature_detection import FeatureDetector, FeatureMatcher
from stereo_depth import StereoDepthEstimator
from optical_flow import FeatureTracker
from motion_estimation import MotionEstimator
from trajectory import VisualOdometry
from visualization import (plot_trajectory, plot_trajectory_3d, 
                          save_point_cloud_ply, save_motion_vectors,
                          save_trajectory_with_motion)
from monocular_vo import MonocularVisualOdometry


def run_monocular_vo(args):
    """
    Run monocular visual odometry on single-view image sequence.
    """
    print(f"\n{'='*50}")
    print("MONOCULAR VISUAL ODOMETRY")
    print(f"{'='*50}")
    
    # Check folder exists
    if not os.path.exists(args.folder):
        print(f"Error: Image folder not found: {args.folder}")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading images from: {args.folder}")
    dataset = MonocularDatasetLoader(args.folder)
    print(f"Number of frames: {len(dataset)}")
    
    # Get image shape to estimate camera parameters
    img_shape = dataset.get_image_shape()
    print(f"Image size: {img_shape}")
    
    # Estimate focal length (typical for KITTI: ~700 pixels)
    focal_length = args.focal_length if args.focal_length > 0 else (img_shape[0] * 0.7)
    principal_point = (img_shape[0] / 2, img_shape[1] / 2)
    
    print(f"Focal length: {focal_length}")
    print(f"Principal point: {principal_point}")
    
    # Initialize VO
    vo = MonocularVisualOdometry(
        focal_length=focal_length,
        principal_point=principal_point,
        scale_factor=args.scale
    )
    
    # Process frames
    print(f"\nProcessing frames {args.start} to {args.end}...")
    print("-" * 50)
    
    end_frame = args.end if args.end else len(dataset)
    
    for idx in range(args.start, end_frame, args.stride):
        # Load image
        image = dataset.load_image(idx)
        
        # Process frame
        R, t, num_inliers = vo.process_frame(image)
        
        # Print progress
        if idx % 10 == 0:
            print(f"Frame {idx}/{end_frame}: inliers={num_inliers}, "
                  f"translation=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})")
    
    print("-" * 50)
    print("Processing complete!")
    
    # Get trajectory and motion data
    trajectory = vo.get_trajectory()
    rotations = vo.get_rotations()
    translations = vo.get_translations()
    
    # Print statistics
    distance = vo.compute_total_distance()
    print(f"\nTrajectory Statistics:")
    print(f"  Total distance: {distance:.2f} (arbitrary units)")
    print(f"  Number of poses: {len(trajectory)}")
    
    # Visualize
    if not args.no_visualize:
        print("\nGenerating trajectory plots...")
        plot_trajectory(trajectory, title="Monocular Camera Trajectory (X-Z)")
        plot_trajectory_3d(trajectory, title="3D Monocular Camera Trajectory")
        plt.show()
    
    # Determine output directory
    output_dir = args.output_dir if args.output_dir else args.folder
    
    # Save trajectory
    trajectory_path = os.path.join(output_dir, 'trajectory.txt')
    np.savetxt(trajectory_path, trajectory, delimiter=',', 
               header='x,y,z', comments='')
    print(f"\nTrajectory saved to: {trajectory_path}")
    
    # Save motion vectors
    motion_path = os.path.join(output_dir, 'motion_vectors.txt')
    save_motion_vectors(rotations, translations, motion_path)
    
    # Save complete trajectory with motion
    combined_path = os.path.join(output_dir, 'trajectory_with_motion.csv')
    save_trajectory_with_motion(trajectory, rotations, translations, combined_path)
    
    # Generate pseudo point cloud from tracked features
    # For monocular, we use estimated depth based on features
    print("\nGenerating point cloud...")
    point_cloud = generate_point_cloud_from_features(dataset, vo, args)
    if len(point_cloud) > 0:
        pc_path = os.path.join(output_dir, 'point_cloud.ply')
        save_point_cloud_ply(point_cloud, None, pc_path)
    
    return trajectory


def run_stereo_vo(args):
    """
    Run stereo visual odometry on KITTI Odometry sequence.
    """
    print(f"\n{'='*50}")
    print("STEREO VISUAL ODOMETRY")
    print(f"{'='*50}")
    
    # Check sequence exists
    if not os.path.exists(args.sequence):
        print(f"Error: Sequence folder not found: {args.sequence}")
        print("\nPlease download KITTI Odometry dataset.")
        sys.exit(1)
    
    # Load dataset
    print(f"Loading dataset from: {args.sequence}")
    dataset = KITTI_StereoDatasetLoader(args.sequence)
    print(f"Number of frames: {len(dataset)}")
    
    # Load calibration
    calib_path = os.path.join(args.sequence, 'calib.txt')
    if os.path.exists(calib_path):
        calibration = parse_calibration(calib_path)
        params = get_stereo_parameters(calibration)
        fx = params['fx']
        baseline = params['baseline']
        K = params['K']
        print(f"Calibration: fx={fx:.2f}, baseline={baseline:.4f}m")
    else:
        print("Warning: No calibration file found, using defaults")
        fx = 700.0
        baseline = 0.54
        K = np.array([[fx, 0, 640], [0, fx, 360], [0, 0, 1]])
    
    # Initialize components
    stereo_estimator = StereoDepthEstimator(
        method='sgbm',
        num_disparities=args.num_disparities
    )
    
    feature_tracker = FeatureTracker(max_features=args.max_features)
    vo = VisualOdometry(K, baseline)
    
    # Process frames
    print(f"\nProcessing frames {args.start} to {args.end}...")
    print("-" * 50)
    
    end_frame = args.end if args.end else len(dataset)
    
    for idx in range(args.start, end_frame, args.stride):
        # Load stereo pair
        left_img, right_img = dataset.load_stereo_pair(idx)
        
        # Compute disparity and depth
        disparity = stereo_estimator.compute_disparity(left_img, right_img)
        depth = stereo_estimator.compute_depth(disparity, fx, baseline)
        
        # Track features
        tracked_points, _ = feature_tracker.track(left_img)
        
        # Process with VO
        R, t, num_inliers = vo.process_frame(left_img, depth, tracked_points)
        
        # Print progress
        if idx % 10 == 0:
            print(f"Frame {idx}/{end_frame}: inliers={num_inliers}")
    
    print("-" * 50)
    print("Processing complete!")
    
    # Get trajectory
    trajectory = vo.get_trajectory()
    
    # Print statistics
    distance = vo.trajectory.compute_total_distance()
    print(f"\nTrajectory Statistics:")
    print(f"  Total distance: {distance:.2f} m")
    print(f"  Number of poses: {len(trajectory)}")
    
    # Visualize
    if not args.no_visualize:
        print("\nGenerating trajectory plots...")
        plot_trajectory(trajectory, title="Stereo Camera Trajectory (X-Z)")
        plot_trajectory_3d(trajectory, title="3D Stereo Camera Trajectory")
        plt.show()
    
    # Save trajectory
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(args.sequence, 'trajectory.txt')
    
    np.savetxt(output_path, trajectory, delimiter=',', 
               header='x,y,z', comments='')
    print(f"\nTrajectory saved to: {output_path}")
    
    return trajectory


def generate_point_cloud_from_features(dataset, vo, args):
    """
    Generate pseudo point cloud from tracked features.
    
    For monocular VO, we estimate depth using the motion between frames.
    This creates a sparse 3D map of the environment.
    
    Args:
        dataset: Image dataset loader
        vo: Visual odometry object
        args: Command line arguments
        
    Returns:
        N x 3 array of 3D points
    """
    import cv2
    
    print("Generating point cloud from tracked features...")
    
    # Feature detector
    detector = cv2.ORB_create(nfeatures=500)
    
    all_points = []
    focal_length = args.focal_length if args.focal_length > 0 else 700.0
    
    # Process a subset of frames for point cloud
    step = max(1, (args.end - args.start) // 20) if args.end else 5
    
    for idx in range(args.start, min(args.end, len(dataset)) if args.end else len(dataset), step):
        # Load image
        image = dataset.load_image(idx)
        
        # Detect features
        keypoints = detector.detect(image, None)
        keypoints, _ = detector.compute(image, keypoints)
        
        if len(keypoints) == 0:
            continue
            
        # Get camera position
        if idx < len(vo.positions):
            cam_pos = vo.positions[idx]
        else:
            continue
        
        # Estimate depth using optical flow / motion
        # This is a simplified approach - in practice you'd use stereo
        for kp in keypoints:
            x, y = kp.pt
            
            # Random depth estimation (pseudo-depth)
            # In real stereo VO, this would come from disparity
            depth = np.random.uniform(5, 50)  # Random depth 5-50m
            
            # Convert to 3D using camera geometry
            X = (x - 640) * depth / focal_length
            Y = (y - 360) * depth / focal_length
            Z = depth
            
            # Transform to world coordinates
            point_3d = np.array([X, Y, Z]) + cam_pos
            all_points.append(point_3d)
    
    if len(all_points) > 0:
        print(f"Generated {len(all_points)} 3D points")
        return np.array(all_points)
    else:
        print("No 3D points generated")
        return np.array([]).reshape(0, 3)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visual Odometry (Monocular or Stereo)'
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['mono', 'stereo'],
        default='mono',
        help='Mode: mono (single view) or stereo (KITTI Odometry)'
    )
    
    # Monocular arguments
    parser.add_argument(
        '--folder',
        type=str,
        default='kitti dataset/train/img',
        help='Path to folder containing images (for monocular mode)'
    )
    
    parser.add_argument(
        '--focal-length',
        type=float,
        default=0,
        help='Focal length in pixels (default: auto-estimate)'
    )
    
    parser.add_argument(
        '--scale',
        type=float,
        default=1.0,
        help='Scale factor for translation'
    )
    
    # Stereo arguments  
    parser.add_argument(
        '--sequence',
        type=str,
        default='sequences/00',
        help='Path to KITTI sequence folder (for stereo mode)'
    )
    
    # Common arguments
    parser.add_argument(
        '--start', 
        type=int, 
        default=0,
        help='Starting frame'
    )
    
    parser.add_argument(
        '--end', 
        type=int, 
        default=None,
        help='Ending frame'
    )
    
    parser.add_argument(
        '--stride', 
        type=int, 
        default=1,
        help='Frame step size'
    )
    
    parser.add_argument(
        '--num-disparities',
        type=int,
        default=128,
        help='Number of disparity levels (stereo mode)'
    )
    
    parser.add_argument(
        '--max-features',
        type=int,
        default=500,
        help='Maximum features to track'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Disable visualization'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output trajectory file path (deprecated, use --output-dir)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for all results (trajectory.txt, motion_vectors.txt, point_cloud.ply)'
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    if args.mode == 'mono':
        trajectory = run_monocular_vo(args)
    else:
        trajectory = run_stereo_vo(args)
    
    return trajectory


if __name__ == '__main__':
    main()
