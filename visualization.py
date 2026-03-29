"""
Visualization Module
====================
Visualizes stereo visual odometry results.

Visualizations:
- Feature matches between stereo pairs
- Disparity and depth maps
- Camera trajectory (2D and 3D)
- 3D point cloud
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple


def show_stereo_pair(left_image: np.ndarray, 
                      right_image: np.ndarray,
                      window_name: str = "Stereo Pair") -> None:
    """
    Display stereo image pair side by side.
    
    Args:
        left_image: Left grayscale image
        right_image: Right grayscale image
        window_name: Window title
    """
    # Stack horizontally
    combined = np.hstack([left_image, right_image])
    
    cv2.imshow(window_name, combined)
    cv2.waitKey(1)


def show_matches(image1: np.ndarray,
                 image2: np.ndarray,
                 keypoints1: List,
                 keypoints2: List,
                 matches: List,
                 max_matches: int = 50,
                 window_name: str = "Feature Matches") -> None:
    """
    Display feature matches between two images.
    
    Args:
        image1: First image
        image2: Second image
        keypoints1: Keypoints from first image
        keypoints2: Keypoints from second image
        matches: List of matches
        max_matches: Maximum number to display
        window_name: Window title
    """
    # Convert to color if grayscale
    if len(image1.shape) == 2:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    if len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)
    
    # Draw matches
    result = cv2.drawMatches(
        image1, keypoints1, 
        image2, keypoints2,
        matches[:max_matches],
        None,
        matchColor=(0, 255, 0),
        singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    cv2.imshow(window_name, result)
    cv2.waitKey(1)


def plot_trajectory(trajectory: np.ndarray,
                    ground_truth: Optional[np.ndarray] = None,
                    title: str = "Camera Trajectory",
                    save_path: Optional[str] = None) -> None:
    """
    Plot camera trajectory in X-Z plane (top-down view).
    
    Args:
        trajectory: N x 3 array of camera positions
        ground_truth: Optional ground truth trajectory
        title: Plot title
        save_path: Optional path to save figure
    """
    plt.figure(figsize=(10, 8))
    
    # Plot estimated trajectory
    if len(trajectory) > 0:
        plt.plot(trajectory[:, 0], trajectory[:, 2], 
                 'b-', linewidth=2, label='Estimated')
        plt.scatter(trajectory[0, 0], trajectory[0, 2], 
                   c='green', s=100, marker='o', label='Start', zorder=5)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 2], 
                   c='red', s=100, marker='x', label='End', zorder=5)
    
    # Plot ground truth if provided
    if ground_truth is not None and len(ground_truth) > 0:
        plt.plot(ground_truth[:, 0], ground_truth[:, 2], 
                 'r--', linewidth=1, alpha=0.7, label='Ground Truth')
    
    plt.xlabel('X (m)', fontsize=12)
    plt.ylabel('Z (m)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_trajectory_3d(trajectory: np.ndarray,
                       ground_truth: Optional[np.ndarray] = None,
                       title: str = "3D Camera Trajectory",
                       save_path: Optional[str] = None) -> None:
    """
    Plot camera trajectory in 3D.
    
    Args:
        trajectory: N x 3 array of camera positions
        ground_truth: Optional ground truth trajectory
        title: Plot title
        save_path: Optional path to save figure
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot estimated trajectory
    if len(trajectory) > 0:
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                'b-', linewidth=2, label='Estimated')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], 
                  c='green', s=100, marker='o', label='Start', zorder=5)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                  c='red', s=100, marker='x', label='End', zorder=5)
    
    # Plot ground truth if provided
    if ground_truth is not None and len(ground_truth) > 0:
        ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2], 
                'r--', linewidth=1, alpha=0.7, label='Ground Truth')
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_point_cloud(points_3d: np.ndarray,
                     colors: Optional[np.ndarray] = None,
                     title: str = "3D Point Cloud",
                     subsample: int = 1,
                     save_path: Optional[str] = None) -> None:
    """
    Plot 3D point cloud.
    
    Args:
        points_3d: N x 3 array of 3D points
        colors: Optional N x 3 array of colors
        title: Plot title
        subsample: Subsample factor for visualization
        save_path: Optional path to save figure
    """
    if len(points_3d) == 0:
        print("No points to visualize")
        return
    
    # Subsample
    points = points_3d[::subsample]
    if colors is not None:
        colors = colors[::subsample]
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if colors is not None and len(colors) > 0:
        # Normalize colors
        colors_norm = colors / 255.0 if colors.max() > 1 else colors
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c=colors_norm, s=1, alpha=0.5)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                  c='blue', s=1, alpha=0.5)
    
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_box_aspect([1, 1, 1])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_disparity(disparity: np.ndarray, 
                         save_path: Optional[str] = None) -> None:
    """
    Visualize disparity map.
    
    Args:
        disparity: Disparity map
        save_path: Optional path to save figure
    """
    # Normalize
    disparity_norm = disparity.copy()
    disparity_norm = np.clip(disparity_norm, 0, None)
    
    if disparity_norm.max() > 0:
        disparity_norm = (disparity_norm / disparity_norm.max() * 255).astype(np.uint8)
    else:
        disparity_norm = np.zeros_like(disparity, dtype=np.uint8)
    
    # Apply colormap
    disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
    
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB))
    plt.title('Disparity Map')
    plt.colorbar(label='Disparity (pixels)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def visualize_depth(depth: np.ndarray,
                     save_path: Optional[str] = None) -> None:
    """
    Visualize depth map.
    
    Args:
        depth: Depth map in meters
        save_path: Optional path to save figure
    """
    # Clip and normalize
    depth_vis = depth.copy()
    depth_vis = np.clip(depth_vis, 0, 50)  # Clip to 50m
    
    plt.figure(figsize=(12, 6))
    plt.imshow(depth_vis, cmap='jet')
    plt.title('Depth Map')
    plt.colorbar(label='Depth (meters)')
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def plot_errors(trajectory: np.ndarray,
                ground_truth: np.ndarray,
                save_path: Optional[str] = None) -> None:
    """
    Plot position errors over time.
    
    Args:
        trajectory: Estimated trajectory
        ground_truth: Ground truth trajectory
        save_path: Optional path to save figure
    """
    if len(trajectory) != len(ground_truth):
        print("Trajectory lengths don't match")
        return
    
    # Compute errors
    errors = np.linalg.norm(trajectory - ground_truth, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(errors, 'b-', linewidth=1)
    plt.xlabel('Frame', fontsize=12)
    plt.ylabel('Position Error (m)', fontsize=12)
    plt.title('Position Error Over Time')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_trajectory_video(trajectory: np.ndarray,
                          output_path: str,
                          fps: int = 30) -> None:
    """
    Save trajectory as video.
    
    Args:
        trajectory: N x 3 array of positions
        output_path: Output video path
        fps: Frames per second
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (800, 600))
    
    for i in range(len(trajectory)):
        ax.clear()
        ax.plot(trajectory[:i+1, 0], trajectory[:i+1, 2], 'b-', linewidth=2)
        ax.scatter(trajectory[i, 0], trajectory[i, 2], c='red', s=50)
        ax.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=50)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Z (m)')
        ax.set_title(f'Frame {i}')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Save frame
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    plt.close()


def create_comparison_plot(trajectory: np.ndarray,
                           ground_truth: np.ndarray,
                           save_path: Optional[str] = None) -> None:
    """
    Create comparison plot between estimated and ground truth trajectory.
    
    Args:
        trajectory: Estimated trajectory
        ground_truth: Ground truth trajectory
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # X-Z plane (top view)
    axes[0].plot(trajectory[:, 0], trajectory[:, 2], 'b-', label='Estimated', linewidth=2)
    axes[0].plot(ground_truth[:, 0], ground_truth[:, 2], 'r--', label='Ground Truth', linewidth=1)
    axes[0].scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=100, marker='o', zorder=5)
    axes[0].set_xlabel('X (m)')
    axes[0].set_ylabel('Z (m)')
    axes[0].set_title('Top View (X-Z)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].axis('equal')
    
    # X-Y plane (side view)
    axes[1].plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Estimated', linewidth=2)
    axes[1].plot(ground_truth[:, 0], ground_truth[:, 1], 'r--', label='Ground Truth', linewidth=1)
    axes[1].set_xlabel('X (m)')
    axes[1].set_ylabel('Y (m)')
    axes[1].set_title('Side View (X-Y)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].axis('equal')
    
    # Z-Y plane (front view)
    axes[2].plot(trajectory[:, 2], trajectory[:, 1], 'b-', label='Estimated', linewidth=2)
    axes[2].plot(ground_truth[:, 2], ground_truth[:, 1], 'r--', label='Ground Truth', linewidth=1)
    axes[2].set_xlabel('Z (m)')
    axes[2].set_ylabel('Y (m)')
    axes[2].set_title('Front View (Z-Y)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].axis('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()


def save_point_cloud_ply(points_3d: np.ndarray,
                         colors: Optional[np.ndarray] = None,
                         filepath: str = "point_cloud.ply") -> None:
    """
    Save 3D point cloud to PLY format.
    
    PLY (Polygon File Format) is a standard format for 3D data
    that can be opened in MeshLab, CloudCompare, Blender, etc.
    
    Args:
        points_3d: N x 3 array of 3D points
        colors: Optional N x 3 array of RGB colors (0-255)
        filepath: Output file path
    """
    if len(points_3d) == 0:
        print("No points to save")
        return
    
    # Subsample if too many points (for performance)
    max_points = 500000
    if len(points_3d) > max_points:
        print(f"Subsampling from {len(points_3d)} to {max_points} points")
        indices = np.random.choice(len(points_3d), max_points, replace=False)
        points_3d = points_3d[indices]
        if colors is not None:
            colors = colors[indices]
    
    num_points = len(points_3d)
    
    with open(filepath, 'w') as f:
        # Write PLY header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {num_points}\n")
        
        if colors is not None and len(colors) > 0:
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        else:
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
        
        f.write("end_header\n")
        
        # Write vertices
        for i in range(num_points):
            x, y, z = points_3d[i]
            if colors is not None and len(colors) > 0:
                r, g, b = colors[i] if len(colors.shape) > 1 else (255, 255, 255)
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
            else:
                f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    
    print(f"Point cloud saved to: {filepath}")
    print(f"Total points: {num_points}")


def save_point_cloud_csv(points_3d: np.ndarray,
                         colors: Optional[np.ndarray] = None,
                         filepath: str = "point_cloud.csv") -> None:
    """
    Save 3D point cloud to CSV format.
    
    Simple CSV format with x, y, z coordinates and optional RGB.
    
    Args:
        points_3d: N x 3 array of 3D points
        colors: Optional N x 3 array of RGB colors
        filepath: Output file path
    """
    if len(points_3d) == 0:
        print("No points to save")
        return
    
    with open(filepath, 'w') as f:
        # Header
        if colors is not None and len(colors) > 0:
            f.write("x,y,z,r,g,b\n")
        else:
            f.write("x,y,z\n")
        
        # Write points
        for i in range(len(points_3d)):
            x, y, z = points_3d[i]
            if colors is not None and len(colors) > 0:
                r, g, b = colors[i] if len(colors.shape) > 1 else (255, 255, 255)
                f.write(f"{x:.6f},{y:.6f},{z:.6f},{int(r)},{int(g)},{int(b)}\n")
            else:
                f.write(f"{x:.6f},{y:.6f},{z:.6f}\n")
    
    print(f"Point cloud saved to: {filepath}")
    print(f"Total points: {len(points_3d)}")


def save_motion_vectors(rotations: List[np.ndarray],
                        translations: List[np.ndarray],
                        filepath: str = "motion_vectors.txt") -> None:
    """
    Save camera motion vectors to text file.
    
    Output format:
    frame,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz
    
    Where:
    - R11..R33 are the 3x3 rotation matrix elements
    - tx,ty,tz are the translation vector components
    
    Args:
        rotations: List of 3x3 rotation matrices
        translations: List of 1x3 or 3x1 translation vectors
        filepath: Output file path
    """
    if len(rotations) == 0:
        print("No motion data to save")
        return
    
    with open(filepath, 'w') as f:
        # Header
        f.write("frame,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz\n")
        
        # Write each frame
        for i in range(len(rotations)):
            R = rotations[i]
            t = translations[i]
            
            # Flatten rotation matrix
            if t.shape != (3,):
                t = t.flatten()
            
            # Format: frame,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz
            row = f"{i},{R[0,0]:.8f},{R[0,1]:.8f},{R[0,2]:.8f}," \
                  f"{R[1,0]:.8f},{R[1,1]:.8f},{R[1,2]:.8f}," \
                  f"{R[2,0]:.8f},{R[2,1]:.8f},{R[2,2]:.8f}," \
                  f"{t[0]:.8f},{t[1]:.8f},{t[2]:.8f}\n"
            
            f.write(row)
    
    print(f"Motion vectors saved to: {filepath}")
    print(f"Total frames: {len(rotations)}")


def save_trajectory_with_motion(trajectory: np.ndarray,
                                 rotations: List[np.ndarray],
                                 translations: List[np.ndarray],
                                 filepath: str = "trajectory_with_motion.csv") -> None:
    """
    Save complete trajectory with motion data.
    
    Output format:
    frame,x,y,z,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz
    
    Args:
        trajectory: N x 3 array of camera positions
        rotations: List of 3x3 rotation matrices
        translations: List of translation vectors
        filepath: Output file path
    """
    if len(trajectory) == 0:
        print("No trajectory data to save")
        return
    
    with open(filepath, 'w') as f:
        # Header
        f.write("frame,x,y,z,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz\n")
        
        # Write each frame
        for i in range(len(trajectory)):
            x, y, z = trajectory[i]
            
            R = rotations[i] if i < len(rotations) else np.eye(3)
            t = translations[i] if i < len(translations) else np.zeros(3)
            
            if t.shape != (3,):
                t = t.flatten()
            
            row = f"{i},{x:.8f},{y:.8f},{z:.8f}," \
                  f"{R[0,0]:.8f},{R[0,1]:.8f},{R[0,2]:.8f}," \
                  f"{R[1,0]:.8f},{R[1,1]:.8f},{R[1,2]:.8f}," \
                  f"{R[2,0]:.8f},{R[2,1]:.8f},{R[2,2]:.8f}," \
                  f"{t[0]:.8f},{t[1]:.8f},{t[2]:.8f}\n"
            
            f.write(row)
    
    print(f"Trajectory with motion saved to: {filepath}")
    print(f"Total frames: {len(trajectory)}")
