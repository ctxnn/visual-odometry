# Code Documentation

This document explains how the visual odometry code works, module by module.

---

## Project Structure

```
project/
├── __init__.py              # Package initialization
├── dataset_loader.py         # Load images from disk
├── calibration.py            # Parse camera parameters
├── feature_detection.py      # Find keypoints in images
├── stereo_depth.py           # Compute depth from stereo
├── optical_flow.py           # Track features between frames
├── motion_estimation.py      # Estimate camera motion
├── trajectory.py             # Build camera path
├── visualization.py          # Plot results
├── monocular_vo.py           # Monocular VO (single camera)
└── main.py                   # Main entry point
```

---

## 1. Dataset Loader (`dataset_loader.py`)

### Purpose
Loads image sequences from the KITTI dataset format.

### Key Classes

#### `MonocularDatasetLoader`
For single-view images (your dataset):
```python
loader = MonocularDatasetLoader("path/to/images")
image = loader.load_image(frame_idx)      # Grayscale
image_color = loader.load_image_color(idx) # Color
```

#### `KITTI_StereoDatasetLoader` 
For stereo sequences:
```python
loader = KITTI_StereoDatasetLoader("sequences/00")
left_img, right_img = loader.load_stereo_pair(idx)
```

### How It Works
1. Scans folder for PNG/JPG files
2. Sorts by numeric filename (000071.png → frame 0)
3. Uses OpenCV to load images as grayscale arrays

---

## 2. Calibration (`calibration.py`)

### Purpose
Parses camera calibration parameters from `calib.txt`.

### Key Functions
```python
calib = parse_calibration("sequences/00/calib.txt")
params = get_stereo_parameters(calib)

# Returns:
# params['fx'], params['fy']   # Focal lengths
# params['cx'], params['cy']   # Principal point
# params['baseline']           # Camera baseline (meters)
# params['K']                   # 3x3 intrinsic matrix
```

### Calibration File Format
```
P0: 7.188560000000e+02 0.0 6.071928000000e+02 0.0
    0.0 7.188560000000e+02 1.852157000000e+02 0.0
    0.0 0.0 1.0 0.0
```

---

## 3. Feature Detection (`feature_detection.py`)

### Purpose
Detect distinctive points (corners) in images and compute descriptors.

### Key Classes

#### `FeatureDetector`
Uses Harris corners + ORB descriptors:
```python
detector = FeatureDetector(max_corners=500)
keypoints, descriptors = detector.detect_and_compute(image)
```

#### `FeatureMatcher`
Matches features between images:
```python
matcher = FeatureMatcher()
matches = matcher.match(descriptors1, descriptors2)
# Uses Hamming distance + Lowe's ratio test
```

### Pipeline
1. **Harris Corner Detection**: Finds points with high intensity change in all directions
2. **Non-Maximum Suppression**: Keeps well-separated corners
3. **ORB Descriptors**: Computes binary descriptors for matching

---

## 4. Stereo Depth (`stereo_depth.py`)

### Purpose
Compute depth from stereo image pairs.

### Key Class: `StereoDepthEstimator`

```python
estimator = StereoDepthEstimator(method='sgbm', num_disparities=128)

# Step 1: Compute disparity
disparity = estimator.compute_disparity(left_img, right_img)

# Step 2: Convert to depth
depth = estimator.compute_depth(disparity, focal_length, baseline)

# Step 3: Optional point cloud
points_3d, colors = estimator.compute_point_cloud(disparity, fx, baseline)
```

### Methods
- **SGBM (Semi-Global Block Matching)**: Matches blocks between images
- **Block Matching**: Faster but lower quality

### Formula Used
```
Depth = (focal_length × baseline) / disparity
```

---

## 5. Optical Flow (`optical_flow.py`)

### Purpose
Track features across consecutive frames.

### Key Class: `FeatureTracker`

```python
tracker = FeatureTracker(max_features=500)

# First frame
tracked_points, indices = tracker.track(image)

# Subsequent frames
# Automatically handles:
# - Feature tracking from previous frame
# - Detecting new features when needed
```

### KLT Algorithm
- Tracks using Lucas-Kanade method with image pyramids
- Filters out points with large displacement or high error

---

## 6. Motion Estimation (`motion_estimation.py`)

### Purpose
Estimate camera motion from feature correspondences.

### Key Class: `MotionEstimator`

```python
estimator = MotionEstimator()

# Estimate motion between two sets of points
R, t, mask, num_inliers = estimator.estimate_motion(
    points1,        # Nx2 array from frame 1
    points2,       # Nx2 array from frame 2
    camera_matrix
)
```

### Pipeline
1. **Find Fundamental Matrix** using RANSAC
2. **Convert to Essential Matrix** using camera intrinsics
3. **Decompose** to get rotation R and translation t
4. **Return inliers** for successful matches

---

## 7. Trajectory (`trajectory.py`)

### Purpose
Accumulate camera poses over time.

### Key Class: `VisualOdometry`

```python
vo = VisualOdometry(camera_matrix, baseline)

# Process each frame
R, t, num_inliers = vo.process_frame(image, depth, points)

# Get results
trajectory = vo.get_trajectory()  # Nx3 array
```

### What It Stores
- List of 4x4 transformation matrices
- 3D camera positions (X, Y, Z)
- Total distance traveled

---

## 8. Monocular VO (`monocular_vo.py`)

### Purpose
Visual odometry with single camera (no depth).

### Key Class: `MonocularVisualOdometry`

```python
vo = MonocularVisualOdometry(
    focal_length=700,
    principal_point=(640, 360),
    scale_factor=1.0
)

# Process each frame
R, t, inliers = vo.process_frame(image)

trajectory = vo.get_trajectory()
```

### Note on Scale
Monocular VO has scale ambiguity - translation is in arbitrary units. Use `scale_factor` to adjust.

---

## 9. Main Entry Point (`main.py`)

### Usage

**Monocular mode (your dataset):**
```bash
python main.py --mode mono --folder "../kitti dataset/train/img"
```

**Stereo mode (KITTI Odometry):**
```bash
python main.py --mode stereo --sequence "sequences/00"
```

### Arguments
| Argument | Description |
|----------|-------------|
| `--mode` | 'mono' or 'stereo' |
| `--folder` | Image folder (mono) |
| `--sequence` | Sequence folder (stereo) |
| `--start` | Starting frame |
| `--end` | Ending frame |
| `--stride` | Frame step |
| `--no-visualize` | Skip plotting |

---

## Data Flow

```
Input Images
    │
    ▼
┌─────────────────┐
│ dataset_loader  │  Load images
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Detect │  Find keypoints + descriptors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Matcher/OF    │  Match features / Track
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Motion Estimate │  Compute R, t
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Trajectory    │  Accumulate poses
└────────┬────────┘
         │
         ▼
    Output: 3D Trajectory
```

---

## Running the Code

### Step 1: Navigate to project
```bash
cd /Users/chiragtaneja/Codes/repos/visualodometry/project
```

### Step 2: Run with your dataset
```bash
python main.py --mode mono --folder "../kitti dataset/train/img" --start 0 --end 100
```

### Step 3: Output Files Generated

The program generates 4 output files in the output directory:

| File | Description | Format |
|------|-------------|--------|
| `trajectory.txt` | Camera position (X, Y, Z) | CSV |
| `motion_vectors.txt` | Rotation matrix + translation per frame | CSV |
| `trajectory_with_motion.csv` | Combined position + motion data | CSV |
| `point_cloud.ply` | 3D point cloud of environment | PLY |

#### Output Format Details

**trajectory.txt:**
```
x,y,z
0.000,0.000,0.000
0.038,0.048,0.998
-0.840,0.010,1.477
```

**motion_vectors.txt:**
```
frame,R11,R12,R13,R21,R22,R23,R31,R32,R33,tx,ty,tz
0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0
1,0.99,-0.01,0.0,0.01,0.99,0.0,0.0,0.0,1.0,0.5,0.1,2.3
```

**point_cloud.ply:**
```
ply
format ascii 1.0
element vertex 15000
property float x
property float y
property float z
end_header
-7.903055 -11.098827 30.230267
-4.060523 -6.885235 15.447642
...
```

### Step 4: Visualize (Optional)
- Plots shown in window (if not using `--no-visualize`)
