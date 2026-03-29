# Mathematics of Visual Odometry

This document explains the mathematical foundations of stereo visual odometry.

---

## 1. Camera Geometry

### 1.1 Pinhole Camera Model

A camera projects 3D world points onto a 2D image plane:

```
World Point P = [X, Y, Z]^T
Image Point p = [u, v]^T
```

The projection is given by:

```
[u]   [fx  0  cx] [X]   [X*fx/Z + cx]
[v] = [ 0 fy  cy] [Y] = [Y*fy/Z + cy]
[1]   [ 0  0   1] [Z]   [     1     ]
```

In matrix form: `p = K * [R|t] * P`

Where:
- `K` = Camera intrinsic matrix (focal lengths fx, fy; principal point cx, cy)
- `R` = Rotation matrix (3x3)
- `t` = Translation vector (3x1)

### 1.2 Stereo Camera Geometry

Two cameras separated by baseline `B`:

```
Left Camera:   P_L = [X_L, Y_L, Z_L]
Right Camera:  P_R = [X_R, Y_R, Z_R]

X_R = X_L - (B * fx / Z_L)
```

---

## 2. Disparity and Depth

### 2.1 Disparity

Disparity `d` is the horizontal offset between matching points in stereo images:

```
d = u_L - u_R
```

### 2.2 Depth from Disparity

Using similar triangles:

```
Z = (f * B) / d
```

Where:
- `Z` = Depth (distance to point)
- `f` = Focal length
- `B` = Stereo baseline (distance between cameras)
- `d` = Disparity in pixels

**Derivation:**

```
From similar triangles:
- (u_L - cx) / fx = X / Z
- (u_R - cx) / fx = X / Z

Subtracting:
(u_L - u_R) / fx = (X - X') / Z = B / Z

Therefore:
d / fx = B / Z
Z = (fx * B) / d
```

---

## 3. Feature Detection

### 3.1 Harris Corner Detection

Harris corners detect points with significant intensity changes in all directions.

**Compute the gradient:**

```
Ix = ∂I/∂x  (gradient in x)
Iy = ∂I/∂y  (gradient in y)
```

**Structure Matrix M:**

```
M = [ Ix^2    Ix*Iy ]
    [ Ix*Iy   Iy^2  ]
```

**Harris Response R:**

```
R = det(M) - k * trace(M)^2
  = (Ix^2 * Iy^2 - (Ix*Iy)^2) - k * (Ix^2 + Iy^2)^2
```

Where `k` is a constant (typically 0.04-0.06).

**Corner Classification:**
- R > threshold: Corner
- R < -threshold: Edge
- |R| < threshold: Flat

### 3.2 ORB Feature Descriptor

ORB (Oriented FAST and Rotated BRIEF):
- Uses FAST for keypoint detection
- Computes orientation using intensity centroid
- BRIEF descriptor with rotation compensation

---

## 4. Feature Matching

### 4.1 Brute Force Matcher

For each descriptor in set A, find the closest descriptor in set B using a distance metric.

**Hamming Distance (for ORB):**

```
d(a, b) = Σ (a_i XOR b_i)
```

Count of different bits.

### 4.2 Lowe's Ratio Test

Filter matches using the ratio of nearest to second-nearest neighbor:

```
if distance(matches[0]) < 0.75 * distance(matches[1]):
    keep_match()
```

This eliminates ambiguous matches.

---

## 5. Motion Estimation

### 5.1 Essential Matrix

The essential matrix `E` relates points in two camera views:

```
p2^T * E * p1 = 0
```

`E` encodes the relative rotation and translation:

```
E = [t]_x * R = skew(t) * R
```

Where `[t]_x` is the skew-symmetric matrix:

```
[t]_x = [  0  -tz   ty ]
        [  tz   0  -tx ]
        [ -ty  tx   0  ]
```

### 5.2 Estimating Essential Matrix (5-point Algorithm)

Using RANSAC for robustness:

1. Randomly select 5 point correspondences
2. Compute Essential matrix E
3. Count inliers (points satisfying p2^T * E * p1 ≈ 0)
4. Repeat N times, keep best E

**Inlier threshold:**
```
|p2^T * E * p1| < threshold
```

### 5.3 Recover Pose from Essential Matrix

Decompose E to get R and t:

1. **SVD of E:**
   ```
   E = U * S * V^T
   ```

2. **Two possible solutions for R:**
   ```
   R1 = U * W * V^T
   R2 = U * W^T * V^T
   
   Where W = [ 0 -1  0 ]
             [ 1  0  0 ]
             [ 0  0  1 ]
   ```

3. **Translation:**
   ```
   t = ±U(:, 2)
   ```

4. **Choose correct solution** using cheirality check (points must be in front of camera).

---

## 6. Triangulation

Given two views with known camera poses, find 3D point location.

### 6.1 Linear Triangulation

For point `p` in image 1 and `p'` in image 2:

```
x = P * X
x' = P' * X
```

Cross product gives:

```
[x]_x * P * X = 0
[x']_x * P' * X = 0
```

Solve using SVD to get X.

### 6.2 Direct Formula (Simplified)

For calibrated cameras with baseline B:

```
Z = (f * B) / d
X = (u - cx) * Z / f
Y = (v - cy) * Z / f
```

---

## 7. Trajectory Computation

### 7.1 Accumulating Transforms

Given relative motion between frames, accumulate to get global pose:

```
T_total = T_1 * T_2 * T_3 * ... * T_n
```

Where each T is a 4x4 transformation matrix:

```
T = [ R  t ]
    [ 0  1 ]
```

### 7.2 Camera Position

Camera position in world coordinates:

```
C = -R^T * t
```

---

## 8. Summary of Formulas

| Concept | Formula |
|---------|---------|
| Depth from disparity | Z = (f × B) / d |
| Essential matrix | E = [t]× × R |
| Point projection | p = K × [R\|t] × P |
| Harris response | R = det(M) - k×trace(M)² |
| Camera position | C = -R^T × t |
| Triangulation | Solve [x]×P×X = 0 |

---

## References

1. Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision
2. Nistér, D. (2004). An efficient solution to the five-point relative pose problem
3. Klein, G., & Murray, D. (2007). Parallel tracking and mapping for small AR workspaces
