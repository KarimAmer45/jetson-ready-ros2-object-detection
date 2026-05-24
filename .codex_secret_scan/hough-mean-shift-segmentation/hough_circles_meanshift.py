"""
Task 2: Hough Transform for Circle Detection
Task 3: Mean Shift for Peak Detection in Hough Accumulator
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os


def _nms2d(acc2d, thresh, min_dist):
    h, w = acc2d.shape
    peaks = []
    visited = np.zeros_like(acc2d, dtype=bool)

    ys, xs = np.where(acc2d >= thresh)
    vals = acc2d[ys, xs]
    order = np.argsort(-vals)

    for idx in order:
        y, x = ys[idx], xs[idx]
        if visited[y, x]:
            continue
        v = acc2d[y, x]
        peaks.append((x, y, v))

        y0, y1 = max(0, y - min_dist), min(h, y + min_dist + 1)
        x0, x1 = max(0, x - min_dist), min(w, x + min_dist + 1)
        visited[y0:y1, x0:x1] = True

    return peaks


def myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz):
    """
    Detect circles with a custom Hough accumulator.
    
    Args:
        edges: single-channel binary source image (e.g: edges)
        min_radius: minimum circle radius
        max_radius: maximum circle radius
        param threshold: minimum number of votes to consider a detection
        min_dist: minimum distance between two centers of the detected circles. 
        r_ssz: stepsize of r
        theta_ssz: stepsize of theta
        return: list of detected circles as (a, b, r, v), accumulator as [r, y_c, x_c]
    """
    assert edges.ndim == 2
    h, w = edges.shape
    max_radius = min(max_radius, int(np.floor(np.hypot(h, w))))

    # Discrete radii
    radii = np.arange(min_radius, max_radius + 1, r_ssz, dtype=int)
    n_r = len(radii)
    accumulator = np.zeros((n_r, h, w), dtype=np.uint16)

    # Edge points
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return [], accumulator

    # Precompute angles
    thetas = np.deg2rad(np.arange(0, 360, theta_ssz, dtype=float))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # Vote
    for ri, r in enumerate(radii):
        # parametric circle: a = x - r*cos(theta), b = y - r*sin(theta)
        for t_idx in range(len(thetas)):
            a = (xs - r * cos_t[t_idx]).round().astype(int)
            b = (ys - r * sin_t[t_idx]).round().astype(int)
            # keep in bounds
            mask = (a >= 0) & (a < w) & (b >= 0) & (b < h)
            a_in = a[mask]
            b_in = b[mask]
            accumulator[ri, b_in, a_in] += 1

    # Collect detections per radius with NMS
    detected = []
    for ri in range(n_r):
        peaks = _nms2d(accumulator[ri], threshold, min_dist)
        for (a, b, v) in peaks:
            detected.append((a, b, radii[ri], int(v)))

    # Sort by votes descending
    detected.sort(key=lambda t: -t[3])
    return detected, accumulator


def myMeanShift(accumulator, bandwidth, threshold=None):
    """
    Find peaks in Hough accumulator using mean shift.
    
    Args:
        accumulator: 3D Hough accumulator (n_radii, h, w)
        bandwidth: Bandwidth for mean shift in voxels (r,y,x)
        threshold: Minimum value to consider (if None, use 0.5 * max)
        
    Returns:
        peaks: List of (x, y, r_idx, value) tuples
    """
    n_r, h, w = accumulator.shape
    if threshold is None:
        threshold = 0.5 * float(accumulator.max())

    # Seed points: voxels above threshold
    r_idx, ys, xs = np.where(accumulator >= threshold)
    vals = accumulator[r_idx, ys, xs].astype(np.float32)

    if len(vals) == 0:
        return []

    seeds = np.stack([r_idx.astype(np.float32),
                      ys.astype(np.float32),
                      xs.astype(np.float32)], axis=1)

    def mean_shift_one(p0):
        p = p0.copy()
        # Iterate fixed small number for stability
        for _ in range(10):
            r0, y0, x0 = p
            r_min = int(max(0, np.floor(r0 - bandwidth)))
            r_max = int(min(n_r - 1, np.ceil(r0 + bandwidth)))
            y_min = int(max(0, np.floor(y0 - bandwidth)))
            y_max = int(min(h - 1, np.ceil(y0 + bandwidth)))
            x_min = int(max(0, np.floor(x0 - bandwidth)))
            x_max = int(min(w - 1, np.ceil(x0 + bandwidth)))

            cube = accumulator[r_min:r_max+1, y_min:y_max+1, x_min:x_max+1]
            if cube.size == 0:
                break
            # coordinates grid
            rr = np.arange(r_min, r_max+1, dtype=np.float32)[:, None, None]
            yy = np.arange(y_min, y_max+1, dtype=np.float32)[None, :, None]
            xx = np.arange(x_min, x_max+1, dtype=np.float32)[None, None, :]

            # weights = cube (flat kernel)
            wts = cube.astype(np.float32)
            total = float(wts.sum())
            if total <= 0:
                break
            r_new = (wts * rr).sum() / total
            y_new = (wts * yy).sum() / total
            x_new = (wts * xx).sum() / total

            shift = np.linalg.norm([r_new - r0, y_new - y0, x_new - x0])
            p = np.array([r_new, y_new, x_new], dtype=np.float32)
            if shift < 0.5:
                break
        val = accumulator[int(round(p[0])), int(round(p[1])), int(round(p[2]))]
        return p, float(val)

    # Run mean shift from top-K seeds to limit runtime
    K = min(200, len(seeds))
    top_idx = np.argsort(-vals)[:K]
    modes = []
    for idx in top_idx:
        mode, v = mean_shift_one(seeds[idx])
        modes.append((mode, v))

    # Merge modes within a small radius
    merged = []
    taken = [False] * len(modes)
    for i, (m_i, v_i) in enumerate(modes):
        if taken[i]:
            continue
        cluster = [i]
        for j, (m_j, v_j) in enumerate(modes[i+1:], start=i+1):
            if taken[j]:
                continue
            if np.linalg.norm(m_i - m_j) <= max(1.0, 0.75 * bandwidth):
                taken[j] = True
                cluster.append(j)
        # pick best in cluster
        best = max(cluster, key=lambda k: modes[k][1])
        taken[best] = True
        mr, my, mx = modes[best][0]
        mv = modes[best][1]
        merged.append((int(round(mx)), int(round(my)), int(round(mr)), float(mv)))

    # Unique by voxel
    uniq = []
    seen = set()
    for x, y, r_i, v in merged:
        key = (x, y, r_i)
        if key not in seen:
            seen.add(key)
            uniq.append((x, y, r_i, v))

    # Sort by value
    uniq.sort(key=lambda t: -t[3])
    return uniq


def main():
    
    print("=" * 70)
    print("Task 2: Hough Transform for Circle Detection")
    print("=" * 70)
        
    img_path = 'data/coins.jpg'
    
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found!")
        return
        
    # Load image and convert to grayscale
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"Error: failed to load {img_path}")
        return
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(img_gray, 80, 160)
    
    # Detect circles - parameters tuned for coins image
    print("\nDetecting circles...")
    min_radius = 30
    max_radius = 90
    threshold  = 25   # vote threshold per (r-slice)
    min_dist   = 12   # suppress close centers
    r_ssz      = 1    # radius step
    theta_ssz  = 1    # angular step in degrees
    
    detected_circles, accumulator = myHoughCircles(edges, min_radius, max_radius, threshold, min_dist, r_ssz, theta_ssz)

    # Visualize detected circles
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (a, b, r, v) in detected_circles[:20]:  # draw top 20
        cv2.circle(vis, (int(a), int(b)), int(r), (0, 255, 0), 2)
        cv2.circle(vis, (int(a), int(b)), 2, (0, 0, 255), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title("Detected circles (Hough)")
    plt.axis("off")
    plt.show()
    
    # Visualize accumulator slices
    n_r, h, w = accumulator.shape
    radii = np.arange(min_radius, max_radius + 1, r_ssz, dtype=int)
    sample_idx = [0, n_r//2, n_r-1]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, ri in zip(axes, sample_idx):
        ax.imshow(accumulator[ri], cmap="viridis")
        ax.set_title(f"Accumulator slice r={radii[ri]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    
    # Visualize peak radius slice
    # Find (r,y,x) with maximum vote
    ri_max, y_max, x_max = np.unravel_index(np.argmax(accumulator), accumulator.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow(accumulator[ri_max], cmap="viridis")
    plt.title(f"Accumulator at peak radius r={radii[ri_max]} (max votes)")
    plt.axis("off")
    plt.show()
    
    print("\n" + "=" * 70)
    print("Parameter Analysis:")
    print("  - Canny thresholds control edge density: too low → many spurious votes; too high → weak edges lost.")
    print("  - Radius range must cover the true coin sizes; narrower ranges or coarse steps (r_ssz) can miss small coins.")
    print("  - Finer theta steps (1–2°) improve circle continuity but slow down accumulation.")
    print("  - Lowering the Hough threshold increases sensitivity (detects faint or small coins) but may add false circles.")
    print("  - In the coins image, reducing threshold from 40→25 and extending radii to 10–90 improved recall of smaller coins.")
    print("=" * 70)
    print("=" * 70)
    print("Task 2 complete!")


    # =============================================================
    print("=" * 70)
    print("Task 3: Mean Shift for Peak Detection in Hough Accumulator")
    print("=" * 70)

    print("Applying mean shift to find peaks...")
    peaks = myMeanShift(accumulator, bandwidth=5, threshold=None)
    # Map r-index to radius
    radii = np.arange(min_radius, max_radius + 1, r_ssz, dtype=int)

    vis_ms = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for (x, y, r_i, v) in peaks[:20]:
        r = int(radii[r_i])
        cv2.circle(vis_ms, (int(x), int(y)), r, (255, 0, 0), 2)
        cv2.circle(vis_ms, (int(x), int(y)), 2, (0, 0, 255), 2)

    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(vis_ms, cv2.COLOR_BGR2RGB))
    plt.title("Circles from Mean Shift peaks on Hough accumulator")
    plt.axis("off")
    plt.show()
    
    print("\n" + "=" * 70)
    print("Bandwidth Parameter Analysis:")
    print("  - Too small bandwidth -> many nearby peaks (over-segmentation).")
    print("  - Too large bandwidth -> merged peaks, missed adjacent circles.")
    print("  - 3D bandwidth operates across (radius, y, x); choose wrt accumulator voxel scale.")
    print("=" * 70)
    print("Task 3 complete!")
    

if __name__ == "__main__":
    main()
