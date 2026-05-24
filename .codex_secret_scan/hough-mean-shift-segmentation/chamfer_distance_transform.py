"""
Task 1: Distance Transform using Chamfer 5-7-11
Template for MA-INF 2201 Computer Vision WS25/26
Exercise 03
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


def chamfer_distance_transform_5_7_11(binary_image):
    """
    Compute Chamfer distance transform using 5-7-11 mask.
    
    Based on Borgefors "Distance transformations in digital images" (1986).
    
    Chamfer 5-7-11:
    - Horizontal/vertical neighbors: weight = 5
    - Diagonal neighbors: weight = 7
    - Knight's move neighbors: weight = 11
    
    Args:
        binary_image: Binary image where features are 255, background is 0
    
    Returns:
        Distance transform image
    """
    assert binary_image.ndim == 2
    h, w = binary_image.shape
    # Initialize: 0 if feature pixel, infinity otherwise
    dt = np.full((h, w), np.inf, dtype=np.float32)
    dt[binary_image > 0] = 0.0

    # Neighborhoods with weights
    hv = 5
    dg = 7
    kn = 11

    # Define forward and backward masks with (row_offset, col_offset, distance)
    # Forward mask (as shown in slide 37)
    forward_offsets = [
        (-1,  0, hv), (0, -1, hv),
        (-1, -1, dg), (-1,  1, dg),
        (-2, -1, kn), (-1, -2, kn),
        (-2,  1, kn), (-1,  2, kn),
    ]

    # Backward mask (as shown in slide 37)
    backward_offsets = [
        ( 1,  0, hv), (0,  1, hv),
        ( 1,  1, dg), ( 1, -1, dg),
        ( 2,  1, kn), ( 1,  2, kn),
        ( 2, -1, kn), ( 1, -2, kn),
    ]

    # Forward pass
    for r in range(h):
        for c in range(w):
            d0 = dt[r, c]
            if d0 == 0:
                continue
            best = d0
            for dr, dc, wgt in forward_offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    cand = dt[rr, cc] + wgt
                    if cand < best:
                        best = cand
            dt[r, c] = best

    # Backward pass
    for r in range(h - 1, -1, -1):
        for c in range(w - 1, -1, -1):
            d0 = dt[r, c]
            best = d0
            for dr, dc, wgt in backward_offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    cand = dt[rr, cc] + wgt
                    if cand < best:
                        best = cand
            dt[r, c] = best

    return dt


def main():    
    
    print("=" * 70)
    print("Task 1: Distance Transform using Chamfer 5-7-11")
    print("=" * 70)
    
    img_path = 'data/bonn.jpg'
    # img_path = 'data/circle.png'      # play with different images
    # img_path = 'data/square.png'      
    # img_path = 'data/triangle.png'    
    
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
    edges = cv2.Canny(img_gray, 100, 200)  # reasonable defaults; can be tuned
    
    # Compute distance transform with the function chamfer_distance_transform_5_7_11
    chamfer_dt = chamfer_distance_transform_5_7_11(edges)
    chamfer_dt_disp = chamfer_dt / max(1.0, chamfer_dt.max())  # normalize for display
    
    # Compute distance transform using cv2.distanceTransform
    # cv2.distanceTransform computes distance to the nearest zero pixel.
    # We want distance to edges (non-zero). So we invert: set edges as zeros.
    inv = (edges == 0).astype(np.uint8) * 255
    opencv_dt = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    opencv_dt_disp = opencv_dt / max(1e-6, opencv_dt.max())
    
    # Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original")
    axes[0].axis("off")
    
    axes[1].imshow(edges, cmap="gray")
    axes[1].set_title("Canny edges")
    axes[1].axis("off")
    
    im3 = axes[2].imshow(chamfer_dt_disp, cmap="hot")
    axes[2].set_title("Chamfer 5-7-11 (normalized)")
    axes[2].axis("off")
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    im4 = axes[3].imshow(opencv_dt_disp, cmap="hot")
    axes[3].set_title("OpenCV distanceTransform (normalized)")
    axes[3].axis("off")
    fig.colorbar(im4, ax=axes[3], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.show()
        
   

if __name__ == "__main__":
    main()
