
# Template for Exercise 5 – Canny Edge Detector

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


def _gaussian_kernel_1d(sigma):
    # Choose size as in OpenCV: k = 2*ceil(3*sigma)+1 (minimum 3)
    k = int(2 * np.ceil(3 * max(sigma, 1e-6)) + 1)
    if k % 2 == 0:
        k += 1
    radius = k // 2
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel


def gaussian_smoothing(img, sigma):
    """
    Apply Gaussian smoothing to reduce noise.
    """
    if sigma <= 0:
        return img.astype(np.float32)
    # separable convolution to match common implementations
    k = _gaussian_kernel_1d(sigma)
    imgf = img.astype(np.float32)
    # horizontal
    tmp = cv2.filter2D(imgf, -1, k.reshape(1, -1), borderType=cv2.BORDER_REPLICATE)
    # vertical
    smoothed = cv2.filter2D(tmp, -1, k.reshape(-1, 1), borderType=cv2.BORDER_REPLICATE)
    return smoothed


def compute_gradients(img):
    """
    Compute gradient magnitude and direction (Sobel-based).
    Return gradient_magnitude, gradient_angle.
    """
    # Sobel with ksize=3, same as default apertureSize in Canny
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_DEFAULT)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_DEFAULT)
    # Magnitude with L1 norm to mimic default Canny (L2gradient=False)
    mag = np.abs(gx) + np.abs(gy)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0,180)
    return mag, ang


def _quantize_angles(ang):
    """Map gradient angle to 4 principal directions: 0,45,90,135 (indices 0..3)."""
    q = np.zeros_like(ang, dtype=np.uint8)
    # Define bins centered at 0,45,90,135 with half-width 22.5
    q[(ang >= 22.5) & (ang < 67.5)] = 1   # 45
    q[(ang >= 67.5) & (ang < 112.5)] = 2  # 90
    q[(ang >= 112.5) & (ang < 157.5)] = 3 # 135
    # 0 already default for [157.5..180) and [0..22.5)
    return q


def nonmax_suppression(mag, ang):
    """
    Perform non-maximum suppression to thin edges.
    """
    H, W = mag.shape
    out = np.zeros((H, W), dtype=np.float32)
    q = _quantize_angles(ang)

    for y in range(1, H - 1):
        # vectorize across x for speed using neighbors per orientation
        x = np.arange(1, W - 1)

        # neighbors for each direction
        # 0 deg -> compare with left/right
        m = mag[y, x]
        # initialize neighbors as zeros
        n1 = np.zeros_like(m)
        n2 = np.zeros_like(m)

        # 0°
        mask0 = (q[y, x] == 0)
        n1[mask0] = mag[y, x - 1][mask0]
        n2[mask0] = mag[y, x + 1][mask0]

        # 45°
        mask1 = (q[y, x] == 1)
        n1[mask1] = mag[y - 1, x + 1][mask1]
        n2[mask1] = mag[y + 1, x - 1][mask1]

        # 90°
        mask2 = (q[y, x] == 2)
        n1[mask2] = mag[y - 1, x][mask2]
        n2[mask2] = mag[y + 1, x][mask2]

        # 135°
        mask3 = (q[y, x] == 3)
        n1[mask3] = mag[y - 1, x - 1][mask3]
        n2[mask3] = mag[y + 1, x + 1][mask3]

        keep = (m >= n1) & (m >= n2)
        out[y, x] = m * keep
    return out


def double_threshold(nms, low, high):
    """
    Apply double thresholding to classify strong, weak, and non-edges.
    Return thresholded edge map.
    """
    strong_val = 255
    weak_val = 75

    strong = (nms >= high).astype(np.uint8) * strong_val
    weak = ((nms >= low) & (nms < high)).astype(np.uint8) * weak_val
    return (strong + weak).astype(np.uint8), weak_val, strong_val


def hysteresis(edge_map, weak, strong):
    """
    Perform edge tracking by hysteresis.
    Return final binary edge map.
    """
    H, W = edge_map.shape
    out = edge_map.copy()

    # BFS from strong edges; promote connected weak pixels to strong
    q = deque()
    strong_coords = np.argwhere(out == strong)
    for y, x in strong_coords:
        q.append((y, x))

    # 8-connectivity
    nbrs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W:
                if out[ny, nx] == weak:
                    out[ny, nx] = strong
                    q.append((ny, nx))

    # Suppress anything that is not strong
    out[out != strong] = 0
    return out


def compute_metrics(manual_edges, cv_edges):
    """
    Compute MAD, precision, recall, and F1-score between two binary edge maps.
    """
    # Normalize to {0,1}
    m = (manual_edges > 0).astype(np.uint8)
    c = (cv_edges > 0).astype(np.uint8)

    mad = np.mean(np.abs(m.astype(np.float32) - c.astype(np.float32)))

    tp = np.sum((m == 1) & (c == 1))
    fp = np.sum((m == 1) & (c == 0))
    fn = np.sum((m == 0) & (c == 1))

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return mad, precision, recall, f1


# ==========================================================

if __name__ == "__main__":
    # 1. Load the grayscale image 'bonn.jpg'
    img = cv2.imread("data/bonn.jpg", cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Could not load 'data/bonn.jpg'. Ensure the image exists.")

    # 2. Smooth the image using your Gaussian function
    # A common Canny pre-smoothing is sigma ~ 1.0
    smoothed = gaussian_smoothing(img, sigma=1.0)

    # 3. Compute gradients (magnitude and direction)
    mag, ang = compute_gradients(smoothed)

    # 4. Apply non-maximum suppression
    nms = nonmax_suppression(mag, ang)

    # 5. Apply double threshold (choose suitable low/high values)
    # We choose thresholds relative to the distribution of NMS magnitudes to be robust.
    # Use high at 90th percentile and low as 40% of high; this pairs well with OpenCV's use below.
    high_thresh = np.percentile(nms[nms > 0], 90) if np.any(nms > 0) else 0.0
    low_thresh = 0.4 * high_thresh
    thresh_map, WEAK, STRONG = double_threshold(nms, low_thresh, high_thresh)

    # 6. Perform hysteresis to obtain final edges
    manual_edges = hysteresis(thresh_map, WEAK, STRONG)

    # 7. Compare your result with cv2.Canny using MAD and F1-score
    # Use the same thresholds (scaled to 0..255 range) for fair comparison.
    # OpenCV Canny thresholds expect 8-bit gradients; align with our relative thresholds.
    # Convert our percentiles to absolute thresholds on the (pre-smoothed) image using Canny.
    # We emulate by passing the same numeric low/high as integers.
    low_cv = int(max(0, min(255, low_thresh)))
    high_cv = int(max(0, min(255, high_thresh)))
    if high_cv <= low_cv:
        high_cv = min(255, low_cv + 1)
    cv_edges = cv2.Canny(smoothed.astype(np.uint8), threshold1=low_cv, threshold2=high_cv, L2gradient=False)

    mad, precision, recall, f1 = compute_metrics(manual_edges, cv_edges)

    print(f"MAD: {mad:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if mad > 0.07 or f1 < 0.6:
        print("Warning: Current thresholds may not meet the targets (MAD ≤ 0.07 and F1 ≥ 0.6). "
              "You can tune the percentile or the low/high ratio above if needed.")

    # 8. Display original image, your edges, and OpenCV edges
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.imshow(img, cmap='gray'); plt.title('Original'); plt.axis('off')
    plt.subplot(1, 3, 2); plt.imshow(manual_edges, cmap='gray'); plt.title('Manual Canny'); plt.axis('off')
    plt.subplot(1, 3, 3); plt.imshow(cv_edges, cmap='gray'); plt.title('OpenCV Canny'); plt.axis('off')
    plt.tight_layout()
    plt.show()
