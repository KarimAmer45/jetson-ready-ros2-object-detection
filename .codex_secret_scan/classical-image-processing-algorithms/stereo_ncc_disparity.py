# Template for Exercise 4 – NCC Stereo Matching (Solved + Annotated)

import cv2
import numpy as np
import matplotlib.pyplot as plt


WINDOW_SIZE = 11       # NCC patch size
MAX_DISPARITY = 64     # Maximum search range


def compute_manual_ncc_map(left_image, right_image, window_size, max_disparity):
    """
    Compute a dense disparity map using Normalized Cross-Correlation (NCC)
    """
    h, w = left_image.shape
    pad = window_size // 2
    left_padded = cv2.copyMakeBorder(left_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    right_padded = cv2.copyMakeBorder(right_image, pad, pad, pad, pad, cv2.BORDER_REFLECT)

    disparity_map = np.zeros((h, w), np.float32)

    for y in range(pad, h + pad):
        for x in range(pad, w + pad):
            ncc_scores = []

            # Extract patch from left image
            left_patch = left_padded[y - pad:y + pad + 1, x - pad:x + pad + 1].astype(np.float32)
            left_mean = np.mean(left_patch)
            left_norm = left_patch - left_mean
            left_denom = np.sqrt(np.sum(left_norm ** 2)) + 1e-5

            # Search corresponding patch in right image
            for d in range(max_disparity):
                xr = x - d
                if xr - pad < 0:
                    continue

                right_patch = right_padded[y - pad:y + pad + 1, xr - pad:xr + pad + 1].astype(np.float32)
                right_mean = np.mean(right_patch)
                right_norm = right_patch - right_mean
                right_denom = np.sqrt(np.sum(right_norm ** 2)) + 1e-5

                ncc = np.sum(left_norm * right_norm) / (left_denom * right_denom)
                ncc_scores.append(ncc)

            if len(ncc_scores) == 0:
                continue

            ncc_scores = np.array(ncc_scores)
            best_d = np.argmax(ncc_scores)

            # Sub-pixel quadratic interpolation
            if 1 <= best_d < len(ncc_scores) - 1:
                l, c, r = ncc_scores[best_d - 1], ncc_scores[best_d], ncc_scores[best_d + 1]
                denom = 2 * (2 * c - l - r)
                if abs(denom) > 1e-5:
                    delta = (r - l) / denom
                    best_d = best_d + delta

            disparity_map[y - pad, x - pad] = best_d

    return disparity_map


def compute_mae(a, b, mask=None):
    """
    Compute Mean Absolute Error (MAE) between two disparity maps.
    Optionally, use a mask to exclude invalid pixels.
    """
    if mask is not None:
        diff = np.abs(a[mask] - b[mask])
    else:
        diff = np.abs(a - b)
    return np.mean(diff)


# ==========================================================

# Load the stereo image pair in grayscale.
left_image = cv2.imread("data/left.jpg", cv2.IMREAD_GRAYSCALE)
right_image = cv2.imread("data/right.jpg", cv2.IMREAD_GRAYSCALE)
if left_image is None or right_image is None:
    raise FileNotFoundError("Stereo pair (data/left.jpg, data/right.jpg) not found.")

# Compute the manual NCC disparity map.
print("Computing manual NCC disparity map (this may take a bit)...")
manual_disp = compute_manual_ncc_map(left_image, right_image, WINDOW_SIZE, MAX_DISPARITY)
manual_disp = cv2.blur(manual_disp, (3, 3))

# Compute a benchmark map using OpenCV StereoBM with the same parameters.
stereo = cv2.StereoBM_create(numDisparities=MAX_DISPARITY, blockSize=WINDOW_SIZE)
bm_disp = stereo.compute(left_image, right_image).astype(np.float32) / 16.0

# Quantitatively compare both maps with mean absolute error.
mask = bm_disp > 0
mae_value = compute_mae(manual_disp, bm_disp, mask)
print(f"Mean Absolute Error (MAE): {mae_value:.4f}")

# Save and display normalized disparity maps for visual comparison.
manual_norm = cv2.normalize(manual_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
bm_norm = cv2.normalize(bm_disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imwrite("data/manual_ncc.jpg", manual_norm)
cv2.imwrite("data/benchmark_bm.jpg", bm_norm)

cv2.imshow("Manual NCC Disparity", manual_norm)
cv2.imshow("StereoBM Benchmark", bm_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()

# The printed MAE provides the numerical comparison against the StereoBM baseline.
