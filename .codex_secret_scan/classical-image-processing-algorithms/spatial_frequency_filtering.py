
# Template for Exercise 3 – Spatial and Frequency Domain Filtering
import cv2
import numpy as np
import matplotlib.pyplot as plt


def make_box_kernel(k):
    """
    Create a normalized k×k box filter kernel.
    """
    h = np.ones((k, k), dtype=np.float32)
    h /= h.sum()
    return h


def make_gauss_kernel(k, sigma):
    """
    Create a normalized 2D Gaussian filter kernel of size k×k.
    """
    r = (k - 1) // 2
    y, x = np.mgrid[-r:r+1, -r:r+1]
    h = np.exp(-(x*x + y*y) / (2.0 * sigma * sigma)).astype(np.float32)
    h /= np.sum(h)
    return h


def conv2_same_zero(img, h):
    """
    Perform 2D spatial convolution using zero padding.
    Output should have the same size as the input image.
    (Do NOT use cv2.filter2D)
    """
    H, W = img.shape
    kH, kW = h.shape
    rH, rW = kH // 2, kW // 2

    padded = np.pad(img.astype(np.float32), ((rH, rH), (rW, rW)), mode='constant')
    out = np.zeros_like(img, dtype=np.float32)

    # Flip kernel for convolution
    hf = h[::-1, ::-1]

    for i in range(H):
        for j in range(W):
            region = padded[i:i+kH, j:j+kW]
            out[i, j] = np.sum(region * hf)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def freq_linear_conv(img, h):
    """
    Perform linear convolution in the frequency domain.
    (You can use numpy.fft)
    Returns a 'same' sized output as img.
    """
    H, W = img.shape
    kH, kW = h.shape
    outH, outW = H + kH - 1, W + kW - 1

    # Zero-pad
    pad_img = np.zeros((outH, outW), dtype=np.float32)
    pad_h = np.zeros((outH, outW), dtype=np.float32)
    pad_img[:H, :W] = img.astype(np.float32)
    pad_h[:kH, :kW] = h.astype(np.float32)

    F_img = np.fft.fft2(pad_img)
    F_h = np.fft.fft2(pad_h)
    conv_full = np.fft.ifft2(F_img * F_h)
    conv_full = np.real(conv_full)

    # Crop to 'same' size
    rH, rW = kH // 2, kW // 2
    out = conv_full[rH:rH+H, rW:rW+W]
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def compute_mad(a, b):
    """
    Compute Mean Absolute Difference (MAD) between two images.
    """
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

# ==========================================================

def main():
    # 1. Load the grayscale image (e.g., lena.png)
    img = cv2.imread("data/lena.png", cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not load data/lena.png")
        return

    # 2. Construct 9×9 box and Gaussian kernels (choose sigma for 9x9, e.g., 1.5)
    k = 9
    sigma = 1.5
    h_box = make_box_kernel(k)
    h_gauss = make_gauss_kernel(k, sigma)

    # 3. Apply both filters spatially (manual convolution)
    box_spatial = conv2_same_zero(img, h_box)
    gauss_spatial = conv2_same_zero(img, h_gauss)

    # 4. Apply both filters in the frequency domain
    box_freq = freq_linear_conv(img, h_box)
    gauss_freq = freq_linear_conv(img, h_gauss)

    # 5. Compute and print MAD between spatial and frequency outputs
    mad_box = compute_mad(box_spatial, box_freq)
    mad_gauss = compute_mad(gauss_spatial, gauss_freq)
    print(f"MAD(box spatial vs freq)   = {mad_box:.10f}")
    print(f"MAD(gauss spatial vs freq) = {mad_gauss:.10f}")
    if mad_box < 1e-7 and mad_gauss < 1e-7:
        print("Requirement satisfied: MAD < 1e-7 for both filters.")

    # 6. Save results
    out_box_spatial = "data/box_spatial.png"
    out_box_freq = "data/box_freq.png"
    out_gauss_spatial = "data/gauss_spatial.png"
    out_gauss_freq = "data/gauss_freq.png"
    cv2.imwrite(out_box_spatial, box_spatial)
    cv2.imwrite(out_box_freq, box_freq)
    cv2.imwrite(out_gauss_spatial, gauss_spatial)
    cv2.imwrite(out_gauss_freq, gauss_freq)

    # 7. Show the saved PNGs
    cv2.imshow("box_spatial", cv2.imread(out_box_spatial, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("box_freq", cv2.imread(out_box_freq, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("gauss_spatial", cv2.imread(out_gauss_spatial, cv2.IMREAD_GRAYSCALE))
    cv2.imshow("gauss_freq", cv2.imread(out_gauss_freq, cv2.IMREAD_GRAYSCALE))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
