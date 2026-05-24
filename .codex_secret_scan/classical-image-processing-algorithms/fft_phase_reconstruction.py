
# Template for Exercise 2 –  Fourier Transform and Image Reconstruction
import cv2
import numpy as np
import matplotlib.pyplot as plt


def compute_fft(img):
    """
    Compute the Fourier Transform of an image and return:
    - The shifted complex spectrum
    - The magnitude
    - The phase
    """
    f = np.fft.fft2(img.astype(np.float32))
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    phase = np.angle(fshift)
    return fshift, mag, phase


def reconstruct_from_mag_phase(mag, phase):
    """
    Reconstruct an image from given magnitude and phase.
    Expects magnitude and phase from the *shifted* spectrum.
    """
    complex_shifted = mag * np.exp(1j * phase)
    f = np.fft.ifftshift(complex_shifted)
    img_rec = np.fft.ifft2(f)
    img_rec = np.real(img_rec)
    img_rec = np.clip(img_rec, 0, 255).astype(np.uint8)
    return img_rec


def compute_mad(a, b):
    """
    Compute the Mean Absolute Difference (MAD) between two images.
    """
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))

# ==========================================================

def main():
    # 1. Load the two grayscale images (1.png and 2.png)
    img1 = cv2.imread("data/1.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("data/2.png", cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Error: Could not load data/1.png or data/2.png.")
        return

    # 2. Compute magnitude and phase of both images
    f1, mag1, phase1 = compute_fft(img1)
    f2, mag2, phase2 = compute_fft(img2)

    # 3. Swap magnitude and phase between the two images
    rec_mag1_phase2 = reconstruct_from_mag_phase(mag1, phase2)
    rec_mag2_phase1 = reconstruct_from_mag_phase(mag2, phase1)

    # 4. Reconstruct and save the swapped results
    out1_path = "data/reconstructed_mag1_phase2.png"
    out2_path = "data/reconstructed_mag2_phase1.png"
    cv2.imwrite(out1_path, rec_mag1_phase2)
    cv2.imwrite(out2_path, rec_mag2_phase1)

    # 5. Compute and print the MAD values between originals and reconstructions
    mad1 = compute_mad(img1, rec_mag1_phase2)
    mad2 = compute_mad(img2, rec_mag2_phase1)
    print(f"MAD(img1, reconstructed_mag1_phase2) = {mad1:.4f}")
    print(f"MAD(img2, reconstructed_mag2_phase1) = {mad2:.4f}")

    # 6. Visualize all images (show only the saved reconstructions as requested)
    rec1 = cv2.imread(out1_path, cv2.IMREAD_GRAYSCALE)
    rec2 = cv2.imread(out2_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("reconstructed_mag1_phase2", rec1)
    cv2.imshow("reconstructed_mag2_phase1", rec2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
