import sys
import os
import numpy as np
import cv2
from scipy.stats import multivariate_normal  # allowed by template, but not required

'''
BG_pivot is the same shape as the input image but with single channel, all pixels have value 1.
'''

class MOG():
    def __init__(self, height=None, width=None, number_of_gaussians=None, background_thresh=None, lr=None):
        if height is None or width is None or number_of_gaussians is None or background_thresh is None or lr is None:
            raise ValueError("MOG init requires height, width, number_of_gaussians, background_thresh, lr")

        self.number_of_gaussians = int(number_of_gaussians)
        self.background_thresh = float(background_thresh)
        self.dist_thresh = 20.0
        self.lr = float(lr)

        self.height = int(height)
        self.width = int(width)

        self.mus = np.zeros((self.height, self.width, self.number_of_gaussians, 3), dtype=np.float32)
        self.sigmaSQs = np.zeros((self.height, self.width, self.number_of_gaussians), dtype=np.float32)
        self.omegas = np.zeros((self.height, self.width, self.number_of_gaussians), dtype=np.float32)

        self.mus[:] = np.array([122, 122, 122], dtype=np.float32)
        self.sigmaSQs[:] = 36.0
        self.omegas[:] = 1.0 / self.number_of_gaussians

    @staticmethod
    def _gauss_prob_iso(x, mu, sigma_sq):
        eps = 1e-6
        sigma_sq = np.maximum(sigma_sq, eps)
        diff = x - mu
        d2 = np.sum(diff * diff, axis=-1)
        norm_const = (2.0 * np.pi * sigma_sq) ** (-1.5)
        return norm_const * np.exp(-0.5 * d2 / sigma_sq)

    def updateParam(self, img, BG_pivot):
        if img is None:
            raise ValueError("updateParam got img=None")

        if img.shape[0] != self.height or img.shape[1] != self.width or img.shape[2] != 3:
            raise ValueError("Image shape does not match initialized MOG size")

        x = img.astype(np.float32)
        K = self.number_of_gaussians
        lr = self.lr

        x_exp = x[:, :, None, :]
        diff = x_exp - self.mus
        d2 = np.sum(diff * diff, axis=-1)

        sigma_sq = np.maximum(self.sigmaSQs, 1e-6)
        match = d2 <= (self.dist_thresh ** 2) * sigma_sq
        any_match = np.any(match, axis=-1)

        d2_masked = np.where(match, d2, np.inf)
        best_k = np.argmin(d2_masked, axis=-1)

        M = np.zeros((self.height, self.width, K), dtype=np.float32)
        rows = np.arange(self.height)[:, None]
        cols = np.arange(self.width)[None, :]
        M[rows, cols, best_k] = any_match.astype(np.float32)

        self.omegas = (1.0 - lr) * self.omegas + lr * M

        no_match = ~any_match
        if np.any(no_match):
            min_k = np.argmin(self.omegas, axis=-1)
            nk = min_k[no_match]
            rr, cc = np.where(no_match)
            self.mus[rr, cc, nk, :] = x[rr, cc, :]
            self.sigmaSQs[rr, cc, nk] = 36.0
            self.omegas[rr, cc, nk] = 0.05

        omega_sum = np.sum(self.omegas, axis=-1, keepdims=True)
        self.omegas = self.omegas / np.maximum(omega_sum, 1e-8)

        probs = self._gauss_prob_iso(x_exp, self.mus, self.sigmaSQs)
        rho = np.clip(lr * probs, 0.0, 0.5)

        for k in range(K):
            mk = (any_match & (best_k == k))
            if not np.any(mk):
                continue
            rr, cc = np.where(mk)
            r = rho[rr, cc, k][:, None]
            mu_old = self.mus[rr, cc, k, :]
            x_pix = x[rr, cc, :]

            mu_new = (1.0 - r) * mu_old + r * x_pix
            self.mus[rr, cc, k, :] = mu_new

            d2_new = np.sum((x_pix - mu_new) ** 2, axis=-1)
            sig_old = self.sigmaSQs[rr, cc, k]
            sig_new = (1.0 - rho[rr, cc, k]) * sig_old + rho[rr, cc, k] * d2_new
            self.sigmaSQs[rr, cc, k] = np.clip(sig_new, 4.0, 5e4)

        rank = self.omegas / np.sqrt(np.maximum(self.sigmaSQs, 1e-6))
        order = np.argsort(rank, axis=-1)[:, :, ::-1]

        omegas_sorted = np.take_along_axis(self.omegas, order, axis=-1)
        cumsum = np.cumsum(omegas_sorted, axis=-1)

        bg_mask_sorted = cumsum <= self.background_thresh
        bg_mask_sorted[:, :, 0] = True

        bg_mask = np.zeros_like(bg_mask_sorted, dtype=bool)
        for pos in range(K):
            ok = order[:, :, pos]
            bg_mask[rows, cols, ok] = bg_mask_sorted[:, :, pos]

        matched_is_bg = np.zeros((self.height, self.width), dtype=bool)
        if np.any(any_match):
            matched_is_bg = bg_mask[rows, cols, best_k] & any_match

        label_img = np.zeros((self.height, self.width), dtype=np.uint8)
        label_img[~matched_is_bg] = 255
        return label_img


def main():
    # Only first 3 images as required by the sheet
    NUM_FRAMES = 3

    # Required fixed parameters
    K = 3
    background_thresh = 0.5
    lr = 0.01

    imgs_dir = 'imgs'
    out_dir = 'output'
    os.makedirs(out_dir, exist_ok=True)

    first_path = os.path.join(imgs_dir, '{:04d}.jpg'.format(1))
    first = cv2.imread(first_path)
    if first is None:
        raise FileNotFoundError(f"Could not read {first_path}. Check that imgs/0001.jpg exists.")

    H, W = first.shape[:2]
    mog = MOG(height=H, width=W, number_of_gaussians=K, background_thresh=background_thresh, lr=lr)

    for i in range(1, NUM_FRAMES + 1):
        img_path = os.path.join(imgs_dir, '{:04d}.jpg'.format(i))
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read {img_path}.")

        label_img = mog.updateParam(img, np.ones(img.shape[:2], dtype=np.uint8))

        # Save mask to output folder
        mask_path = os.path.join(out_dir, f"label{i:04d}.jpg")
        cv2.imwrite(mask_path, label_img)

        # Show original and output side-by-side (single window)
        try:
            mask_bgr = cv2.cvtColor(label_img, cv2.COLOR_GRAY2BGR)
            vis = np.hstack([img, mask_bgr])

            win_name = f"Original | Mask  {i:04d}.jpg"
            cv2.imshow(win_name, vis)

            # Wait for a keypress:
            # - any key continues
            # - 'q' quits
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

            cv2.destroyWindow(win_name)
        except cv2.error:
            # headless environment: skip displaying
            pass

    try:
        cv2.destroyAllWindows()
    except cv2.error:
        pass


if __name__ == "__main__":
    main()
