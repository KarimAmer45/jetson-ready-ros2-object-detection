import os
import argparse
import sys
import numpy as np
import cv2


# -----------------------------------------------------------------------------
# Mixture of Gaussians (MOG) background subtraction (Task 1 core, reused here)
# -----------------------------------------------------------------------------
class MOG:
    """
    Per-pixel Mixture of Gaussians background model (isotropic variance).

    Background decision:
    - Sort components by omega/sigma (descending).
    - Mark the first B components as background s.t. cumulative omega >= background_thresh.
    - Pixel is background if it matches any of the background components; otherwise foreground.
    """
    def __init__(self, height, width, number_of_gaussians=3, background_thresh=0.5, lr=0.01):
        self.number_of_gaussians = int(number_of_gaussians)
        self.background_thresh = float(background_thresh)
        self.dist_thresh = 20.0
        self.lr = float(lr)

        self.height = int(height)
        self.width = int(width)

        K = self.number_of_gaussians
        self.mus = np.zeros((self.height, self.width, K, 3), dtype=np.float32)
        self.sigmaSQs = np.zeros((self.height, self.width, K), dtype=np.float32)
        self.omegas = np.zeros((self.height, self.width, K), dtype=np.float32)

        # Default init (overwritten by init_from_first_frame()).
        self.mus[:] = np.array([122, 122, 122], dtype=np.float32)
        self.sigmaSQs[:] = 36.0
        self.omegas[:] = 1.0 / K

    def init_from_first_frame(self, first_bgr, init_sigma_sq=36.0):
        """
        Strong initialization:
        - Set all component means to the first frame colors.
        - Give most weight to component 0.
        This reduces "everything is foreground" in early frames.
        """
        if first_bgr is None:
            raise ValueError("first_bgr is None")
        if first_bgr.shape[:2] != (self.height, self.width) or first_bgr.shape[2] != 3:
            raise ValueError("first_bgr shape does not match initialized MOG size")

        x = first_bgr.astype(np.float32)
        K = self.number_of_gaussians

        self.mus[:] = x[:, :, None, :]
        self.sigmaSQs[:] = float(init_sigma_sq)

        self.omegas[:] = (0.1 / max(K - 1, 1))
        self.omegas[:, :, 0] = 0.9

    @staticmethod
    def _gauss_prob_iso(x, mu, sigma_sq):
        eps = 1e-6
        sigma_sq = np.maximum(sigma_sq, eps)
        diff = x - mu
        d2 = np.sum(diff * diff, axis=-1)
        norm_const = (2.0 * np.pi * sigma_sq) ** (-1.5)
        return norm_const * np.exp(-0.5 * d2 / sigma_sq)

    def updateParam(self, img, BG_pivot=None):
        """
        Update model and return a foreground mask:
        255 = foreground, 0 = background.
        """
        if img is None:
            raise ValueError("updateParam got img=None")
        if img.shape[:2] != (self.height, self.width) or img.shape[2] != 3:
            raise ValueError("Image shape does not match initialized MOG size")

        x = img.astype(np.float32)
        K = self.number_of_gaussians
        lr = self.lr

        x_exp = x[:, :, None, :]              # (H,W,1,3)
        diff = x_exp - self.mus               # (H,W,K,3)
        d2 = np.sum(diff * diff, axis=-1)     # (H,W,K)

        sigma_sq = np.maximum(self.sigmaSQs, 1e-6)
        match = d2 <= (self.dist_thresh ** 2) * sigma_sq
        any_match = np.any(match, axis=-1)

        d2_masked = np.where(match, d2, np.inf)
        best_k = np.argmin(d2_masked, axis=-1)

        M = np.zeros((self.height, self.width, K), dtype=np.float32)
        rows = np.arange(self.height)[:, None]
        cols = np.arange(self.width)[None, :]
        M[rows, cols, best_k] = any_match.astype(np.float32)

        # update omegas
        self.omegas = (1.0 - lr) * self.omegas + lr * M

        # replace weakest component on no-match
        no_match = ~any_match
        if np.any(no_match):
            min_k = np.argmin(self.omegas, axis=-1)
            nk = min_k[no_match]
            rr, cc = np.where(no_match)
            self.mus[rr, cc, nk, :] = x[rr, cc, :]
            self.sigmaSQs[rr, cc, nk] = 36.0
            self.omegas[rr, cc, nk] = 0.05

        # renormalize
        omega_sum = np.sum(self.omegas, axis=-1, keepdims=True)
        self.omegas = self.omegas / np.maximum(omega_sum, 1e-8)

        # update mean/variance for best matched component
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

        # background set selection per pixel
        rank = self.omegas / np.sqrt(np.maximum(self.sigmaSQs, 1e-6))
        order = np.argsort(rank, axis=-1)[:, :, ::-1]

        omegas_sorted = np.take_along_axis(self.omegas, order, axis=-1)
        cumsum = np.cumsum(omegas_sorted, axis=-1)

        bg_count = np.argmax(cumsum >= self.background_thresh, axis=-1) + 1
        pos = np.arange(K, dtype=np.int32)[None, None, :]
        bg_mask_sorted = pos < bg_count[:, :, None]

        bg_mask = np.zeros_like(bg_mask_sorted, dtype=bool)
        rows = np.arange(self.height)[:, None]
        cols = np.arange(self.width)[None, :]
        for p in range(K):
            ok = order[:, :, p]
            bg_mask[rows, cols, ok] = bg_mask_sorted[:, :, p]

        matched_bg_any = np.any(match & bg_mask, axis=-1)

        label_img = np.zeros((self.height, self.width), dtype=np.uint8)
        label_img[~matched_bg_any] = 255
        return label_img


# -----------------------------------------------------------------------------
# Post-processing + detections
# -----------------------------------------------------------------------------
def postprocess_mask(mask_u8, min_area_px, open_ksize=3, close_ksize=5, close_iters=2):
    """
    Clean raw foreground mask with thresholding, morphology, and area filtering.
    """
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    fg = (mask_u8 > 127).astype(np.uint8) * 255

    open_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))

    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_k, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_k, iterations=close_iters)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    out = np.zeros_like(fg)
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area_px:
            out[labels == i] = 255
    return out


def detections_from_mask(fg_u8, H, W):
    """
    Convert cleaned binary mask into bounding-box detections (x,y,w,h).

    Heuristics:
    - People bboxes are typically taller than wide (cars are wider than tall).
    - Filter by relative size and aspect ratio.
    """
    contours, _ = cv2.findContours(fg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    img_area = float(H * W)

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)

        area = float(w * h)
        if area < 0.0008 * img_area:
            continue

        ar = h / (w + 1e-6)

        if h < 0.06 * H:
            continue
        if w < 0.01 * W:
            continue
        if w > 0.45 * W:
            continue
        if ar < 1.15:
            continue

        dets.append((x, y, w, h))
    return dets


# -----------------------------------------------------------------------------
# Simple tracker
# -----------------------------------------------------------------------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return float(inter) / float(union + 1e-6)


class Track:
    def __init__(self, tid, bbox):
        self.id = int(tid)
        self.bbox = bbox
        self.hits = 1
        self.missed = 0
        self.confirmed = False

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        self.start_center = (cx, cy)
        self.max_disp = 0.0  # maximum displacement from start center

    def update(self, bbox):
        self.bbox = bbox
        self.hits += 1
        self.missed = 0
        if self.hits >= 2:
            self.confirmed = True

        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0
        dx = cx - self.start_center[0]
        dy = cy - self.start_center[1]
        disp = float(np.sqrt(dx * dx + dy * dy))
        if disp > self.max_disp:
            self.max_disp = disp

    def mark_missed(self):
        self.missed += 1


def update_tracks(tracks, detections, next_id, iou_thr=0.2, max_missed=3):
    """
    Greedy IoU matching.
    """
    if len(tracks) == 0:
        for d in detections:
            tracks.append(Track(next_id, d))
            next_id += 1
        return tracks, next_id

    used = [False] * len(detections)

    for t in tracks:
        best_j = -1
        best_iou = 0.0
        for j, d in enumerate(detections):
            if used[j]:
                continue
            val = iou_xywh(t.bbox, d)
            if val > best_iou:
                best_iou = val
                best_j = j

        if best_j >= 0 and best_iou >= iou_thr:
            t.update(detections[best_j])
            used[best_j] = True
        else:
            t.mark_missed()

    for j, d in enumerate(detections):
        if not used[j]:
            tracks.append(Track(next_id, d))
            next_id += 1

    tracks = [t for t in tracks if t.missed <= max_missed]
    return tracks, next_id


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def list_frames(imgs_dir, n_frames=15):
    """
    Return paths to frames 0001..0015 in numeric order.

    This avoids accidentally picking up masks/debug images if they exist in the same folder.
    """
    if not os.path.isdir(imgs_dir):
        raise FileNotFoundError(
            f"Input directory not found: {imgs_dir}. "
            f"Put frames in that folder (e.g., imgs/0001.jpg..0015.jpg)."
        )

    exts = [".jpg", ".jpeg", ".png", ".bmp"]
    paths = []
    for i in range(1, n_frames + 1):
        found = None
        for ext in exts:
            p = os.path.join(imgs_dir, f"{i:04d}{ext}")
            if os.path.exists(p):
                found = p
                break
        if found is not None:
            paths.append(found)

    # Fallback: if naming isn't strictly 0001..0015, take first n_frames numeric files
    if len(paths) == 0:
        files = []
        for f in os.listdir(imgs_dir):
            base, ext = os.path.splitext(f)
            if ext.lower() not in exts:
                continue
            if len(base) == 4 and base.isdigit():
                files.append(f)
        files.sort()
        paths = [os.path.join(imgs_dir, f) for f in files[:n_frames]]

    return paths


def make_debug_vis(img_bgr, fg_u8, tracks):
    vis = img_bgr.copy()

    for t in tracks:
        if not t.confirmed:
            continue
        x, y, w, h = t.bbox
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(vis, f"ID {t.id}", (x, max(0, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    mask_small = cv2.resize(fg_u8, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
    mask_small_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
    h0, w0 = mask_small_bgr.shape[:2]
    vis[0:h0, 0:w0] = mask_small_bgr

    return vis


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgs_dir", type=str, default="imgs", help="Directory with input frames (15 images).")
    parser.add_argument("--out_dir", type=str, default="output_task2", help="Where to save masks and debug images.")

    # Debug images (per-frame overlay PNGs)
    parser.add_argument("--no_debug", action="store_true", help="Disable debug image export.")

    # Optional video stitching
    parser.add_argument("--save_video", action="store_true", help="Save an MP4 video of the debug visualization.")
    parser.add_argument("--video_name", type=str, default="result.mp4", help="Video filename saved inside out_dir.")
    parser.add_argument("--video_fps", type=float, default=5.0, help="FPS for the output video.")

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # Required fixed parameters (Task 1)
    K = 3
    background_thresh = 0.5
    lr = 0.01

    frame_paths = list_frames(args.imgs_dir, n_frames=15)
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No images found in {args.imgs_dir}.")
    if len(frame_paths) < 15:
        print(f"[Warning] Found only {len(frame_paths)} images; counting on available frames.", file=sys.stderr)

    first = cv2.imread(frame_paths[0])
    if first is None:
        raise FileNotFoundError(f"Could not read {frame_paths[0]}")

    H, W = first.shape[:2]
    mog = MOG(height=H, width=W, number_of_gaussians=K, background_thresh=background_thresh, lr=lr)

    # Option A warm-start: initialize from frame 1, then treat frame 1 as warm-up only.
    mog.init_from_first_frame(first)

    tracks = []
    next_id = 1

    # Post-processing parameter (relative -> pixels)
    min_area_px = int(0.0007 * H * W)

    # Optional video writer (starts from frame 2)
    writer = None
    video_path = os.path.join(args.out_dir, args.video_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for idx, p in enumerate(frame_paths):
        img = cv2.imread(p)
        if img is None:
            raise FileNotFoundError(f"Could not read {p}")

        # Warm-up frame: update model but do NOT export mask/debug or update tracking on this same frame.
        if idx == 0:
            _ = mog.updateParam(img, BG_pivot=None)
            continue

        base = os.path.splitext(os.path.basename(p))[0]

        raw_mask = mog.updateParam(img, BG_pivot=None)

        # Save a representative raw mask for the report (frame 0002) as JPG
        # (idx==0 is 0001 warm-up, idx==1 is 0002)
        if idx == 1:
            cv2.imwrite(os.path.join(args.out_dir, f"{base}_rawmask.jpg"), raw_mask)

        fg = postprocess_mask(raw_mask, min_area_px=min_area_px)
        dets = detections_from_mask(fg, H, W)

        tracks, next_id = update_tracks(tracks, dets, next_id, iou_thr=0.2, max_missed=3)

        cv2.imwrite(os.path.join(args.out_dir, f"{base}_mask.png"), fg)

        vis = make_debug_vis(img, fg, tracks)
        if not args.no_debug:
            cv2.imwrite(os.path.join(args.out_dir, f"{base}_debug.png"), vis)

        if args.save_video:
            if writer is None:
                writer = cv2.VideoWriter(video_path, fourcc, float(args.video_fps), (vis.shape[1], vis.shape[0]))
                if not writer.isOpened():
                    writer = None
                    print("[Warning] Could not open VideoWriter. Video will not be saved.", file=sys.stderr)
            if writer is not None:
                writer.write(vis)

    if writer is not None:
        writer.release()

    # ---------------------------
    # Static-blob rejection (motion gate)
    # ---------------------------
    MIN_HITS_FOR_MOTION_TEST = 6   # apply motion test only to long-lived tracks
    MIN_DISP_PX = 8.0             # require some motion to reject static artifacts

    valid_ids = set()
    for t in tracks:
        if not t.confirmed:
            continue
        if t.hits >= MIN_HITS_FOR_MOTION_TEST and t.max_disp < MIN_DISP_PX:
            continue
        valid_ids.add(t.id)

    people_count = len(valid_ids)
    print(people_count)

    with open(os.path.join(args.out_dir, "people_count.txt"), "w", encoding="utf-8") as f:
        f.write(str(people_count) + "\n")



if __name__ == "__main__":
    main()
