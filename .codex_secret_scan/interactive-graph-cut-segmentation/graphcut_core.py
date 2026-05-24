import cv2
import numpy as np
import maxflow
import os
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


# ================================================================
#  Color models
# ================================================================

class HistogramColorModel:
    """
    Simple 3D histogram with -log probability lookup.
    """
    def __init__(self, n_bins=16):
        self.n_bins = n_bins
        self.hist = None
        self.total = None
        self.bin_width = 1.0 / n_bins

    def fit(self, colors):
        """
        colors: (N, 3) uint8 in some color space (here: Lab-like 0..255).
        """
        if colors.size == 0:
            raise ValueError("HistogramColorModel.fit got 0 pixels")

        data = colors.reshape(-1, 3).astype(np.float32) / 255.0

        hist, _ = np.histogramdd(
            data,
            bins=self.n_bins,
            range=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
        )

        hist = hist.astype(np.float64) + 1e-6  # smoothing
        self.hist = hist
        self.total = np.sum(hist)
        self.bin_width = 1.0 / self.n_bins

    def neg_log_likelihood(self, colors):
        """
        Returns -log P(color | model) for each pixel.
        colors: (N, 3) uint8 in same space as fit().
        """
        if self.hist is None:
            raise RuntimeError("HistogramColorModel used before fit()")

        data = colors.reshape(-1, 3).astype(np.float32) / 255.0
        idx = np.floor(data / self.bin_width).astype(np.int32)
        idx = np.clip(idx, 0, self.n_bins - 1)

        probs = self.hist[idx[:, 0], idx[:, 1], idx[:, 2]] / self.total
        probs = np.clip(probs, 1e-12, None)
        return -np.log(probs)


class GMMColorModel:
    """
    Wrapper around sklearn.mixture.GaussianMixture to get -log likelihoods.
    """
    def __init__(self, n_components=5, reg_covar=1e-6, covariance_type="full"):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.covariance_type = covariance_type
        self.gmm = None

    def fit(self, colors):
        """
        colors: (N, 3) uint8 in some color space (here: Lab-like 0..255).
        """
        if colors.size == 0:
            raise ValueError("GMMColorModel.fit got 0 pixels")

        data = colors.reshape(-1, 3).astype(np.float32)
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=0,
        )
        self.gmm.fit(data)

    def neg_log_likelihood(self, colors):
        """
        Returns -log P(color | model) for each pixel.
        colors: (N, 3) uint8 in same space as fit().
        """
        if self.gmm is None:
            raise RuntimeError("GMMColorModel used before fit()")

        data = colors.reshape(-1, 3).astype(np.float32)
        logp = self.gmm.score_samples(data)  # log P(x)
        return -logp


# ================================================================
#  Core Graph Cut class for 2.1
# ================================================================

class GraphCutCore:
    """
    Core Graph Cut segmenter for 2.1:
      image  + predefined scribbles  ->  binary mask

    Supports two color models:
        - 'hist' : HistogramColorModel
        - 'gmm'  : GMMColorModel
    Uses Lab color space internally for modeling/pairwise.
    """

    def __init__(
        self,
        color_model_type="gmm",
        n_hist_bins=16,
        n_gmm_components=5,
        lambda_smooth=50.0,
        sigma=10.0,
        connectivity=8,
        use_lab=True,
    ):
        assert color_model_type in ("hist", "gmm")
        assert connectivity in (4, 8)

        self.color_model_type = color_model_type
        self.n_hist_bins = n_hist_bins
        self.n_gmm_components = n_gmm_components
        self.lambda_smooth = float(lambda_smooth)
        self.sigma = float(sigma)
        self.connectivity = connectivity
        self.use_lab = use_lab

        self.fg_model = None
        self.bg_model = None

        # scribble label values
        self.fg_value = None
        self.bg_value = None

        # large capacity for hard constraints
        self.INF = 1e9

    # ------------------------------------------------------------
    # Color space helper
    # ------------------------------------------------------------

    def _get_color_image(self, image_bgr):
        """
        Return the image in the color space used for modeling/pairwise.
        Currently Lab (uint8) if use_lab=True, else original BGR.
        """
        if self.use_lab:
            img_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
            return img_lab  # uint8 0..255
        else:
            return image_bgr

    # ------------------------------------------------------------
    # Scribble handling
    # ------------------------------------------------------------

    def _infer_fg_bg_values(self, scribbles_gray):
        """
        Scribbles have: 0 (unlabeled), one value for BG, one for FG.
        We detect the two non-zero values and assume:
           smaller  -> BG
           larger   -> FG
        """
        values = np.unique(scribbles_gray)
        non_zero = values[values != 0]
        if non_zero.size < 2:
            raise ValueError(
                "Scribble map must contain at least two non-zero values "
                "(one FG and one BG)."
            )
        sorted_vals = np.sort(non_zero)
        self.bg_value = int(sorted_vals[0])
        self.fg_value = int(sorted_vals[-1])

    def _get_scribble_masks(self, scribbles_gray):
        """
        Return boolean masks: fg_mask, bg_mask, using original scribbles.
        """
        if self.fg_value is None or self.bg_value is None:
            self._infer_fg_bg_values(scribbles_gray)

        fg_mask = scribbles_gray == self.fg_value
        bg_mask = scribbles_gray == self.bg_value
        return fg_mask, bg_mask

    # ------------------------------------------------------------
    # Color model fitting
    # ------------------------------------------------------------

    def _fit_color_models(self, color_img, scribbles_gray):
        """
        Fit FG and BG color models using pixels around scribbles.
        We slightly dilate scribbles for *training* only (not constraints)
        to get more stable color models.
        """
        h, w = scribbles_gray.shape
        fg_mask, bg_mask = self._get_scribble_masks(scribbles_gray)

        if not np.any(fg_mask):
            raise ValueError("No FG scribbles found")
        if not np.any(bg_mask):
            raise ValueError("No BG scribbles found")

        # Dilate masks to collect more training samples
        kernel = np.ones((3, 3), np.uint8)
        fg_dil = cv2.dilate(fg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
        bg_dil = cv2.dilate(bg_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        # Avoid overlap: FG wins over BG for training
        bg_dil = np.logical_and(bg_dil, np.logical_not(fg_dil))

        img_flat = color_img.reshape(-1, 3)
        fg_colors = img_flat[fg_dil.reshape(-1)]
        bg_colors = img_flat[bg_dil.reshape(-1)]

        if self.color_model_type == "hist":
            self.fg_model = HistogramColorModel(self.n_hist_bins)
            self.bg_model = HistogramColorModel(self.n_hist_bins)
        else:
            self.fg_model = GMMColorModel(self.n_gmm_components)
            self.bg_model = GMMColorModel(self.n_gmm_components)

        self.fg_model.fit(fg_colors)
        self.bg_model.fit(bg_colors)

    # ------------------------------------------------------------
    # Unary cost:  -log P(color | FG/BG model)
    # ------------------------------------------------------------

    def _compute_unary_costs(self, color_img, scribbles_gray):
        """
        Returns:
            D_fg, D_bg : each (H, W) arrays of unary costs
        """
        h, w = scribbles_gray.shape
        img_flat = color_img.reshape(-1, 3)

        fg_nll = self.fg_model.neg_log_likelihood(img_flat)
        bg_nll = self.bg_model.neg_log_likelihood(img_flat)

        D_fg = fg_nll.reshape(h, w)
        D_bg = bg_nll.reshape(h, w)
        return D_fg, D_bg

    # ------------------------------------------------------------
    # Pairwise term
    # ------------------------------------------------------------

    def _add_pairwise_edges(self, graph, node_ids, color_img):
        """
        Adds n-links between neighboring pixels.
        Weight w_pq is high if colors are similar, low if different:
            w_pq = lambda_smooth * exp( - ||I_p - I_q||^2 / (2 * sigma^2) )
        Computed in Lab space.
        """
        h, w = color_img.shape[:2]
        img = color_img.astype(np.float32)
        lam = self.lambda_smooth
        sigma2 = self.sigma ** 2

        def weight(p, q):
            diff = img[p] - img[q]
            d2 = float(np.dot(diff, diff))
            return lam * np.exp(-d2 / (2.0 * sigma2))

        # 4-connected neighbours
        for y in range(h):
            for x in range(w):
                p_idx = node_ids[y, x]

                if x + 1 < w:
                    q_idx = node_ids[y, x + 1]
                    w_pq = weight((y, x), (y, x + 1))
                    if w_pq > 0:
                        graph.add_edge(p_idx, q_idx, w_pq, w_pq)

                if y + 1 < h:
                    q_idx = node_ids[y + 1, x]
                    w_pq = weight((y, x), (y + 1, x))
                    if w_pq > 0:
                        graph.add_edge(p_idx, q_idx, w_pq, w_pq)

        # 8-connected: diagonals
        if self.connectivity == 8:
            diag_scale = 1.0 / np.sqrt(2.0)
            for y in range(h):
                for x in range(w):
                    p_idx = node_ids[y, x]

                    if x + 1 < w and y + 1 < h:
                        q_idx = node_ids[y + 1, x + 1]
                        w_pq = weight((y, x), (y + 1, x + 1)) * diag_scale
                        if w_pq > 0:
                            graph.add_edge(p_idx, q_idx, w_pq, w_pq)

                    if x - 1 >= 0 and y + 1 < h:
                        q_idx = node_ids[y + 1, x - 1]
                        w_pq = weight((y, x), (y + 1, x - 1)) * diag_scale
                        if w_pq > 0:
                            graph.add_edge(p_idx, q_idx, w_pq, w_pq)

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def segment(self, image_bgr, scribbles_gray):
        """
        Main entry point for 2.1:

            mask = GraphCutCore(...).segment(image_bgr, scribbles_gray)

        Args:
            image_bgr      : HxWx3 uint8 BGR image
            scribbles_gray : HxW uint8, values {0, BG, FG}

        Returns:
            mask : HxW uint8, {0,255}  (255 = foreground)
        """
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("image_bgr must be HxWx3 BGR color image")
        if scribbles_gray.shape[:2] != image_bgr.shape[:2]:
            raise ValueError("scribbles_gray and image_bgr must have same size")

        h, w = scribbles_gray.shape

        # Work in Lab (or BGR) for modeling/pairwise
        color_img = self._get_color_image(image_bgr)

        # 1) fit color models from scribbles
        self._fit_color_models(color_img, scribbles_gray)

        # 2) unary costs
        D_fg, D_bg = self._compute_unary_costs(color_img, scribbles_gray)

        # 3) build graph
        n_pixels = h * w
        graph = maxflow.Graph[float]()
        nodes = graph.add_nodes(n_pixels)
        node_ids = nodes.reshape(h, w)

        fg_mask, bg_mask = self._get_scribble_masks(scribbles_gray)

        for y in range(h):
            for x in range(w):
                idx = node_ids[y, x]

                if fg_mask[y, x]:
                    # FG scribble: D_bg = INF, D_fg = 0
                    graph.add_tedge(idx, self.INF, 0.0)
                elif bg_mask[y, x]:
                    # BG scribble: D_bg = 0, D_fg = INF
                    graph.add_tedge(idx, 0.0, self.INF)
                else:
                    # label 0 (sink) uses cap_source (BG), label 1 (source) uses cap_sink (FG)
                    cost_fg = float(D_fg[y, x])
                    cost_bg = float(D_bg[y, x])
                    graph.add_tedge(idx, cost_bg, cost_fg)

        # 4) pairwise edges
        self._add_pairwise_edges(graph, node_ids, color_img)

        # 5) max-flow / min-cut
        _ = graph.maxflow()

        # 6) extract mask
        mask = np.zeros((h, w), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                idx = node_ids[y, x]
                if graph.get_segment(idx) == 0:  # source (FG)
                    mask[y, x] = 255
                else:                            # sink (BG)
                    mask[y, x] = 0

        return mask


# ================================================================
#  Evaluation helpers
# ================================================================

def compute_iou(pred_mask, gt_mask):
    """
    Intersection over Union between two masks.

    GT has values {0, 128, 255}. We treat >=128 as foreground.
    """
    pred = pred_mask > 0
    gt = gt_mask >= 128

    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _collect_dataset_paths(dataset_root="dataset"):
    """
    Collect (image_path, scribble_path, gt_path) triplets from:
        dataset_root/images
        dataset_root/images-labels
        dataset_root/images-gt
    """
    images_dir = os.path.join(dataset_root, "images")
    scribbles_dir = os.path.join(dataset_root, "images-labels")
    gt_dir = os.path.join(dataset_root, "images-gt")

    image_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    triplets = []
    for img_name in image_files:
        base = os.path.splitext(img_name)[0]
        img_path = os.path.join(images_dir, img_name)
        scrib_path = os.path.join(scribbles_dir, base + "-anno.png")
        gt_path = os.path.join(gt_dir, base + ".png")
        if os.path.exists(scrib_path) and os.path.exists(gt_path):
            triplets.append((img_path, scrib_path, gt_path))
    return triplets


def _visualize_example(image_bgr, scribbles, pred_mask, gt_mask, iou, title_prefix=""):
    """
    Show a 4-panel figure:
      Input | Scribbles | Graph-Cut Result (IoU=..) | Ground Truth
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    h, w = scribbles.shape
    scrib_vis = np.zeros((h, w, 3), dtype=np.uint8)

    vals = np.unique(scribbles)
    non_zero = vals[vals != 0]
    if non_zero.size >= 1:
        bg_val = int(np.min(non_zero))
        fg_val = int(np.max(non_zero))
        # background scribbles in red
        scrib_vis[scribbles == bg_val] = (255, 0, 0)        # RGB red
        # foreground scribbles in white
        scrib_vis[scribbles == fg_val] = (255, 255, 255)    # RGB white

    pred_disp = (pred_mask > 0).astype(np.uint8) * 255
    gt_disp = (gt_mask >= 128).astype(np.uint8) * 255

    plt.figure(figsize=(10, 3))

    plt.subplot(1, 4, 1)
    plt.imshow(image_rgb)
    plt.axis("off")
    plt.title("Input Image")

    plt.subplot(1, 4, 2)
    plt.imshow(scrib_vis)
    plt.axis("off")
    plt.title("User Annotation\n(Red=BG, White=FG)")

    plt.subplot(1, 4, 3)
    plt.imshow(pred_disp, cmap="gray")
    plt.axis("off")
    plt.title(f"Graph Cut Result\nIoU: {iou:.2f}")

    plt.subplot(1, 4, 4)
    plt.imshow(gt_disp, cmap="gray")
    plt.axis("off")
    plt.title("Ground Truth")

    if title_prefix:
        plt.suptitle(title_prefix, y=1.05)

    plt.tight_layout()
    plt.show()


def evaluate_dataset(dataset_root="dataset",
                     color_model_type="gmm",
                     save_dir=None,
                     display=True):
    """
    Run 2.1 core algorithm on all dataset images and print / return IoUs.

    If display=True, show a 4-panel figure for each image:
       Input | Scribbles | Prediction | GT  (with IoU in title)
    """
    triplets = _collect_dataset_paths(dataset_root)
    if not triplets:
        print("No dataset images found in", dataset_root)
        return []

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    segmenter = GraphCutCore(color_model_type=color_model_type)

    ious = []
    print(f"\n=== Evaluating color model: {color_model_type} ===")
    for img_path, scrib_path, gt_path in triplets:
        base = os.path.splitext(os.path.basename(img_path))[0]
        print("Processing:", base)

        image_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        scribbles = cv2.imread(scrib_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or scribbles is None or gt_mask is None:
            print("  -> Could not read one of the files, skipping.")
            continue

        pred_mask = segmenter.segment(image_bgr, scribbles)
        iou = compute_iou(pred_mask, gt_mask)
        ious.append(iou)
        print(f"  IoU: {iou:.4f}")

        if save_dir is not None:
            out_path = os.path.join(save_dir, base + f"_{color_model_type}.png")
            cv2.imwrite(out_path, pred_mask)

        if display:
            _visualize_example(
                image_bgr,
                scribbles,
                pred_mask,
                gt_mask,
                iou,
                title_prefix=f"{base} ({color_model_type})",
            )

    if ious:
        avg = float(np.mean(ious))
        print(f"\nAverage IoU ({color_model_type}) over {len(ious)} images: {avg:.4f}")
    else:
        print("No IoUs computed – check dataset path.")
    return ious


if __name__ == "__main__":
    # Example: evaluate both histogram and GMM models on the dataset
    root = "dataset"

    evaluate_dataset(root,
                     color_model_type="hist",
                     save_dir=os.path.join(root, "results_hist"),
                     display=True)

    evaluate_dataset(root,
                     color_model_type="gmm",
                     save_dir=os.path.join(root, "results_gmm"),
                     display=True)
