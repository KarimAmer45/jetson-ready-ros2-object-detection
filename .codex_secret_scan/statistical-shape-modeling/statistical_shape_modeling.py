#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exercise 08 (Shapes) - MA-INF 2201 / MA-MOROB-M04 Computer Vision WS25/26

Single-file solution:
  - Task 1: Shape template matching on rat.webp using rat.txt
  - Task 2: GPA (affine), PCA SSM, PPCA SSM, inference on hands_test.npy

Allowed tooling philosophy:
  - No sklearn / no pre-built PCA/PPCA helpers
  - Uses: numpy, cv2, matplotlib, os, math (and basic Python stdlib)

Outputs:
  - Writes images/plots into ./output
  - Prints key numeric results (e.g., chosen N, MSEs)
"""

import os
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt


# -----------------------------
# Utility / IO
# -----------------------------

def _try_paths(*candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_rat_landmarks(txt_path):
    pts = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            xs = line.split()
            if len(xs) < 2:
                continue
            pts.append([float(xs[0]), float(xs[1])])
    return np.asarray(pts, dtype=np.float32)  # (N,2)


def save_overlay_polyline(img_bgr, pts_xy, out_path, color=(0, 255, 0), thickness=2):
    img = img_bgr.copy()
    pts = np.round(pts_xy).astype(np.int32)
    pts[:, 0] = np.clip(pts[:, 0], 0, img.shape[1] - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, img.shape[0] - 1)

    # draw closed polyline + points
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], isClosed=True, color=color, thickness=thickness)
    for (x, y) in pts:
        cv2.circle(img, (int(x), int(y)), 2, color, -1)
    cv2.imwrite(out_path, img)


def plot_shapes(ax, shapes, title=None):
    """
    shapes: list of (P,2) arrays
    """
    for s in shapes:
        ax.plot(s[:, 0], s[:, 1])
    ax.set_aspect("equal", adjustable="box")
    ax.invert_yaxis()
    if title:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


# -----------------------------
# Task 1: Shape Template Models
# -----------------------------

def canny_edges(img_bgr, blur_ksize=5, low=60, high=140):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if blur_ksize and blur_ksize > 0:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    edges = cv2.Canny(gray, low, high)
    return edges


def distance_transform_from_edges(edges_u8):
    """
    edges_u8: uint8 edges with 255 on edges
    returns float32 distance-to-edge for each pixel
    """
    inv = (edges_u8 == 0).astype(np.uint8)  # 1 where background (non-edge)
    dist = cv2.distanceTransform(inv, distanceType=cv2.DIST_L2, maskSize=3)
    return dist.astype(np.float32)


def bilinear_sample(img_float, xs, ys):
    """
    img_float: (H,W) float32
    xs, ys: arrays of same length, float coords
    returns sampled values (N,)
    """
    h, w = img_float.shape[:2]

    x0 = np.floor(xs).astype(np.int32)
    y0 = np.floor(ys).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    Ia = img_float[y0c, x0c]
    Ib = img_float[y1c, x0c]
    Ic = img_float[y0c, x1c]
    Id = img_float[y1c, x1c]

    wa = (x1 - xs) * (y1 - ys)
    wb = (x1 - xs) * (ys - y0)
    wc = (xs - x0) * (y1 - ys)
    wd = (xs - x0) * (ys - y0)

    return Ia * wa + Ib * wb + Ic * wc + Id * wd


def similarity_transform_points(pts, s, theta, tx, ty):
    """
    pts: (N,2) template points
    similarity about centroid of pts.
    """
    c = pts.mean(axis=0)
    p = pts - c
    ct = math.cos(theta)
    st = math.sin(theta)
    R = np.array([[ct, -st], [st, ct]], dtype=np.float32)
    q = (s * (p @ R.T)) + np.array([tx, ty], dtype=np.float32)
    return q


def template_match_distance_transform(dist_map, template_pts, init_params=None,
                                      iters=40, shrink=0.75,
                                      dt0=40.0, dtheta0=math.radians(20.0), ds0=0.20):
    """
    Simple coordinate search on (s, theta, tx, ty) to minimize mean distance transform value.
    """
    if init_params is None:
        # put translation at template centroid (identity-like)
        c = template_pts.mean(axis=0)
        s, theta, tx, ty = 1.0, 0.0, float(c[0]), float(c[1])
    else:
        s, theta, tx, ty = init_params

    dt = float(dt0)
    dtheta = float(dtheta0)
    ds = float(ds0)

    def cost(_s, _theta, _tx, _ty):
        pts_t = similarity_transform_points(template_pts, _s, _theta, _tx, _ty)
        xs = pts_t[:, 0]
        ys = pts_t[:, 1]
        # outside penalty
        h, w = dist_map.shape[:2]
        outside = (xs < 0) | (xs > (w - 1)) | (ys < 0) | (ys > (h - 1))
        xs_clip = np.clip(xs, 0, w - 1)
        ys_clip = np.clip(ys, 0, h - 1)
        vals = bilinear_sample(dist_map, xs_clip, ys_clip)
        # large penalty for outside points
        vals[outside] += 50.0
        return float(np.mean(vals))

    best_c = cost(s, theta, tx, ty)

    for _ in range(iters):
        # neighborhood candidates
        cand = []
        for ds_i in [0.0, -ds, ds]:
            for dt_i in [0.0, -dt, dt]:
                for dt_j in [0.0, -dt, dt]:
                    for dth_i in [0.0, -dtheta, dtheta]:
                        cand.append((s + ds_i, theta + dth_i, tx + dt_i, ty + dt_j))

        # evaluate and pick best
        best_local = (s, theta, tx, ty)
        for (ss, th, txx, tyy) in cand:
            if ss <= 0.05:
                continue
            cval = cost(ss, th, txx, tyy)
            if cval < best_c:
                best_c = cval
                best_local = (ss, th, txx, tyy)

        s, theta, tx, ty = best_local
        dt *= shrink
        dtheta *= shrink
        ds *= shrink

        if dt < 0.5 and dtheta < math.radians(0.5) and ds < 0.005:
            break

    return (s, theta, tx, ty), best_c


def run_task1(output_dir, rat_img_path, rat_txt_path):
    img = cv2.imread(rat_img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {rat_img_path}")

    template = load_rat_landmarks(rat_txt_path)

    # edge detector + distance transform
    edges = canny_edges(img, blur_ksize=5, low=60, high=140)
    dist = distance_transform_from_edges(edges)

    # optimize similarity transform
    init_c = template.mean(axis=0)
    init_params = (1.0, 0.0, float(init_c[0]), float(init_c[1]))
    params, best_cost = template_match_distance_transform(dist, template, init_params=init_params)

    s, theta, tx, ty = params
    warped = similarity_transform_points(template, s, theta, tx, ty)

    # save visuals
    cv2.imwrite(os.path.join(output_dir, "task1_edges.png"), edges)
    # visualize distance map
    dist_vis = dist.copy()
    dist_vis = dist_vis / (dist_vis.max() + 1e-6)
    dist_vis = (dist_vis * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, "task1_dist.png"), dist_vis)

    save_overlay_polyline(img, warped, os.path.join(output_dir, "task1_overlay.png"), color=(0, 255, 0), thickness=2)

    print("[Task 1] Best params:")
    print("  s=%.4f theta(deg)=%.2f tx=%.2f ty=%.2f cost=%.4f" % (s, math.degrees(theta), tx, ty, best_cost))

    return warped, params, best_cost


# -----------------------------
# Task 2: Statistical Shape Models
# -----------------------------

def best_affine_transform(src_pts, dst_pts):
    """
    Solve for affine transform that maps src -> dst:
        [x y 1] @ M = [x' y']
    src_pts, dst_pts: (P,2)
    Returns M: (3,2)
    """
    P = src_pts.shape[0]
    X = np.ones((P, 3), dtype=np.float64)
    X[:, 0:2] = src_pts.astype(np.float64)
    Y = dst_pts.astype(np.float64)  # (P,2)
    # least squares
    M, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    # M is (3,2)
    return M


def apply_affine(pts, M):
    """
    pts: (P,2), M: (3,2)
    """
    P = pts.shape[0]
    X = np.ones((P, 3), dtype=np.float64)
    X[:, 0:2] = pts.astype(np.float64)
    Y = X @ M  # (P,2)
    return Y.astype(np.float32)


def normalize_shape(pts):
    """
    Remove global translation + scale for mean shape stability.
    """
    p = pts - pts.mean(axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.sum(p * p, axis=1))) + 1e-8
    return (p / scale).astype(np.float32)


def generalized_procrustes_affine(shapes, max_iters=50, tol=1e-6):
    """
    shapes: (N,P,2)
    Aligns via affine transforms to evolving mean.
    Returns:
      aligned: (N,P,2)
      mean_shape: (P,2) normalized
      transforms: list of affine M_i (3,2) mapping original -> aligned
    """
    N, P, _ = shapes.shape
    # init: normalized first shape
    mean_shape = normalize_shape(shapes[0])

    aligned = np.zeros_like(shapes, dtype=np.float32)
    transforms = [None] * N

    prev_err = None
    for it in range(max_iters):
        # align all to current mean
        for i in range(N):
            M = best_affine_transform(shapes[i], mean_shape)
            si = apply_affine(shapes[i], M)
            aligned[i] = si
            transforms[i] = M

        new_mean = np.mean(aligned, axis=0)
        new_mean = normalize_shape(new_mean)

        # compute mean change
        err = float(np.mean((new_mean - mean_shape) ** 2))
        mean_shape = new_mean

        if prev_err is not None and abs(prev_err - err) < tol:
            break
        prev_err = err

    # final pass: align all to final mean
    for i in range(N):
        M = best_affine_transform(shapes[i], mean_shape)
        aligned[i] = apply_affine(shapes[i], M)
        transforms[i] = M

    return aligned, mean_shape, transforms


def pca_fit(X, energy_keep=0.90):
    """
    Implement PCA from scratch using SVD.

    X: (n,d) data
    Returns:
      mu: (d,)
      Phi: (d,k) principal components (orthonormal)
      lambdas: (k,) eigenvalues
      k: chosen number to preserve energy_keep
      explained: (d,) all eigenvalues sorted desc (full)
      Vt_full: (d,d) right-singular vectors (full)
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu

    # SVD on centered data
    # Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    # eigenvalues of covariance
    n = X.shape[0]
    evals = (S ** 2) / max(n - 1, 1)
    # energy selection
    total = float(np.sum(evals))
    if total <= 0:
        k = 1
    else:
        cumsum = np.cumsum(evals)
        k = int(np.searchsorted(cumsum / total, energy_keep) + 1)

    Phi = Vt[:k].T  # (d,k)
    lambdas = evals[:k].copy()
    return mu.astype(np.float32), Phi.astype(np.float32), lambdas.astype(np.float32), k, evals.astype(np.float32), Vt.astype(np.float32)


def ppca_from_cov(X, q):
    """
    PPCA maximum likelihood solution using eigendecomposition of sample covariance.
    X: (n,d)
    q: latent dimension
    Returns:
      mu (d,)
      W  (d,q)
      sigma2 (scalar)
      Uq (d,q) eigenvectors
      lam (d,) eigenvalues sorted desc
    """
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(axis=0)
    Xc = X - mu
    n, d = Xc.shape

    # sample covariance
    S = (Xc.T @ Xc) / max(n - 1, 1)

    # eigendecomposition (S is symmetric)
    lam, U = np.linalg.eigh(S)
    # sort descending
    idx = np.argsort(lam)[::-1]
    lam = lam[idx]
    U = U[:, idx]

    q = int(min(max(q, 1), d))
    Uq = U[:, :q]
    lam_q = lam[:q]

    if q < d:
        sigma2 = float(np.mean(lam[q:]))
    else:
        sigma2 = 0.0

    # numerical floor
    sigma2 = max(sigma2, 1e-8)

    # W = Uq (Lambda_q - sigma^2 I)^{1/2}
    diag = np.maximum(lam_q - sigma2, 0.0)
    W = Uq @ np.diag(np.sqrt(diag + 1e-12))

    return mu.astype(np.float32), W.astype(np.float32), float(sigma2), Uq.astype(np.float32), lam.astype(np.float32)


def pca_project_reconstruct(x, mu, Phi):
    """
    Standard PCA projection and reconstruction.
    """
    x = x.astype(np.float32).reshape(-1)
    mu = mu.reshape(-1)
    Phi = Phi.astype(np.float32)
    b = Phi.T @ (x - mu)
    x_hat = mu + Phi @ b
    return b, x_hat


def ppca_posterior_reconstruct(x, mu, W, sigma2):
    """
    PPCA posterior mean of latent z and reconstruction.
    """
    x = x.astype(np.float32).reshape(-1)
    mu = mu.reshape(-1)
    W = W.astype(np.float32)
    q = W.shape[1]
    Iq = np.eye(q, dtype=np.float32)
    M = (W.T @ W) + (sigma2 * Iq)
    # solve M z = W^T (x-mu)
    rhs = W.T @ (x - mu)
    z = np.linalg.solve(M.astype(np.float64), rhs.astype(np.float64)).astype(np.float32)
    x_hat = mu + W @ z
    return z, x_hat


def plot_gpa_before_after(shapes_raw, shapes_aligned, out_path_before, out_path_after, max_plot=12):
    N = shapes_raw.shape[0]
    idx = list(range(min(N, max_plot)))

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_shapes(ax, [shapes_raw[i] for i in idx], title="Hands: before GPA (subset)")
    fig.tight_layout()
    fig.savefig(out_path_before, dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_shapes(ax, [shapes_aligned[i] for i in idx], title="Hands: after GPA (subset)")
    fig.tight_layout()
    fig.savefig(out_path_after, dpi=200)
    plt.close(fig)


def plot_mode_variations(mu_shape, basis_vec, out_path, title, alphas=(-3, -1, 0, 1, 3)):
    """
    mu_shape: (P,2)
    basis_vec: (d,) flatten direction in data space
    """
    P = mu_shape.shape[0]
    mu = mu_shape.reshape(-1)
    fig = plt.figure(figsize=(10, 2.4))
    for j, a in enumerate(alphas):
        ax = fig.add_subplot(1, len(alphas), j + 1)
        s = (mu + a * basis_vec).reshape(P, 2)
        plot_shapes(ax, [s], title=f"{a:+d}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def run_task2(output_dir, hands_train_path, hands_test_path):
    train = np.load(hands_train_path).astype(np.float32)  # (38,56,2)
    test = np.load(hands_test_path).astype(np.float32)    # (56,2)

    # GPA
    aligned, mean_shape, _ = generalized_procrustes_affine(train, max_iters=60, tol=1e-7)

    plot_gpa_before_after(train, aligned,
                          os.path.join(output_dir, "task2_gpa_before.png"),
                          os.path.join(output_dir, "task2_gpa_after.png"))

    # Flatten aligned training data
    X = aligned.reshape(aligned.shape[0], -1)  # (n,d)
    mu, Phi, lambdas, k, evals_full, _ = pca_fit(X, energy_keep=0.90)

    print("[Task 2] PCA:")
    print(f"  d={X.shape[1]} n={X.shape[0]}  chosen N={k} (>=90% energy)")

    # Save explained variance plot
    total = float(np.sum(evals_full)) + 1e-12
    cum = np.cumsum(evals_full) / total
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(cum)
    ax.set_xlabel("component")
    ax.set_ylabel("cumulative energy")
    ax.set_title("PCA energy")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_pca_energy.png"), dpi=200)
    plt.close(fig)

    # Visualize first 3 PCA modes
    mu_shape = mu.reshape(-1, 2)
    for kk in range(min(3, k)):
        # direction = sqrt(lambda) * phi_k
        basis = (math.sqrt(float(lambdas[kk])) * Phi[:, kk]).astype(np.float32)
        plot_mode_variations(mu_shape, basis, os.path.join(output_dir, f"task2_pca_mode_{kk+1}.png"),
                             title=f"PCA mode {kk+1}")

    # PPCA using same q=k
    mu_pp, W, sigma2, Uq, lam_full = ppca_from_cov(X, q=k)
    print("[Task 2] PPCA:")
    print(f"  q={k} sigma^2={sigma2:.6f}")

    # Visualize first 3 PPCA modes (use Uq and eigenvalues for comparable scaling)
    # Using lam_full which are covariance eigenvalues
    mu_shape_pp = mu_pp.reshape(-1, 2)
    for kk in range(min(3, k)):
        basis = (math.sqrt(float(lam_full[kk])) * Uq[:, kk]).astype(np.float32)
        plot_mode_variations(mu_shape_pp, basis, os.path.join(output_dir, f"task2_ppca_mode_{kk+1}.png"),
                             title=f"PPCA mode {kk+1}")

    # Inference on test: align to mean_shape first
    M_test = best_affine_transform(test, mean_shape)
    test_aligned = apply_affine(test, M_test).astype(np.float32)
    x_test = test_aligned.reshape(-1)

    # PCA reconstruction
    b, xhat = pca_project_reconstruct(x_test, mu, Phi)
    mse_pca = float(np.mean((xhat - x_test) ** 2))

    # PPCA reconstruction
    z, xhat_pp = ppca_posterior_reconstruct(x_test, mu_pp, W, sigma2)
    mse_ppca = float(np.mean((xhat_pp - x_test) ** 2))

    print("[Task 2] Test reconstruction MSE:")
    print(f"  PCA  MSE = {mse_pca:.6f}")
    print(f"  PPCA MSE = {mse_ppca:.6f}")

    # Plot original vs reconstructions
    P = test_aligned.shape[0]
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_shapes(ax, [test_aligned], title="Test shape (aligned)")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_test_aligned.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_shapes(ax, [xhat.reshape(P, 2)], title=f"Test reconstruction (PCA) MSE={mse_pca:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_test_recon_pca.png"), dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(1, 1, 1)
    plot_shapes(ax, [xhat_pp.reshape(P, 2)], title=f"Test reconstruction (PPCA) MSE={mse_ppca:.4f}")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_test_recon_ppca.png"), dpi=200)
    plt.close(fig)

    # Combined comparison figure
    fig = plt.figure(figsize=(12, 4))
    for j, (s, ttl) in enumerate([
        (test_aligned, "Original (aligned)"),
        (xhat.reshape(P, 2), f"PCA recon\nMSE={mse_pca:.4f}"),
        (xhat_pp.reshape(P, 2), f"PPCA recon\nMSE={mse_ppca:.4f}")
    ]):
        ax = fig.add_subplot(1, 3, j + 1)
        plot_shapes(ax, [s], title=ttl)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "task2_test_compare.png"), dpi=200)
    plt.close(fig)

    return {
        "N": k,
        "mse_pca": mse_pca,
        "mse_ppca": mse_ppca,
        "sigma2": sigma2
    }


def main():
    # Resolve data paths (expects data files near this script; also tries current working dir)
    here = os.path.dirname(os.path.abspath(__file__))

    rat_img = _try_paths(os.path.join(here, "rat.webp"), "rat.webp")
    rat_txt = _try_paths(os.path.join(here, "rat.txt"), "rat.txt")
    hands_train = _try_paths(os.path.join(here, "hands_train.npy"), "hands_train.npy")
    hands_test = _try_paths(os.path.join(here, "hands_test.npy"), "hands_test.npy")

    if rat_img is None or rat_txt is None:
        raise RuntimeError("Missing rat.webp or rat.txt (place them next to the script).")
    if hands_train is None or hands_test is None:
        raise RuntimeError("Missing hands_train.npy or hands_test.npy (place them next to the script).")

    outdir = os.path.join(here, "output")
    ensure_dir(outdir)

    # Task 1
    run_task1(outdir, rat_img, rat_txt)

    # Task 2
    stats = run_task2(outdir, hands_train, hands_test)

    # Print for report convenience
    print("\n[Summary numbers]")
    print(f"N (90% energy) = {stats['N']}")
    print(f"MSE PCA  = {stats['mse_pca']:.6f}")
    print(f"MSE PPCA = {stats['mse_ppca']:.6f}")
    print(f"sigma^2 (PPCA) = {stats['sigma2']:.6f}")


if __name__ == "__main__":
    main()
