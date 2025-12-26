#  Copyright (c) 2021-2025  The University of Texas Southwestern Medical Center.
#  All rights reserved.
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted for academic and research use only (subject to the
#  limitations in the disclaimer below) provided that the following conditions are met:
#       * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#       * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#       * Neither the name of the copyright holders nor the names of its
#       contributors may be used to endorse or promote products derived from this
#       software without specific prior written permission.
#  NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
#  THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
#  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
#  PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
#  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
#  IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
#  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#  POSSIBILITY OF SUCH DAMAGE.

# Standard Library Imports

# Third Party Imports
from skimage.feature import BRIEF, match_descriptors, plot_matched_features
from skimage.measure import ransac
from skimage.transform import SimilarityTransform, AffineTransform
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# Local Imports
from clearex.detect.particles import preprocess, detect_particles, intensity_weighted_centroids



def brief_descriptors_at_keypoints(img, keypoints_rc, patch_size=31, n_bits=256):
    """
    Compute BRIEF descriptors at given (row, col) keypoints.
    BRIEF produces binary descriptors; match with Hamming distance.  [oai_citation:1‡Scikit-image](https://scikit-image.org/docs/0.25.x/auto_examples/features_detection/plot_brief.html?utm_source=chatgpt.com)
    """
    keypoints_rc = np.asarray(keypoints_rc, dtype=float)
    kpi = np.round(keypoints_rc).astype(int)

    half = patch_size // 2
    H, W = img.shape
    ok = (
        (kpi[:, 0] >= half) & (kpi[:, 0] < H - half) &
        (kpi[:, 1] >= half) & (kpi[:, 1] < W - half)
    )
    kpi = kpi[ok]

    brief = BRIEF(descriptor_size=n_bits, patch_size=patch_size)
    brief.extract(img, kpi)
    # brief.mask indicates which keypoints got a descriptor (should be all after border filtering)
    return kpi[brief.mask], brief.descriptors


def xy_from_rc(rc):
    """Convert (row, col) -> (x, y) == (col, row) for skimage.transform models."""
    rc = np.asarray(rc)
    return np.stack([rc[:, 1], rc[:, 0]], axis=1).astype(float)


def fit_transform_ransac(src_xy, dst_xy, model="similarity", residual_threshold=2.0, max_trials=1000):
    """
    Fit robust transform using RANSAC, as recommended when using match_descriptors
    due to spurious matches.  [oai_citation:2‡Scikit-image](https://scikit-image.org/docs/0.25.x/user_guide/geometrical_transform.html?utm_source=chatgpt.com)
    """
    if model == "affine":
        ModelClass = AffineTransform
        min_samples = 3
    else:
        ModelClass = SimilarityTransform
        min_samples = 2  # Similarity can be estimated with 2 point pairs (though 3+ is safer)

    # Use at least 3 if you can, for stability:
    min_samples = max(min_samples, 3) if len(src_xy) >= 3 else min_samples

    model_robust, inliers = ransac(
        (src_xy, dst_xy),
        ModelClass,
        min_samples=min_samples,
        residual_threshold=residual_threshold,
        max_trials=max_trials
    )
    return model_robust, inliers


def registration_errors(model, src_xy, dst_xy):
    pred = model(src_xy)
    residuals = np.linalg.norm(pred - dst_xy, axis=1)
    rms = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else np.nan
    med = float(np.median(residuals)) if residuals.size else np.nan
    p95 = float(np.percentile(residuals, 95)) if residuals.size else np.nan
    return residuals, rms, med, p95


def cv_tre_proxy(src_xy, dst_xy, n_splits=20, test_fraction=0.2, random_state=0, model="similarity"):
    """
    Cross-validated proxy for TRE:
      - repeatedly hold out a subset of matches
      - fit transform on the rest (least squares estimate)
      - compute RMS error on held-out points

    This is closer to TRE than reporting residuals on the same points used to fit (FRE).
    """
    rng = np.random.default_rng(random_state)
    n = len(src_xy)
    if n < 4:
        return np.nan, np.array([])

    ModelClass = AffineTransform if model == "affine" else SimilarityTransform

    test_size = max(1, int(round(test_fraction * n)))
    errs = []

    for _ in range(n_splits):
        idx = rng.permutation(n)
        test_idx = idx[:test_size]
        train_idx = idx[test_size:]

        if len(train_idx) < 3:
            continue

        m = ModelClass()
        ok = m.estimate(src_xy[train_idx], dst_xy[train_idx])
        if not ok:
            continue

        pred = m(src_xy[test_idx])
        e = np.linalg.norm(pred - dst_xy[test_idx], axis=1)
        errs.extend(e.tolist())

    errs = np.array(errs, dtype=float)
    tre_rms = float(np.sqrt(np.mean(errs ** 2))) if errs.size else np.nan
    return tre_rms, errs


def particle_registration_tre(
    img1,
    img2,
    fwhm_px=10.0,
    invert=False,
    thresholds=(0.03, 0.03),
    brief_patch_size=31,
    model="similarity",
    ransac_residual_threshold=2.0,
    show_debug=False,
    optimize=False,
):
    """
    Returns:
      dict with blobs, matches, inliers, transform, FRE, CV-TRE proxy
    """
    # If the image dimensions are >2, raise an error. Only implemented in 2D currently.
    if img1.ndim != 2 or img2.ndim != 2:
        raise ValueError("Only 2D images are supported.")

    # 1) preprocess
    I1 = preprocess(img1, 50)
    I2 = preprocess(img2, 50)

    if invert:
        I1 = 1.0 - I1
        I2 = 1.0 - I2

    # 2) detect blobs
    # exclude_border should be >= half patch size so BRIEF can be computed
    exclude_border = brief_patch_size // 2
    img1_thresh = thresholds[0]
    img2_thresh = thresholds[1]

    b1 = detect_particles(
        I1,
        fwhm_px=fwhm_px,
        threshold=img1_thresh,
        exclude_border=exclude_border
    )
    b2 = detect_particles(
        I2,
        fwhm_px=fwhm_px,
        threshold=img2_thresh,
        exclude_border=exclude_border
    )
    print(f"Detected {len(b1)} blobs in image 1 and {len(b2)} blobs in image 2.")

    if optimize:
        # Visualize blobs on round 1
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(I1, cmap='gray', vmin=0, vmax=np.percentile(I1, 99))
        ax.scatter(b1[:, 1], b1[:, 0], s=5, facecolors='none', edgecolors='r')
        ax.axis('off')
        plt.show()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(I2, cmap='gray', vmin=0, vmax=np.percentile(I2, 99))
        ax.scatter(b2[:, 1], b2[:, 0], s=5, facecolors='none', edgecolors='r')
        ax.axis('off')
        plt.show()

        return None

    # 3) refine centers (optional but recommended)
    b1r = intensity_weighted_centroids(I1, b1)
    b2r = intensity_weighted_centroids(I2, b2)

    # 4) descriptors at blob centers
    kp1_rc, d1 = brief_descriptors_at_keypoints(I1, b1r[:, :2], patch_size=brief_patch_size)
    kp2_rc, d2 = brief_descriptors_at_keypoints(I2, b2r[:, :2], patch_size=brief_patch_size)

    # 5) match descriptors
    # Hamming is appropriate for BRIEF/binary descriptors; match_descriptors supports it.  [oai_citation:3‡Scikit-image](https://scikit-image.org/docs/0.25.x/api/skimage.feature.html)
    matches = match_descriptors(
        d1, d2,
        metric="hamming",
        cross_check=False,
        max_ratio=1.0  # keep all matches for RANSAC to filter
    )

    src_rc = kp1_rc[matches[:, 0]]
    dst_rc = kp2_rc[matches[:, 1]]

    src_xy = xy_from_rc(src_rc)
    dst_xy = xy_from_rc(dst_rc)

    # 6) robust transform (RANSAC)
    model_robust, inliers = fit_transform_ransac(
        src_xy,
        dst_xy,
        model=model,
        residual_threshold=ransac_residual_threshold,
        max_trials=1*10**6
    )

    # Refit on inliers (least squares) for final residuals
    in_src = src_xy[inliers]
    in_dst = dst_xy[inliers]

    final_model = (AffineTransform() if model == "affine" else SimilarityTransform())
    final_model.estimate(in_src, in_dst)

    # 7) errors
    in_resid, fre_rms, fre_med, fre_p95 = registration_errors(final_model, in_src, in_dst)
    cv_rms, cv_errs = cv_tre_proxy(in_src, in_dst, model=model)

    out = {
        "n_blobs_img1": int(len(b1)),
        "n_blobs_img2": int(len(b2)),
        "n_keypoints_img1": int(len(kp1_rc)),
        "n_keypoints_img2": int(len(kp2_rc)),
        "n_matches": int(len(matches)),
        "n_inliers": int(np.sum(inliers)),
        "transform_model": final_model,
        "fre_inlier_residuals_px": in_resid,
        "fre_rms_px": fre_rms,
        "fre_median_px": fre_med,
        "fre_p95_px": fre_p95,
        "cv_tre_rms_px": cv_rms,
        "cv_tre_errors_px": cv_errs,
        "blobs1": b1r,
        "blobs2": b2r,
        "kp1_rc": kp1_rc,
        "kp2_rc": kp2_rc,
        "matches": matches,
        "inliers": inliers,
    }

    if show_debug:
        m_in = matches[inliers]  # shape (n_inliers, 2), indices into kp1_rc/kp2_rc

        # Optional: subsample so plotting doesn't hang with hundreds of lines
        if len(m_in) > 200:
            rng = np.random.default_rng(0)
            m_in = m_in[rng.choice(len(m_in), size=200, replace=False)]

        fig, ax = plt.subplots(figsize=(10, 5))
        plot_matched_features(
            I1, I2,
            keypoints0=kp1_rc,
            keypoints1=kp2_rc,
            matches=m_in,
            ax=ax
        )
        ax.set_title(f"Inlier matches (n={len(m_in)})")
        plt.tight_layout()
        plt.show()

        print(f"Blobs: img1={out['n_blobs_img1']}, img2={out['n_blobs_img2']}")
        print(f"Keypoints: img1={out['n_keypoints_img1']}, img2={out['n_keypoints_img2']}")
        print(f"Matches={out['n_matches']}, Inliers={out['n_inliers']}")
        print(f"FRE RMS (inliers) = {out['fre_rms_px']:.3f} px; median={out['fre_median_px']:.3f} px; p95={out['fre_p95_px']:.3f} px")
        print(f"CV-TRE proxy RMS   = {out['cv_tre_rms_px']:.3f} px")

    return out


def mutual_nn_pairs(fixed_rc, moving_rc, max_dist=None):
    """
    Find mutual nearest neighbor pairs between two point sets.

    Parameters:
    -----------
    fixed_rc : ndarray (N, 2)
        Fixed points in (row, col) format
    moving_rc : ndarray (M, 2)
        Moving points in (row, col) format (should already be transformed to fixed space)
    max_dist : float, optional
        Maximum distance threshold for valid pairs

    Returns:
    --------
    pairs : ndarray (K, 2)
        Indices of matched pairs: [moving_idx, fixed_idx]
    dists : ndarray (K,)
        Distances for each pair
    """
    tree_fixed = cKDTree(fixed_rc)
    tree_moving = cKDTree(moving_rc)

    # For each moving point, find nearest fixed point
    d_m2f, idx_m2f = tree_fixed.query(moving_rc, k=1)

    # For each fixed point, find nearest moving point
    d_f2m, idx_f2m = tree_moving.query(fixed_rc, k=1)

    # Find mutual matches: moving[i] -> fixed[j] AND fixed[j] -> moving[i]
    idx_moving = np.arange(len(moving_rc))
    mutual_mask = idx_f2m[idx_m2f] == idx_moving

    if max_dist is not None:
        mutual_mask &= d_m2f < max_dist

    pairs = np.column_stack([idx_moving[mutual_mask], idx_m2f[mutual_mask]])
    dists = d_m2f[mutual_mask]

    return pairs, dists