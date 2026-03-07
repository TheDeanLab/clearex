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

import numpy as np
from numpy.typing import NDArray
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.feature import blob_dog
from skimage.filters import gaussian

# Local Imports
from clearex.filter.filters import fwhm_to_sigma


def detect_particles(
    img: NDArray[np.floating | np.integer],
    fwhm_px: float = 10.0,
    sigma_min_factor: float = 0.7,
    sigma_max_factor: float = 1.7,
    sigma_ratio: float = 1.2,
    threshold: float = 0.03,
    overlap: float = 0.5,
    exclude_border: int | tuple[int, ...] | bool = 0,
) -> NDArray[np.floating]:
    """Detect bright blob-like particles in a normalized 2D image.

    Parameters
    ----------
    img : numpy.ndarray
        Input image containing bright particles on a dark background. If the
        source data are dark-on-bright, invert the image before calling this
        helper.
    fwhm_px : float, default=10.0
        Expected particle full width at half maximum, in pixels.
    sigma_min_factor : float, default=0.7
        Multiplier applied to the inferred sigma when computing the minimum
        scale for Difference-of-Gaussians detection.
    sigma_max_factor : float, default=1.7
        Multiplier applied to the inferred sigma when computing the maximum
        scale for Difference-of-Gaussians detection.
    sigma_ratio : float, default=1.2
        Geometric ratio between successive scales evaluated by
        :func:`skimage.feature.blob_dog`.
    threshold : float, default=0.03
        Absolute detection threshold passed to :func:`blob_dog`.
    overlap : float, default=0.5
        Maximum allowed overlap fraction between detected blobs before the
        weaker blob is suppressed.
    exclude_border : int or tuple of int or bool, default=0
        Border exclusion rule forwarded to :func:`blob_dog`.

    Returns
    -------
    numpy.ndarray
        Detected blobs with columns ``(row, col, sigma)``.
    """
    sigma0 = fwhm_to_sigma(fwhm_px)
    min_sigma = sigma_min_factor * sigma0
    max_sigma = sigma_max_factor * sigma0

    blobs = blob_dog(
        img,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        sigma_ratio=sigma_ratio,
        threshold=threshold,
        overlap=overlap,
        exclude_border=exclude_border,
    )
    # blobs: array of shape (N, 3) with columns [row, col, sigma]
    return blobs


def preprocess(
    img: NDArray[np.floating | np.integer], bg_sigma: float = 20
) -> NDArray[np.floating]:
    """Normalize an image for particle detection.

    Parameters
    ----------
    img : numpy.ndarray
        Input 2D image or 3D image with the channel axis last.
    bg_sigma : float, default=20
        Gaussian sigma used to estimate the low-frequency background.

    Returns
    -------
    numpy.ndarray
        Background-subtracted image rescaled to the ``[0, 1]`` range.
    """
    img = img_as_float(img)
    if img.ndim == 3:
        # If RGB/multi-channel, use mean. Adjust if you have a specific channel.
        img = img.mean(axis=-1)

    bg = gaussian(img, sigma=bg_sigma, preserve_range=True)
    hp = img - bg
    hp = rescale_intensity(hp, in_range="image", out_range=(0, 1))
    return hp


def intensity_weighted_centroids(
    img: NDArray[np.floating | np.integer],
    blobs: NDArray[np.floating | np.integer],
    radius_factor: float = 2.0,
    local_baseline_percentile: float = 20,
) -> NDArray[np.floating]:
    """Refine blob centroids with local intensity weighting.

    Parameters
    ----------
    img : numpy.ndarray
        Input 2D image used to compute local weighted centroids.
    blobs : numpy.ndarray
        Blob detections with columns ``(row, col, sigma)``.
    radius_factor : float, default=2.0
        Multiplier that expands the centroiding window relative to blob scale.
    local_baseline_percentile : float, default=20
        Local intensity percentile subtracted before computing weighted
        coordinates.

    Returns
    -------
    numpy.ndarray
        Refined blob coordinates with the same ``(row, col, sigma)`` layout as
        the input detections.
    """
    refined = []
    H, W = img.shape

    for r, c, s in blobs:
        # use a window proportional to scale; r≈sqrt(2)*sigma, and expand
        rad = int(np.ceil(radius_factor * np.sqrt(2) * s))
        r0 = int(np.round(r))
        c0 = int(np.round(c))

        r1, r2 = max(r0 - rad, 0), min(r0 + rad + 1, H)
        c1, c2 = max(c0 - rad, 0), min(c0 + rad + 1, W)

        patch = img[r1:r2, c1:c2]
        baseline = np.percentile(patch, local_baseline_percentile)
        w = patch - baseline
        w[w < 0] = 0

        if w.sum() <= 0:
            refined.append((float(r), float(c), float(s)))
            continue

        rr, cc = np.mgrid[r1:r2, c1:c2]
        r_ref = float((rr * w).sum() / w.sum())
        c_ref = float((cc * w).sum() / w.sum())
        refined.append((r_ref, c_ref, float(s)))

    return np.array(refined, dtype=float)
