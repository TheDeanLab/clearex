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
from skimage import img_as_float
from skimage.exposure import rescale_intensity
from skimage.filters import gaussian
from skimage.feature import blob_dog
import numpy as np

# Local Imports
from clearex.filter.filters import fwhm_to_sigma


def detect_particles(
        img,
        fwhm_px=10.0,
        sigma_min_factor=0.7,
        sigma_max_factor=1.7,
        sigma_ratio=1.2,
        threshold=0.03,
        overlap=0.5, exclude_border=0):
    """
    blob_dog assumes blobs are bright-on-dark. If your mitochondria are dark-on-bright,
    invert first: img = 1 - img.

    Uses threshold_rel (supported in modern scikit-image) for stability after normalization.
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
        exclude_border=exclude_border
    )
    # blobs: array of shape (N, 3) with columns [row, col, sigma]
    return blobs


def preprocess(img, bg_sigma=20):
    """ Preprocess an image by converting to float and background subtracting.

    Parameters

    """
    img = img_as_float(img)
    if img.ndim == 3:
        # If RGB/multi-channel, use mean. Adjust if you have a specific channel.
        img = img.mean(axis=-1)

    bg = gaussian(img, sigma=bg_sigma, preserve_range=True)
    hp = img - bg
    hp = rescale_intensity(hp, in_range="image", out_range=(0, 1))
    return hp

def intensity_weighted_centroids(img, blobs, radius_factor=2.0, local_baseline_percentile=20):
    """
    Optional: refine blob centers to subpixel-ish locations using intensity-weighted centroid.
    This can improve registration error estimates.
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
