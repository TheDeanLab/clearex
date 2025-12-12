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
from scipy import ndimage as ndi
import numpy as np
from numpy.typing import NDArray
from skimage import filters, morphology
from skimage.restoration import denoise_nl_means, estimate_sigma

# Local Imports


def segment_tubular_structures(
    img: NDArray[np.floating],
    sigmas: tuple[float, ...] = (1, 2, 3, 4),
    use_sato: bool = True,
    bright_ridges: bool = True,
    min_obj_voxels: int = 200,
    clean_ball_radius: int = 1,
) -> tuple[NDArray[np.bool_], NDArray[np.float32]]:
    """Segment fine, tubular structures using vesselness filtering.

    Detects ridge-like structures (e.g., neurites, axons, blood vessels,
    filaments) using Sato or Frangi vesselness filters followed by
    thresholding and morphological cleanup.

    Parameters
    ----------
    img : NDArray[np.floating]
        3D input image array. Will be normalized to [0, 1] internally.
    sigmas : tuple[float, ...], optional
        Scales (in pixels) for multi-scale vesselness detection. Should be
        tuned to match the expected radii of tubular structures in your data.
        Default is (1, 2, 3, 4).
    use_sato : bool, optional
        If True, use Sato vesselness filter. If False, use Frangi filter.
        Sato is generally more robust for neurites/axons, while Frangi
        suppresses blob/plate responses more strongly. Default is True.
    bright_ridges : bool, optional
        If True, detect bright tubular structures on dark background.
        If False, detect dark structures on bright background.
        Default is True.
    min_obj_voxels : int, optional
        Minimum object size in voxels. Objects smaller than this are removed.
        Default is 200.
    clean_ball_radius : int, optional
        Radius of the ball structuring element used for morphological closing
        to connect small gaps. Default is 1.

    Returns
    -------
    binary_mask : NDArray[np.bool_]
        Binary segmentation mask of detected tubular structures.
    vesselness_response : NDArray[np.float32]
        Normalized vesselness response image in range [0, 1].

    Notes
    -----
    The pipeline consists of:
        1. Intensity normalization to [0, 1] using 1st/99th percentiles
        2. Non-local means denoising to preserve ridge structures
        3. White top-hat transform for background subtraction
        4. Multi-scale Sato or Frangi vesselness filtering
        5. Adaptive thresholding (max of Otsu and 75th percentile)
        6. Morphological closing and small object removal

    Examples
    --------
    >>> mask, response = segment_tubular_structures(volume, sigmas=(1, 2, 3))
    >>> viewer.add_labels(mask.astype(int), name='filaments')
    """
    # 0) Ensure float in [0,1]
    img = np.asarray(img, dtype=np.float32)
    p1, p99 = np.percentile(img, (1, 99))
    img = np.clip((img - p1) / (p99 - p1 + 1e-6), 0, 1)

    # 1) Denoise (3D NL-means is good for preserving ridges)
    sigma_est = float(np.mean(estimate_sigma(img, channel_axis=None)))
    den = denoise_nl_means(
        img,
        h=0.8 * sigma_est,
        fast_mode=True,
        patch_size=3,
        patch_distance=5,
        channel_axis=None,
    ).astype(np.float32)

    # Use a white tophat with a 3D ball structuring element roughly larger than axon radius
    bg_removed = morphology.white_tophat(den, footprint=morphology.ball(5))

    # 2) Vesselness: Sato or Frangi (both support 3D)
    if use_sato:
        resp = filters.sato(bg_removed, sigmas=sigmas, black_ridges=not bright_ridges)
    else:
        resp = filters.frangi(bg_removed, sigmas=sigmas, black_ridges=not bright_ridges)

    # 3) Normalize vesselness response
    resp = (resp - resp.min()) / (resp.max() - resp.min() + 1e-6)

    # 4) Binarize—Otsu is a decent start, but a high percentile often works better
    t_otsu = filters.threshold_otsu(resp)
    t = max(t_otsu, float(np.percentile(resp, 75)))  # more conservative than Otsu
    bw = resp > t

    # 5) Clean up
    # Small morphological close to connect tiny gaps
    bw = morphology.binary_closing(bw, footprint=morphology.ball(clean_ball_radius))

    # Remove small specks
    bw = morphology.remove_small_objects(bw, min_size=min_obj_voxels)

    # (Optional) fill tiny holes inside the tube
    bw = ndi.binary_fill_holes(bw)

    return bw, resp.astype(np.float32)
