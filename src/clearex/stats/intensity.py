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

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance

from typing import Any


def compare_intensity(
    fixed: Any,
    moving: Any,
    *,
    max_sample: int = 100_000,
    use_hist_wasserstein: bool = True,
    hist_bins: int = 256,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compare two intensity distributions with KS and Wasserstein metrics.

    Parameters
    ----------
    fixed, moving : array-like or ANTsImage
        Images or array-like objects. Any object exposing ``.numpy()`` is also
        accepted.
    max_sample : int, default=100000
        Maximum number of pixels drawn from each image before statistics are
        computed.
    use_hist_wasserstein : bool, default=True
        If ``True``, approximate the Wasserstein distance from histograms to
        reduce memory use.
    hist_bins : int, default=256
        Number of histogram bins used when ``use_hist_wasserstein`` is enabled.
    alpha : float, default=0.05
        Significance threshold used for the printed KS-test verdict.
    seed : int, default=0
        Seed for the random subsampling step.

    Returns
    -------
    tuple of float
        ``(D, p_value, emd)`` where ``D`` is the KS statistic, ``p_value`` is
        the KS p-value, and ``emd`` is the Wasserstein distance.
    """

    def to_1d(x: Any, name: str) -> np.ndarray:
        """Convert an image-like input to a flat ``float32`` NumPy vector.

        Parameters
        ----------
        x : Any
            Input image or array-like object.
        name : str
            Name used in error messages.

        Returns
        -------
        numpy.ndarray
            Flattened ``float32`` array view of the input.

        Raises
        ------
        TypeError
            If ``x`` cannot be interpreted as a NumPy array.
        ValueError
            If ``x`` is empty.
        """
        if hasattr(x, "numpy"):
            x = x.numpy()
        if not isinstance(x, np.ndarray):
            raise TypeError(f"{name} must be array-like, got {type(x)}")
        if x.size == 0:
            raise ValueError(f"{name} is empty.")
        return x.ravel().astype(np.float32, copy=False)

    fixed = to_1d(fixed, "fixed")
    moving = to_1d(moving, "moving")

    # Down-sample, if necessary
    rng = np.random.default_rng(seed)
    n = min(max_sample, fixed.size, moving.size)

    fixed_samp = rng.choice(fixed, n, replace=False)
    moving_samp = rng.choice(moving, n, replace=False)

    # Kolmogorov–Smirnov test
    D, p = ks_2samp(fixed_samp, moving_samp, mode="asymp")

    # Wasserstein / Earth-Mover distance
    if use_hist_wasserstein:
        # Compute on *histograms* – O(bins) memory
        lo = min(fixed_samp.min(), moving_samp.min())
        hi = max(fixed_samp.max(), moving_samp.max())
        bins = np.linspace(lo, hi, hist_bins + 1)

        h_fixed, _ = np.histogram(fixed_samp, bins=bins, density=True)
        h_moving, _ = np.histogram(moving_samp, bins=bins, density=True)

        # Mid-points of each bin
        mid = 0.5 * (bins[:-1] + bins[1:])
        emd = wasserstein_distance(mid, mid, h_fixed, h_moving)
    else:
        # Can use a lot of RAM when n is large
        emd = wasserstein_distance(fixed_samp, moving_samp)

    verdict = (
        "Reject H₀ → distributions differ"
        if p < alpha
        else "Fail to reject H₀ → no evidence of a difference"
    )

    print(f"KS D = {D:.4f} | p = {p:.3g} | {verdict}")
    print(f"Wasserstein distance = {emd:.2f} ADU\n")

    return D, p, emd
