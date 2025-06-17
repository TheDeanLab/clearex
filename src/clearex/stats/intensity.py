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

def compare_intensity(fixed, moving, *,
                        max_sample=100_000,
                        use_hist_wasserstein=True,
                        hist_bins=256,
                        alpha=0.05,
                        seed=0):
    """
    KS + Wasserstein comparison of two images, memory-safe.

    Compare two image-intensity populations with a distribution-free test
    and a complementary effect-size metric.

    Why these tests?
    ----------------
    * Two-sample Kolmogorov–Smirnov (KS) test
      A non-parametric test that compares the entire empirical cumulative
      distribution functions (ECDFs) of the two samples.
      It makes no assumption about normality or equal variances and is
      sensitive to differences in location, spread, and tail behaviour—
      ideal for raw pixel intensities that are integer-quantised and often
      skewed.

    * 1-Wasserstein / Earth-Mover distance (EMD)
      A scalar effect size that measures how far one distribution must be
      “moved” to match the other (same units as the data, e.g. ADU).
      Reporting EMD alongside the KS p-value prevents over-interpreting
      trivial but statistically significant differences when the sample size
      is large.

    Null hypotheses
    ---------------
    * KS testEMSPH₀ : The two intensity distributions are identical.
      We reject H₀ when the maximum vertical distance between their ECDFs
      (statistic D) is large enough that the associated p-value
      <p α (default α = 0.05).

    * EMDEMSPNo formal hypothesis—EMD is descriptive.
      Smaller values imply more similar distributions; thresholds for
      “practical equivalence” should be defined by domain expertise.

    Sampling strategy
    -----------------
    * Convert each volume to a flat NumPy `float32` vector.
    * Randomly sample up to `max_sample` pixels without replacement;
      this preserves distribution shape while limiting `n`.
    * Optionally compute EMD on coarse histograms (`hist_bins` ≤ 256),
        reducing RAM from O(n) to O(bins) while leaving KS unaffected.

    Parameters
    ----------
    fixed, moving : array-like or ANTsImage
        2-D/3-D images.  Anything with `.numpy()` is accepted.
    max_sample : int
        Cap on the *pixel* sample taken from each image.
    use_hist_wasserstein : bool
        If True, compute the Wasserstein distance on coarse histograms
        (fast, <10 MB); otherwise use the exact sample (can be heavy).
    hist_bins : int
        Number of bins per histogram when `use_hist_wasserstein=True`.
    alpha : float
        Significance level for the KS test.
    seed : int
        RNG seed for reproducibility.
    """

    def to_1d(x, name):
        """ Helper function to convert o a flat 1d float32 array. """
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

    verdict = ("Reject H₀ → distributions differ"
               if p < alpha else
               "Fail to reject H₀ → no evidence of a difference")

    print(f"KS D = {D:.4f} | p = {p:.3g} | {verdict}")
    print(f"Wasserstein distance = {emd:.2f} ADU\n")

    return D, p, emd