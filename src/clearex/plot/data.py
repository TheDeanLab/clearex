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
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import numpy as np

# Local Imports


def histograms(*images, bins=256, sample=200_000, labels=None):
    """
    Plot side-by-side histograms for an arbitrary number of images,
    using identical bin edges so the distributions are directly comparable.

    Parameters
    ----------
    *images : array_like or ants.core.ants_image.ANTsImage
        One or more 2-D / 3-D images (or already-flat arrays) to visualise.
        Each input is converted to a 1-D ``float32`` NumPy vector.
        ANTsImages are handled transparently via their ``.numpy()`` method.
    bins : int or sequence or str, optional
        Controls the binning rule passed to
        :func:`numpy.histogram_bin_edges`.
        *int* → that many equal-width bins; *str* (e.g. ``"auto"``,
        ``"fd"``) → NumPy’s data-driven rules. Default is ``"auto"``.
    sample : int
        Number of samples to use from the data. Randomly retrieved.
    labels : list of str, optional
        Titles for the per-image subplots (and legend, if added later).
        If *None*, defaults to ``["Image 1", "Image 2", …]``.

    Notes
    -----
    *Each histogram uses the same ``common_bins`` array calculated from the
    concatenation of *all* input vectors to ensure a fair comparison.*

    * Bars are drawn semi-transparent (``alpha = 0.85``) with white edges.

    Examples
    --------
    >>> histograms(img_raw, img_flat, img_matched,
    ...            bins=256,
    ...            labels=['Raw', 'Bias-corrected', 'Histogram-matched'])
    """
    plt.rcParams.update(
        {
            "figure.dpi": 120,  # crisp on screens & slides
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "axes.grid": True,
            "grid.alpha": 0.3,
        }
    )

    rng = np.random.default_rng(0)
    flat = []
    vmin, vmax = np.inf, -np.inf

    # 1. Flatten, sample, track global min/max
    for im in images:
        if hasattr(im, "numpy"):  # ANTsImage
            im = im.numpy()

        vec = np.asarray(im).ravel()  # no .astype
        if vec.size > sample:
            vec = rng.choice(vec, sample, replace=False)

        flat.append(vec)
        vmin = min(vmin, vec.min())
        vmax = max(vmax, vec.max())

    # 2. Common equal-width edges (fast)
    edges = np.linspace(vmin, vmax, bins + 1)

    # 3. Plot
    n = len(flat)
    fig, axes = plt.subplots(1, n, sharex=True, sharey=True, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]

    for ax, vec, title in zip(
        axes, flat, labels or [f"Img {i}" for i in range(1, n + 1)]
    ):
        ax.hist(vec, bins=edges, color="#4472c4", alpha=0.85, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("Intensity")
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    axes[0].set_ylabel("Pixel count")

    fig.suptitle("Histogram Comparison", y=1.03, fontsize=15, weight="semibold")
    fig.tight_layout()
    plt.show()
