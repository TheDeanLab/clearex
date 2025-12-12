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


def get_bounding_box(
    image_shape: tuple[int, int, int],
    roi_slice: tuple[slice, slice, slice],
    buffer: int = 0,
) -> tuple[slice, slice, slice]:
    """
    Expand a regionprops slice tuple (z, y, x) by `buffer` voxels, clamped to image bounds.

    Parameters
    ----------
    image_shape: tuple of int
        Shape of the image (z, y, x).
    roi_slice: tuple of slice
        Original regionprops slice (z, y, x).
    buffer: int
        Number of voxels to expand the slice in each direction.

    Returns
    -------
    tuple of slice
        (sz, sy, sx) expanded and clamped slices.
    """
    z, y, x = image_shape
    sz, sy, sx = roi_slice

    z0: int = max(0, (sz.start or 0) - buffer)
    y0: int = max(0, (sy.start or 0) - buffer)
    x0: int = max(0, (sx.start or 0) - buffer)

    z1: int = min(z, (sz.stop if sz.stop is not None else z) + buffer)
    y1: int = min(y, (sy.stop if sy.stop is not None else y) + buffer)
    x1: int = min(x, (sx.stop if sx.stop is not None else x) + buffer)

    # Guarantee at least one voxel
    if z1 <= z0:
        z1: int = min(z, z0 + 1)
    if y1 <= y0:
        y1: int = min(y, y0 + 1)
    if x1 <= x0:
        x1: int = min(x, x0 + 1)

    return slice(z0, z1), slice(y0, y1), slice(x0, x1)
