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

import os

# import pylab as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def mips(
    *channels,
    bounding_boxes=None,
    filepath=None,
    filename=None,
    block_info=None,
    points=None,
    scale_intensity=False,
    lut="gray",
):
    """Display Maximum Intensity Projections (MIPs) for multiple channels.

    Optionally draws bounding box(es) on the MIPs: one bounding box per channel.

    Parameters
    ----------
    *channels : np.ndarray or bool
        Multiple channels to display

    bounding_boxes : list or tuple of 6-tuples, optional
        Each bounding box should be (zmin, ymin, xmin, zmax, ymax, xmax).
        The i-th bounding box corresponds to channels[i].
        If None, no bounding boxes are drawn.

    filepath : str, optional
        If provided, save the plot to this directory.

    filename : str, optional
        If provided, save the plot with this filename.

    block_info : list of dict, optional
        If provided, save the plot with this block information

    points : list of np.ndarray, optional
        Each element is an array of shape (N,3) containing (z, y, x) coordinates
        for detected blobs corresponding to each channel.

    scale_intensity : bool or float, optional
        If False, do not scale the intensity of the images.
        If a float, scale the intensity of the images to this value (e.g., 0.0 - 1.0)

    lut : str, optional
        The lookup table to use for the image. Default is 'gray'. Other options
        include 'nipy_spectral', 'viridis', 'plasma', 'inferno', 'magma', 'cividis'...
    """
    num_channels = len(channels)

    # If bounding_boxes is provided, make sure it has the same number of elements as
    # channels
    if bounding_boxes is not None:
        if len(bounding_boxes) != num_channels:
            raise ValueError(
                f"bounding_boxes has length {len(bounding_boxes)}, "
                f"but there are {num_channels} channels."
            )
    else:
        # Use a list of None so we can uniformly handle below
        bounding_boxes = [None] * num_channels

    plt.figure(figsize=(10, 6))
    for i, channel in enumerate(channels):
        bbox = bounding_boxes[i]
        point = points[i] if points is not None else None

        for axis in range(3):
            # Compute MIPs
            mip_channel = np.max(channel, axis=axis).astype(np.float64)

            # Normalize the images
            if np.max(mip_channel) > 0:
                mip_channel /= np.max(mip_channel)

            # Plot MIPs for each channel
            ax = plt.subplot(num_channels, 3, i * 3 + axis + 1)
            if scale_intensity is False:
                ax.imshow(mip_channel, cmap=lut)
            else:
                ax.imshow(mip_channel, cmap=lut, vmin=0, vmax=scale_intensity)
            ax.axis("off")

            # If we have a bbox for this channel, draw it
            if bbox is not None:
                draw_bbox(ax, bbox, axis)

            # Overlay points if provided
            if point is not None and point.size > 0:
                draw_points(ax, point, axis)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    if filepath is None:
        # Display the plot
        plt.show()
    else:
        # Save the plot
        if filename is None:
            filename = "mip.pdf"

        if block_info is None:
            save_path = os.path.join(filepath, filename)
        else:
            # Dask block information is a list of dictionaries.
            info = block_info[0]
            chunk_location = info["chunk-location"]
            chunk_id = "_".join(map(str, chunk_location))
            filename = f"{chunk_id}_{filename}"
            save_path = os.path.join(filepath, filename)
        plt.savefig(save_path, dpi=300)
        plt.close()


def draw_points(ax: plt.subplot, point: np.ndarray, axis: int) -> None:
    """Draw points on the MIP image.

    Parameters
    ----------
    ax : plt.subplot
        The axis to draw the points on.
    point : np.ndarray
        The points to draw.
    axis : int
        The axis along which the MIP was taken (0, 1, or 2)
    """
    # Project 3D point coordinates onto the 2D plane of the current MIP
    if axis == 0:
        ax.scatter(
            point[:, 2], point[:, 1], color="red", marker="x", linewidths=1, alpha=0.75
        )
    elif axis == 1:
        ax.scatter(
            point[:, 2], point[:, 0], color="red", marker="x", linewidths=1, alpha=0.75
        )
    elif axis == 2:
        ax.scatter(
            point[:, 1], point[:, 0], color="red", marker="x", linewidths=1, alpha=0.75
        )


def draw_bbox(ax: plt.subplot, bbox: tuple, axis: int) -> None:
    """Draw a bounding-box rectangle on the MIP image, given the axis (0, 1, or 2).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to draw the bounding box on.
    bbox : tuple of 6 ints
        The bounding box to draw. (zmin, ymin, xmin, zmax, ymax, xmax)
    axis : int
        The axis along which the MIP was taken (0, 1, or 2)
    """
    zmin, ymin, xmin, zmax, ymax, xmax = bbox

    if axis == 0:
        # MIP over Z => 2D image is (y, x)
        x0, y0 = xmin, ymin
        width, height = (xmax - xmin), (ymax - ymin)
    elif axis == 1:
        # MIP over Y => 2D image is (z, x)
        x0, y0 = xmin, zmin
        width, height = (xmax - xmin), (zmax - zmin)
    else:
        # MIP over X => 2D image is (z, y)
        x0, y0 = ymin, zmin
        width, height = (ymax - ymin), (zmax - zmin)

    rect = patches.Rectangle(
        (x0, y0), width, height, linewidth=1.5, edgecolor="r", facecolor="none"
    )
    ax.add_patch(rect)
