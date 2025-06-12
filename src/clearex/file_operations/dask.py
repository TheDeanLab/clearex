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


# Standard imports
import logging
from typing import Optional

# Third-party imports
import dask.array as da
import zarr
import tifffile

# Local imports


def report_chunks(data: da.Array) -> None:
    """Log the number of chunks and their sizes for each dimension in the volumes.

    Parameters
    ----------
    data : dask.array.Array
        Dask array containing the data
    """
    if isinstance(data, da.Array):
        chunks = data.chunks
        dim_labels = ["Position", "Channel", "Z", "Y", "X"]
        dim = len(chunks)

        if dim > len(dim_labels):
            logging.warning(f"Unsupported number of dimensions: {dim}")
            return

        chunk_info = "Chunk Information:\n"
        for i in range(dim):
            chunk_info += (
                f"{dim_labels[-dim + i]} - Chunk Size: {chunks[i][0]}, "
                f"Chunk Length: {len(chunks[i])}\n"
            )

        logging.info(chunk_info)
    else:
        logging.warning("Input is not a Dask array.")


def tiff_to_zarr(
    data_path: str,
    output_path: str,
    position: Optional[int] = 0,
    channel: Optional[int] = 0,
) -> None:
    """Convert a set of TIFF files to a Zarr dataset.

    Parameters
    ----------
    data_path : str
        The path to the TIFF file to convert.
    output_path : str
        The path to save the Zarr dataset.
    position : int
        The position index to save the data to.
    channel : int
        The channel index to save the data to.
    """

    data = tifffile.imread(data_path)  # Load TIFF into memory
    size_z, size_y, size_x = data.shape  # Image dimensions

    # Open or create the Zarr dataset
    zarr_store = zarr.open(
        output_path,
        mode="a",
        shape=(1, 1, size_z, size_y, size_x),  # Default initial shape
        chunks=(1, 1, 256, 256, 256),
        dtype="uint16",
    )

    # Ensure the dataset is large enough
    new_shape = (
        max(zarr_store.shape[0], position + 1),
        max(zarr_store.shape[1], channel + 1),
        size_z,
        size_y,
        size_x,
    )
    if new_shape != zarr_store.shape:
        zarr_store.resize(new_shape)

    zarr_store[position, channel, :, :, :] = data
