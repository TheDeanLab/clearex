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
