# Standard Library Imports
import os

# Third Party Imports
import numpy as np
import skimage.filters as skfilters
import dask.array as da
from dask_jobqueue.slurm import SLURMRunner
from dask.distributed import Client

# Local Imports


def apply_meijering(slice2d: np.ndarray, sigmas: list[float], black_ridges: bool) -> np.ndarray:
    """ Apply the Meijering filter to a 2D slice.

    Parameters
    ----------
    slice2d : np.ndarray
        A 2D slice of the data.
    sigmas : list of float
        Standard deviations for Gaussian smoothing.
    black_ridges : bool
        If True, return black ridges on a white background.

    Returns
    -------
    np.ndarray
        The filtered 2D slice.
    """
    return skfilters.meijering(slice2d, sigmas=sigmas, black_ridges=black_ridges)

if __name__ == '__main__':
    base_path = "/archive/bioinformatics/Danuser_lab/Dean/publication/2024-multiscale/Figure5/38x"
    schedule_path = os.path.join(base_path, "nerves_2025_01_02_v2.json")

    with SLURMRunner(scheduler_file=schedule_path) as runner:
        with Client(runner) as client:
            client.wait_for_workers(runner.n_workers)

            # Read data as a Dask array
            data_path = os.path.join(base_path, 'output.zarr')
            data = da.from_zarr(data_path).astype(np.float32)
            segmentation_channel = 1
            data = data[:, segmentation_channel, :, :, :]  # shape: (positions, z, y, x)
            data = data.rechunk((1, 1, 2048, 2048))

            # Apply the filter slice-by-slice in the yx plane.
            filtered = da.map_blocks(
                apply_meijering,
                data,
                sigmas=[2, 5, 7],
                black_ridges=False,
                dtype=np.float32,
            )

            # Write out once using Dask's to_zarr method.
            save_path = os.path.join(base_path, 'nerves_v2.zarr')
            filtered.to_zarr(
                save_path,
                overwrite=True  # or mode='w'
            )
