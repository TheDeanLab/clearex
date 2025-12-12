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
import dask.array as da
from dask_jobqueue.slurm import SLURMRunner
from dask.distributed import Client

from clearex.filter.filters import meijering_filter

if __name__ == "__main__":
    base_path = (
        "/archive/bioinformatics/Danuser_lab/"
        "Dean/publication/2024-multiscale/Figure5/38x"
    )
    schedule_path = os.path.join(base_path, "nerves_2025_01_02_v2.json")

    with SLURMRunner(scheduler_file=schedule_path) as runner:
        with Client(runner) as client:
            client.wait_for_workers(runner.n_workers)

            # Read data as a Dask array
            data_path = os.path.join(base_path, "output.zarr")
            data = da.from_zarr(data_path).astype(np.float32)
            segmentation_channel = 1
            data = data[:, segmentation_channel, :, :, :]  # shape: (positions, z, y, x)
            data = data.rechunk((1, 1, 2048, 2048))

            # Apply the filter slice-by-slice in the yx plane.
            filtered = da.map_blocks(
                meijering_filter,
                data,
                sigmas=[2, 5, 7],
                black_ridges=False,
                dtype=np.float32,
            )

            # Write out once using Dask's to_zarr method.
            save_path = os.path.join(base_path, "nerves_v2.zarr")
            filtered.to_zarr(save_path, overwrite=True)  # or mode='w'
