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
import logging
import os
import argparse
from pathlib import Path
from typing import Optional

# Third Party Imports

# Local Imports
# from clearex.registration import Registration
from clearex.io.read import ImageOpener
from clearex import initiate_logger

CLEAR_EX_LOGO = r"""
       _                          
      | |                         
   ___| | ___  __ _ _ __ _____  __
  / __| |/ _ \/ _` | '__/ _ \ \/ /
 | (__| |  __/ (_| | | |  __/>  < 
  \___|_|\___|\__,_|_|  \___/_/\_\
                                  
"""


def main():
    """Run the ClearEx command line interface."""
    print(CLEAR_EX_LOGO)

    base_path = os.getcwd()
    initiate_logger(base_path)
    logging.info("Starting ClearEx")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Clearex Command Line Arguments")
    input_args = parser.add_argument_group("Input Arguments")

    input_args.add_argument(
        "-r",
        "--registration",
        required=False,
        default=False,
        action="store_true",
        help="Registration Workflow",
    )

    input_args.add_argument(
        "-v",
        "--visualization",
        required=False,
        default=False,
        action="store_true",
        help="Visualization of the data with Neuroglancer",
    )

    parser.add_argument(
        "-f",
        "--file",
        help="Path to image (TIFF/OME‑TIFF, .zarr, .npy/.npz)",
        type=str,
        required=True
    )

    parser.add_argument(
        "--dask",
        action="store_true",
        default=True,
        type=bool,
        help="Return a Dask array when possible"
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Chunk spec for Dask, e.g. '256,256,64' or single int"
    )
    args = parser.parse_args()

    chunks_opt: Optional[Union[int, Tuple[int, ...]]] = None
    if args.chunks:
        if "," in args.chunks:
            chunks_opt = tuple(int(x) for x in args.chunks.split(","))
        else:
            chunks_opt = int(args.chunks)

    opener = ImageOpener()
    arr, info = opener.open(args.file, prefer_dask=args.dask, chunks=chunks_opt)

    print("Loaded:", info.path.name)
    print("  shape:", info.shape)
    print("  dtype:", info.dtype)
    if info.axes:
        print("  axes :", info.axes)
    if info.metadata:
        print("  meta :", {k: type(v).__name__ for k, v in (info.metadata or {}).items()})

    # Example: force compute if Dask
    try:
        import dask.array as da  # re-import safe
        if isinstance(arr, da.Array):
            print("  (Dask) computing small checksum...")
            # tiny checksum to avoid full materialization
            print("  checksum:", da.nanmean(arr[:8]).compute())
    except Exception:
        pass



    if args.registration:
        print("Registration")
        # Registration()
    elif args.visualization:
        print("Launching visualization")
    else:
        exit()


if __name__ == "__main__":
    print("what's up?")
    main()