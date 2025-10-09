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
import argparse


def create_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Command Line Arguments")
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
        required=False,
    )

    parser.add_argument(
        "--dask",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Return a Dask array when possible (use --dask to enable, --no-dask to disable)",
    )
    parser.add_argument(
        "--chunks",
        type=str,
        default=None,
        help="Chunk spec for Dask, e.g. '256,256,64' or single int",
    )

    return parser


def display_logo():
    logo = r"""
           _                          
          | |                         
       ___| | ___  __ _ _ __ _____  __
      / __| |/ _ \/ _` | '__/ _ \ \/ /
     | (__| |  __/ (_| | | |  __/>  < 
      \___|_|\___|\__,_|_|  \___/_/\_\
    
    """
    print(logo)
