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
import os
from typing import Optional, Union, Tuple

# Third Party Imports

# Local Imports
from clearex.registration import Registration
from clearex.io.read import ImageOpener
from clearex import initiate_logger, display_logo, create_parser


def main():
    """Run the ClearEx command line interface."""
    display_logo()

    # Initialize Logging
    logger = initiate_logger(os.getcwd())
    logger.info("Starting ClearEx")

    # Parse command line arguments.
    parser = create_parser()
    args = parser.parse_args()
    logger.info(f"Command line arguments: {args}")

    if args.file:
        # Specify chunking strategy.
        chunks_opt: Optional[Union[int, Tuple[int, ...]]] = None
        if args.chunks:
            if "," in args.chunks:
                chunks_opt = tuple(int(x) for x in args.chunks.split(","))
            else:
                chunks_opt = int(args.chunks)

        opener = ImageOpener()
        arr, info = opener.open(args.file, prefer_dask=args.dask, chunks=chunks_opt)

        logger.info(f"Image shape: {info.shape}")
        logger.info(f"Image dtype:, {info.dtype}")
        if info.axes:
            logger.info(f"Image axes: {info.axes}")
        if info.metadata:
            metadata = {k: type(v).__name__ for k, v in (info.metadata or {}).items()}
            logger.info(f"Image metadata: {metadata}")

    if args.registration:
        Registration()

    if args.visualization:
        print("Launching visualization")


if __name__ == "__main__":
    main()