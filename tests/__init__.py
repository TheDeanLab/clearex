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

import requests
import zipfile
import os
from pathlib import Path


def download_test_registration_data() -> str:
    """Download test registration data from the cloud server.

    The data is downloaded as a zip file and extracted to the project root directory
    (clearex/). If the data already exists, it will not be downloaded again.
    The zip file is deleted after extraction.

    Returns
    -------
    str
        Path to the extracted data directory.
    """
    # Get the project root directory (clearex/)
    # This function is in clearex/tests/__init__.py, so go up two levels
    project_root = Path(__file__).resolve().parent.parent

    # Define the output directory path in the project root
    output_dir = project_root / "downloaded_data"

    # Check if the data already exists
    # The zip extracts files directly into downloaded_data/
    if output_dir.exists() and output_dir.is_dir():
        # Verify that the expected files exist
        expected_files = ["cropped_fixed.tif", "cropped_moving.tif", "GenericAffine.mat"]
        files_exist = all((output_dir / f).exists() for f in expected_files)

        if files_exist:
            print(f"Test data already exists at {output_dir}. Skipping download.")
            return str(output_dir)

    # Data doesn't exist or is incomplete, proceed with download
    url = "https://zenodo.org/api/records/17591393/files-archive"
    zip_path = project_root / "downloaded_data.zip"

    print(f"Downloading test data from {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    print(f"Downloaded to {zip_path}")

    print(f"Extracting to {output_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Extracted to {output_dir}")

    # Clean up zip file
    os.remove(zip_path)
    print(f"Removed temporary zip file")

    return str(output_dir)
