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

# Third Party Imports
import tifffile

# Local Imports
from clearex.registration.linear import register_image, inspect_affine_transform
from clearex.registration.common import export_affine_transform, export_tiff
from tests import download_test_registration_data


def main(
    path_to_fixed_data: str, path_to_moving_data: str, path_to_save_to: str
) -> None:
    """Perform linear registration of a moving image to a fixed image.

    Parameters
    ----------
    path_to_fixed_data : str
        Path to the fixed image (target).
    path_to_moving_data : str
        Path to the moving image (to be transformed).
    path_to_save_to : str
        Directory to save the results (transform and registered image).
    """
    print(f"Loading fixed image: {path_to_fixed_data}")
    fixed_roi = tifffile.imread(path_to_fixed_data)

    print(f"Loading moving image: {path_to_moving_data}")
    moving_roi = tifffile.imread(path_to_moving_data)

    print("Registering the data:")
    transformed_image, transform = register_image(
        moving_image=moving_roi,
        fixed_image=fixed_roi,
        registration_type="TRSAA",
        accuracy="low",
        verbose=True,
    )

    print("Inspecting transform:")
    inspect_affine_transform(transform)

    print("Exporting the affine transform to:", path_to_save_to)
    export_affine_transform(transform, path_to_save_to)

    print("Exporting the registered data to:", path_to_save_to)
    export_tiff(
        image=transformed_image,
        data_path=os.path.join(path_to_save_to, "registered.tif"),
    )

    print("Done.")


if __name__ == "__main__":
    output_path = download_test_registration_data()
    fixed_path = os.path.join(output_path, "cropped_fixed.tif")
    moving_path = os.path.join(output_path, "cropped_moving.tif")
    main(fixed_path, moving_path, output_path)
