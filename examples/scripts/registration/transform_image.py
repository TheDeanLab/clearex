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
from clearex.registration.linear import inspect_affine_transform, transform_image
from clearex.registration.common import export_tiff, import_affine_transform
from tests import download_test_registration_data


def main(
    fixed_path: str,
    moving_path: str,
    transform_path: str,
    output_path: str,
):
    """Apply an affine transform to a moving image to align it with a fixed image.

    Parameters
    ----------
    fixed_path : str
        Path to the fixed image (target).
    moving_path : str
        Path to the moving image (to be transformed).
    transform_path : str
        Path to the affine transform file (in .mat format).
    output_path : str
        Path to save the transformed moving image.
    """
    print(f"Loading fixed image: {fixed_path}")
    fixed_roi = tifffile.imread(fixed_path)

    print(f"Loading moving image: {moving_path}")
    moving_roi = tifffile.imread(moving_path)

    print(f"Importing affine transform from: {transform_path}")
    transform = import_affine_transform(transform_path)

    print("Inspecting transform:")
    inspect_affine_transform(transform)

    print("Applying transform to moving image...")
    transformed_image = transform_image(moving_roi, fixed_roi, transform)

    print(f"Exporting registered image to: {output_path}")
    export_tiff(image=transformed_image, data_path=output_path)

    print("Done.")


if __name__ == "__main__":
    output_path = download_test_registration_data()
    fixed_path = os.path.join(output_path, "cropped_fixed.tif")
    moving_path = os.path.join(output_path, "cropped_moving.tif")
    transform_path = os.path.join(output_path, "GenericAffine.mat")
    save_path = os.path.join(output_path, "transformed.tif")
    main(
        fixed_path=fixed_path,
        moving_path=moving_path,
        transform_path=transform_path,
        output_path=save_path,
    )
