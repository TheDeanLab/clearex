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
from clearex.registration.linear import (
    inspect_affine_transform
)
from clearex.registration.common import transform_image, export_tiff, \
    import_affine_transform


def main(
    fixed_path: str,
    moving_path: str,
    transform_path: str,
    output_path: str,
):
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
    # Example file paths — consider replacing with argparse or env vars in real use
    channel = 1
    fixed_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/Sytox_ppm1/250320/new_Cell4/1_CH0{channel}_000000.tif'
    moving_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/restained/Cell1/1_CH0{channel}_000000.tif'
    base_path = "/archive/bioinformatics/Danuser_lab/Dean/dean/2025-05-28-registration"
    transform_path = os.path.join(base_path, "AffineTransform.mat")
    output_path = os.path.join(base_path, "registered_test.tif")

    main(fixed_path, moving_path, transform_path, output_path)