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

"""
Example script demonstrating the ImageRegistration class.

This script shows how to use the ImageRegistration class to register
a moving image to a fixed reference image using combined linear and
nonlinear transformations.
"""

# Standard Library Imports
import os

# Local Imports
from clearex.registration import ImageRegistration
from tests import download_test_registration_data


def main_class_based():
    """Perform registration using the ImageRegistration class."""
    print("=== Class-Based Registration Example ===\n")

    # Download test data
    output_path = download_test_registration_data()
    fixed_path = os.path.join(output_path, "clearex", "cropped_fixed.tif")
    moving_path = os.path.join(output_path, "clearex", "cropped_moving.tif")

    print(f"Fixed image: {fixed_path}")
    print(f"Moving image: {moving_path}")
    print(f"Output directory: {output_path}\n")

    # Method 1: Using class with parameters at initialization
    print("Method 1: Initialize with parameters, then call register()")
    registrar = ImageRegistration(
        fixed_image_path=fixed_path,
        moving_image_path=moving_path,
        save_directory=output_path,
        imaging_round=1,
        crop=False,
        enable_logging=True,
    )

    # Perform registration
    # Uncomment to actually run registration:
    # registrar.register()
    print("Registration configured (not executed in this example)\n")

    # Method 2: Using class with parameters passed to register()
    print("Method 2: Initialize without parameters, pass them to register()")
    registrar2 = ImageRegistration(enable_logging=True)

    # Uncomment to actually run registration:
    # registrar2.register(
    #     fixed_image_path=fixed_path,
    #     moving_image_path=moving_path,
    #     save_directory=output_path,
    #     imaging_round=2,
    # )
    print("Registration configured (not executed in this example)\n")

    # Method 3: Reusing the same registrar for multiple rounds
    print("Method 3: Reuse registrar for multiple rounds")
    registrar3 = ImageRegistration(
        fixed_image_path=fixed_path,
        save_directory=output_path,
        enable_logging=True,
    )

    # Register multiple rounds
    # for round_num in range(1, 4):
    #     print(f"Registering round {round_num}...")
    #     registrar3.register(
    #         moving_image_path=f"path/to/round_{round_num}.tif",
    #         imaging_round=round_num,
    #     )
    print("Multi-round registration configured (not executed in this example)\n")

    print("Done! See examples/scripts/registration/register_round_function.py")
    print("for the functional approach using register_round().")


if __name__ == "__main__":
    main_class_based()

