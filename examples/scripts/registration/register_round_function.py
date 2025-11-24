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
Example script demonstrating the register_round function.

This script shows how to use the register_round convenience function
to register a moving image to a fixed reference image using combined
linear and nonlinear transformations.
"""

# Standard Library Imports
import os

# Local Imports
from clearex.registration import register_round
from tests import download_test_registration_data


def main_functional():
    """Perform registration using the register_round function."""
    print("=== Functional Registration Example ===\n")

    # Download test data
    output_path = download_test_registration_data()
    fixed_path = os.path.join(output_path, "clearex", "cropped_fixed.tif")
    moving_path = os.path.join(output_path, "clearex", "cropped_moving.tif")

    print(f"Fixed image: {fixed_path}")
    print(f"Moving image: {moving_path}")
    print(f"Output directory: {output_path}\n")

    # Simple one-line registration
    print("Performing registration...")

    # Uncomment to actually run registration:
    # register_round(
    #     fixed_image_path=fixed_path,
    #     moving_image_path=moving_path,
    #     save_directory=output_path,
    #     imaging_round=1,
    #     crop=False,
    #     enable_logging=True,
    # )

    print("Registration configured (not executed in this example)\n")
    print("Done! See examples/scripts/registration/image_registration_class.py")
    print("for the class-based approach using ImageRegistration.")


if __name__ == "__main__":
    main_functional()

