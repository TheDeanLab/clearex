"""
Example script demonstrating chunked registration for large volumes.

This script shows how to use the ChunkedRegistration class to register
large 3D image volumes using a multi-scale hierarchical approach.
"""

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
from pathlib import Path
from clearex.registration.chunked import ChunkedRegistration


# Example usage
def main():
    # Define paths to your data
    fixed_image_path = Path("path/to/fixed_image.tif")  # or .zarr
    moving_image_path = Path("path/to/moving_image.tif")  # or .zarr
    output_dir = Path("output/chunked_registration")

    # Create chunked registration object
    chunked_reg = ChunkedRegistration(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        output_dir=output_dir,
        downsample_factor=4,  # 4x downsampling for coarse registration
        chunk_size=256,  # 256^3 voxel chunks
        overlap_fraction=0.15,  # 15% overlap between chunks
        prefer_dask=True,  # Use dask for lazy loading
    )

    # Run the complete pipeline
    output_image_path = output_dir / "registered_volume.tif"
    chunked_reg.run_full_pipeline(
        output_path=output_image_path,
        linear_type="Affine",
        nonlinear_type_coarse="SyNOnly",
        nonlinear_type_fine="SyNOnly",
        linear_accuracy="low",
        coarse_accuracy="medium",
        fine_accuracy="high",
        interpolator="linear",
        blend_sigma=10.0,
    )

    print(f"Registration complete! Output saved to: {output_image_path}")


# Alternative: Step-by-step usage
def step_by_step_example():
    fixed_image_path = Path("path/to/fixed_image.tif")
    moving_image_path = Path("path/to/moving_image.tif")
    output_dir = Path("output/chunked_registration")

    chunked_reg = ChunkedRegistration(
        fixed_image_path=fixed_image_path,
        moving_image_path=moving_image_path,
        output_dir=output_dir,
        downsample_factor=8,  # More aggressive downsampling
        chunk_size=(128, 256, 256),  # Different sizes per dimension
        overlap_fraction=0.2,
        prefer_dask=True,
    )

    # Step 1: Perform coarse global registration
    print("Step 1: Coarse global registration...")
    linear_path, nonlinear_path = chunked_reg.perform_coarse_registration(
        linear_type="Affine",
        nonlinear_type="SyNOnly",
        linear_accuracy="low",
        nonlinear_accuracy="medium",
    )
    print(f"  Linear transform: {linear_path}")
    print(f"  Nonlinear transform: {nonlinear_path}")

    # Step 2: Register all chunks
    print("\nStep 2: Fine registration on chunks...")
    chunked_reg.register_all_chunks(
        nonlinear_type="SyNOnly",
        nonlinear_accuracy="high",
    )
    print(f"  Registered {len(chunked_reg.chunk_info_list)} chunks")

    # Step 3: Apply transforms and blend
    print("\nStep 3: Applying transforms and blending...")
    output_path = output_dir / "registered_volume.zarr"
    chunked_reg.apply_transforms_and_blend(
        output_path=output_path,
        interpolator="linear",
        blend_sigma=10.0,
    )
    print(f"  Output saved to: {output_path}")


if __name__ == "__main__":
    # Run the simple example
    main()

    # Or run the step-by-step example
    # step_by_step_example()
