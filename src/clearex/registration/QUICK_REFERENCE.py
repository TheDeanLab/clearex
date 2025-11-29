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
Quick Reference: ChunkedRegistration
=====================================

BASIC USAGE
-----------
from clearex.registration.chunked import ChunkedRegistration

chunked_reg = ChunkedRegistration(
    fixed_image_path="reference.tif",
    moving_image_path="moving.tif",
    output_dir="output",
    downsample_factor=4,      # 4-8 recommended
    chunk_size=256,           # 256-512 depending on RAM
    overlap_fraction=0.15,    # 10-20% recommended
    prefer_dask=True,         # Essential for large data
)

# One-line execution
chunked_reg.run_full_pipeline(output_path="registered.tif")


STEP-BY-STEP EXECUTION
----------------------
# Step 1: Coarse global registration (downsampled)
linear, nonlinear = chunked_reg.perform_coarse_registration(
    linear_type="Affine",
    nonlinear_type="SyNOnly",
    linear_accuracy="low",
    nonlinear_accuracy="medium",
)

# Step 2 & 3: Fine registration on chunks
chunked_reg.register_all_chunks(
    nonlinear_type="SyNOnly",
    nonlinear_accuracy="high",
)

# Step 4: Apply transforms and blend
chunked_reg.apply_transforms_and_blend(
    output_path="registered.tif",
    interpolator="linear",
    blend_sigma=10.0,
)


PARAMETER PRESETS
-----------------

Small Dataset (< 2GB):
    downsample_factor=4
    chunk_size=512
    overlap_fraction=0.10

Medium Dataset (2-10GB):
    downsample_factor=4
    chunk_size=256
    overlap_fraction=0.15

Large Dataset (10-50GB):
    downsample_factor=6
    chunk_size=256
    overlap_fraction=0.15
    Output: .zarr format

Very Large (50GB+):
    downsample_factor=8
    chunk_size=(128, 256, 256)
    overlap_fraction=0.20
    Output: .zarr format


KEY FEATURES
------------
✓ Downsamples BEFORE converting to ANTsImage (memory efficient!)
✓ Dask integration for lazy loading
✓ Zarr support for inputs/outputs
✓ Overlapping chunks with smooth blending
✓ Transform composition (global + local)
✓ Step-by-step or automated execution


TROUBLESHOOTING
---------------
Memory errors during coarse registration:
  → Increase downsample_factor (try 6 or 8)
  → Ensure prefer_dask=True

Discontinuities at chunk boundaries:
  → Increase overlap_fraction (try 0.20)
  → Increase blend_sigma (try 15-20)

Poor registration quality:
  → Check coarse registration first
  → Increase accuracy levels
  → Try smaller chunks for more local refinement

Too slow:
  → Increase chunk_size
  → Decrease accuracy levels
  → Reduce number of chunks


OUTPUT STRUCTURE
----------------
output_dir/
├── transforms/
│   ├── global_linear_transform.mat
│   └── global_nonlinear_warp.nii.gz
└── chunks/
    ├── chunk_0000_warp.nii.gz
    ├── chunk_0001_warp.nii.gz
    └── ...


COMMON PATTERNS
---------------

# Check image size before processing
from clearex.io.read import ImageOpener
opener = ImageOpener()
_, info = opener.open("image.tif", prefer_dask=True)
print(f"Shape: {info.shape}, Dtype: {info.dtype}")

# Inspect chunk grid before registration
chunks = chunked_reg._compute_chunk_grid(image_shape)
print(f"Will create {len(chunks)} chunks")

# Save intermediate results
chunked_reg.perform_coarse_registration()
# ... can stop here and resume later
chunked_reg.register_all_chunks()

# Use Zarr for large outputs
chunked_reg.run_full_pipeline(output_path="result.zarr")


ACCURACY LEVELS
---------------
linear_accuracy: "low", "medium", "high"
  - Use "low" for coarse registration (faster)

nonlinear_accuracy: "low", "medium", "high"
  - Use "medium" for coarse, "high" for fine
  - Higher = slower but more accurate


REGISTRATION TYPES
------------------
Linear:
  - "Translation", "Rigid", "Similarity", "Affine"

Nonlinear:
  - "SyNOnly" (recommended), "SyN", "Elastic"


INTERPOLATORS
-------------
- "linear" (default, good for most cases)
- "nearestNeighbor" (for label/segmentation images)
- "bSpline" (slower but smoother)


EXAMPLES
--------
See:
- examples/scripts/registration/chunked_registration.py
- examples/notebooks/registration/chunked_registration.ipynb
- src/clearex/registration/CHUNKED_REGISTRATION.md
"""
