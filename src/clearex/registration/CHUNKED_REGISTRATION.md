# Chunked Registration for Large Volumes

This module implements a multi-scale hierarchical registration strategy for large 3D image volumes that may not fit in memory.

## Strategy Overview

The chunked registration approach follows a four-step pipeline:

1. **Coarse Global Registration**: Perform registration on heavily downsampled data (4-8x) to get a global alignment
2. **Chunk Decomposition**: Break the full-resolution volume into overlapping chunks
3. **Fine Local Registration**: Perform high-accuracy registration on each chunk independently
4. **Transform Composition & Blending**: Compose global and local transforms, apply to chunks, and blend overlapping regions

## Key Features

- **Memory Efficient**: Uses Dask for lazy loading and processing
- **Addresses Your Concern**: Downsampling happens *before* converting to ANTsImage, keeping memory usage low
- **Parallel Processing**: Each chunk can be processed independently (future enhancement)
- **Smooth Boundaries**: Overlap and blending prevent discontinuities at chunk edges
- **Flexible Configuration**: Customizable chunk sizes, overlap, and registration parameters

## Basic Usage

```python
from clearex.registration.chunked import ChunkedRegistration

# Create the registration object
chunked_reg = ChunkedRegistration(
    fixed_image_path="reference_volume.tif",
    moving_image_path="moving_volume.tif",
    output_dir="output/registration",
    downsample_factor=4,      # Downsample by 4x for coarse registration
    chunk_size=256,           # 256³ voxel chunks
    overlap_fraction=0.15,    # 15% overlap between chunks
    prefer_dask=True,         # Use Dask for lazy loading
)

# Run the complete pipeline
chunked_reg.run_full_pipeline(
    output_path="output/registered_volume.tif",
    linear_type="Affine",
    nonlinear_type_coarse="SyNOnly",
    nonlinear_type_fine="SyNOnly",
    linear_accuracy="low",
    coarse_accuracy="medium",
    fine_accuracy="high",
)
```

## Step-by-Step Usage

For more control, you can run each step separately:

```python
# Step 1: Coarse global registration
linear_path, nonlinear_path = chunked_reg.perform_coarse_registration(
    linear_type="Affine",
    nonlinear_type="SyNOnly",
    linear_accuracy="low",
    nonlinear_accuracy="medium",
)

# Step 2: Register all chunks
chunked_reg.register_all_chunks(
    nonlinear_type="SyNOnly",
    nonlinear_accuracy="high",
)

# Step 3: Apply transforms and blend
chunked_reg.apply_transforms_and_blend(
    output_path="output/registered.tif",
    interpolator="linear",
    blend_sigma=10.0,
)
```

## Configuration Parameters

### ChunkedRegistration Constructor

- `fixed_image_path`: Path to reference image (TIFF, Zarr, etc.)
- `moving_image_path`: Path to image to be registered
- `output_dir`: Directory for outputs (transforms, chunks, logs)
- `downsample_factor`: Factor for downsampling (default: 4)
  - Higher = faster coarse registration but less accurate initial alignment
  - Recommended: 4-8 for most datasets
- `chunk_size`: Size of chunks in voxels (default: 256)
  - Can be int (cubic) or tuple (z, y, x)
  - Larger = fewer chunks but more memory per chunk
  - Recommended: 256-512 depending on your RAM
- `overlap_fraction`: Fraction of chunk size to overlap (default: 0.15)
  - Higher = smoother blending but more computation
  - Recommended: 0.10-0.20
- `prefer_dask`: Whether to use Dask arrays (default: True)
  - True = lazy loading, lower memory usage
  - False = load entire volumes into memory

### Pipeline Parameters

- `linear_type`: Type of linear registration (default: "Affine")
  - Options: "Translation", "Rigid", "Similarity", "Affine"
- `nonlinear_type_coarse/fine`: Type of nonlinear registration
  - Options: "SyNOnly", "SyN", "Elastic"
- `linear_accuracy`: Accuracy for linear step (default: "low")
  - Options: "low", "medium", "high"
- `coarse_accuracy`: Accuracy for coarse step (default: "medium")
- `fine_accuracy`: Accuracy for fine step (default: "high")
- `interpolator`: Interpolation method (default: "linear")
  - Options: "linear", "nearestNeighbor", "bSpline"
- `blend_sigma`: Gaussian sigma for boundary blending (default: 10.0)
  - Higher = smoother transitions but more blurring at boundaries

## How It Addresses Your Concerns

### Memory Management with Large Datasets

1. **Downsampling Before ANTs Conversion**
   - Your concern: "I'm still a bit concerned that we would want to down-sample before converting to an AntsImage"
   - Solution: The `_downsample_for_coarse_registration()` method downsamples the data while it's still a numpy/dask array, *before* converting to ANTsImage
   - This keeps memory usage minimal during the coarse registration phase

2. **Dask Integration**
   - Uses your existing `ImageOpener` class with `prefer_dask=True`
   - Data remains lazy until needed
   - Only the chunk being processed is loaded into memory

3. **Chunked Processing**
   - Each chunk is processed independently
   - Can be extended to parallel processing with Dask delayed/distributed

### Transform Composition

The transforms are applied in sequence:
```python
transformlist = [
    global_linear_transform,      # From coarse registration
    global_nonlinear_transform,   # From coarse registration
    local_nonlinear_transform,    # From chunk-specific registration
]
```

ANTsPy handles the composition internally via `ants.apply_transforms()`.

## Output Structure

```
output_dir/
├── transforms/
│   ├── global_linear_transform.mat
│   └── global_nonlinear_warp.nii.gz
└── chunks/
    ├── chunk_0000_warp.nii.gz
    ├── chunk_0001_warp.nii.gz
    └── ...
```

## Performance Tips

1. **Start with aggressive downsampling** (8x) for very large datasets
2. **Use larger chunks** if you have sufficient RAM (512³ or more)
3. **Reduce accuracy** for coarse registration ("low" or "medium")
4. **Save intermediate results** by running steps separately
5. **Monitor memory usage** during chunk processing
6. **Consider Zarr output** for very large results instead of TIFF

## Comparison to BigWarp

You mentioned BigWarp was slow. This approach is fundamentally different:

- **BigWarp**: Manual landmark-based registration with interactive refinement
- **ChunkedRegistration**: Automated intensity-based registration with hierarchical optimization

Advantages over BigWarp:
- Fully automated (no manual landmarks)
- Better suited for intensity-based alignment
- Parallelizable chunk processing
- Works with very large datasets via Dask/Zarr

## Future Enhancements

Potential improvements to consider:

1. **Parallel chunk processing** using `dask.delayed` or `concurrent.futures`
2. **Adaptive chunk parameters** based on local image content
3. **Multi-resolution chunk pyramid** for even larger datasets
4. **GPU acceleration** for supported ANTs operations
5. **Resume capability** to restart from failed chunks
6. **Quality metrics** to identify poorly registered chunks

## Example Workflow for Very Large Data

```python
# For a 10GB dataset that doesn't fit in RAM
chunked_reg = ChunkedRegistration(
    fixed_image_path="large_volume.zarr",  # Already in Zarr format
    moving_image_path="moving_volume.zarr",
    output_dir="output",
    downsample_factor=8,         # Aggressive downsampling
    chunk_size=(128, 256, 256),  # Smaller chunks
    overlap_fraction=0.20,       # More overlap for safety
    prefer_dask=True,            # Essential for large data
)

chunked_reg.run_full_pipeline(
    output_path="output/registered.zarr",  # Save as Zarr
    linear_accuracy="low",
    coarse_accuracy="medium",
    fine_accuracy="medium",  # Can reduce if needed
)
```

## Troubleshooting

**Memory errors during coarse registration:**
- Increase `downsample_factor` (e.g., 8 or 16)
- Ensure `prefer_dask=True`

**Discontinuities at chunk boundaries:**
- Increase `overlap_fraction` (e.g., 0.20)
- Increase `blend_sigma` (e.g., 15-20)

**Registration quality issues:**
- Increase accuracy levels for coarse/fine registration
- Check that global registration is reasonable first
- Consider smaller chunks for more local refinement

**Process taking too long:**
- Reduce number of chunks (increase `chunk_size`)
- Decrease accuracy levels
- Consider parallelization (future feature)

