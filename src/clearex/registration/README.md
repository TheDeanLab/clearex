# Registration Module

The registration module provides tools for aligning images using combined linear (affine) and nonlinear (deformable) transformations.

## Module Structure

- `__init__.py` - High-level registration interface (`ImageRegistration` class and `register_round` function)
- `linear.py` - Linear/affine registration functions
- `nonlinear.py` - Nonlinear/deformable registration functions  
- `common.py` - Shared utilities (transform I/O, cropping, etc.)

## Quick Start

### Using the ImageRegistration Class (Recommended)

```python
from clearex.registration import ImageRegistration

# Create a registrar
registrar = ImageRegistration(
    fixed_image_path="reference.tif",
    moving_image_path="round_1.tif",
    save_directory="./output",
    imaging_round=1,
    enable_logging=True
)

# Perform registration
registrar.register()
```

### Using the register_round Function

```python
from clearex.registration import register_round

# One-line registration
register_round(
    fixed_image_path="reference.tif",
    moving_image_path="round_1.tif",
    save_directory="./output",
    imaging_round=1
)
```

## Advanced Usage

### Registering Multiple Rounds

Reuse the same `ImageRegistration` instance:

```python
registrar = ImageRegistration(
    fixed_image_path="reference.tif",
    save_directory="./output"
)

for i in range(1, 10):
    registrar.register(
        moving_image_path=f"round_{i}.tif",
        imaging_round=i
    )
```

### Using Lower-Level Functions

For more control, use the functions in `linear.py` and `nonlinear.py`:

```python
from clearex.registration.linear import register_image as linear_registration
from clearex.registration.nonlinear import register_image as nonlinear_registration

# Perform linear registration
transformed, transform = linear_registration(
    moving_image=moving,
    fixed_image=fixed,
    registration_type="TRSAA",
    accuracy="low"
)

# Perform nonlinear registration
warped, warp_path = nonlinear_registration(
    moving_image=transformed,
    fixed_image=fixed,
    accuracy="high"
)
```

## Output Files

After registration, the following files are saved:

- `linear_transform_{round}.mat` - Affine transformation matrix
- `nonlinear_warp_{round}.nii.gz` - Deformable warp field
- Log files (if logging is enabled)

## Examples

See the `examples/` directory for complete examples:

- `examples/scripts/registration/image_registration_class.py` - Class-based approach
- `examples/scripts/registration/register_round_function.py` - Functional approach
- `examples/scripts/registration/linear_registration.py` - Linear registration only
- `examples/scripts/registration/nonlinear_registration.py` - Nonlinear registration
- `examples/notebooks/registration/image_registration_class.ipynb` - Interactive notebook
- `examples/notebooks/registration/chunked_nonlinear_registration.ipynb` - Chunked registration demo

### Chunked Registration Example

```python
from clearex.registration import ChunkedImageRegistration
from pathlib import Path

# Setup paths
output_dir = Path("./registration_output")
output_dir.mkdir(exist_ok=True)

# Initialize registrar
registrar = ChunkedImageRegistration(
    fixed_image_path="data/reference_brain.tif",
    moving_image_path="data/brain_round_1.tif",
    save_directory=str(output_dir),
    imaging_round=1,
    crop=False,
    enable_logging=True,
    force_override=False  # Skip already processed chunks
)

# Configure chunking for your data
# For a 2048x2048x1000 image with 64GB RAM:
registrar.chunk_size = (512, 512, 512)
registrar.overlap_fraction = 0.15

# Run registration
registrar.register()

print(f"Registration complete!")
print(f"Output files in: {output_dir}")
print(f"Chunks processed: {len(registrar.chunk_info_list)}")
```

### Parallel Chunked Registration Example

```python
from clearex.registration import ParallelChunkedImageRegistration
import multiprocessing as mp

# Determine optimal workers
available_cores = mp.cpu_count()
num_workers = max(1, int(available_cores * 0.75))

print(f"Using {num_workers} workers out of {available_cores} cores")

registrar = ParallelChunkedImageRegistration(
    fixed_image_path="data/huge_brain.tif",
    moving_image_path="data/huge_brain_round_1.tif",
    save_directory="./output",
    imaging_round=1,
    num_workers=num_workers,
    enable_logging=True
)

# Smaller chunks for parallel processing
registrar.chunk_size = (256, 256, 256)
registrar.overlap_fraction = 0.15

registrar.register()
```

## Practical Considerations

### Choosing Chunk Size

**Rule of thumb:** Each chunk should be ~500MB-2GB in memory

```python
# Calculate approximate chunk memory usage
import numpy as np

def estimate_chunk_memory(chunk_size, dtype=np.float32):
    """Estimate memory usage per chunk in GB."""
    z, y, x = chunk_size
    bytes_per_voxel = np.dtype(dtype).itemsize
    # Multiply by 3 for fixed, moving, and warp field
    total_bytes = z * y * x * bytes_per_voxel * 3
    return total_bytes / (1024**3)  # Convert to GB

# For 512^3 chunks:
print(f"Memory per chunk: {estimate_chunk_memory((512, 512, 512)):.2f} GB")

# For 256^3 chunks:
print(f"Memory per chunk: {estimate_chunk_memory((256, 256, 256)):.2f} GB")
```

**Recommendations:**
- **Small RAM (16-32GB)**: Use 256×256×256 chunks
- **Medium RAM (64-128GB)**: Use 512×512×512 chunks
- **Large RAM (256GB+)**: Use 512×512×512 or larger chunks
- **Anisotropic data**: Adjust chunk size per dimension (e.g., (128, 512, 512) for thin Z slices)

### Choosing Number of Workers

```python
import multiprocessing as mp

# Conservative (leaves CPU for other tasks)
num_workers = max(1, mp.cpu_count() - 2)

# Aggressive (uses most cores)
num_workers = max(1, int(mp.cpu_count() * 0.9))

# Based on available memory
available_ram_gb = 64  # Your system RAM
chunk_memory_gb = estimate_chunk_memory((512, 512, 512))
max_workers_by_ram = int(available_ram_gb * 0.7 / chunk_memory_gb)

# Use the minimum of CPU-based and RAM-based limits
num_workers = min(
    max(1, mp.cpu_count() - 2),
    max_workers_by_ram
)

print(f"Recommended workers: {num_workers}")
```

### Overlap Fraction Guidelines

- **0.10-0.15**: Good for most applications (recommended)
- **0.20-0.25**: Better for highly deformable tissues
- **0.05-0.10**: Minimal overlap for speed (may show boundaries)
- **>0.25**: Excessive, slows processing without much benefit

### Force Override Behavior

```python
# First run: processes all chunks
registrar = ChunkedImageRegistration(
    fixed_image_path="brain.tif",
    moving_image_path="brain_r1.tif",
    save_directory="./output",
    imaging_round=1,
    force_override=False
)
registrar.register()

# If interrupted, resume from where it left off
# Only unprocessed chunks will be computed
registrar.register()  # Skips completed chunks

# Force recompute everything
registrar.force_override = True
registrar.register()  # Recomputes all chunks
```

## Chunked Registration for Large Images

For very large images (>10GB), use the chunked registration classes which divide the volume into overlapping chunks.

### ChunkedImageRegistration Class

```python
from clearex.registration import ChunkedImageRegistration

# Create a chunked registrar
registrar = ChunkedImageRegistration(
    fixed_image_path="large_reference.tif",
    moving_image_path="large_round_1.tif",
    save_directory="./output",
    imaging_round=1,
    enable_logging=True
)

# Customize chunk parameters
registrar.chunk_size = (512, 512, 512)  # (z, y, x) in pixels
registrar.overlap_fraction = 0.15  # 15% overlap between chunks

# Perform registration
registrar.register()
```

**How It Works:**
1. Performs linear registration on the full image
2. Divides the volume into overlapping chunks
3. Registers each chunk independently using SyN
4. Assembles chunk warp fields into a full-volume warp field
5. Reconstructs the final registered image

**Output Files:**
- `linear_transform_{round}.mat` - Full-image affine transform
- `nonlinear_warp_{round}.nii.gz` - Assembled full-volume warp field
- `nonlinear_registered_{round}.tif` - Final registered image
- `round_{round}_chunks/` - Directory with per-chunk transforms:
  - `chunk_0000_warp.nii.gz`
  - `chunk_0001_warp.nii.gz`
  - ...

**Quality Features:**
- Automatically skips chunks with insufficient variance
- Handles chunks with NaN or infinite values
- Caches individual chunk transforms for reuse
- Validates chunk boundaries before extraction

### ParallelChunkedImageRegistration Class

For even faster processing, use parallel chunk processing:

```python
from clearex.registration import ParallelChunkedImageRegistration
import multiprocessing as mp

# Use 75% of available CPU cores
num_workers = max(1, int(mp.cpu_count() * 0.75))

registrar = ParallelChunkedImageRegistration(
    fixed_image_path="huge_image.tif",
    moving_image_path="huge_round_1.tif",
    save_directory="./output",
    imaging_round=1,
    num_workers=num_workers,
    enable_logging=True
)

# Adjust chunk size based on RAM
registrar.chunk_size = (256, 256, 256)  # Smaller for limited RAM

registrar.register()
```

**Performance Tips:**
- Set `num_workers` to available CPU cores minus 1-2
- Smaller chunks = less memory per worker but more chunks
- Fast storage (SSD/NVMe) benefits parallel processing most
- Each worker processes one chunk at a time

**When to Use Chunked Registration:**
- Images too large for memory (>10GB)
- Need fine-grained local registration
- Limited memory resources
- Very high-resolution volumes

## API Reference

### ImageRegistration Class

**Initialization Parameters:**
- `fixed_image_path` (str | Path, optional): Path to fixed reference image
- `moving_image_path` (str | Path, optional): Path to moving image
- `save_directory` (str | Path, optional): Output directory
- `imaging_round` (int, default=0): Round number for naming transforms
- `crop` (bool, default=False): Crop moving image before registration
- `enable_logging` (bool, default=True): Enable logging
- `force_override` (bool, default=False): Force re-registration even if transforms exist

**Methods:**
- `register(fixed_image_path=None, moving_image_path=None, save_directory=None, imaging_round=None)`: Perform registration

### ChunkedImageRegistration Class

Extends `ImageRegistration` for large-volume chunk-based registration.

**Additional Initialization Parameters:**
- All `ImageRegistration` parameters, plus:
- `force_override` (bool, default=False): Force re-computation of existing chunk transforms

**Additional Attributes:**
- `chunk_size` (tuple, default=(512, 512, 512)): Chunk dimensions in (z, y, x)
- `overlap_fraction` (float, default=0.15): Fraction of chunk overlap (0.0-1.0)
- `chunk_info_list` (list): List of ChunkInfo objects after grid computation
- `chunks_dir` (Path): Directory containing individual chunk transforms

**Methods:**
- `register()`: Perform linear + chunked nonlinear registration
- `_compute_chunk_grid(image_shape)`: Compute overlapping chunk grid
- `register_chunk(chunk, fixed_image, moving_image, ...)`: Register a single chunk
- `extract_chunk(chunk, fixed_image, moving_image)`: Extract chunk regions from images

### ParallelChunkedImageRegistration Class

Extends `ChunkedImageRegistration` with parallel processing.

**Additional Initialization Parameters:**
- All `ChunkedImageRegistration` parameters, plus:
- `num_workers` (int, default=4): Number of parallel workers for chunk processing

**Performance Considerations:**
- Total memory ≈ chunk_size × num_workers
- Optimal workers = CPU cores × 0.75
- I/O speed becomes bottleneck with many workers

### register_round Function

Convenience function that creates an `ImageRegistration` instance and calls `register()`.

**Parameters:**
- `fixed_image_path` (str | Path): Path to fixed reference image
- `moving_image_path` (str | Path): Path to moving image
- `save_directory` (str | Path): Output directory
- `imaging_round` (int, default=0): Round number
- `crop` (bool, default=False): Crop moving image
- `enable_logging` (bool, default=True): Enable logging

### ChunkInfo Dataclass

Represents information about a single chunk in the volume.

**Attributes:**
- `chunk_id` (int): Unique chunk identifier
- `z_start`, `z_end` (int): Core Z boundaries (without overlap)
- `y_start`, `y_end` (int): Core Y boundaries (without overlap)
- `x_start`, `x_end` (int): Core X boundaries (without overlap)
- `z_start_ext`, `z_end_ext` (int): Extended Z boundaries (with overlap)
- `y_start_ext`, `y_end_ext` (int): Extended Y boundaries (with overlap)
- `x_start_ext`, `x_end_ext` (int): Extended X boundaries (with overlap)
- `linear_transform_path` (Path, optional): Path to linear transform
- `nonlinear_transform_path` (Path, optional): Path to nonlinear warp field

## Design Philosophy

The registration module follows a layered design:

1. **High-level interface** (`ImageRegistration` class): Complete workflow for whole images
2. **Convenience function** (`register_round`): Simple functional interface
3. **Mid-level functions** (`linear.py`, `nonlinear.py`): Individual registration steps
4. **Utility functions** (`common.py`): Shared helpers

This design allows users to choose the appropriate level of abstraction for their needs.

