# Registration Module

The registration module provides tools for aligning images using combined linear (affine) and nonlinear (deformable) transformations.

## Module Structure

- `__init__.py` - High-level registration interface (`ImageRegistration` class and `register_round` function)
- `linear.py` - Linear/affine registration functions
- `nonlinear.py` - Nonlinear/deformable registration functions  
- `common.py` - Shared utilities (transform I/O, cropping, etc.)
- `pipeline.py` - Chunked tile-registration workflow for canonical 6D OME-Zarr
  stores (Dask + Zarr v3)

## ClearEx Runtime Contract

These notes are mandatory when editing `pipeline.py` and related runtime
integration:

- Canonical runtime inputs are OME-Zarr v3 `*.ome.zarr` stores.
- `input_source` is a logical workflow alias by default (`data`,
  `flatfield`, `deconvolution`, `shear_transform`, `usegment3d`,
  `registration`, `fusion`) and resolves to the matching ClearEx runtime or
  metadata component.
- `registration` is metadata-only and writes transform/layout artifacts under
  `clearex/results/registration/latest`:
  - `affines_tpx44`
  - `edges_pe2`
  - `pairwise_affines_tex44`
  - `edge_status_te`
  - `edge_residual_te`
  - `transformed_bboxes_tpx6`
- `fusion` consumes the latest registration result and writes stitched image
  data to `clearex/runtime_cache/results/fusion/latest/data`.
- `fusion` publishes its public image result as the OME collection
  `results/fusion/latest`.
- `fusion` auxiliary artifacts stay under `clearex/results/fusion/latest`:
  - `blend_weights`
  - `intensity_gains_tp`
  - `intensity_offsets_tp`
- Do not reintroduce `results/registration/latest/data` as a canonical image
  target. Registration no longer owns a reusable image output.
- Pairwise registration must apply the `max_pairwise_voxels` budget before
  source reads and overlap resampling. Large overlap crops should be sampled
  onto a coarser registration grid instead of allocating the full-resolution
  crop and downsampling afterward.

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
5. **Tile-registration pipeline** (`pipeline.py`): Dask + Zarr distributed tile registration

This design allows users to choose the appropriate level of abstraction for their needs.

## Performance Tuning (pipeline.py)

The chunked tile-registration pipeline (`pipeline.py`) now runs as two
separate operations:

- `run_registration_analysis(...)`
  Estimates pairwise overlaps, solves global transforms, and persists layout
  metadata for later rendering.
- `run_fusion_analysis(...)`
  Consumes the persisted registration result and renders the stitched image
  volume.

This split is intentional: registration usually needs much more memory per
worker than fusion, so ClearEx now lets you run them in separate workflow
executions with different Dask worker counts and memory limits.

### Dask/Zarr Design Principles

- **No large arrays through Dask pipes.** Workers receive `zarr_path` (string)
  and component paths, then open the store locally and read only the minimal
  sub-volume needed for each task.
- **Cropped source reads.** Both pairwise registration and fusion use
  `_source_subvolume_for_overlap` to compute the minimal source Zarr slice
  that covers each output region, dramatically reducing I/O for large tiles.
- **Cached blend profiles.** The separable blend-weight profiles are
  pre-computed once during fusion and stored under
  `clearex/results/fusion/latest/blend_weights` in the analysis store.
  Fusion workers lazily reconstruct only the sub-volume they need.

### Registration Parameters

These parameters belong to the `registration` operation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `registration_channel` | `0` | Channel index used for pairwise overlap registration. Choose a structurally stable, high-SNR channel that is present in every tile; avoid sparse labels or channels with strong bleaching. |
| `registration_type` | `"translation"` | Pairwise ANTsPy transform family: `translation` (fastest), `rigid`, or `similarity`. Translation is the default for maximum throughput; combined with FFT initial alignment this converges in very few iterations. |
| `input_resolution_level` | `0` | Pyramid level used only for pairwise overlap registration (`0` = full resolution). Final fusion always resamples the full-resolution source tiles. |
| `pairwise_overlap_zyx` | `(8, 32, 32)` | Overlap crop padding in `(z, y, x)` order used for pairwise registration. Larger values give ANTs more shared context, but they increase I/O and registration cost. |
| `anchor_mode` | `"central"` | Global graph anchor policy. `central` fixes the most central tile automatically; `manual` uses `anchor_position`. |
| `anchor_position` | `None` | Tile index held fixed when `anchor_mode="manual"`. Use this only when you need a specific reference tile. |
| `use_fft_initial_alignment` | `True` | Run FFT phase correlation to pre-align the moving crop before ANTs. Gives ANTs a much better starting point so it converges faster with fewer iterations. |
| `max_pairwise_voxels` | `500_000` | Maximum voxel budget for pairwise overlap crops. Crops exceeding this budget are isotropically downsampled before ANTs estimation. Set to `0` to disable. |
| `ants_iterations` | `(200, 100, 50, 25)` | ANTsPy multi-resolution iteration schedule. Smaller values are faster; the legacy schedule `(2000, 1000, 500, 250)` is available as `_ANTS_AFF_ITERATIONS_LEGACY`. |
| `ants_sampling_rate` | `0.20` | ANTsPy random voxel sampling rate. Reducing this saves per-iteration cost with minimal accuracy impact for tile registration. |
| `use_phase_correlation` | `False` | When `True` and `registration_type` is `"translation"`, skip ANTs entirely and use FFT phase correlation only. Falls back to ANTs on failure. |

### Fusion Parameters

These parameters belong to the `fusion` operation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_source` | `"registration"` | Registration result consumed by fusion. In the normal workflow this is the latest `registration` output. |
| `blend_overlap_zyx` | `(8, 32, 32)` | Blend-ramp width in `(z, y, x)` order. Larger values widen the fusion transition and usually hide seams better, but they spread mismatch over a larger region. |
| `blend_mode` | `"feather"` | Fusion mode for overlapping tiles. `average` uses uniform weights, `feather` uses cosine edge ramps, `center_weighted` steepens the feather falloff, `content_aware` modulates feather weights by local gradient content, and `gain_compensated_feather` estimates a moving-to-fixed intensity match before feather blending. |
| `blend_exponent` | `1.0` | Exponent applied to the spatial blend profile. Values above `1.0` make the edge transition steeper; values below `1.0` make it gentler. |
| `gain_clip_range` | `(0.25, 4.0)` | Minimum and maximum allowed multiplicative gain for gain-compensated feather. The fitted overlap gain is clipped into this range before fusion. |

### Practical Parameter Guidance

#### Choosing the registration signal

- Use `registration_channel` for the channel with the most stable anatomical structure across tiles.
- Good choices are dense structural channels with consistent contrast in the overlap.
- Poor choices are sparse labels, channels dominated by isolated bright puncta, or channels with strong edge-only signal.

#### Choosing the transform model

- `translation`
  Use when tiles differ mostly by stage offset. This is the fastest option and is often enough after upstream geometry correction.
- `rigid`
  Use when the seam suggests a small residual rotation in addition to translation. This is usually the best quality-throughput tradeoff for real stitched volumes.
- `similarity`
  Use when the overlap suggests slight scale drift in addition to rotation and translation. This is slower and should usually be reserved for difficult seams.

#### Choosing the registration resolution

- `input_resolution_level: 0`
  Best accuracy. Use this when chasing a stubborn seam.
- `input_resolution_level: 1`
  Good compromise for large datasets when coarse structures are still well preserved at the next pyramid level.
- `input_resolution_level >= 2`
  Best treated as a preview or emergency-throughput setting. Fine local mismatch can survive into the fused result even though final fusion still reads level 0 data.

#### Choosing overlap width

`pairwise_overlap_zyx` and `blend_overlap_zyx` are now separate knobs. Increase
pairwise overlap when registration needs more shared context; increase blend
overlap when alignment is already acceptable but the seam transition is still
too abrupt.

| Goal | Suggested `pairwise_overlap_zyx` | Suggested `blend_overlap_zyx` | Notes |
|------|----------------------------------|-------------------------------|-------|
| Fast preview / throughput | `[8, 32, 32]` | `[8, 32, 32]` | Current default. Good when seams are already minor. |
| Balanced seam reduction | `[12, 64, 64]` | `[16, 128, 128]` | Good starting point when the seam is visible but not severe. |
| Aggressive seam reduction | `[16, 96, 96]` | `[24, 192, 192]` | Use when overlap has enough tissue context and the seam is still obvious. |

Practical notes:

- Broader `y/x` overlap is usually more helpful than broader `z` overlap for anisotropic light-sheet data.
- Increasing overlap improves both registration context and blend smoothness, but it also increases read volume and ANTs cost.
- Very large values are rarely useful once they approach a substantial fraction of the tile size; the blend ramp is clipped internally at half the tile extent per axis.

#### Choosing the blend mode

- `feather`
  Default choice. Use when the overlap is reasonably well aligned and you mainly want a smooth edge transition.
- `center_weighted`
  Use when tile centers look cleaner than tile borders and the seam appears to come from boundary quality loss. This uses a stronger falloff than plain feathering.
- `content_aware`
  Use when one side of the overlap is visibly sharper or has better local SNR. The fusion weights are modulated by local gradient content, so sharper structure contributes more strongly.
- `gain_compensated_feather`
  Use when the seam looks like an intensity jump after flatfield correction, not just a geometric mismatch. The pipeline estimates a per-edge linear moving-to-fixed intensity map, solves per-position gain/offset corrections, and then applies feather blending.
- `average`
  Mostly useful as a debugging or control mode. If a seam is already visible, average blending usually makes residual mismatch easier to see rather than harder to hide.

Mode-specific notes:

- `blend_exponent` has no effect for `average`.
- `gain_clip_range` is only used for `gain_compensated_feather`.
- `center_weighted` applies a stronger effective exponent internally, so start conservatively.

#### Recommended blend exponent ranges

| Use case | Suggested `blend_exponent` | Notes |
|----------|----------------------------|-------|
| Gentle transition | `0.7-1.0` | Useful when the overlap is already clean and you want the widest possible blend zone. |
| General use | `1.0-1.5` | Good default range for `feather`, `content_aware`, and `gain_compensated_feather`. |
| Stronger center preference | `1.5-3.0` | Useful when boundary quality is poor, but can make the handoff between tiles more abrupt if pushed too high. |

For `center_weighted`, start at `1.0` before pushing upward because the implementation already doubles the effective exponent internally.

#### Recommended gain clipping ranges

| Situation | Suggested `gain_clip_range` | Notes |
|-----------|-----------------------------|-------|
| Mild intensity mismatch | `[0.7, 1.4]` | Conservative. Good when flatfield correction is already close and you only need small correction. |
| Moderate mismatch | `[0.5, 2.0]` | Good starting range for real overlaps with noticeable but not extreme intensity differences. |
| Strong or uncertain mismatch | `[0.25, 4.0]` | Broadest supported default. Use when you do not yet know how large the overlap gain error is. |

Avoid very broad ranges unless the overlap contains strong, representative signal. If the overlap is weak or low-SNR, broad gain limits can overfit noise.

#### Recommended registration-cost ranges

| Parameter | Fast / preview | Balanced | High quality / seam chasing |
|-----------|----------------|----------|-----------------------------|
| `max_pairwise_voxels` | `250_000-500_000` | `1_000_000` | `2_000_000` or `0` to disable the cap |
| `ants_sampling_rate` | `0.10-0.20` | `0.25-0.35` | `0.35-0.50` |
| `ants_iterations` | `[200, 100, 50, 25]` | `[500, 250, 125, 50]` | `[1000, 500, 250, 100]` |

Practical notes:

- Leave `use_fft_initial_alignment=True` unless you have a specific reason to disable it.
- Use `use_phase_correlation=True` only for translation-only preview runs or when ANTs is unnecessary. It is not a substitute for `rigid` or `similarity`.
- If a seam remains visible, it is usually better to increase overlap and pairwise fidelity before jumping directly to much more expensive transform models.

### Recommended Starting Presets

#### Fast preview

Use this when you want a quick check of seam direction and gross tile placement.

```yaml
registration:
  registration_type: translation
  input_resolution_level: 1
  pairwise_overlap_zyx: [8, 32, 32]
  max_pairwise_voxels: 250000
  ants_sampling_rate: 0.10
  ants_iterations: [200, 100, 50, 25]
  use_fft_initial_alignment: true
  use_phase_correlation: true
fusion:
  input_source: registration
  blend_overlap_zyx: [8, 32, 32]
  blend_mode: feather
  blend_exponent: 1.0
```

#### Balanced high-resolution stitching

Use this when the seam is visible and you want a stronger default without going to the most expensive settings.

```yaml
registration:
  registration_type: rigid
  input_resolution_level: 0
  pairwise_overlap_zyx: [12, 64, 64]
  max_pairwise_voxels: 1000000
  ants_sampling_rate: 0.25
  ants_iterations: [500, 250, 125, 50]
  use_fft_initial_alignment: true
  use_phase_correlation: false
fusion:
  input_source: registration
  blend_overlap_zyx: [16, 128, 128]
  blend_mode: feather
  blend_exponent: 1.25
```

#### Intensity-mismatch seam

Use this when the seam looks like a brightness or background jump across otherwise similar anatomy.

```yaml
registration:
  registration_type: rigid
  input_resolution_level: 0
  pairwise_overlap_zyx: [12, 64, 64]
  max_pairwise_voxels: 1000000
  ants_sampling_rate: 0.25
  ants_iterations: [500, 250, 125, 50]
  use_fft_initial_alignment: true
  use_phase_correlation: false
fusion:
  input_source: registration
  blend_overlap_zyx: [16, 128, 128]
  blend_mode: gain_compensated_feather
  blend_exponent: 1.0
  gain_clip_range: [0.5, 2.0]
```

#### Difficult seam with residual geometric mismatch

Use this when the seam remains visible after default fusion and the overlap still looks slightly rotated or mis-scaled.

```yaml
registration:
  registration_type: similarity
  input_resolution_level: 0
  pairwise_overlap_zyx: [16, 96, 96]
  max_pairwise_voxels: 2000000
  ants_sampling_rate: 0.35
  ants_iterations: [1000, 500, 250, 100]
  use_fft_initial_alignment: true
  use_phase_correlation: false
fusion:
  input_source: registration
  blend_overlap_zyx: [24, 192, 192]
  blend_mode: content_aware
  blend_exponent: 1.0
```

### GPU Acceleration (future)

The core resampling function `_resample_source_to_world_grid` uses
`scipy.ndimage.affine_transform`, which is single-threaded CPU code.  A
drop-in GPU replacement via `cupyx.scipy.ndimage.affine_transform` is marked
as a TODO for future integration.  This would benefit both pairwise
registration and fusion.  The deconvolution subsystem already supports
GPU-pinned `LocalCluster` workers; registration would reuse the same backend.
