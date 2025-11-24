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

## API Reference

### ImageRegistration Class

**Initialization Parameters:**
- `fixed_image_path` (str | Path, optional): Path to fixed reference image
- `moving_image_path` (str | Path, optional): Path to moving image
- `save_directory` (str | Path, optional): Output directory
- `imaging_round` (int, default=0): Round number for naming transforms
- `crop` (bool, default=False): Crop moving image before registration
- `enable_logging` (bool, default=True): Enable logging

**Methods:**
- `register(fixed_image_path=None, moving_image_path=None, save_directory=None, imaging_round=None)`: Perform registration

### register_round Function

Convenience function that creates an `ImageRegistration` instance and calls `register()`.

**Parameters:**
- `fixed_image_path` (str | Path): Path to fixed reference image
- `moving_image_path` (str | Path): Path to moving image
- `save_directory` (str | Path): Output directory
- `imaging_round` (int, default=0): Round number
- `crop` (bool, default=False): Crop moving image
- `enable_logging` (bool, default=True): Enable logging

## Design Philosophy

The registration module follows a layered design:

1. **High-level interface** (`ImageRegistration` class): Complete workflow for whole images
2. **Convenience function** (`register_round`): Simple functional interface
3. **Mid-level functions** (`linear.py`, `nonlinear.py`): Individual registration steps
4. **Utility functions** (`common.py`): Shared helpers

This design allows users to choose the appropriate level of abstraction for their needs.

