# Standard Library Imports
import os

# Third Party Imports
import tifffile

# Local Imports
from clearex.registration.linear import (
    inspect_affine_transform,
    import_affine_transform,
    export_tiff,
    transform_image
)

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