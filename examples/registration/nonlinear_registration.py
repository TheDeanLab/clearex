# Standard Library Imports
import os

# Third Party Imports
import tifffile

# Local Imports
from clearex.registration.linear import (
    import_affine_transform,
    transform_image,
    export_tiff,
)
from clearex.registration.nonlinear import register_image
from clearex.file_operations.tools import crop_overlapping_datasets

def main(
    path_to_fixed_data: str,
    path_to_moving_data: str,
    path_to_affine_transform: str,
    path_to_save_to: str,
):
    print(f"Loading fixed image: {path_to_fixed_data}")
    fixed_roi = tifffile.imread(path_to_fixed_data)

    print(f"Loading moving image: {path_to_moving_data}")
    moving_roi = tifffile.imread(path_to_moving_data)

    print(f"Importing affine transform from: {path_to_affine_transform}")
    transform = import_affine_transform(path_to_affine_transform)

    print("Applying transform to moving image...")
    transformed_image = transform_image(moving_roi, fixed_roi, transform)

    print("Identifying smallest ROI that encapsulates the data.")
    fixed_roi, transformed_image = crop_overlapping_datasets(
        fixed_roi,
        transformed_image
    )
    print(f"Cropped image shape: {fixed_roi.shape}")

    print("Exporting cropped reference and moving images.")
    export_tiff(
        image=transformed_image,
        data_path=os.path.join(path_to_save_to, "cropped_moving.tif"))
    export_tiff(
        image=fixed_roi,
        data_path=os.path.join(path_to_save_to, "cropped_fixed.tif"))

    print("Performing nonlinear registration on the cropped data.")
    transformed_image, transform = register_image(
        moving_image=moving_roi,
        fixed_image=fixed_roi,
        accuracy="high",
        verbose=True)

    print("Exporting the registered data to:", path_to_save_to)
    export_tiff(
        image=transformed_image,
        data_path=os.path.join(path_to_save_to, "non_linear_registered.tif"))

    print("Done.")

if __name__ == "__main__":
    # Example file paths — consider replacing with argparse or env vars in real use
    channel = 1
    fixed_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/Sytox_ppm1/250320/new_Cell4/1_CH0{channel}_000000.tif'
    moving_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/restained/Cell1/1_CH0{channel}_000000.tif'
    base_path = "/archive/bioinformatics/Danuser_lab/Dean/dean/2025-05-28-registration"
    transform_path = os.path.join(base_path, "AffineTransform.mat")
    main(fixed_path, moving_path, transform_path, base_path)