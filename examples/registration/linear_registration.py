# Standard Library Imports
import os

# Third Party Imports
import tifffile

# Local Imports
from clearex.registration.linear import (
    register_image,
    inspect_affine_transform,
    export_affine_transform,
    export_tiff
)

def main(
        path_to_fixed_data,
        path_to_moving_data,
        path_to_save_to
):
    print(f"Loading fixed image: {path_to_fixed_data}")
    fixed_roi = tifffile.imread(path_to_fixed_data)

    print(f"Loading moving image: {path_to_moving_data}")
    moving_roi = tifffile.imread(path_to_moving_data)

    print("Registering the data:")
    transformed_image, transform = register_image(
        moving_image=moving_roi,
        fixed_image=fixed_roi,
        registration_type="TRSAA",
        accuracy="high",
        verbose=True)

    print("Inspecting transform:")
    inspect_affine_transform(transform)

    print("Exporting the affine transform to:", path_to_save_to)
    export_affine_transform(transform, path_to_save_to)

    print("Exporting the registered data to:", path_to_save_to)
    export_tiff(
        image=transformed_image,
        data_path=os.path.join(path_to_save_to, "registered.tif"))

    print("Done.")

if __name__ == "__main__":
    channel = 1
    fixed_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/Sytox_ppm1/250320/new_Cell4/1_CH0{channel}_000000.tif'
    moving_path = f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/restained/Cell1/1_CH0{channel}_000000.tif'
    base_path = "/archive/bioinformatics/Danuser_lab/Dean/dean/2025-05-28-registration"
    main(fixed_path, moving_path, base_path)