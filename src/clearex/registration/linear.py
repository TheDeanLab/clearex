# Standard Library Imports
import os

# Local Imports

# Third Party Imports
import ants
import numpy as np
from tifffile import imwrite, imread
from scipy.linalg import polar, rq
from scipy.spatial.transform import Rotation

def register_image(moving_image, fixed_image, accuracy="high", verbose=False):
    """ Perform linear TRSAA-type Image Registration.

    Registration is performed via Translation->Rigid->Similarity->Affine (x2) in a
    4-level multiscale format. By default, it applies shrink factors of 8x4x2x1 and
    smoothing sigmas 3x2x1x0 (voxel units) over the four levels

    """
    # Multi-resolution iteration schedule
    if accuracy=="high":
        reg_iterations = (1000, 500, 250, 100)
    else:
        reg_iterations = (100, 70, 50, 20)

    image_transform = ants.registration(
        fixed=fixed_image,
        moving=moving_image,
        type_of_transform='TRSAA',
        reg_iterations=reg_iterations,
        aff_metric='mattes',
        aff_sampling=32,
        aff_random_sampling_rate=1.0,
        verbose=verbose
    )

    return image_transform

def transform_image(moving_image, fixed_image, affine_transform):
    """ Register an image and perform histogram matching to original.

    Parameters
    ----------
    moving_image: ants.core.ants_image.ANTsImage
        The image that will be transformed.
    fixed_image: ants.core.ants_image.ANTsImage
        The stationary image.
    affine_transform: ants.core.ants_transform.ANTsTransform
        The affine transform to apply to the moving_image.

    Returns
    -------
    registered_image: ants.core.ants_image.ANTsImage
        The registered image.
    """
    warped_image = affine_transform.apply_to_image(
        moving_image,
        reference=fixed_image,
        interpolation='linear'
    )
    return ants.histogram_match_image(warped_image, moving_image)

def export_tiff(image, data_path):
    """ Export an ants.ANTsImage to a 16-bit tiff file.

    Parameters
    ----------
    image: ants.ANTsImage
        Image to export
    data_path: str
        The location and name of the file to save the data to.
    """
    image=image.numpy().astype(np.uint16)
    imwrite(data_path, image)

def import_tiff(data_path):
    """ Import a tiff file and convert to an ANTsImage.

    Parameters
    ----------
    data_path: str
        The path to the file to import.
    """
    return ants.from_numpy(imread(data_path))

def import_affine_transform(data_path: str):
    """ Import an ants Affine Transform.

    Parameters
    ----------
    data_path: str
        The path to the Affine Transform.

    Returns
    ------
    affine_transform: ants.core.ants_transform.ANTsTransform
        The affine transform.
    """
    affine_transform = ants.read_transform(data_path)
    return affine_transform

def export_affine_transform(affine_transform, data_path: str):
    """ Export an ants Affine Transform to disk. """
    np.savetxt(data_path, affine_transform)


def inspect_affine_transform(affine_transform):
    """ Evaluate the affine transform and report it's decomposed parts.

    affine_transform: ants.core.ants_transform.ANTsTransform
    """
    transform_parameters = affine_transform.parameters
    center_of_rotation = affine_transform.fixed_parameters

    # Reshape the matrix and calculate the offset.
    affine_matrix = transform_parameters[:9].reshape(3, 3)
    translation_vector = transform_parameters[9:]
    offset = translation_vector + center_of_rotation - affine_matrix @ center_of_rotation

    # Assemble a complete affine transform matrix.
    final_matrix = np.eye(4)
    final_matrix[:3, :3] = affine_matrix
    final_matrix[:3, 3] = offset
    print("Complete Affine 4x4 matrix:\n", final_matrix)

    # Decompose the Affine Matrix into its Parts
    _extract_translation(final_matrix)
    _extract_rotation(final_matrix)
    _extract_shear(final_matrix)
    _extract_scale(final_matrix)

def _extract_scale(affine_matrix):
    """ Extract the scaling factors from an affine matrix.

    Parameters
    ----------
    affine_matrix: np.ndarray
        4x4 affine transform matrix.
    """
    rotation, scaling_shear = polar(affine_matrix[:, :3])
    # Further decompose the scaling/shear using RQ decomposition to separate scale and shear
    scale, _ = rq(scaling_shear)
    scale_vector = np.diagonal(scale)
    scale_labels = ['Z', 'Y', 'X']
    for label, scale in zip(scale_labels, scale_vector):
        print(f'{label} Scale: {scale:.2f}-fold')

def _extract_shear(affine_matrix):
    """ Extract the shearing factors from an affine matrix.

    Parameters
    ----------
    affine_matrix: np.ndarray
        4x4 affine transform matrix.
    """
    rotation, scaling_shear = polar(affine_matrix[:3, :3])
    _, shear = rq(scaling_shear)

    # Shear factors from off-diagonal elements
    angles_deg = np.degrees(
        [
            np.arctan(shear[0, 1]), np.arctan(shear[0, 2]), np.arctan(shear[1, 0]),
            np.arctan(shear[1, 2]), np.arctan(shear[2, 0]), np.arctan(shear[2, 1])
        ]
    )

    # Clearly print the results
    shear_labels = ['XY', 'XZ', 'YX', 'YZ', 'ZX', 'ZY']
    for label, angle in zip(shear_labels, angles_deg):
        print(f'Shear angle {label}: {angle:.2f} degrees')

def _extract_translation(affine_matrix):
    """ Extract the translation factors from an affine matrix.

    Parameters
    ----------
    affine_matrix: np.ndarray
        4x4 affine transform matrix.
    """
    # Extract the translation
    translation = affine_matrix[:, 3]
    translation_labels = ['Z', 'Y', 'X']
    for label, distance in zip(translation_labels, translation):
        print(f'Translation distance in {label}: {distance:.2f} voxels')

def _extract_rotation(affine_matrix):
    """ Extract the rotation factors from an affine matrix.

    Parameters
    ----------
    affine_matrix: np.ndarray
        4x4 affine transform matrix.
    """
    rotation, _ = polar(affine_matrix[:3, :3])

    # Create a rotation object
    r = Rotation.from_matrix(rotation)

    # Extract Euler angles (XYZ order) in degrees
    euler_angles_deg = r.as_euler('xyz', degrees=True)

    # Print angles clearly
    axis_labels = ['X (roll)', 'Y (pitch)', 'Z (yaw)']
    for axis, angle in zip(axis_labels, euler_angles_deg):
        print(f'Rotation around {axis} axis: {angle:.2f} degrees')


if __name__ == "__main__":
    export_path = '/archive/bioinformatics/Danuser_lab/Dean/dean/2025-04-29-registration'
    affine_transform = import_affine_transform(
        os.path.join(export_path, 'ants_initial_affine.mat'))
    print("Affine Transform Loaded.")

    inspect_affine_transform(affine_transform)

    channel = 1
    fixed_image = import_tiff(
        f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/Sytox_ppm1/250320/new_Cell4/1_CH0{channel}_000000.tif')
    moving_image = import_tiff(
        f'/archive/bioinformatics/Danuser_lab/Dean/Seweryn/s2/restained/Cell1/1_CH0{channel}_000000.tif')
    print("Data loaded and converted to ANTs format.")

    affine_transform = register_image(moving_image,
                                      fixed_image,
                                      accuracy="low",
                                      verbose=True)
    print("Registration Complete")
    inspect_affine_transform(affine_transform)
