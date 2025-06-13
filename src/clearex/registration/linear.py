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


# Standard Library Imports
import logging

# Local Imports

# Third Party Imports
import ants
import numpy as np
from scipy.linalg import polar, rq
from scipy.spatial.transform import Rotation

# Set up logging
logger = logging.getLogger('registration')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

def register_image(
        moving_image: ants.core.ants_image.ANTsImage | np.ndarray,
        fixed_image: ants.core.ants_image.ANTsImage | np.ndarray,
        registration_type: str="TRSAA",
        accuracy: str="high",
        verbose: bool=False
) -> tuple[ants.core.ants_image.ANTsImage, ants.core.ants_transform.ANTsTransform]:
    """ Linear Image Registration.

    Perform an image registration between the moving image and the fixed
    image. Registration is by default performed via Translation -> Rigid ->
    Similarity -> Affine -> Affine.

    Parameters
    ----------
    moving_image : ants.core.ants_image.ANTsImage
        The moving image.
    fixed_image : ants.core.ants_image.ANTsImage
        The image which the moving_image will be registered to. It remains fixed.
    registration_type : str
        The type of registration method to use. Options include Translation, Rigid,
        Similarity, Affine, TRSAA, and more.
    accuracy : str
        Used for the TRSAA registration type. Registration performed in a 4-level
        multiscale format. By default, it applies shrink factors of 8x4x2x1 and
        smoothing sigmas 3x2x1x0 (voxel units) over the four levels.
    verbose : bool
        The verbosity of the registration routine, showing iteration index,
        duration, registration error, etc.

    Returns
    -------
    transformed_image : ants.core.ants_image.ANTsImage
        The registered image, moved to the coordinate space of the fixed image.
    transform : ants.core.ants_transform.ANTsTransform
        The affine transform used in the transformation of the transformed_image.

    Raises
    ------
    ValueError
        If the fixed and moving images do not have the same number of dimensions,
        or if an unsupported registration type is specified.

    References
    ----------
    https://antspy.readthedocs.io/en/latest/registration.html
    """
    if fixed_image.ndim != moving_image.ndim:
        logger.error("Fixed and moving images must have the same number of dimensions.")
        raise ValueError("Both images must have the same number of dimensions.")

    if registration_type not in [
        "Translation", "Rigid", "Similarity", "Affine", "TRSAA"
    ]:
        logger.error(f"Unsupported registration type: {registration_type}. "
                     "Supported types are: Translation, Rigid, Similarity, Affine, TRSAA.")
        raise ValueError(f"Unsupported registration type: {registration_type}. "
                         "Supported types are: Translation, Rigid, Similarity, Affine, TRSAA.")

    # Convert images to ANTsImage if they are numpy arrays.
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    kwargs = {
        "fixed": fixed_image,
        "moving": moving_image,
        "type_of_transform": registration_type,
        "aff_metric": "mattes",
        "aff_sampling": 32,
        "aff_random_sampling_rate": 1.0,
        "verbose": verbose,
        }

    if registration_type=="TRSAA":
        # Multi-resolution iteration schedule
        if accuracy == "high":
            kwargs["reg_iterations"] = (1000, 500, 250, 100)
        else:
            kwargs["reg_iterations"] = (100, 70, 50, 20)
        logger.info(f"Using TRSAA registration with {kwargs['reg_iterations']} iterations.")

    # Register the images. This will return a dictionary with the results.
    registered = ants.registration(**kwargs)

    # Read the transform from the temporary disk location.
    transform = ants.read_transform(registered['fwdtransforms'][0])

    # Resample the registered image to the target image.
    transformed_image = ants.resample_image_to_target(
        image=registered["warpedmovout"],
        target=fixed_image,
        interp_type="linear"
    )

    # Histogram match to original data.
    transformed_image = ants.histogram_match_image(
        source_image=transformed_image,
        reference_image=moving_image
        # number_of_match_points=...
        # use_threshold_at_mean_intensity=...
    )
    return transformed_image, transform

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
    logger.info(f"Affine Transform Matrix:\n{final_matrix}")
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
        logger.info(f"{label} Scale: {scale:.2f}-fold")
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
        logger.info(f'Shear Angle {label}: {angle:.2f} degrees')
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
        logger.info(f'Translation {label}: {distance:.2f} voxels')
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
        logger.info(f'Rotation {axis}: {angle:.2f} degrees')
        print(f'Rotation around {axis} axis: {angle:.2f} degrees')
