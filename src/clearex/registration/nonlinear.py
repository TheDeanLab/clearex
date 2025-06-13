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
import os
import logging

# Local Imports

# Third Party Imports
import ants
import numpy as np
from tifffile import imwrite, imread
from scipy.linalg import polar, rq
from scipy.spatial.transform import Rotation

# Set up logging
logger = logging.getLogger('registration')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

def register_image(
        moving_image: ants.core.ants_image.ANTsImage | np.ndarray,
        fixed_image: ants.core.ants_image.ANTsImage | np.ndarray,
        registration_type: str="SyNOnly",
        accuracy: str="high",
        verbose: bool=False
) -> tuple[ants.core.ants_image.ANTsImage, ants.core.ants_transform.ANTsTransform]:
    """ Linear Image Registration.

    Perform nonlinear image registration between the moving image and the fixed
    image. Registration is by default performed with Symmetric Normalization.

    Parameters
    ----------
    moving_image : ants.core.ants_image.ANTsImage | np.ndarray
        The moving image.
    fixed_image : ants.core.ants_image.ANTsImage | np.ndarray
        The image which the moving_image will be registered to. It remains fixed.
    registration_type : str
        The type of registration method to use. Options include Elastic, SyN,
        SyNOnly. Default is SyNOnly.
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
        The nonlinear transform used in the transformation of the transformed_image.

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
        raise ValueError("Both images must have the same number of dimensions.")

    if registration_type not in ["Elastic", "SyNOnly", "SyN"]:
        raise ValueError(f"Unsupported registration type: {registration_type}. "
                         "Supported types are: Elastic, SyNOnly, SyN.")

    # Convert images to ANTsImage if they are numpy arrays.
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    kwargs = {
        "fixed": fixed_image,
        "moving": moving_image,
        "type_of_transform": registration_type,
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "shrink_factors": [2, 1], # 4, 2, 1
        "smoothing_sigmas": [1, 0], #2, 1, 0
        "metric": "CC", # MeanSquares, CC, Mattes, NormalizedMutualInformation
        "singleprecision": False,
        "aff_random_sampling_rate": 1.0,
        "verbose": verbose,
        "reg_iterations": (70, 50), #vector of iterations for syn, 100, 70, 50.
        "initial_transform": 'Identity'
        }

    # Register the images. This will return a dictionary with the results.
    registered = ants.registration(**kwargs)

    # Read the transform from the temporary disk location.
    # transform = ants.read_transform(registered['fwdtransforms'][0])
    transform = registered['fwdtransforms'][0]

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