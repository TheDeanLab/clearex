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

# Third Party Imports
import ants
import numpy as np
from tifffile import imwrite, imread

# Local Imports

# Set up logging
logger = logging.getLogger('registration')
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

def export_affine_transform(
        affine_transform: ants.core.ants_transform.ANTsTransform,
        directory: str
) -> None:
    """ Export an ants Affine Transform to disk.

    Parameters
    ----------
    affine_transform : ants.core.ants_transform.ANTsTransform
        The affine transform to export.
    directory : str
        The directory where the transform will be saved. If it does not exist, it will
        be created.

    Raises
    ------
    ValueError
        If the affine_transform is not an instance of ANTsTransform.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(affine_transform, ants.core.ants_transform.ANTsTransform):
        raise ValueError("The affine_transform must be an instance of ANTsTransform.")

    save_path=os.path.join(directory, str(affine_transform.type) + ".mat")
    ants.write_transform(affine_transform, save_path)


def transform_image(moving_image: ants.core.ants_image.ANTsImage,
                    fixed_image: ants.core.ants_image.ANTsImage,
                    affine_transform: ants.core.ants_transform.ANTsTransform) -> (
        ants.core.ants_image.ANTsImage):
    """ Use a pre-existing affine transform to transform on a naive image to the
    coordinate space of the fixed_image. Performs histogram matching to the original.

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
    # Convert images to ANTsImage if they are numpy arrays.
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    warped_image = affine_transform.apply_to_image(
        moving_image,
        reference=fixed_image,
        interpolation='linear'
    )
    return ants.histogram_match_image(warped_image, moving_image)


def export_tiff(image: ants.core.ants_image.ANTsImage, data_path: str) -> None:
    """ Export an ants.ANTsImage to a 16-bit tiff file.

    Parameters
    ----------
    image: ants.core.ants_image.ANTsImage
        Image to export
    data_path: str
        The location and name of the file to save the data to.
    """
    if isinstance(image, ants.core.ants_image.ANTsImage):
        image=image.numpy().astype(np.uint16)
    elif isinstance(image, np.ndarray):
        pass
    else:
        raise TypeError(f"Unsupported file type {type(image)}")
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
