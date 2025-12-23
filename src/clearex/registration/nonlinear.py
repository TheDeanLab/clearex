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
import tifffile

# Set up logging
logger = logging.getLogger("registration")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def register_image(
    moving_image: ants.core.ants_image.ANTsImage | np.ndarray,
    fixed_image: ants.core.ants_image.ANTsImage | np.ndarray,
    moving_mask: ants.core.ants_image.ANTsImage | None = None,
    fixed_mask: ants.core.ants_image.ANTsImage | None = None,
    registration_type: str = "SyNOnly",
    accuracy: str = "high",
    verbose: bool = False,
) -> tuple[ants.core.ants_image.ANTsImage, ants.core.ants_transform.ANTsTransform]:
    """Linear Image Registration.

    Perform nonlinear image registration between the moving image and the fixed
    image. Registration is by default performed with Symmetric Normalization.

    Parameters
    ----------
    moving_image : ants.core.ants_image.ANTsImage | np.ndarray
        The moving image.
    fixed_image : ants.core.ants_image.ANTsImage | np.ndarray
        The image which the moving_image will be registered to. It remains fixed.
    moving_mask : ants.core.ants_image.ANTsImage | None
        An optional mask for the moving image to limit registration to a region
        of interest.
    fixed_mask : ants.core.ants_image.ANTsImage | None
        An optional mask for the fixed image to limit registration to a region
        of interest.
    registration_type : str
        The type of registration method to use. Options include Elastic, SyN,
        SyNOnly. Default is SyNOnly.
    accuracy : str
        Controls the number of registration iterations. Options are "high"
        (100, 70, 50 iterations), "low" (5, 5, 5 iterations), or "dry run"
        (1, 1, 1 iterations). Default is "high".
    verbose : bool
        The verbosity of the registration routine, showing iteration index,
        duration, registration error, etc.

    Returns
    -------
    transformed_image : ants.core.ants_image.ANTsImage
        The registered image, moved to the coordinate space of the fixed image.
    transformed_mask : ants.core.ants_image.ANTsImage | None
        The registered mask, moved to the coordinate space of the fixed image.
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
    if registration_type not in ["Elastic", "SyNOnly", "SyN"]:
        raise ValueError(
            f"Unsupported registration type: {registration_type}. "
            "Supported types are: Elastic, SyNOnly, SyN."
        )

    # Convert images to ANTsImage if they are numpy arrays.
    if isinstance(fixed_image, np.ndarray):
        fixed_image = ants.from_numpy(fixed_image)
    if isinstance(moving_image, np.ndarray):
        moving_image = ants.from_numpy(moving_image)

    if fixed_image.dimension != moving_image.dimension:
        raise ValueError("Both images must have the same number of dimensions.")

    kwargs = {
        "fixed": fixed_image,
        "moving": moving_image,
        "fixed_mask": fixed_mask,
        "moving_mask": moving_mask,
        "type_of_transform": registration_type,
        "flow_sigma": 3.0,
        "total_sigma": 0.5,
        "shrink_factors": [4, 2, 1],  # Previously [2, 1]
        "smoothing_sigmas": [2, 1, 0],  # Previously [1,0]
        "syn_metric": "mattes",  # Previously, thought to be CC, but probably incorrect.
        "metric_weight": 1.0,
        "number_of_bins": 32,  # For mattes metric
        "singleprecision": False,  # True possibly helpful for larger data.
        "aff_random_sampling_rate": 1.0,
        "verbose": verbose,
        "initial_transform": "Identity",
        "mask_all_stages": True,
        # "sampling_strategy": "regular",  # More stable than random. New.
        # "metric": "CC", # MeanSquares, CC, Mattes, NormalizedMutualInformation
        # "radius_or_number_of_bins": 4,  # For CC metric (if used)
        # "sampling_percentage": 0.25,  # Reduce memory/computation. New.
    }

    if accuracy == "high":
        kwargs["reg_iterations"] = (100, 70, 50)  # Previously 100, 70
    elif accuracy == "medium":
        kwargs["reg_iterations"] = (50, 35, 25)
    elif accuracy == "low":
        kwargs["reg_iterations"] = (5, 5, 5)
    elif accuracy == "dry run":
        kwargs["reg_iterations"] = (1, 1, 1)

    # Register the images. This will return a dictionary with the results.
    registered = ants.registration(**kwargs)

    # Read the transform from the temporary disk location.
    transform = registered["fwdtransforms"][0]

    # # Transform the masks if provided
    # if moving_mask is not None:
    #     # Read transform from disk.
    #     mask_transform: ants.ANTsTransform = ants.read_transform(transform)
    #
    #     transformed_mask: ants.ANTsImage = mask_transform.apply_to_image(
    #         image=moving_mask, reference=fixed_image, interpolation="nearestNeighbor"
    #     )
    #     transformed_mask: ants.ANTsImage = ants.threshold_image(
    #         transformed_mask, low_thresh=0.5, high_thresh=1.0, inval=1, outval=0
    #     )

    # Resample the registered image to the target image.
    transformed_image = ants.resample_image_to_target(
        image=registered["warpedmovout"], target=fixed_image, interp_type="linear"
    )

    # Histogram match to original data.
    transformed_image = ants.histogram_match_image(
        source_image=transformed_image,
        reference_image=moving_image,
        # number_of_match_points=...
        # use_threshold_at_mean_intensity=...
    )
    return transformed_image, transform


def transform_image(
    moving_image: ants.core.ants_image.ANTsImage,
    fixed_image: ants.core.ants_image.ANTsImage,
    transformed_image: ants.core.ants_image.ANTsImage,
) -> ants.core.ants_image.ANTsImage:
    """Use a pre-existing warp transform to transform on a naive image to the
    coordinate space of the fixed_image. Performs histogram matching to the original.

    Parameters
    ----------
    moving_image: ants.core.ants_image.ANTsImage
        The image that will be transformed.
    fixed_image: ants.core.ants_image.ANTsImage
        The stationary image.
    transformed_image: ants.core.ants_image.ANTsImage
        The previously registered image, which contains a reference to the warp
        transform.
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

    # Apply the linear transform
    warped_image = ants.apply_transforms(
        fixed=fixed_image,
        moving=moving_image,
        transformlist=transformed_image["fwdtransforms"],
        interpolator="linear",
    )

    return ants.histogram_match_image(warped_image, moving_image)


def inspect_warp_transform(
    image_path: str, transform_path: str
) -> dict[str, float | int]:
    """Inspect a warp (displacement) transform and summarize displacements inside an image mask.

    Parameters
    ----------
    image_path : str
        Path to the registered image file. The image is read with ``tifffile.imread`` and
        converted to an ANTs image for mask extraction.
    transform_path : str
        Path to the warp/displacement field image. This is read with ``ants.image_read`` and
        expected to be a vector image with one component per spatial dimension.

    Returns
    -------
    stats : dict
        Dictionary with summary statistics of the displacement magnitude (within the image mask):
        - ``mean`` (float): mean displacement magnitude
        - ``min`` (float): minimum displacement magnitude
        - ``std`` (float): standard deviation of displacement magnitude
        - ``max`` (float): maximum displacement magnitude
        - ``n_vox`` (int): number of voxels included in the mask

    Raises
    ------
    ValueError
        If the warp array shape is not compatible with the expected number of dimensions
        (i.e. components not on the last axis or first axis).

    Notes
    -----
    The function uses ANTs for reading the warp (to preserve vector field semantics)
    and for creating a binary mask from the provided image. The image is expected to be
    readable by ``tifffile``.
    """
    # 1) Read the warp as an image (vector displacement field), not as a Transform object
    warp: ants.ANTsImage = ants.image_read(filename=transform_path)

    # 2) Read your registered image and create a mask
    img: ants.ANTsImage = ants.from_numpy(data=tifffile.imread(files=image_path))
    mask: ants.ANTsImage = ants.get_mask(img)  # 0/1 mask

    # 3) Compute amplitude (magnitude) of the displacement vector at each voxel
    warp_np: np.ndarray = warp.numpy()

    # Make sure components are on the last axis: (..., dim)
    dim: tuple[int, int, int] = warp.dimension
    if warp_np.ndim == dim + 1 and warp_np.shape[-1] == dim:
        disp = warp_np
    elif warp_np.ndim == dim + 1 and warp_np.shape[0] == dim:
        disp = np.moveaxis(warp_np, 0, -1)
    else:
        raise ValueError(f"Unexpected warp array shape {warp_np.shape} for dim={dim}")

    # Amplitude in the warp’s native units (usually physical units, e.g. mm)
    amp = np.linalg.norm(disp, axis=-1)

    # 4) Mask and summarize
    m = mask.numpy().astype(bool)
    vals = amp[m]
    vals = vals[np.isfinite(vals)]  # safety

    stats: dict = {
        "mean": float(vals.mean()),
        "min": float(vals.min()),
        "std": float(vals.std(ddof=0)),  # ddof=1 if you prefer sample std
        "max": float(vals.max()),
        "n_vox": int(vals.size),
    }

    print(stats)
    return stats
