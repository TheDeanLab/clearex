# Standard Library Imports

# Local Imports

# Third Party Imports
import ants

def transform_image(moving_image, fixed_image, affine_transform):
    """ Register an image and perform histogram matching to original.

    Parameters
    ----------
    moving_image : ants.ANTsImage
    """
    warped_image = affine_transform.apply_to_image(
        moving_image,
        reference=fixed_image,
        interpolation='linear'
    )

    registered_image = ants.histogram_match_image(warped_image, moving_image)

    return registered_image

def export_tiff(image):
    pass

