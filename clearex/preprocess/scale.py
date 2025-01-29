from skimage.transform import resize
import numpy as np

def resize_data(data: np.array, axial_pixel_size: float, lateral_pixel_size: float) -> np.array:
    """Resize the data to isotropic resolution with linear interpolation.

    Parameters
    ----------
    data : np.array
        The 3D image data to resize.
    axial_pixel_size : float
        The axial pixel size of the data.
    lateral_pixel_size : float
        The lateral pixel size of the data.

    Returns
    -------
    np.array
        The resized 3D image data.
    """

    # Assert that the data is 3D
    assert data.ndim == 3, "Data must be 3D"

    original_dims = data.shape
    if axial_pixel_size == lateral_pixel_size:
        return data
    elif axial_pixel_size > lateral_pixel_size:
        # Make axial match lateral
        new_dims = (
            int(original_dims[0] * (axial_pixel_size / lateral_pixel_size)),
            original_dims[1],
            original_dims[2]
        )
    else:
        # Make lateral match axial
        new_dims = (
            original_dims[0],
            int(original_dims[1] * (lateral_pixel_size / axial_pixel_size)),
            original_dims[2] * (lateral_pixel_size / axial_pixel_size)
        )

    # linear interpolation specified by order. Should anti_aliasing=False?
    return resize(data, new_dims, preserve_range=True, order=1)
