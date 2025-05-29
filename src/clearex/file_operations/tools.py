# Standard imports
import sys
import pickle
import os

# Third-party imports

# Local imports


def get_variable_size(variable: any) -> float:
    """Get the size of a variable in MB.

    Parameters
    ----------
    variable : any
        The variable to get the size of.

    Returns
    -------
    float
        The size of the variable in MB.
    """
    return sys.getsizeof(variable) / 1024**2


def save_variable_to_disk(variable: any, path: str) -> None:
    """Save a variable to disk.

    Parameters
    ----------
    variable : any
        The variable to save.
    path : str
        The path to save the variable to.
    """
    with open(path, "wb") as f:
        pickle.dump(variable, f)


def load_variable_from_disk(path: str) -> any:
    """Load a variable from disk.

    Parameters
    ----------
    path : str
        The path to load the variable from.

    Returns
    -------
    any
        The loaded variable

    Raises
    ------
    FileNotFoundError
        If the file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "rb") as f:
        return pickle.load(f)


def delete_filetype(data_path: str, filetype: str) -> None:
    """Delete any files with the designated filetype in the specified directory.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the files.
    filetype : str
        The filetype to delete. E.g. 'pdf'
    """
    if filetype[0] != ".":
        filetype = "." + filetype

    files = [f for f in os.listdir(data_path) if f.endswith(filetype)]
    for file in files:
        os.remove(os.path.join(data_path, file))

def get_roi_indices(image, roi_size=256):
    """   Get indices for a centered ROI of size roi_size x roi_size x roi_size

    Parameters
    ----------
    image : np.ndarray
        The input image from which to extract the ROI.
    roi_size : int, optional
        The size of the ROI to extract from the center of the image. Default is 256.

    Returns
    -------
    tuple
        A tuple containing the start and end indices for each dimension (z_start, z_end, y_start, y_end, x_start, x_end).
    """
    if len(image.shape) != 3:
        raise ValueError("Input image must be a 3D array.")
    if roi_size <= 0:
        raise ValueError("ROI size must be a positive integer.")
    if roi_size > min(image.shape):
        raise ValueError("ROI size must be less than or equal to the smallest dimension of the image.")

    # Calculate the start and end indices for the ROI
    distance = roi_size // 2
    b = image.shape
    z_start = b[0] // 2 - distance
    z_end = b[0] // 2 + distance
    y_start = b[1] // 2 - distance
    y_end = b[1] // 2 + distance
    x_start = b[2] // 2 - distance
    x_end = b[2] // 2 + distance

    # Ensure indices are within bounds
    z_start = max(0, z_start)
    z_end = min(b[0], z_end)
    y_start = max(0, y_start)
    y_end = min(b[1], y_end)
    x_start = max(0, x_start)
    x_end = min(b[2], x_end)
    return z_start, z_end, y_start, y_end, x_start, x_end
