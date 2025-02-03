
# Standard imports
import sys
import pickle
import os

# Third-party imports

# Local imports

def get_variable_size(variable: any) -> float:
    """ Get the size of a variable in MB.

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
    """ Save a variable to disk.

    Parameters
    ----------
    variable : any
        The variable to save.
    path : str
        The path to save the variable to.
    """
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

def load_variable_from_disk(path: str) -> any:
    """ Load a variable from disk.

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
        raise FileNotFoundError(f'File not found: {path}')

    with open(path, 'rb') as f:
        return pickle.load(f)

def delete_filetype(data_path: str, filetype: str) -> None:
    """ Delete any files with the designated filetype in the specified directory.

    Parameters
    ----------
    data_path : str
        The path to the directory containing the files.
    filetype : str
        The filetype to delete. E.g. 'pdf'
    """
    if filetype[0] != '.':
        filetype = '.' + filetype

    files = [f for f in os.listdir(data_path) if f.endswith(filetype)]
    for file in files:
        os.remove(os.path.join(data_path, file))