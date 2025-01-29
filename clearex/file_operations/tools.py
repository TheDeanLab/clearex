import sys
import pickle
import os

def get_variable_size(variable):
    """ Get the size of a variable in MB. """
    return sys.getsizeof(variable) / 1024**2

def save_variable_to_disk(variable, path):
    """ Save a variable to disk. """
    with open(path, 'wb') as f:
        pickle.dump(variable, f)

def load_variable_from_disk(path):
    """ Load a variable from disk. """
    with open(path, 'rb') as f:
        return pickle.load(f)

def delete_pdfs(data_path: str) -> None:
    """ Delete any PDFs in the specified directory."""
    pdf_files = [f for f in os.listdir(data_path) if f.endswith('.pdf')]
    for pdf_file in pdf_files:
        os.remove(os.path.join(data_path, pdf_file))