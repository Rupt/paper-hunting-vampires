"""Utilities for working with hdf5."""
import h5py


def save_h5(key_to_array, path, **kwargs):
    """Save the dict key_to_array to an h5 file at path."""
    with h5py.File(path, "w") as file_:
        for key, array in key_to_array.items():
            file_.create_dataset(
                key, array.shape, array.dtype, array, **kwargs
            )


def load_h5(path):
    """Load a dict of arrays from the h5 file at path."""
    with h5py.File(path, "r") as file_:
        key_to_array = {key: dataset[:] for key, dataset in file_.items()}

    return key_to_array
