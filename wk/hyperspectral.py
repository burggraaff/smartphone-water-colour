"""
Module with functions etc for processing hyperspectral reference data.
"""
from spectacle import io, calibrate, spectral
from spectacle.io import load_exif
import numpy as np
from datetime import datetime, timedelta
from astropy import table
from scipy.linalg import block_diag
from os import walk
from functools import partial

from . import colours
from . import statistics as stats


read = table.Table.read


def get_reference_name(path_reference):
    """
    From a given filename, extract the name of the sensor.
    """
    if "So-Rad" in path_reference.stem:
        reference = "So-Rad"
        ref_small = "sorad"
    elif "wisp" in path_reference.stem:
        reference = "WISP-3"
        ref_small = "wisp"
    elif "TriOS" in path_reference.stem:
        reference = "TriOS"
        ref_small = "trios"
    else:
        raise ValueError(f"Unknown reference sensor for file {path_reference}")

    return reference, ref_small


def get_keys_for_parameter(data, parameter, keys_exclude=[*"XYZxyRGB", "hue", "FU", "sR", "sG", "sB"]):
    """
    For a given parameter `parameter`, e.g. 'R_rs', get all keys in a table `data` that include that parameter, but exclude all of the `keys_exclude`.
    This is used for example to get hyperspectral R_rs from a reference data table without also getting convolved data.
    """
    return [col for col in data.keys() if parameter in col and not any(f"({label})" in col for label in keys_exclude)]


def get_wavelengths_from_keys(cols, key):
    """
    For a given list of column names `cols`, get the corresponding wavelengths by removing a constant `key` from them.
    """
    return np.array([float(col.split(key)[1][1:]) for col in cols])


def convert_columns_to_array(data, column_names, dtype=np.float64):
    """
    Extract data from a given list of column_names in a data table, and cast them to a pure numpy array.
    """
    data_as_array = np.array(data[column_names]).view(dtype).reshape((-1, len(column_names)))
    return data_as_array
