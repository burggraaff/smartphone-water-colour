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


# Function for reading astropy tables - short-hand
read = table.Table.read

# Parameters measured by our hyperspectral sensors
parameters = ["Ed", "Lsky", "Lu", "R_rs"]

# Standard wavelength range to interpolate to.
wavelengths_interpolation = np.arange(400, 701, 1)


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


def interpolate_hyperspectral_table(data, parameters=parameters, wavelengths=wavelengths_interpolation):
    """
    For a table containing hyperspectral data, extract the data corresponding to each `parameter` and interpolate these to a new wavelength range.
    Then creates a new table that contains the interpolated hyperspectral data, but not the original hyperspectral data.
    """
    # Get the number of spectra in each parameter
    nr_spectra = len(data)

    # Extract the data for each parameter
    columns = [get_keys_for_parameter(data, param) for param in parameters]
    columns_flat = sum(columns, start=[])
    wavelengths_old = get_wavelengths_from_keys(columns[0], key=parameters[0])
    data_old = convert_columns_to_array(data, columns_flat)
    data_old = data_old.reshape((-1, len(wavelengths_old)))

    # Interpolate
    data_new = spectral.interpolate_spectral_data(wavelengths_old, data_old, wavelengths)
    data_new = data_new.reshape((len(data), -1))

    # Put back into the data table
    columns_new = [[f"{param}_{wvl:.1f}" for wvl in wavelengths] for param in parameters]
    columns_new = sum(columns_new, start=[])
    data.remove_columns(columns_flat)
    table_data_new = table.Table(data=data_new, names=columns_new)
    table_data_combined = table.hstack([data, table_data_new])

    return table_data_combined
