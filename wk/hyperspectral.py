"""
Module with functions etc for processing hyperspectral reference data.
"""
from spectacle import io, calibrate, spectral
from spectacle.io import load_exif
import numpy as np
from datetime import datetime, timedelta
from astropy import table
from scipy.linalg import block_diag
from functools import partial

from . import statistics as stats, hydrocolor as hc

# Add max_time_diff as a dictionary here, with keys corresponding to the different cases (ref-ref, ref-phone, NZ)

# Function for reading astropy tables - short-hand
read = table.Table.read

# Parameters measured by our hyperspectral sensors
parameters = ["Ed", "Lsky", "Lu", "R_rs"]
parameters_uncertainty = [param+"_err" for param in parameters]

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


def find_elements_within_range(data, reference_value, maximum_difference=60, key="UTC"):
    """
    Find elements in a table containing hyperspectral data that are within a certain margin from a reference value.
    Typically used to find elements within a certain timespan from a reference time, for example when matching observations between data sets.
    """
    # Calculate the differences
    differences_all = np.abs(data[key] - reference_value)

    # Find elements that are within the given range, and the closest element
    close_enough_indices = np.where(differences_all <= maximum_difference)[0]
    closest_index = differences_all.argmin()
    minimum_difference = differences_all[closest_index]
    nr_matches = len(close_enough_indices)

    # Return the results
    return nr_matches, close_enough_indices, closest_index, minimum_difference


def extend_keys_to_wavelengths(keys, wavelengths=wavelengths_interpolation):
    """
    For a given set of keys, e.g. ["Lu", "Lsky"], generate variants for each given wavelengths.
    """
    # If only one key was given, put it into a list
    if isinstance(keys, str):
        keys = [keys]

    # Add suffixes
    list_wvl = [key + "_{wvl:.1f}" for key in keys]
    list_wavelengths_full = [[s.format(wvl=wvl) for wvl in wavelengths] for s in list_wvl]
    list_wavelengths_flat = sum(list_wavelengths_full, start=[])

    return list_wavelengths_flat


def average_hyperspectral_data(data, *, parameters=parameters, wavelengths=wavelengths_interpolation, colour_keys=[*"RGBXYZxy", "sR", "sG", "sB", "hue", "FU"], default_row=0, func_average=np.nanmedian, func_uncertainty=np.nanstd):
    """
    Calculate the average (default: median) across multiple rows of a given data table.
    The uncertainties are also estimated using np.nanstd by default.

    The columns being averaged are any that match the pattern "{key}_{wvl:.1f}" for all keys in `parameters` and wavelengths in `wavelengths`, as well as any that match "{key} ({key_colour})" for all keys in `parameters` and `colour_keys`.
    """
    # Create a copy of the data table with a single placeholder row
    data_averaged = table.Table(data[default_row])

    # Get the keys that need to be averaged over
    parameters_err = [param+"_err" for param in parameters]
    keys = extend_keys_to_wavelengths(parameters, wavelengths=wavelengths) + hc.extend_keys_to_RGB(parameters, colour_keys)
    keys_err = extend_keys_to_wavelengths(parameters_err, wavelengths=wavelengths) + hc.extend_keys_to_RGB(parameters_err, colour_keys)

    # Calculate the averages - this needs to be done in a loop to allow in-place editing of the astropy table
    for k in keys:
        data_averaged[0][k] = func_average(data[k])

    # Calculate the uncertainties and put them into a new table
    uncertainties = np.array([func_uncertainty(data[k]) for k in keys])
    table_uncertainties = table.Table(data=uncertainties, names=keys_err)

    # Combine the averages and uncertainties and return the result
    data_averaged = table.hstack([data_averaged, table_uncertainties])
    return data_averaged
