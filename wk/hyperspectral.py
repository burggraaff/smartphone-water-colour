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

from . import statistics as stats, hydrocolor as hc, wacodi as wa, colours

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


def get_keys_for_parameter(data, parameter, keys_exclude=[*"XYZxyRGB", *hc.bands_sRGB, "hue", "FU"]):
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


def get_wavelengths_from_table(data, parameter, **kwargs):
    """
    For a given data table, get the corresponding wavelengths.
    This just combines get_keys_for_parameter and get_wavelengths_from_keys.
    """
    cols = get_keys_for_parameter(data, parameter, **kwargs)
    wavelengths = get_wavelengths_from_keys(cols, parameter)
    return wavelengths


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


def average_hyperspectral_data(data, *, parameters=parameters, wavelengths=wavelengths_interpolation, colour_keys=[*"RGBXYZxy", *hc.bands_sRGB, "hue", "FU"], default_row=0, func_average=np.nanmedian, func_uncertainty=np.nanstd):
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


def add_hyperspectral_matchup_metadata(data, nr_matches, min_time_diff):
    """
    Add columns relating to the number of successful match-ups and the minimum time difference to a data table.
    """
    # Generate a new table containing these metadata
    metadata = table.Table(names=["nr_matches", "closest_match"], data=np.array([nr_matches, min_time_diff]), dtype=[int, np.float64])

    # Create and return a new table that contains the original data and the new metadata
    data_with_metadata = table.hstack([metadata, data])
    return data_with_metadata


def print_matchup_metadata(instrument_name, nr_matches, min_time_diff):
    """
    Simply print the number of match-ups and the closest match-up for a given sensor.
    """
    print(f"{instrument_name}: Number of matches: {nr_matches:>3.0f}; Closest match: {min_time_diff:>4.0f} s")


def find_single_and_multiple_matchups(data):
    """
    Find the indices of the rows in `data` where nr_matches == 1 and > 1.
    """
    indices_single_match = np.where(data["nr_matches"] == 1)[0]
    indices_multiple_matches = np.where(data["nr_matches"] > 1)[0]
    return indices_single_match, indices_multiple_matches


def fill_in_median_uncertainties(data):
    """
    Find rows in `data` that only had a single match, and fill in the uncertainties in those rows with the median of that column.
    """
    # Get the keys corresponding to uncertainties
    keys_uncertainties = (key for key in data.keys() if "_err" in key)

    # Find the rows with 1 vs multiple matches
    indices_single_match, indices_multiple_matches = find_single_and_multiple_matchups(data)

    # If no rows with only a single match were found, do nothing
    if len(indices_single_match) < 1:
        return data

    # Else, go ahead and calculate the medians
    for key in keys_uncertainties:
        data[key][indices_single_match] = np.nanmedian(data[key][indices_multiple_matches])

    return data


def add_dummy_columns(data, key_source="R_rs", keys_goal=["Ed", "Lu", "Lsky"], value=-1.):
    """
    Add dummy columns for missing quantities, for example Ed and Lu
    in a data set that only contained R_rs.
    The dummy columns will contain a given value, by default -1.
    """
    # Find the original column names
    columns_source_keys = [key for key in data.keys() if key_source in key]

    # Generate the new columns
    columns_goal_keys = np.ravel([[key.replace(key_source, key_goal) for key_goal in keys_goal] for key in columns_source_keys])
    dummy_data = np.tile(value, (len(data), len(columns_goal_keys)))
    dummy_data = table.Table(data=dummy_data, names=columns_goal_keys)

    # Combine the dummy data with the main data
    data = table.hstack([data, dummy_data])

    return data


def add_bandratios_to_hyperspectral_data(data, parameter="R_rs"):
    """
    Calculate RGB band ratios and add them to a data table.
    """
    # Calculate the band ratios and put them in a table
    bandratio_data = hc.calculate_bandratios(data[f"{parameter} (R)"], data[f"{parameter} (G)"], data[f"{parameter} (B)"]).T
    bandratio_names = hc.extend_keys_to_RGB(parameter, hc.bandratio_labels)

    bandratios = table.Table(data=bandratio_data, names=bandratio_names)

    # Calculate the uncertainties in these band ratios and put them in a table
    parameter_uncertainty = parameter + "_err"
    bandratio_uncertainties_data = [bandratios[col] * np.sqrt(data[f"{parameter_uncertainty} ({bands[0]})"]**2/data[f"{parameter} ({bands[0]})"]**2 + data[f"{parameter_uncertainty} ({bands[1]})"]**2/data[f"{parameter} ({bands[1]})"]**2) for col, bands in zip(bandratios.colnames, hc.bandratio_pairs)]
    bandratio_uncertainties_names = hc.extend_keys_to_RGB(parameter_uncertainty, hc.bandratio_labels)

    bandratio_uncertainties = table.Table(data=bandratio_uncertainties_data, names=bandratio_uncertainties_names)

    # Combine everything into one table
    data_combined = table.hstack([data, bandratios, bandratio_uncertainties])
    return data_combined


def add_colour_data_to_hyperspectral_data(data, key="R_rs"):
    """
    Add colour data (XYZ, xy, hue angle, FU, sRGB) to a data table.
    """
    # Spectral convolution to XYZ
    cols = [col for col in data.keys() if key in col]  # Find the relevant keys

    # Check that columns were actually found
    assert len(cols) > 0, f"No columns were found for key '{key}'."
    wavelengths = np.array([float(col.split(key)[1][1:]) for col in cols])  # Data wavelengths
    data_array = np.array(data[cols]).view(np.float64).reshape((-1, len(wavelengths)))  # Cast the relevant data to a numpy array

    # Convolve to XYZ
    data_XYZ = np.array([spectral.convolve_multi(spectral.cie_wavelengths, band, wavelengths, data_array) for band in spectral.cie_xyz]).T

    # Calculate xy from XYZ
    data_xy = wa.convert_XYZ_to_xy(data_XYZ)

    # Calculate the hue angle and associated FU index
    hue_angles = wa.convert_xy_to_hue_angle(data_xy)
    FU_indices = wa.convert_hue_angle_to_ForelUle(hue_angles)

    # Convert to sRGB
    data_sRGB = wa.convert_XYZ_to_sRGB(data_XYZ, axis=-1)

    # Put WACODI data in a table
    data_WACODI = [*data_XYZ.T, *data_xy.T, *data_sRGB.T, hue_angles, FU_indices]
    header_WACODI = [f"{key} ({label})" for label in [*"XYZxy", *hc.bands_sRGB, "hue", "FU"]]
    table_WACODI = table.Table(data=data_WACODI, names=header_WACODI)

    # Merge convolved data table with original data table
    data = table.hstack([data, table_WACODI])

    return data


def add_colour_data_to_hyperspectral_data_multiple_keys(data, keys=["R_rs", "Lu", "Lsky", "Ld", "Ed"]):
    """
    Add colour data (XYZ, xy, hue angle, FU, sRGB) to a data table.
    Applies `add_colour_data_to_table` for each of the given keys.
    Skips keys that were not found.
    """
    # Loop over keys
    for key in keys:
        try:
            data = add_colour_data_to_hyperspectral_data(data, key)
        except AssertionError:
            print(f"Key '{key}' was not found; continuing.")

    return data
