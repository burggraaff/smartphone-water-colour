"""
Module with functions etc for WACODI
"""
from spectacle.general import apply_to_multiple_args
from spectacle.spectral import convolve_multi, cie_wavelengths, cie_xyz, convert_to_XYZ
import numpy as np
from astropy import table
from .statistics import MAD

M_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])

M_XYZ_D65_to_XYZ_E = np.array([[ 1.0502616,  0.0270757, -0.0232523],
                               [ 0.0390650,  0.9729502, -0.0092579],
                               [-0.0024047,  0.0026446,  0.9180873]])

M_sRGB_to_XYZ_E = M_XYZ_D65_to_XYZ_E @ M_sRGB_to_XYZ

FU_hueangles = np.array([234.55, 227.168, 220.977, 209.994, 190.779, 163.084, 132.999, 109.054, 94.037, 83.346, 74.572, 67.957, 62.186, 56.435, 50.665, 45.129, 39.769, 34.906, 30.439, 26.337, 22.741])

def _convert_error_to_XYZ(RGB_errors, XYZ_matrix):
    """
    Convert RGB errors to XYZ
    Simple for now, assume given data are (3,)
    Simply square the XYZ matrix (element-wise) and matrix-multiply it
    with the square of the RGB errors, then take the square root
    """
    XYZ_errors = np.sqrt(XYZ_matrix**2 @ RGB_errors**2)
    return XYZ_errors


def convert_errors_to_XYZ(XYZ_matrix, *RGB_errors):
    """
    Apply _convert_error_to_XYZ to multiple arguments
    """
    XYZ_errors = apply_to_multiple_args(_convert_error_to_XYZ, RGB_errors, XYZ_matrix=XYZ_matrix)
    return XYZ_errors


def convert_XYZ_to_xy(*XYZ_data):
    """
    Convert data from XYZ to xy (chromaticity)
    """
    def _convert_single(XYZ):
        xy = XYZ[:2] / np.sum(XYZ, axis=0)
        return xy
    xy_all = apply_to_multiple_args(_convert_single, XYZ_data)
    return xy_all


def convert_XYZ_to_xy_covariance(XYZ_covariance, XYZ_data):
    """
    Convert XYZ covariances to xy using the Jacobian.
    """
    X, Y, Z = XYZ_data  # Split the elements out
    S = np.sum(XYZ_data, axis=0)  # Sum, used in denominators
    J = np.array([[(Y+Z)/S, -X/S**2, -X/S**2],
                  [-Y/S**2, (X+Z)/S, -Y/S**2]])
    xy_covariance = J @ XYZ_covariance @ J.T
    return xy_covariance


def convert_xy_to_hue_angle(*xy_data):
    """
    Convert data from xy (chromaticity) to hue angle (in degrees)
    """
    def _convert_single(xy):
        hue_angle = np.rad2deg(np.arctan2(xy[1]-1/3, xy[0]-1/3) % (2*np.pi))
        return hue_angle
    hue_angle_all = apply_to_multiple_args(_convert_single, xy_data)
    return hue_angle_all


def convert_xy_to_hue_angle_covariance(xy_covariance, xy_data):
    """
    Convert xy covariances to an uncertainty in hue angle using the Jacobian.
    """
    x, y = xy_data
    xc, yc = x-1/3, y-1/3  # Subtract the white point
    denominator = xc**2 + yc**2
    J = np.array([yc/denominator, -xc/denominator])
    hue_angle_uncertainty_rad = J @ xy_covariance @ J.T
    hue_angle_uncertainty = np.rad2deg(hue_angle_uncertainty_rad)
    return hue_angle_uncertainty


@np.vectorize
def convert_hue_angle_to_ForelUle(hue_angle):
    """
    Use a look-up table to convert a given hue angle to a Forel-Ule index.
    """
    try:  # Simply look for the lowest FU colour whose hue angle is less than ours
        bigger_than = np.where(hue_angle > FU_hueangles)[0][0]
    except IndexError:  # If the index is out of bounds, return 21
        bigger_than = 21
    return bigger_than


def convert_hue_angle_to_ForelUle_uncertainty(hue_angle_uncertainty, hue_angle):
    """
    Use a look-up table to convert a hue angle and its uncertainty into a range
    of Forel-Ule indices.
    """
    # Calculate the hue angles corresponding to +-1 sigma
    minmax_hueangle = hue_angle - hue_angle_uncertainty, hue_angle + hue_angle_uncertainty

    # Convert the minimum and maximum to Forel-Ule indices
    # Sort because hue angle and FU are inversely related
    minmax_FU = np.sort(convert_hue_angle_to_ForelUle(minmax_hueangle))

    return minmax_FU


def add_colour_data_to_table(data, key="R_rs"):
    """
    Add colour data (XYZ, xy, hue angle, FU) to a data table.
    """
    # Spectral convolution to XYZ
    cols = [col for col in data.keys() if key in col]  # Find the relevant keys
    wavelengths = np.array([float(col.split(key)[1][1:]) for col in cols])  # Data wavelengths
    data_array = np.array(data[cols]).view(np.float64).reshape((-1, len(wavelengths)))  # Cast the relevant data to a numpy array

    # Convolve to XYZ
    data_XYZ = np.array([convolve_multi(cie_wavelengths, band, wavelengths, data_array) for band in cie_xyz])

    # Calculate xy from XYZ
    data_xy = convert_XYZ_to_xy(data_XYZ)

    # Calculate the hue angle and associated FU index
    hue_angles = convert_xy_to_hue_angle(data_xy)
    FU_indices = convert_hue_angle_to_ForelUle(hue_angles)

    # Put WACODI data in a table
    table_WACODI = table.Table(data=[*data_XYZ, *data_xy, hue_angles, FU_indices], names=[f"{key} ({label})" for label in [*"XYZxy", "hue", "FU"]])

    # Merge convolved data table with original data table
    data = table.hstack([data, table_WACODI])

    return data


def compare_FU_matches_from_hue_angle(x, y):
    """
    Count the percentage of matching FU colours in x and y.
    Return the percentage that are the same (e.g. 1,1)
    the percentage within 1 (e.g. 1, 2), and the MAD.
    """
    assert len(x) == len(y), f"x and y have different lengths: {len(x)} and {len(y)}."

    # Convert hue angles to Forel-Ule colours
    x_FU, y_FU = convert_hue_angle_to_ForelUle([x, y])

    # Count the number of (near-)matching FU colours
    matches = np.where(x_FU == y_FU)[0]
    near_matches = np.where(np.abs(x_FU - y_FU) <= 1)[0]
    mad = MAD(x_FU, y_FU)

    # Convert counts to percentages
    matches_percent = 100*len(matches)/len(x)
    near_matches_percent = 100*len(near_matches)/len(x)

    return matches_percent, near_matches_percent, mad
