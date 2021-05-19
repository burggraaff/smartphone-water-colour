from spectacle.general import apply_to_multiple_args
import numpy as np


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
        xy = XYZ[:2] / XYZ.sum(axis=0)
        return xy
    xy_all = apply_to_multiple_args(_convert_single, XYZ_data)
    return xy_all


def convert_XYZ_to_xy_covariance(XYZ_covariance, XYZ_data):
    """
    Convert XYZ covariances to xy using the Jacobian.
    """
    X, Y, Z = XYZ_data  # Split the elements out
    S = XYZ_data.sum(axis=0)  # Sum, used in denominators
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


def convert_XYZ_error_to_hue_angle(XYZ_data, XYZ_error):
    """
    For a single XYZ vector and error vector, convert this to an error in hue angle
    """
    X,Y,Z = XYZ_data
    Xerr, Yerr, Zerr = XYZ_error
    ddX = (3*Z - 3*Y) / (5*X**2 - 2*X*(4*Y + Z) + 5*Y**2 - 2*Y*Z + 2*Z**2)
    ddY = (3*X - 3*Z) / (5*X**2 - 2*X*(4*Y + Z) + 5*Y**2 - 2*Y*Z + 2*Z**2)
    hue_err = np.sqrt(Xerr**2 * ddX**2 + Yerr**2 * ddY**2)  # Radians
    hue_err = np.rad2deg(hue_err)
    return hue_err
