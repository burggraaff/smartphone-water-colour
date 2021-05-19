from spectacle.general import apply_to_multiple_args
import numpy as np
from matplotlib import pyplot as plt, transforms
from matplotlib.patches import Ellipse
from colorio._tools import plot_flat_gamut
from .hydrocolor import correlation_from_covariance

M_sRGB_to_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                          [0.2126729, 0.7151522, 0.0721750],
                          [0.0193339, 0.1191920, 0.9503041]])

M_XYZ_D65_to_XYZ_E = np.array([[ 1.0502616,  0.0270757, -0.0232523],
                               [ 0.0390650,  0.9729502, -0.0092579],
                               [-0.0024047,  0.0026446,  0.9180873]])

M_sRGB_to_XYZ_E = M_XYZ_D65_to_XYZ_E @ M_sRGB_to_XYZ

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


def _confidence_ellipse(center, covariance, ax, covariance_scale=1, **kwargs):
    """
    Plot a confidence ellipse from a given (2x2) covariance matrix.
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    correlation = correlation_from_covariance(covariance)[0,1]
    ell_radius_x = np.sqrt(1 + correlation)
    ell_radius_y = np.sqrt(1 - correlation)
    ellipse = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, **kwargs)

    scale_x = np.sqrt(covariance[0,0])*covariance_scale
    scale_y = np.sqrt(covariance[1,1])*covariance_scale

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(*center)
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_xy_on_gamut_covariance(xy, xy_covariance, covariance_scale=1):
    """
    Plot xy coordinates on the gamut including their covariance ellipse.
    """
    fig = plt.figure(figsize=(3,3))
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    _confidence_ellipse(xy, xy_covariance, plt.gca(), covariance_scale=covariance_scale, edgecolor="k", fill=False, linestyle="--")
    plt.scatter(*xy, c="k", s=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
    plt.close()
