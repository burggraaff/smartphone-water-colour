"""
Module with functions etc for HydroColor
"""
from spectacle import io, calibrate, spectral
import numpy as np
from datetime import datetime, timedelta
from astropy import table
from scipy.linalg import block_diag

from . import colours
from . import statistics as stats


def add_Rref_to_covariance(covariance, Rref_uncertainty=0.01):
    """
    Add a column and row for R_ref to a covariance matrix.
    The input Rref_uncertainty is assumed fully uncorrelated
    to the other elements.
    """
    covariance_with_Rref = block_diag(covariance, [Rref_uncertainty**2])

    return covariance_with_Rref


def convert_Ld_to_Ed(Ld, R_ref=0.18):
    """
    Convert downwelling radiance from a grey card (Ld) to downwelling
    irradiance (Ed) using the reference reflectance (R_ref).
    """
    Ed = Ld / R_ref
    return Ed


def convert_Ld_to_Ed_covariance(Ld_covariance, Ed, R_ref=0.18, R_ref_uncertainty=0.01):
    """
    Convert the covariance in downwelling radiance (Ld) and the
    reference reflectance (R_ref) to a covariance in downwelling
    irradiance (Ed).
    """
    nr_bands = len(Ed)  # Number of bands - 3 for RGB, 4 for RGBG2
    total_covariance = add_Rref_to_covariance(Ld_covariance, R_ref_uncertainty)
    J = np.block([np.eye(nr_bands)/R_ref, (-Ed/R_ref)[:,np.newaxis]])  # Jacobian matrix
    Ed_covariance = J @ total_covariance @ J.T

    return Ed_covariance


def split_combined_radiances(radiances):
    """
    For a combined radiance array, e.g. [Lu(R), Lu(G), Lu(B), Ls(R), ..., Ld(G), Ld(B)],
    split it into three separate arrays: [Lu(R), Lu(G), Lu(B)], [Ls(R), ...], ...
    """
    n = len(radiances)//3
    Lu, Ls, Ld = radiances[:n], radiances[n:2*n], radiances[2*n:]
    return Lu, Ls, Ld


def R_RS(L_u, L_s, L_d, rho=0.028, R_ref=0.18):
    """
    Calculate the remote sensing reflectance (R_rs) from upwelling radiance L_u,
    sky radiance L_s, downwelling radiance L_d.
    Additional parameters are surface reflectivity rho (default 0.028), grey card
    reflectance R_ref (0.18).
    L_u, L_s, L_d can be NumPy arrays.
    """
    return (L_u - rho * L_s) / ((np.pi / R_ref) * L_d)


def R_RS_error(L_u, L_s, L_d, L_u_err, L_s_err, L_d_err, rho=0.028, R_ref=0.18):
    """
    Calculate the uncertainty in R_rs from the uncertainty in L_u, L_s, L_d.
    Note this does NOT account for uncertainty in R_ref nor covariance.
    """
    # Calculate squared errors individually
    R_rs_err_water = L_u_err**2 * ((0.18/np.pi) * L_d**-1)**2
    R_rs_err_sky = L_s_err**2 * ((0.18/np.pi) * 0.028 * L_d**-1)**2
    R_rs_err_card = L_d_err**2 * ((0.18/np.pi) * (L_u - 0.028 * L_s) * L_d**-2)**2

    R_rs_err = np.sqrt(R_rs_err_water + R_rs_err_sky + R_rs_err_card)
    return R_rs_err


def R_rs_covariance(L_Rref_covariance, R_rs, L_d, rho=0.028, R_ref=0.18):
    """
    Propagate the covariance in radiance and R_ref into a covariance matrix
    for R_rs. Automatically determine the number of bands and return an
    appropriately sized matrix.
    """
    # Determine the number of bands and create an appropriate identity matrix
    nr_bands = len(R_rs)
    I = np.eye(nr_bands)

    # Calculate the four parts of the Jacobian matrix
    J1 = 1/np.pi * R_ref * I * (1/L_d)
    J2 = -1/np.pi * R_ref * rho * I * (1/L_d)
    J3 = -1 * I * (R_rs / L_d)
    JR = R_rs[:,np.newaxis] / R_ref

    # Combine the parts of the Jacobian
    J = np.block([J1, J2, J3, JR])

    # Propagate the individual covariances
    R_rs_cov = J @ L_Rref_covariance @ J.T

    return R_rs_cov


def data_type_RGB(filename):
    """
    Find out if a given filename has RAW, JPEG, or linearised JPEG data.
    """
    name = filename.stem
    if "raw" in name:
        return "RAW"
    elif "jpeg" in name:
        if "linear" in name:
            return "JPEG (Linear)"
        else:
            return "JPEG"
    else:
        raise ValueError(f"File `{filename}` does not match known patterns ('raw', 'jpeg', 'jpeg_linear').")


def generate_paths(data_path, extension=".dng"):
    """
    Generate the paths to the water, sky, and greycard images
    """
    paths = [data_path/(photo + extension) for photo in ("water", "sky", "greycard")]
    return paths


def load_raw_images(*filenames):
    raw_images = [io.load_raw_image(filename) for filename in filenames]
    return raw_images


def load_jpeg_images(*filenames):
    jpg_images = [io.load_jpg_image(filename) for filename in filenames]
    return jpg_images


def load_exif(*filenames):
    exif = [io.load_exif(filename) for filename in filenames]
    return exif


def load_raw_thumbnails(*filenames):
    thumbnails = [io.load_raw_image_postprocessed(filename, half_size=True, user_flip=0) for filename in filenames]
    return thumbnails


box_size = 100
def central_slice_jpeg(*images, size=box_size):
    central_x, central_y = images[0].shape[0]//2, images[0].shape[1]//2
    central_slice = np.s_[central_x-size:central_x+size+1, central_y-size:central_y+size+1, :]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {2*size}x{2*size} pixels in the JPEG data")

    return images_cut


def central_slice_raw(*images, size=box_size):
    half_size = size//2

    central_x, central_y = images[0].shape[1]//2, images[0].shape[2]//2
    central_slice = np.s_[:, central_x-half_size:central_x+half_size+1, central_y-half_size:central_y+half_size+1]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {size}x{size} pixels in the RAW data")

    return images_cut


def RGBG2_to_RGB(*arrays):
    RGB_lists = [[array[0].ravel(), array[1::2].ravel(), array[2].ravel()] for array in arrays]
    return RGB_lists


def effective_wavelength(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]
    RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

    return RGB_wavelengths


def effective_bandwidth(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]

    RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
    effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths, axis=1)

    return effective_bandwidths


def get_radiances(data, parameters=["Lu", "Lsky", "Ld"]):
    """
    From a given data set containing radiances, return a table that has combined
    all relevant entries.
    """
    # Use ravel_table to get the elements out. That function is normally used to
    # ravel an RGB table for one parameter, e.g. combine Lu (R), Lu (G), Lu (B).
    # Here, we have to flip the order of operations and use a different loop_keys
    # argument to get the results in the right order.
    radiance_RGB = [stats.ravel_table(data, "{c} "+f"({c})", loop_keys=parameters) for c in colours]
    radiance_RGB_err = [stats.ravel_table(data, "{c}_err "+f"({c})", loop_keys=parameters) for c in colours]
    cols = [f"L ({c})" for c in colours] + [f"L_err ({c})" for c in colours]
    radiance = np.concatenate([radiance_RGB, radiance_RGB_err]).T

    radiance = table.Table(data=radiance, names=cols)
    return radiance


def UTC_timestamp(water_exif, conversion_to_utc=timedelta(hours=2)):
    try:
        timestamp = water_exif["EXIF DateTimeOriginal"].values
    except KeyError:
        timestamp = water_exif["Image DateTimeOriginal"].values
    # Convert to ISO format
    timestamp_ISO = timestamp[:4] + "-" + timestamp[5:7] + "-" + timestamp[8:10] + "T" + timestamp[11:]
    UTC = datetime.fromisoformat(timestamp_ISO)
    UTC = UTC - conversion_to_utc

    return UTC


def _convert_symmetric_matrix_to_list(sym):
    """
    Convert a symmetric matrix `sym` to a list that contains its
    upper-triangular (including diagonal) elements.
    """
    return sym[np.triu_indices_from(sym)]


def _convert_list_to_symmetric_matrix(symlist):
    """
    Convert a list containing elemens of a symmetric matrix
    (e.g. generated using _convert_symmetric_matrix_to_list) back
    into a matrix.
    """
    # Number of independent elements in symmetric matrix of size nxn is
    # L = n*(n+1)/2
    # Inverted gives n = -0.5 + 0.5*sqrt(1 + 8L)
    nr_columns = int(-0.5 + 0.5*np.sqrt(1 + 8*len(symlist)))

    # Create the array
    arr = np.zeros((nr_columns, nr_columns))
    arr[np.triu_indices(nr_columns)] = symlist  # Add the upper-triangular elements
    arr = arr + arr.T - np.diag(np.diag(arr))  # Add the lower-triangular elements without doubling the diagonal

    return arr


def _generic_header(elements, prefix=""):
    """
    Generate a generic header (list of names) for a list `elements`.
    Optionally use a prefix to identify them.
    """
    header = [f"{prefix}_{j:04d}" for j, ele in enumerate(elements)]
    return header


def write_results(saveto, timestamp, radiances, radiances_covariance, Ed, Ed_covariance, R_rs, R_rs_covariance, band_ratios, band_ratios_covariance, R_rs_xy, R_rs_xy_covariance, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, R_rs_FU_range):
    # assert len(water) == len(water_err) == len(sky) == len(sky_err) == len(grey) == len(grey_err) == len(Rrs) == len(Rrs_err), "Not all input arrays have the same length"

    # Split the covariance matrices out
    radiances_covariance_list = _convert_symmetric_matrix_to_list(radiances_covariance)
    Ed_covariance_list = _convert_symmetric_matrix_to_list(Ed_covariance)
    R_rs_covariance_list = _convert_symmetric_matrix_to_list(R_rs_covariance)
    band_ratios_covariance_list = _convert_symmetric_matrix_to_list(band_ratios_covariance)
    R_rs_xy_covariance_list = _convert_symmetric_matrix_to_list(R_rs_xy_covariance)

    # Headers for the covariance matrices
    radiances_covariance_header = _generic_header(radiances_covariance_list, "cov_L")
    Ed_covariance_header = _generic_header(Ed_covariance_list, "cov_Ed")
    R_rs_covariance_header = _generic_header(R_rs_covariance_list, "cov_R_rs_RGB")
    band_ratios_covariance_header = _generic_header(band_ratios_covariance_list, "cov_band_ratio")
    R_rs_xy_covariance_header = _generic_header(R_rs_xy_covariance_list, "cov_R_rs_xy")

    # Make a header with the relevant items
    header_RGB = ["Lu ({c})", "Lsky ({c})", "Ld ({c})", "Ed ({c})", "R_rs ({c})"]
    bands = "RGB"
    header_RGB_full = [[s.format(c=c) for c in bands] for s in header_RGB]
    header_hue = ["R_rs (G/R)", "R_rs (G/B)", "R_rs (x)", "R_rs (y)", "R_rs (hue)", "R_rs_err (hue)", "R_rs (FU)", "R_rs_min (FU)", "R_rs_max (FU)"]
    header = ["UTC", "UTC (ISO)"] + [item for sublist in header_RGB_full for item in sublist] + header_hue + radiances_covariance_header + Ed_covariance_header + R_rs_covariance_header + band_ratios_covariance_header + R_rs_xy_covariance_header

    # Add the data to a row, and that row to a table
    data = [[timestamp.timestamp(), timestamp.isoformat(), *radiances, *Ed, *R_rs, *band_ratios, *R_rs_xy, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, *R_rs_FU_range, *radiances_covariance_list, *Ed_covariance_list, *R_rs_covariance_list, *band_ratios_covariance_list, *R_rs_xy_covariance_list]]
    result = table.Table(rows=data, names=header)

    # Write the result to file
    result.write(saveto, format="ascii.fast_csv")
    print(f"Saved results to `{saveto}`")


def _convert_matrix_to_uncertainties_column(covariance_matrices, labels):
    """
    Take a column containing covariance matrices and generate a number of columns
    containing the uncertainties on its diagonal.
    """
    assert len(labels) == len(covariance_matrices[0]), f"Number of labels (len{labels}) does not match matrix dimensionality ({len(covariance_matrices[0])})."
    diagonals = np.array([np.diag(matrix) for matrix in covariance_matrices])
    uncertainties = np.sqrt(diagonals)
    columns = [table.Column(name=label, data=uncertainties[:,j]) for j, label in enumerate(labels)]
    return columns


def read_results(filename):
    """
    Read a results file generated with write_results.
    """
    # Read the file
    data = table.Table.read(filename)

    # Iterate over the different covariance columns and make them into arrays again
    covariance_keys = ["cov_L", "cov_Ed", "cov_R_rs_RGB", "cov_band_ratio", "cov_R_rs_xy"]
    for key_cov in covariance_keys:
        keys = sorted([key for key in data.keys() if key_cov in key])
        # [*row] puts the row data into a list; otherwise the iteration does not work
        covariance_matrices = [_convert_list_to_symmetric_matrix([*row]) for row in data[keys]]

        # Add a new column with these matrices and remove the raw data columns
        data.add_column(table.Column(name=key_cov, data=covariance_matrices))
        data.remove_columns(keys)

    # Iterate over the covariance matrices and calculate simple uncertainties from them
    covariance_keys_split = [np.ravel([[f"L{sub}_err ({c})" for c in colours] for sub in ["u", "sky", "d"]]),
                             [f"Ed_err ({c})" for c in colours],
                             [f"R_rs_err ({c})" for c in colours],
                             [f"R_rs_err ({ratio})" for ratio in ["G/R", "G/B"]],
                             [f"R_rs_err ({c})" for c in "xy"]]
    for key, keys_split in zip(covariance_keys, covariance_keys_split):
        uncertainties = _convert_matrix_to_uncertainties_column(data[key], keys_split)
        data.add_columns(uncertainties)

    return data
