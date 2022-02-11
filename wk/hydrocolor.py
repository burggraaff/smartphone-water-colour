"""
Module with functions etc for HydroColor
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

# Matrix for converting RGBG2 to RGB data
M_RGBG2_to_RGB = np.array([[1, 0  , 0, 0  ],
                           [0, 0.5, 0, 0.5],
                           [0, 0  , 1, 0  ]])

# Labels for band ratios, in the correct order
bandratio_pairs = [("G", "R"), ("B", "G"), ("R", "B")]
bandratio_labels = [f"{bands[0]}/{bands[1]}" for bands in bandratio_pairs]
bandratio_labels_latex = [r"$\frac{" + bands[0] + r"}{" + bands[1] + r"}$" for bands in bandratio_pairs]

# Grey card reflectance, empirically determined
# R_ref, R_ref_uncertainty = 0.21872167469852127, 0.02428559578772454  # From comparing Lu and Ed
# R_ref, R_ref_uncertainty = 0.185, 0.01  # From comparing measurements with a cosine collector
R_ref, R_ref_uncertainty = 0.18, 0.01


def add_Rref_to_covariance(covariance, R_ref_uncertainty=R_ref_uncertainty):
    """
    Add a column and row for R_ref to a covariance matrix.
    The input Rref_uncertainty is assumed fully uncorrelated
    to the other elements.
    """
    covariance_with_Rref = block_diag(covariance, [R_ref_uncertainty**2])

    return covariance_with_Rref


def convert_Ld_to_Ed(Ld, R_ref=R_ref):
    """
    Convert downwelling radiance from a grey card (Ld) to downwelling
    irradiance (Ed) using the reference reflectance (R_ref).
    """
    Ed = Ld * np.pi / R_ref
    return Ed


def convert_Ld_to_Ed_covariance(Ld_covariance, Ed, R_ref=R_ref, R_ref_uncertainty=R_ref_uncertainty):
    """
    Convert the covariance in downwelling radiance (Ld) and the
    reference reflectance (R_ref) to a covariance in downwelling
    irradiance (Ed).
    """
    nr_bands = len(Ed)  # Number of bands - 3 for RGB, 4 for RGBG2
    total_covariance = add_Rref_to_covariance(Ld_covariance, R_ref_uncertainty)
    J = np.block([np.eye(nr_bands)*np.pi/R_ref, (-Ed/R_ref)[:,np.newaxis]])  # Jacobian matrix
    Ed_covariance = J @ total_covariance @ J.T

    return Ed_covariance


def split_combined_radiances(radiances):
    """
    For a combined radiance array, e.g. [Lu(R), Lu(G), Lu(B), Ls(R), ..., Ld(G), Ld(B)],
    split it into three separate arrays: [Lu(R), Lu(G), Lu(B)], [Lsky(R), ...], ...
    """
    n = len(radiances)//3
    Lu, Lsky, Ld = radiances[:n], radiances[n:2*n], radiances[2*n:]
    return Lu, Lsky, Ld


def R_RS(L_u, L_s, L_d, rho=0.028, R_ref=R_ref):
    """
    Calculate the remote sensing reflectance (R_rs) from upwelling radiance L_u,
    sky radiance L_s, downwelling radiance L_d.
    Additional parameters are surface reflectivity rho (default 0.028), grey card
    reflectance R_ref.
    L_u, L_s, L_d can be NumPy arrays.
    """
    return (L_u - rho * L_s) / ((np.pi / R_ref) * L_d)


def R_rs_covariance(L_Rref_covariance, R_rs, L_d, rho=0.028, R_ref=R_ref):
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


def generate_folders(folders, pattern):
    """
    Given a list of folders and a pattern, look for subfolders in `folders`
    that match the given pattern.
    Example:
        for folder in generate_folders("water-colour-data/Balaton", "iPhone_SE")
    """
    for folder_main in folders:
        for folder, *_ in walk(folder_main):
            data_path = io.Path(folder) / pattern
            if data_path.exists():
                yield data_path


def generate_paths(data_path, extension=".dng", image_names=("water", "sky", "greycard"), multiple=False):
    """
    Generate the paths to the water, sky, and greycard images (in that order!).
    If desired, this can be changed to any set of images.
    If `multiple` is False, generate paths only to data_path/water.extension, etc.
    If `multiple` is True, generate paths to *any* files matching the pattern data_path/water*.extension, etc.

    Example: generate_paths("C:/myfolder/", extension=".dng")
    """
    # Add a wildcard to the file extension if desired
    if multiple:
        extension = "*" + extension

    # Find all filenames in the data folder that match the pattern, and concatenate them into one big list
    paths = sum([sorted(data_path.glob(photo + extension)) for photo in image_names], [])

    # Check that the right number of files have been returned
    # Note that this assertion is NOT triggered if, for example, 5 'water' images are present but zero of the others
    assert len(paths) >= 3, f"Fewer than {len(image_names)} images found in folder {data_path}. Only the following files were present:\n{paths}"
    return paths


def load_raw_images(filenames):
    """
    Load RAW images located at `filenames` (iterable).
    The images are stacked into one array.
    """
    raw_images = np.array([io.load_raw_image(filename) for filename in filenames])
    return raw_images


def load_jpeg_images(filenames):
    """
    Load JPEG images located at `filenames` (iterable).
    The images are stacked into one array, with the colour (RGB) axis moved to the front, like load_raw_images.
    """
    jpg_images = np.array([io.load_jpg_image(filename) for filename in filenames])

    # Move the colour axis forwards: (i, x, y, 3) -> (i, 3, x, y)
    jpg_images = np.moveaxis(jpg_images, -1, -3)
    return jpg_images


def load_raw_images_as_jpeg(filenames, user_flip=0, use_camera_wb=True, **kwargs):
    """
    Load RAW images located at `filenames` (iterable) and convert them to JPEG with standard settings.
    Additional **kwargs are passed to `load_raw_image_postprocessed`.
    The images are stacked into one array, with the colour (RGB) axis moved to the front, like load_raw_images.
    """
    jpg_images = np.array([io.load_raw_image_postprocessed(filename, user_flip=user_flip, use_camera_wb=use_camera_wb, **kwargs) for filename in filenames])

    # Move the colour axis forwards: (i, x, y, 3) -> (i, 3, x, y)
    jpg_images = np.moveaxis(jpg_images, -1, -3)
    return jpg_images


# Function specifically for making thumbnails
load_raw_thumbnails = partial(load_raw_images_as_jpeg, half_size=True, user_flip=0)


def convert_RGBG2_to_RGB_without_average(*arrays):
    """
    Convert RGBG2 arrays to RGB, keeping all pixels.
    This concatenates G and G2 rather than averaging them.
    """
    RGB_lists = [[array[0].ravel(), array[1::2].ravel(), array[2].ravel()] for array in arrays]
    return RGB_lists


def convert_RGBG2_to_RGB(data):
    """
    Convert RGBG2 data to RGB by matrix multiplication.
    """
    # Find out how many RGBG2 repetitions are in the data
    # For example, if the data have a shape of (12, N) then there
    # are three sets of RGBG2 data (e.g. Lu, Lsky, Ld).
    # Then repeat the conversion matrix that many times.
    nr_reps = len(data)//4
    M_RGBG2_to_RGB_repeated = block_diag(*[M_RGBG2_to_RGB]*nr_reps)

    # Convert to RGB
    RGB_data = M_RGBG2_to_RGB_repeated @ data
    return RGB_data


def convert_RGBG2_to_RGB_covariance(covariance):
    """
    Convert an RGBG2 covariance matrix to RGB by matrix multiplication.
    """
    # Find out how many RGBG2 repetitions are in the covariance matrix.
    # See convert_RGBG2_to_RGB for more information.
    nr_reps = len(covariance)//4
    M_RGBG2_to_RGB_repeated = block_diag(*[M_RGBG2_to_RGB]*nr_reps)

    # Convert to RGB
    RGB_covariance = M_RGBG2_to_RGB_repeated @ covariance @ M_RGBG2_to_RGB_repeated.T
    return RGB_covariance


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
    radiance_RGB = np.array([stats.ravel_table(data, "{c} "+f"({c})", loop_keys=parameters) for c in colours])
    radiance_RGB_err = np.array([stats.ravel_table(data, "{c}_err "+f"({c})", loop_keys=parameters) for c in colours])
    cols = [f"L ({c})" for c in colours] + [f"L_err ({c})" for c in colours]
    radiance = np.concatenate([radiance_RGB, radiance_RGB_err]).T

    radiance = table.Table(data=radiance, names=cols)

    # Add labels to indicate where the radiance came from
    sources = np.repeat(parameters, len(data))  # Repeats each of the parameter keys once for each row
    sources = table.Column(data=sources, name="source", dtype="S4")
    radiance.add_column(sources)

    return radiance


def calculate_bandratios(data_R, data_G, data_B):
    """
    Calculate the B/G, G/R, and R/B band ratios between given data.
        """
    GR = data_G / data_R  # G/R
    BG = data_B / data_G  # B/G
    RB = data_R / data_B  # R/B

    return np.array([GR, BG, RB])


def calculate_bandratios_covariance(data_R, data_G, data_B, data_covariance):
    """
    Propagate a covariance matrix for band ratios.
    """
    # Calculate the band ratios first
    GR, BG, RB = calculate_bandratios(data_R, data_G, data_B)

    # Construct the Jacobian matrix
    bandratios_J = np.array([[-GR/data_R, 1/data_R, 0         ],
                             [0         , -BG/data_G, 1/data_G],
                             [1/data_B  , 0       , -RB/data_B]])

    # Do the propagation
    bandratios_covariance = bandratios_J @ data_covariance @ bandratios_J.T

    return bandratios_covariance


def print_bandratios(bandratios, bandratios_covariance, key="R_rs"):
    """
    Provide pretty output about band ratio calculations.
    Prints the band ratios, their uncertainties, and correlations.
    """
    # Calculate the diagonal uncertainties and correlation from the covariance matrix
    correlation = stats.correlation_from_covariance(bandratios_covariance)
    uncertainties = stats.uncertainty_from_covariance(bandratios_covariance)

    # Select the off-diagonal elements from the correlation matrix
    indices = tuple([(0, 0, 1), (1, 2, 2)])
    correlation = correlation[indices]

    # Print the band ratios and uncertainties
    print("Calculated band ratios:")
    for label, ratio, uncertainty in zip(bandratio_labels, bandratios, uncertainties):
        print(f"{key} ({label}) = {ratio:.2f} +- {uncertainty:.2f}")

    # Print the correlations
    print("Correlations between band ratios:")
    for index0, index1, corr in zip(*indices, correlation):
        print(f"{key} ({bandratio_labels[index0]})-({bandratio_labels[index1]}): r = {corr:+.2f}")


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


def iso_timestamp(utc):
    """
    Convert a UTC timestamp from the data to ISO format.
    """
    return datetime.fromtimestamp(utc).isoformat()


def UTC_timestamp(water_exif, data_path=""):
    try:
        timestamp = water_exif["EXIF DateTimeOriginal"].values
    except KeyError:
        timestamp = water_exif["Image DateTimeOriginal"].values

    # Time zone: UTC+2 for Balaton data, UTC for NZ data
    if any("NZ" in path.stem for path in data_path.parents):
        conversion_to_utc = timedelta(hours=0)
    else:
        conversion_to_utc = timedelta(hours=2)

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


def extend_keys_to_RGB(keys):
    """
    For a given set of keys, e.g. ["Lu", "Lsky"], generate variants
    for each RGB band.
    """
    # If only one key was given, put it into a list
    if isinstance(keys, str):
        keys = [keys]

    # Add suffixes
    list_RGB = [key + " ({c})" for key in keys]
    list_RGB_full = [[s.format(c=c) for c in colours] for s in list_RGB]
    list_RGB_flat = sum(list_RGB_full, start=[])

    return list_RGB_flat


def write_results(saveto, timestamp, radiances, radiances_covariance, Ed, Ed_covariance, R_rs, R_rs_covariance, band_ratios, band_ratios_covariance, R_rs_xy, R_rs_xy_covariance, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, R_rs_FU_range):
    """
    Write the results from process_raw.py or process_jpeg.py to a CSV file.
    Covariance matrices are unravelled into their constituent elements.
    """
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
    header_RGB = extend_keys_to_RGB(["Lu", "Lsky", "Ld", "Ed", "R_rs"])
    header_hue = [f"R_rs ({key})" for key in [*bandratio_labels, *"xy"]] + ["R_rs (hue)", "R_rs_err (hue)", "R_rs (FU)", "R_rs_min (FU)", "R_rs_max (FU)"]
    header = ["UTC", "UTC (ISO)"] + header_RGB + header_hue + radiances_covariance_header + Ed_covariance_header + R_rs_covariance_header + band_ratios_covariance_header + R_rs_xy_covariance_header

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
                             extend_keys_to_RGB("Ed_err"),
                             extend_keys_to_RGB("R_rs_err"),
                             [f"R_rs_err ({c})" for c in bandratio_labels],
                             [f"R_rs_err ({c})" for c in "xy"]]
    for key, keys_split in zip(covariance_keys, covariance_keys_split):
        uncertainties = _convert_matrix_to_uncertainties_column(data[key], keys_split)
        data.add_columns(uncertainties)

    return data


def _print_or_save_latex(text, saveto=None):
    """
    Helper function to print or save LaTeX output from the functions below.
    """
    # If no saveto was given, print the result
    if saveto is None:
        print(text)
    # Else, print it to file
    else:
        with open(saveto, "w") as file:
            print(text, file=file)


def output_latex_vector(data, label="L", separator=r"\\", saveto=None):
    """
    Save a vector in LaTeX format for the flowchart.
    """
    # Start and end, always the same
    start = "\\mathbf{" + label + "} =\n    \\begin{bmatrix}"
    end = "    \\end{bmatrix}"

    # Put the data in the middle
    # If the data are long, show the first and last two elements
    if len(data) >= 4:
        middle = f"        {data[0]:.2g} {separator} {data[1]:.2g} {separator} \\dots {separator} {data[-1]:.2g}"
    # Else, just show the whole vector
    else:
        middle = "        " + f" {separator} ".join(f"{d:.2g}" for d in data)

    # Combine them with line breaks in between
    combined = "\n".join([start, middle, end])

    # Save or show the result
    _print_or_save_latex(combined, saveto=saveto)


def output_latex_matrix(data, label="L", saveto=None):
    """
    Save a matrix in LaTeX format for the flowchart.
    """
    # Start and end, always the same
    start = "\\mathbf{\\Sigma_{" + label + "}} &=\n    \\begin{bmatrix}"
    end = "    \\end{bmatrix}"

    # Put the data in the middle
    # If the data are long, show the first and last two elements
    if len(data) >= 4:
        convert_row = lambda row: f"        {row[0]:.1g} & {row[1]:.1g} & \\dots & {row[-1]:.1g}"
        dotrow = r"        \vdots & \vdots & \ddots & \vdots"
        middle = "\\\\\n".join([convert_row(data[0]), convert_row(data[1]), dotrow, convert_row(data[-1])])
    # Else, just show the whole vector
    else:
        convert_row = lambda row: "        " + " & ".join(f"{d:.1g}" for d in row)
        middle = " \\\\\n".join(convert_row(row) for row in data)

    # Combine them with line breaks in between
    combined = "\n".join([start, middle, end])

    # Save or show the result
    _print_or_save_latex(combined, saveto=saveto)


def output_latex_hueangle_FU(hueangle, hueangle_uncertainty, FU, FU_range, label=r"R_{rs}", saveto=None):
    """
    Save a LaTeX string for the hue angle and Forel-Ule index.
    """
    # Strings for the two variables
    string_hue = f"        {label} (\\alpha) &= ({hueangle:.0f} \\pm {hueangle_uncertainty:.0f})\\degree"
    string_FU = f"        {label} (FU) &= {FU} \\; ({FU_range[0]} - {FU_range[-1]})"

    # Combine them with line breaks in between
    combined = "\\\\\n".join([string_hue, string_FU])

    # Save or show the result
    _print_or_save_latex(combined, saveto=saveto)
