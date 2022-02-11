"""
Compare smartphone RGB RAW data to hyperspectral data.

Command-line inputs:
    * SPECTACLE calibration folder
    * Smartphone data file
    * Hyperspectral reference data file

Example:
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/combined_iPhone_SE_raw.csv water-colour-data/So-Rad_Balaton2019.csv
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, spectral, load_camera
from astropy import table
from wk import hydrocolor as hc, plot

# Time limit for inclusion
max_time_diff = 60*10  # 10 minutes for everything except NZ data
# max_time_diff = 60*60  # 60 minutes for NZ data

# Get the data folder from the command line
path_calibration, path_phone, path_reference = io.path_from_input(argv)

# Find out if we're doing JPEG or RAW
data_type = hc.data_type_RGB(path_phone)

# Find out what reference sensor we're using
reference, ref_small = hc.get_reference_name(path_reference)

# Get Camera object
camera = load_camera(path_calibration)
print(f"Loaded Camera object:\n{camera}")

# Names of the data being compared and some useful labels
cameralabel = f"{camera.name} {data_type}"
saveto_base = f"results/comparison_{reference}_X_{camera.name}_{data_type}"
print(f"Comparing data from {reference} and {cameralabel}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Find the effective wavelength corresponding to the RGB bands
camera._load_spectral_response()
spectral_response = camera.spectral_response
wavelengths_phone = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths_phone, RGB_responses)

# Effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands

# Read the data
table_phone = hc.read_results(path_phone)
table_reference = table.Table.read(path_reference)
print("Finished reading data")

# Function to extract column names
def get_keys_for_parameter(data, key_include, keys_exclude=[*"XYZxyRGB", "hue", "FU", "sR", "sG", "sB"]):
    return [col for col in data.keys() if key_include in col and not any(f"({label})" in col for label in keys_exclude)]

def get_wavelengths_from_keys(cols, key):
    return np.array([float(col.split(key)[1][1:]) for col in cols])

# Parameters of interest
parameters = ["Ed", "Lsky", "Lu", "R_rs"]
cols_example = get_keys_for_parameter(table_reference, parameters[0])
wavelengths = get_wavelengths_from_keys(cols_example, key=parameters[0])

# Convolve the hyperspectral data
# Convolve R_rs itself for now because of fingerprinting
for key in parameters:
    cols = get_keys_for_parameter(table_reference, key)
    data = np.array(table_reference[cols]).view(np.float64).reshape((-1, len(wavelengths)))  # Cast the relevant data to a numpy array

    # Apply spectral convolution with the RGB (not G2) bands
    data_convolved = camera.convolve_multi(wavelengths, data)[:3].T

    # Put convolved data in a table
    keys_convolved = hc.extend_keys_to_RGB(key)
    table_convolved = table.Table(data=data_convolved, names=keys_convolved)

    # Merge convolved data table with original data table
    table_reference = table.hstack([table_reference, table_convolved])

# Don't use for now -- doesn't seem to work very well
# # If we are using JPEG data, simply use the sRGB values already in the data files
# else:
#     for key in parameters:
#         bands_RGB = hc.extend_keys_to_RGB(key)
#         bands_sRGB = [f"{key} (s{band})" for band in hc.colours]
#         table_reference.rename_columns(bands_sRGB, bands_RGB)

# For the WISP-3, where we have Lu, Lsky, and Ed, don't convolve R_rs directly
if reference == "WISP-3":
    for c in hc.colours:
        table_reference[f"R_rs ({c})"] = (table_reference[f"Lu ({c})"] - 0.028*table_reference[f"Lsky ({c})"])/table_reference[f"Ed ({c})"]


# Find matches
data_phone, data_reference = [], []  # Lists to contain matching table entries
for row in table_phone:  # Loop over the smartphone table to look for matches
    # Find matches within a threshold
    time_differences = np.abs(table_reference["UTC"] - row["UTC"])
    close_enough = np.where(time_differences <= max_time_diff)[0]
    closest = time_differences.argmin()
    min_time_diff = time_differences[closest]
    if min_time_diff > max_time_diff:  # If no close enough matches are found, skip this observation
        continue
    phone_time = hc.iso_timestamp(row['UTC'])
    reference_time = hc.iso_timestamp(table_reference[closest]["UTC"])

    # Calculate the median Lu/Lsky/Ed/R_rs within the matching observations, and uncertainty on this spectrum
    row_reference = table.Table(table_reference[closest])
    for key in parameters:
        # Average over the "close enough" rows
        keys = [f"{key}_{wvl:.1f}" for wvl in wavelengths] + hc.extend_keys_to_RGB(key)
        keys_err = [f"{key}_err_{wvl:.1f}" for wvl in wavelengths] + hc.extend_keys_to_RGB(key+"_err")

        row_reference[keys][0] = [np.nanmedian(table_reference[k][close_enough]) for k in keys]
        uncertainties = np.array([np.nanstd(table_reference[k][close_enough]) for k in keys])
        row_uncertainties = table.Table(data=uncertainties, names=keys_err)
        row_reference = table.hstack([row_reference, row_uncertainties])

    # If the uncertainties on the reference data are above a threshold, disregard this match-up
    # This may cause differences between RAW and JPEG matchup numbers for the same data set
    threshold = 0.1  # relative
    if any(row_reference[f"R_rs_err ({c})"]/row_reference[f"R_rs ({c})"] >= threshold for c in hc.colours):
        continue

    # Add some metadata that may be used to identify the quality of the match
    metadata = table.Table(names=["nr_matches", "closest_match"], data=np.array([len(close_enough), time_differences[closest]]))
    row_reference = table.hstack([metadata, row_reference])
    print("----")
    print(f"Phone time: {phone_time} ; {reference} time: {reference_time}")
    print(f"Number of matches: {row_reference['nr_matches'][0]:.0f}; Closest match: {row_reference['closest_match'][0]:.0f} s difference")

    # Store matched rows in lists
    data_phone.append(row)
    data_reference.append(row_reference)

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small}_comparison/{camera.name}_{data_type}_{phone_time}.pdf".replace(":", "-")

    # Plot the spectrum for comparison
    R_rs_reference = np.array([row_reference[f"R_rs_{wvl:.1f}"][0] for wvl in wavelengths])
    R_rs_reference_uncertainty = np.array([row_reference[f"R_rs_err_{wvl:.1f}"][0] for wvl in wavelengths])

    R_rs_phone = list(row[hc.extend_keys_to_RGB("R_rs")])
    R_rs_phone_err = list(row[hc.extend_keys_to_RGB("R_rs_err")])

    plot.plot_R_rs_RGB(RGB_wavelengths, R_rs_phone, effective_bandwidths, R_rs_phone_err, reference=[wavelengths, R_rs_reference, R_rs_reference_uncertainty], title=f"{cameralabel}\n{phone_time}", saveto=saveto)

# Make new tables from the match-up rows
data_phone = table.vstack(data_phone)
data_reference = table.vstack(data_reference)

# Add typical errors to R_rs (R, G, B) if only a single match was found
indices_single_match, indices_multiple_matches = np.where(data_reference["nr_matches"] == 1), np.where(data_reference["nr_matches"] > 1)
keys_uncertainties = hc.extend_keys_to_RGB([param+"_err" for param in parameters])
for key in keys_uncertainties:
    data_reference[key][indices_single_match] = np.nanmedian(data_reference[key][indices_multiple_matches])

# Add band ratios to reference data
bandratios = table.Table(data=hc.calculate_bandratios(data_reference["R_rs (R)"], data_reference["R_rs (G)"], data_reference["R_rs (B)"]).T, names=[f"R_rs ({label})" for label in hc.bandratio_labels])

bandratio_uncertainties = table.Table(data=[bandratios[col] * np.sqrt(data_reference[f"R_rs_err ({bands[0]})"]**2/data_reference[f"R_rs ({bands[0]})"]**2 + data_reference[f"R_rs_err ({bands[1]})"]**2/data_reference[f"R_rs ({bands[1]})"]**2) for col, bands in zip(bandratios.colnames, hc.bandratio_pairs)], names=[f"R_rs_err ({label})" for label in hc.bandratio_labels])

data_reference = table.hstack([data_reference, bandratios, bandratio_uncertainties])

# Save the comparison table to file
saveto_data = f"{saveto_base}_data.csv"
table_combined = table.hstack([data_reference, data_phone], table_names=["reference", "phone"])
table_combined.remove_columns([key for key in table_combined.keys() if "cov_" in key])
table_combined.write(saveto_data, format="ascii.fast_csv", overwrite=True)
print(f"Saved comparison table to `{saveto_data}`.")

# Correlation plot: Radiances and irradiance
plot.correlation_plot_radiance(data_reference, data_phone, keys=["Lu", "Lsky"], xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data_reference, data_phone, keys=["Lu", "Lsky"], xlabel=f"{reference}\n$L$ {plot.Wnmsr}", ylabel=f"{cameralabel} $L$ [a.u.]", saveto=f"{saveto_base}_radiance_simple.pdf")
plot.correlation_plot_RGB(data_reference, data_phone, "Ed ({c})", "Ed ({c})", xerrlabel="Ed_err ({c})", yerrlabel="Ed_err ({c})", xlabel=f"{reference} {plot.keys_latex['Ed']} {plot.Wnm}", ylabel=f"{cameralabel} {plot.keys_latex['Ed']} {plot.ADUnm}", regression="rgb", saveto=f"{saveto_base}_Ed.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_RGB_equal(data_reference, data_phone, "R_rs", errlabel="R_rs_err", xlabel=f"{reference} {label_R_rs} {plot.persr}", ylabel=f"{cameralabel}\n{label_R_rs} {plot.persr}", regression="all", difference_unit=plot.persr, saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data_reference, data_phone, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data_reference["R_rs (hue)"], data_phone["R_rs (hue)"], xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
