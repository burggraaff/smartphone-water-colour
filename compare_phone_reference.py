"""
Compare smartphone RGB RAW data to hyperspectral data.

Command-line inputs:
    * SPECTACLE calibration folder
    * Smartphone data file
    * Hyperspectral reference data file

Example:
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/combined_iPhone_SE_raw.csv water-colour-data/Balaton_20190703/SoRad/So-Rad_Balaton2019.csv

To do:
    * Histograms of residuals
    * Fit radiances
    * Add MAPD, other deviation measures
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, spectral, load_camera
from astropy import table
from datetime import datetime
from wk import hydrocolor as hc, wacodi as wa

# Time limit for inclusion
max_time_diff = 60*5  # 5 minutes

# Get the data folder from the command line
path_calibration, path_phone, path_reference = io.path_from_input(argv)

# Find out if we're doing JPEG or RAW
data_type = hc.data_type_RGB(path_phone)

# Find out what reference sensor we're using
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

# Spectral convolution
# Convolve R_rs itself for now because of fingerprinting
for key in ["Ed", "Lsky", "Lu", "R_rs"]:
    cols = [col for col in table_reference.keys() if key in col and not any(f"({label})" in col for label in [*"XYZxy", "hue", "FU"])]  # Find the relevant keys
    wavelengths = np.array([float(col.split(key)[1][1:]) for col in cols])  # Data wavelengths
    data = np.array(table_reference[cols]).view(np.float64).reshape((-1, len(wavelengths)))  # Cast the relevant data to a numpy array

    # Apply spectral convolution with the RGB (not G2) bands
    data_convolved = camera.convolve_multi(wavelengths, data)[:3].T

    # Put convolved data in a table
    keys_convolved = [f"{key} ({c})" for c in hc.colours]
    table_convolved = table.Table(data=data_convolved, names=keys_convolved)

    # Merge convolved data table with original data table
    table_reference = table.hstack([table_reference, table_convolved])

# For the WISP-3, where we have Lu, Lsky, and Ed, don't convolve R_rs directly
if reference == "WISP-3":
    for c in hc.colours:
        table_reference[f"R_rs ({c})"] = (table_reference[f"Lu ({c})"] - 0.028*table_reference[f"Lsky ({c})"])/table_reference[f"Ed ({c})"]

# Add band ratios to table
bandratio_GR = table_reference["R_rs (G)"]/table_reference["R_rs (R)"]
bandratio_GR.name = "R_rs (G/R)"
# bandratio_GR_err = bandratio_GR * np.sqrt(table_reference["R_rs_err (G)"]**2/table_reference["R_rs (G)"]**2 + table_reference["R_rs_err (R)"]**2/table_reference["R_rs (R)"]**2)
bandratio_GB = table_reference["R_rs (G)"]/table_reference["R_rs (B)"]
bandratio_GB.name = "R_rs (G/B)"
# bandratio_GB_err = bandratio_GR * np.sqrt(table_reference["R_rs_err (G)"]**2/table_reference["R_rs (G)"]**2 + table_reference["R_rs_err (B)"]**2/table_reference["R_rs (B)"]**2)
table_reference.add_columns([bandratio_GR, bandratio_GB])

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
    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    reference_time = datetime.fromtimestamp(table_reference[closest]["UTC"]).isoformat()

    # Calculate the median Lu/Lsky/Ed/R_rs within the matching observations, and uncertainty on this spectrum
    row_reference = table.Table(table_reference[closest])
    for key in ["Ed", "Lu", "Lsky", "R_rs"]:
        # Average over the "close enough" rows
        keys = [f"{key}_{wvl:.1f}" for wvl in wavelengths] + [f"{key} ({c})" for c in hc.colours]
        keys_err = [f"{key}_err_{wvl:.1f}" for wvl in wavelengths] + [f"{key}_err ({c})" for c in hc.colours]

        row_reference[keys][0] = [np.nanmedian(table_reference[k][close_enough]) for k in keys]
        uncertainties = np.array([np.nanstd(table_reference[k][close_enough]) for k in keys])
        row_uncertainties = table.Table(data=uncertainties, names=keys_err)
        row_reference = table.hstack([row_reference, row_uncertainties])

    # If the uncertainties on the reference data are above a threshold, disregard this match-up
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
    R_rs = np.array([row_reference[f"R_rs_{wvl:.1f}"][0] for wvl in wavelengths])
    R_rs_err = np.array([row_reference[f"R_rs_err_{wvl:.1f}"][0] for wvl in wavelengths])
    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    plt.plot(wavelengths, R_rs, c="k")
    plt.fill_between(wavelengths, R_rs-R_rs_err, R_rs+R_rs_err, facecolor="0.3")
    for j, (c, pc) in enumerate(zip("RGB", hc.plot_colours)):
        plt.errorbar(RGB_wavelengths[j], row[f"R_rs ({c})"], xerr=effective_bandwidths[j]/2, yerr=row[f"R_rs_err ({c})"], fmt="o", c=pc)
        plt.errorbar(RGB_wavelengths[j], row_reference[f"R_rs ({c})"][0], xerr=effective_bandwidths[j]/2, yerr=row_reference[f"R_rs_err ({c})"][0], fmt="^", c=pc)
    plt.grid(True, ls="--")
    plt.xlim(300, 900)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.15)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{cameralabel}\n{phone_time}")
    plt.savefig(saveto)
    plt.show()
    plt.close()

# Make new tables from the match-up rows
data_phone = table.vstack(data_phone)
data_reference = table.vstack(data_reference)

# Correlation plot: Radiances and irradiance
parameters = ["Lu", "Lsky", "Ed"]
labels = ["$L_u$", "$L_{sky}$", "$E_d$"]
units_phone = ["[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$]"]
units_reference = ["[W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "[W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "[W m$^{-2}$ nm$^{-1}$]"]

for param, label, unit_phone, unit_reference in zip(parameters, labels, units_phone, units_reference):
    hc.correlation_plot_RGB(data_reference, data_phone, param+" ({c})", param+" ({c})", xerrlabel=param+"_err ({c})", yerrlabel=param+"_err ({c})", xlabel=f"{reference} {label} {unit_reference}", ylabel=f"{cameralabel} {label} {unit_phone}", regression="all", saveto=f"{saveto_base}_{param}.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = "$R_{rs}$"
unit_R_rs = "[sr$^{-1}$]"
hc.correlation_plot_RGB_equal(data_reference, data_phone, "R_rs ({c})", "R_rs ({c})", xerrlabel="R_rs_err ({c})", yerrlabel="R_rs_err ({c})", xlabel=f"{reference} {label_R_rs} {unit_R_rs}", ylabel=f"{cameralabel}\n{label_R_rs} {unit_R_rs}", regression="all", saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_R_rs.pdf")

# Correlation plot: Band ratios
hc.correlation_plot_bands(data_reference["R_rs (G/R)"], data_phone["R_rs (G/R)"], data_reference["R_rs (G/B)"], data_phone["R_rs (G/B)"], x_err_GR=None, y_err_GR=data_phone["R_rs_err (G/R)"], x_err_GB=None, y_err_GB=data_phone["R_rs_err (G/B)"], quantity="$R_{rs}$", xlabel=reference, ylabel=cameralabel, saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_band_ratio.pdf")

# Correlation plot: Radiance (all combined)
radiance_reference = hc.get_radiances(data_reference, parameters=["Lu", "Lsky"])
radiance_phone = hc.get_radiances(data_phone, parameters=["Lu", "Lsky"])

label = "$L$"
unit = "[ADU nm$^{-1}$ sr$^{-1}$]"
hc.correlation_plot_RGB(radiance_reference, radiance_phone, "L ({c})", "L ({c})", xerrlabel="L_err ({c})", yerrlabel="L_err ({c})", xlabel=f"{reference} {label} {unit}", ylabel=f"{cameralabel} {label} {unit}", regression="all", saveto=f"{saveto_base}_L.pdf")

# Correlation plot: hue angle and Forel-Ule index
wa.correlation_plot_hue_angle_and_ForelUle(data_reference["R_rs (hue)"], data_phone["R_rs (hue)"], xlabel=reference, ylabel=cameralabel, saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_hueangle_ForelUle.pdf")
