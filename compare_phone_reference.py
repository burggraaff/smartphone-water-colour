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
from spectacle.general import RMS
from astropy import table
from datetime import datetime
from wk import hydrocolor as hc

# Time limit for inclusion
max_time_diff = 60*5  # 5 minutes

# Get the data folder from the command line
path_calibration, path_phone, path_reference = io.path_from_input(argv)

# Find out if we're doing JPEG or RAW
data_type = path_phone.stem.split("_")[-1]

# Find out what reference sensor we're using
if "So-Rad" in path_reference.stem:
    reference = "So-Rad"
    ref_small = "sorad"
elif "wisp" in path_reference.stem:
    reference = "WISP-3"
    ref_small = "wisp"
else:
    raise ValueError(f"Unknown reference sensor for file {path_reference}")

# Get Camera object
camera = load_camera(path_calibration)
print(f"Loaded Camera object:\n{camera}")
cameralabel = f"{camera.name} ({data_type.upper()})"

# Find the effective wavelength corresponding to the RGB bands
camera._load_spectral_response()
spectral_response = camera.spectral_response
wavelengths_phone = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths_phone, RGB_responses)

# Effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands

table_phone = table.Table.read(path_phone)
table_reference = table.Table.read(path_reference)

# Spectral convolution
# Convolve Rrs itself for now because of fingerprinting
for key in ["Ed", "Lsky", "Lu", "Rrs"]:
    cols = [col for col in table_reference.keys() if key in col]  # Find the relevant keys
    wavelengths = np.array([float(col.split("_")[1]) for col in cols])  # Data wavelengths
    data = np.array(table_reference[cols]).view(np.float64).reshape((-1, len(wavelengths)))  # Cast the relevant data to a numpy array

    data_convolved = camera.convolve_multi(wavelengths, data).T  # Apply spectral convolution

    # Put convolved data in a table
    keys_convolved = [f"{key} {band}" for band in hc.colours]
    table_convolved = table.Table(data=data_convolved, names=keys_convolved)

    # Merge convolved data table with original data table
    table_reference = table.hstack([table_reference, table_convolved])

# For non-SoRad systems, don't convolve Rrs directly
if reference != "So-Rad":
    for band in hc.colours:
        table_reference[f"Rrs {band}"] = (table_reference[f"Lu {band}"] - 0.028*table_reference[f"Lsky {band}"])/table_reference[f"Ed {band}"]

data_phone = []
data_reference = []

# Match observations between the data sets
for row in table_phone:
    # Find close matches in time
    time_differences = np.abs(table_reference["UTC"] - row["UTC"])
    close_enough = np.where(time_differences <= max_time_diff)
    closest = time_differences.argmin()
    min_time_diff = time_differences[closest]
    if min_time_diff > max_time_diff:  # If no close enough matches are found, skip this observation
        continue
    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    reference_time = datetime.fromtimestamp(table_reference[closest]["UTC"]).isoformat()
    print("----")
    print(f"Phone time: {phone_time} ; {reference} time: {reference_time} ; Closest: {min_time_diff:.1f} seconds")

    # Calculate the median Rrs within the matching observations, and uncertainty on this spectrum
    Rrs = np.array([np.nanmedian(table_reference[f"Rrs_{wvl:.1f}"][close_enough]) for wvl in wavelengths])
    Rrs_err = np.array([np.nanstd(table_reference[f"Rrs_{wvl:.1f}"][close_enough]) for wvl in wavelengths])

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small}_comparison/{camera.name}_{data_type}_{phone_time}.pdf".replace(":", "-")

    # Plot the spectrum for comparison
    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    plt.plot(wavelengths, Rrs, c="k")
    plt.fill_between(wavelengths, Rrs-Rrs_err, Rrs+Rrs_err, facecolor="0.3")
    for j, (c, pc) in enumerate(zip("RGB", hc.plot_colours)):
        plt.errorbar(RGB_wavelengths[j], row[f"Rrs {c}"], xerr=effective_bandwidths[j]/2, yerr=row[f"Rrs_err {c}"], fmt="o", c=pc)
        plt.errorbar(RGB_wavelengths[j], table_reference[closest][f"Rrs {c}"], xerr=effective_bandwidths[j]/2, yerr=0, fmt="^", c=pc)
    plt.grid(True, ls="--")
    plt.xlim(200, 900)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.15)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{cameralabel}\n{phone_time}")
    plt.savefig(saveto)
    plt.show()
    plt.close()

    data_phone.append(row)
    data_reference.append(table_reference[closest])

data_phone = table.vstack(data_phone)
data_reference = table.vstack(data_reference)

sorad_wavelengths_RGB = [wavelengths[np.abs(wavelengths-wvl).argmin()] for wvl in RGB_wavelengths]

parameters = ["Lu", "Lsky", "Ed"]
labels_phone = ["$L_u$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [ADU nm$^{-1}$ sr$^{-1}$]", "$E_d$ [ADU nm$^{-1}$]"]
labels_reference = ["$L_u$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "$E_d$ [W m$^{-2}$ nm$^{-1}$]"]

for param, label_phone, label_reference in zip(parameters, labels_phone, labels_reference):
    hc.correlation_plot_RGB(data_reference, data_phone, param+" {c}", param+" {c}", xerrlabel=None, yerrlabel=param+"_err {c}", xlabel=f"{reference} {label_reference}", ylabel=f"{cameralabel} {label_phone}", saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_{param}.pdf")

label_Rrs = "$R_{rs}$ [sr$^{-1}$]"
hc.correlation_plot_RGB_equal(data_reference, data_phone, "Rrs {c}", "Rrs {c}", xerrlabel=None, yerrlabel="Rrs_err {c}", xlabel=f"{reference} {label_Rrs}", ylabel=f"{cameralabel}\n{label_Rrs}", saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_Rrs.pdf")

# Correlation plot: Band ratios/differences
hc.correlation_plot_bands(data_reference, data_phone, xlabel=reference, ylabel=cameralabel, saveto=f"results/comparison_{reference}_X_{camera.name}_bands.pdf")
