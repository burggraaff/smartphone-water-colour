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
    keys_convolved = [f"{key} {band}" for band in [*"RGB", "G2"]]
    table_convolved = table.Table(data=data_convolved, names=keys_convolved)

    # Merge convolved data table with original data table
    table_reference = table.hstack([table_reference, table_convolved])

data_phone = []
data_reference = []

for row in table_phone:
    time_differences = np.abs(table_reference["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 1000:
        continue
    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    sorad_time = datetime.fromtimestamp(table_reference[closest]["UTC"]).isoformat()
    print("----")
    print(f"Phone time: {phone_time} ; {reference} time: {sorad_time} ; Difference: {time_diff:.1f} seconds")

    Rrs = np.array([table_reference[f"Rrs_{wvl:.1f}"][closest] for wvl in wavelengths])

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small}_comparison/{camera.name}_{data_type}_{phone_time}.pdf".replace(":", "-")

    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    plt.plot(wavelengths, Rrs, c="k")
    for j, c in enumerate("RGB"):
        plt.errorbar(RGB_wavelengths[j], row[f"Rrs {c}"], xerr=effective_bandwidths[j]/2, yerr=row[f"Rrs_err {c}"], fmt="o", c=c.lower())
        plt.errorbar(RGB_wavelengths[j], table_reference[closest][f"Rrs {c}"], xerr=effective_bandwidths[j]/2, yerr=0, fmt="^", c=c.lower())
    plt.grid(True, ls="--")
    plt.xlim(200, 900)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.15)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{camera.name}\n{phone_time}")
    plt.savefig(saveto)
    plt.show()
    plt.close()

    data_phone.append(row)
    data_reference.append(table_reference[closest])

data_phone = table.vstack(data_phone)
data_reference = table.vstack(data_reference)

sorad_wavelengths_RGB = [wavelengths[np.abs(wavelengths-wvl).argmin()] for wvl in RGB_wavelengths]

parameters = ["Lu", "Lsky", "Ed", "Rrs"]
labels_phone = ["$L_u$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [ADU nm$^{-1}$ sr$^{-1}$]", "$E_d$ [ADU nm$^{-1}$]", "$R_{rs}$ [sr$^{-1}$]"]
labels_reference = ["$L_u$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]", "$E_d$ [W m$^{-2}$ nm$^{-1}$]", "$R_{rs}$ [sr$^{-1}$]"]

for param, label_phone, label_reference in zip(parameters, labels_phone, labels_reference):
    aspect = (param == "Rrs")

    MAD_all, MAD_RGB = hc.statistic_RGB(hc.MAD, data_phone, data_reference, param)
    MAPD_all, MAPD_RGB = hc.statistic_RGB(hc.MAPD, data_phone, data_reference, param)
    r_all, r_RGB = hc.statistic_RGB(hc.correlation, data_phone, data_reference, param)

    title_r = f"$r$ = {r_all:.2f}"
    title_MAD = f"    MAD = {MAD_all:.3f} sr$" + "^{-1}$" + f" ({MAPD_all:.0f}%)" if param == "Rrs" else ""
    title = f"{title_r} {title_MAD}"

    hc.correlation_plot_RGB(data_reference, data_phone, param+" {c}", param+" {c}", xerrlabel=None, yerrlabel=param+"_err {c}", xlabel=f"{reference} {label_reference}", ylabel=f"{camera.name} ({data_type.upper()}) {label_phone}", title=title, equal_aspect=aspect, saveto=f"results/comparison_{reference}_X_{camera.name}_{data_type}_{param}.pdf")

# Correlation plot: Rrs G/B (SoRad) vs Rrs G/B (smartphone)
GB_sorad = data_reference["Rrs G"]/data_reference["Rrs B"]
GB_phone = data_phone["Rrs G"]/data_phone["Rrs B"]

rms = RMS(GB_phone - GB_sorad)
r = hc.correlation(GB_phone, GB_sorad)

plt.figure(figsize=(5,5), tight_layout=True)
plt.errorbar(GB_sorad, GB_phone, color="k", fmt="o")
max_val = max(0, GB_phone.max(), GB_sorad.max())
plt.plot([-1, 5], [-1, 5], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel(reference + " $R_{rs}$ G/B")
plt.ylabel(camera.name + " $R_{rs}$ G/B")
plt.title(f"$r$ = {r:.2f}     RMS = {rms:.2f}")
plt.savefig(f"results/comparison_{reference}_X_{camera.name}_GB.pdf")
plt.show()
