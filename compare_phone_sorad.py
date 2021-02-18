"""
Compare smartphone RGB RAW data to So-Rad hyperspectral data.

Command-line inputs:
    * SPECTACLE calibration folder
    * Smartphone data file
    * SoRad data file

Example:
    %run compare_phone_sorad.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/combined_iPhone_SE_raw.csv water-colour-data/Balaton_20190703/SoRad/So-Rad_Balaton2019.csv

To do:
    * Split So-Rad processing into its own script (process_sorad.py)
    * Proper spectral convolution
    * Compare Ed, Lw, Rrs separately
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral, load_camera
from spectacle.general import RMS
from astropy import table
from datetime import datetime

# Get the data folder from the command line
path_calibration, path_phone, path_sorad = io.path_from_input(argv)
phone_name = " ".join(path_phone.stem.split("_")[1:-1])

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
table_sorad = table.Table.read(path_sorad)

wavelengths = np.arange(320, 955, 3.3)

sorad_datetime = [datetime.fromisoformat(DT) for DT in table_sorad["trigger_id"]]
sorad_timestamps = [dt.timestamp() for dt in sorad_datetime]
table_sorad.add_column(table.Column(data=sorad_timestamps, name="UTC"))

def average_row(row):
    Rrs = np.array([row[f"Rrs_{wvl:.1f}"] for wvl in wavelengths])
    RGB_averaged = []

    for response, c in zip(RGB_responses, "RGB"):
        response_interpolated = np.interp(wavelengths, wavelengths_phone, response, left=0, right=0)
        not_nan = np.where(~np.isnan(Rrs))
        Rrs_avg = np.average(Rrs[not_nan], weights=response_interpolated[not_nan])

        RGB_averaged.append(Rrs_avg)

    return RGB_averaged

RGB_averaged_all = np.array([average_row(row) for row in table_sorad])
RGB_averaged_table = table.Table(data=RGB_averaged_all, names=["Rrs_avg (R)", "Rrs_avg (G)", "Rrs_avg (B)"])
table_sorad = table.hstack([table_sorad, RGB_averaged_table])

data_phone = []
data_sorad = []

for row in table_phone:
    time_differences = np.abs(table_sorad["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 1000:
        continue
    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    sorad_time = datetime.fromtimestamp(table_sorad[closest]["UTC"]).isoformat()
    print("----")
    print(f"Phone time: {phone_time} ; SoRad time: {sorad_time} ; Difference: {time_diff:.1f} seconds")
    print(f"Valid: {table_sorad[closest]['valid']} ; offset: {table_sorad[closest]['offset']}")

    Rrs = np.array([table_sorad[f"Rrs_{wvl:.1f}"][closest] for wvl in wavelengths])

    # Convert ":" to - in the filename when saving
    saveto = f"results/sorad_comparison/{phone_name}_{phone_time}.pdf".replace(":", "-")

    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    plt.plot(wavelengths, Rrs, c="k")
    for j, c in enumerate("RGB"):
        plt.errorbar(RGB_wavelengths[j], row[f"R_rs ({c})"], xerr=effective_bandwidths[j]/2, yerr=row[f"R_rs_err ({c})"], fmt="o", c=c.lower())
        plt.errorbar(RGB_wavelengths[j], table_sorad[closest][f"Rrs_avg ({c})"], xerr=effective_bandwidths[j]/2, yerr=0, fmt="^", c=c.lower())
    plt.grid(True, ls="--")
    plt.xlim(200, 900)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.15)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{phone_name}\n{phone_time}")
    plt.savefig(saveto)
    plt.show()
    plt.close()

    data_phone.append(row)
    data_sorad.append(table_sorad[closest])

data_phone = table.vstack(data_phone)
data_sorad = table.vstack(data_sorad)

sorad_wavelengths_RGB = [wavelengths[np.abs(wavelengths-wvl).argmin()] for wvl in RGB_wavelengths]

Rrs_phone = np.hstack([data_phone["R_rs (R)"].data, data_phone["R_rs (G)"].data, data_phone["R_rs (B)"].data])
Rrs_sorad_averaged = np.hstack([data_sorad["Rrs_avg (R)"].data, data_sorad["Rrs_avg (G)"].data, data_sorad["Rrs_avg (B)"].data])
rms = RMS(Rrs_phone - Rrs_sorad_averaged)
r = np.corrcoef(Rrs_phone, Rrs_sorad_averaged)[0, 1]

max_val = 0
plt.figure(figsize=(5,5), tight_layout=True)
for j,c in enumerate("RGB"):
    plt.errorbar(data_sorad[f"Rrs_avg ({c})"], data_phone[f"R_rs ({c})"], xerr=0, yerr=data_phone[f"R_rs_err ({c})"], color=c.lower(), fmt="o")
    max_val = max(max_val, data_phone[f"R_rs ({c})"].max(), data_sorad[f"Rrs_avg ({c})"].max())
plt.plot([-1, 1], [-1, 1], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel("SoRad $R_{rs}$ [sr$^{-1}$]")
plt.ylabel(phone_name + " $R_{rs}$ [sr$^{-1}$]")
plt.title(f"$r$ = {r:.2f}     RMS = {rms:.2f} sr$" + "^{-1}$")
plt.savefig(f"results/comparison_So-Rad_X_{phone_name}.pdf")
plt.show()
