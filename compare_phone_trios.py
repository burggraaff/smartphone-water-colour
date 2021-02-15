"""
Compare RGB RAW data to TriOS hyperspectral data.

To do:
    * Go over entirely
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral
from astropy import table
from datetime import datetime

# Get the data folder from the command line
path_calibration, path_phone, path_trios = io.path_from_input(argv)

# Find the effective wavelength corresponding to the RGB bands
spectral_response = calibrate.load_spectral_response(path_calibration)
wavelengths = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

# SPECTACLE function for effective bandwidth is currently somewhat broken so we
# do it ourselves
RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths, axis=1)

table_phone = table.Table.read(path_phone)

table_trios = table.Table.read(path_trios, format="ascii.csv")
trios_datetime = [datetime.fromisoformat(DT) for DT in table_trios["DateTime"]]
trios_timestamps = [dt.timestamp() for dt in trios_datetime]
table_trios.add_column(table.Column(data=trios_timestamps, name="UTC"))

for row in table_phone:
    time_differences = np.abs(table_trios["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 1000:
        continue
    phone_time = datetime.fromtimestamp(row['UTC']).isoformat()
    trios_time = datetime.fromtimestamp(table_trios[closest]["UTC"]).isoformat()
    print(f"Phone time: {phone_time} ; TriOS time: {trios_time} ; Difference: {time_diff:.1f} seconds")

    wavelengths = np.arange(323.3, 953.7, 3.3)
    Rrs = np.array([table_trios[f"Rrs_{wvl:.1f}"][closest] for wvl in wavelengths])

    plt.figure(figsize=(3,3), tight_layout=True)
    plt.plot(wavelengths, Rrs, c="k")
    for j, c in enumerate("RGB"):
        plt.errorbar(RGB_wavelengths[j], row[f"R_rs ({c})"], xerr=effective_bandwidths[j]/2, yerr=row[f"R_rs_err ({c})"], fmt="o", c=c)
    plt.grid(True, ls="--")
    plt.xlim(390, 700)
    plt.ylim(0, 0.07)
    plt.title(f"iPhone 6S; {phone_time}")
    plt.show()
    plt.close()

#max_val = 0
#
#plt.figure(figsize=(5,5), tight_layout=True)
#for c in "RGB":
#    plt.errorbar(table_raw[f"R_rs ({c})"], table_jpeg[f"R_rs ({c})"], xerr=table_raw[f"R_rs_err ({c})"], yerr=table_jpeg[f"R_rs_err ({c})"], color=c, fmt="o")
#    max_val = max(max_val, table_raw[f"R_rs ({c})"].max(), table_jpeg[f"R_rs ({c})"].max())
#plt.plot([-1, 1], [-1, 1], c='k', ls="--")
#plt.xlim(0, 1.05*max_val)
#plt.ylim(0, 1.05*max_val)
#plt.grid(True, ls="--")
#plt.xlabel("RAW $R_{rs}$ [sr$^{-1}$]")
#plt.ylabel("JPEG $R_{rs}$ [sr$^{-1}$]")
#plt.show()
