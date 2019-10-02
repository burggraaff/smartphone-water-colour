"""
Process three images (water, sky, grey card), calibrated using SPECTACLE, to
calculate the remote sensing reflectance in the RGB channels, following the
HydroColor protocol.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Bias
    - Flat-field
    - Spectral response
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral
from astropy import table
from datetime import datetime

# Get the data folder from the command line
path_calibration, path_phone, path_sorad = io.path_from_input(argv)
phone_name = " ".join(path_phone.stem.split("_")[1:-1])

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

wavelengths = np.arange(320, 955, 3.3)

def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

with open(path_sorad) as file:
    data = file.readlines()
    header = data[0]
    data = data[1:]
    cols = header.split(";")[:-1] + [f"Rrs_{wvl:.1f}" for wvl in wavelengths]

    rows = [convert_row(row) for row in data]
    dtypes = ["S30" for h in header.split(";")[:-1]] + [np.float32 for wvl in wavelengths]

    table_sorad = table.Table(rows=rows, names=cols, dtype=dtypes)

sorad_datetime = [datetime.fromisoformat(DT) for DT in table_sorad["trigger_id"]]
sorad_timestamps = [dt.timestamp() for dt in sorad_datetime]
table_sorad.add_column(table.Column(data=sorad_timestamps, name="UTC"))

table_sorad = table_sorad[26000:27500]

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
    print(f"Valid: {table_sorad[closest]['valid']} ; rho: {table_sorad[closest]['rho']}")

    Rrs = np.array([table_sorad[f"Rrs_{wvl:.1f}"][closest] for wvl in wavelengths])

    plt.figure(figsize=(3.3,3.3), tight_layout=True)
    plt.plot(wavelengths, Rrs, c="k")
    for j, c in enumerate("RGB"):
        plt.errorbar(RGB_wavelengths[j], row[f"R_rs ({c})"], xerr=effective_bandwidths[j]/2, yerr=row[f"R_rs_err ({c})"], fmt="o", c=c)
    plt.grid(True, ls="--")
    plt.xlim(390, 700)
    plt.xlabel("Wavelength [nm]")
    plt.ylim(0, 0.07)
    plt.ylabel("$R_{rs}$ [sr$^{-1}$]")
    plt.title(f"{phone_name}\n{phone_time}")
    plt.savefig(f"SoRad_comparison/{phone_name}_{phone_time}.pdf")
    plt.show()
    plt.close()

    data_phone.append(row)
    data_sorad.append(table_sorad[closest])

data_phone = table.vstack(data_phone)
data_sorad = table.vstack(data_sorad)

sorad_wavelengths_RGB = [wavelengths[np.abs(wavelengths-wvl).argmin()] for wvl in RGB_wavelengths]

max_val = 0

plt.figure(figsize=(5,5), tight_layout=True)
for j,c in enumerate("RGB"):
    plt.errorbar(data_sorad[f"Rrs_{sorad_wavelengths_RGB[j]:.1f}"], data_phone[f"R_rs ({c})"], xerr=0, yerr=data_phone[f"R_rs_err ({c})"], color=c, fmt="o")
    max_val = max(max_val, data_phone[f"R_rs ({c})"].max(), data_sorad[f"Rrs_{sorad_wavelengths_RGB[j]:.1f}"].max())
plt.plot([-1, 1], [-1, 1], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel("SoRad $R_{rs}$ [sr$^{-1}$]")
plt.ylabel(phone_name + " $R_{rs}$ [sr$^{-1}$]")
plt.savefig(f"comparison_SoRad_X_{phone_name}.pdf")
plt.show()


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
