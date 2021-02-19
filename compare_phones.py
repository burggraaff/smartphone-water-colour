"""
Compare RGB RAW data between smartphones.

Command-line inputs:
    * path_phone1: Path to data file for first phone
    * path_phone2: Path to data file for second phone
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, calibrate, spectral
from spectacle.general import RMS
from astropy import table
from datetime import datetime

# Get the data folder from the command line
path_phone1, path_phone2 = io.path_from_input(argv)

phone1_name = " ".join(path_phone1.stem.split("_")[1:-1])
phone2_name = " ".join(path_phone2.stem.split("_")[1:-1])

table_phone1 = table.Table.read(path_phone1)
table_phone2 = table.Table.read(path_phone2)

data_phone1 = []
data_phone2 = []

for row in table_phone1:
    time_differences = np.abs(table_phone2["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 100:
        continue
    phone1_time = datetime.fromtimestamp(row['UTC']).isoformat()
    phone2_time = datetime.fromtimestamp(table_phone2[closest]["UTC"]).isoformat()
    print(f"{phone1_name} time: {phone1_time} ; {phone2_name} time: {phone2_time} ; Difference: {time_diff:.1f} seconds")

    data_phone1.append(row)
    data_phone2.append(table_phone2[closest])

data_phone1 = table.vstack(data_phone1)
data_phone2 = table.vstack(data_phone2)

max_val = 0

plt.figure(figsize=(5,5), tight_layout=True)
for c in "RGB":
    plt.errorbar(data_phone1[f"R_rs ({c})"], data_phone2[f"R_rs ({c})"], xerr=data_phone1[f"R_rs_err ({c})"], yerr=data_phone2[f"R_rs_err ({c})"], color=c.lower(), fmt="o")
    max_val = max(max_val, data_phone1[f"R_rs ({c})"].max(), data_phone2[f"R_rs ({c})"].max())
plt.plot([-1, 1], [-1, 1], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel(phone1_name + " $R_{rs}$ [sr$^{-1}$]")
plt.ylabel(phone2_name + " $R_{rs}$ [sr$^{-1}$]")
plt.savefig(f"comparison_{phone1_name}_X_{phone2_name}.pdf")
plt.show()

differences_RGB = table.hstack([data_phone1[f"R_rs ({c})"] - data_phone2[f"R_rs ({c})"] for c in "RGB"])
RMS_RGB = [RMS(differences_RGB[f"R_rs ({c})"]) for c in "RGB"]
differences_all = np.ravel([differences_RGB[f"R_rs ({c})"].data for c in "RGB"])
RMS_all = RMS(differences_all)
