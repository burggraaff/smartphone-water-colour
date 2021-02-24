"""
Compare RGB RAW data between smartphones.

Command-line inputs:
    * path_phone1: Path to data file for first phone
    * path_phone2: Path to data file for second phone
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io
from spectacle.general import RMS
from astropy import table
from datetime import datetime
from wk import hydrocolor as hc

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

parameters = ["Lu", "Ls", "Ld", "Rrs"]
labels = ["$L_u$ [ADU nm$^{-1}$]", "$L_{sky}$ [ADU nm$^{-1}$]", "$L_d$ [ADU nm$^{-1}$]", "$R_{rs}$ [sr$^{-1}$]"]

for param, label in zip(parameters, labels):
    hc.correlation_plot_RGB(data_phone1, data_phone2, param+" {c}", param+" {c}", xerrlabel=param+"_err {c}", yerrlabel=param+"_err {c}", xlabel=f"{phone1_name} {label}", ylabel=f"{phone2_name} {label}", saveto=f"results/comparison_{phone1_name}_X_{phone2_name}_{param}.pdf")

    differences_RGB = table.hstack([data_phone1[f"{param} {c}"] - data_phone2[f"{param} {c}"] for c in hc.colours])
    RMS_RGB = [RMS(differences_RGB[f"{param} {c}"]) for c in hc.colours]
    differences_all = np.ravel([differences_RGB[f"{param} {c}"].data for c in hc.colours])
    RMS_all = RMS(differences_all)
