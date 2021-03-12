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

parameters = ["Lu", "Lsky", "Ld", "Ed", "Rrs"]
labels = ["$L_u$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_d$ [ADU nm$^{-1}$ sr$^{-1}$]", "$E_d$ [ADU nm$^{-1}$]", "$R_{rs}$ [sr$^{-1}$]"]

for param, label in zip(parameters, labels):
    aspect = (param == "Rrs")

    MAD_all, MAD_RGB = hc.statistic_RGB(hc.MAD, data_phone1, data_phone2, param)
    r_all, r_RGB = hc.correlation_RGB(data_phone1, data_phone2, param)

    title_r = f"$r$ = {r_all:.2f}"
    title_MAD = f"    MAD = {MAD_all:.3f} sr$" + "^{-1}$" if param == "Rrs" else ""
    title = f"{title_r} {title_MAD}"

    hc.correlation_plot_RGB(data_phone1, data_phone2, param+" {c}", param+" {c}", xerrlabel=param+"_err {c}", yerrlabel=param+"_err {c}", xlabel=f"{phone1_name} {label}", ylabel=f"{phone2_name} {label}", title=title, equal_aspect=aspect, saveto=f"results/comparison_{phone1_name}_X_{phone2_name}_{param}.pdf")


# Correlation plot for all radiances combined
def get_radiances(data):
    radiance_RGB = [np.ravel([data[f"{param} {c}"].data for param in parameters[:3]]) for c in hc.colours]
    radiance_RGB_err = [np.ravel([data[f"{param}_err {c}"].data for param in parameters[:3]]) for c in hc.colours]
    cols = [f"L {c}" for c in hc.colours] + [f"L_err {c}" for c in hc.colours]
    radiance = np.concatenate([radiance_RGB, radiance_RGB_err]).T

    radiance = table.Table(data=radiance, names=cols)
    return radiance

radiance_phone1 = get_radiances(data_phone1)
radiance_phone2 = get_radiances(data_phone2)

r_all, r_RGB = hc.correlation_RGB(radiance_phone1, radiance_phone2, "L")
title_r = f"$r$ = {r_all:.2f}"

label = "$L$ [ADU nm$^{-1}$ sr$^{-1}$]"
hc.correlation_plot_RGB(radiance_phone1, radiance_phone2, "L {c}", "L {c}", xerrlabel="L_err {c}", yerrlabel="L_err {c}", xlabel=f"{phone1_name} {label}", ylabel=f"{phone2_name} {label}", title=title_r, equal_aspect=False, saveto=f"results/comparison_{phone1_name}_X_{phone2_name}_L.pdf")
