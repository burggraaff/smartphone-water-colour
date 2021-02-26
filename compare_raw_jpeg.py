"""
Compare RGB RAW and JPEG data from the same smartphone.

Command-line inputs:
    * path_raw: path to table with RAW data summary
    * path_jpeg: path to table with JPEG data summary
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io
from astropy import table
from wk import hydrocolor as hc

# Get the data folder from the command line
path_raw, path_jpeg = io.path_from_input(argv)
phone_name = " ".join(path_raw.stem.split("_")[1:-1])

data_raw = table.Table.read(path_raw)
data_jpeg = table.Table.read(path_jpeg)

parameters = ["Lu", "Lsky", "Ld", "Ed", "Rrs"]
labels = ["$L_u$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_{sky}$ [ADU nm$^{-1}$ sr$^{-1}$]", "$L_d$ [ADU nm$^{-1}$ sr$^{-1}$]", "$E_d$ [ADU nm$^{-1}$]", "$R_{rs}$ [sr$^{-1}$]"]

for param, label in zip(parameters, labels):
    aspect = (param == "Rrs")

    RMS_all, RMS_RGB = hc.RMS_RGB(data_raw, data_jpeg, param)
    r_all, r_RGB = hc.correlation_RGB(data_raw, data_jpeg, param)

    title_r = f"$r$ = {r_all:.2f}"
    title_RMS = f"    RMSE = {RMS_all:.3f} sr$" + "^{-1}$" if param == "Rrs" else ""
    title = f"{title_r} {title_RMS}"

    hc.correlation_plot_RGB(data_raw, data_jpeg, param+" {c}", param+" {c}", xerrlabel=param+"_err {c}", yerrlabel=param+"_err {c}", xlabel=f"RAW {label}", ylabel=f"JPEG {label}", title=title, equal_aspect=aspect, saveto=f"results/comparison_{phone_name}_RAW_X_JPEG_{param}.pdf")
