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
from wk import hydrocolor as hc, wacodi as wa

# Get the data folder from the command line
path_phone1, path_phone2 = io.path_from_input(argv)

# Names of the two phones being compared and some useful labels
phone1_name = " ".join(path_phone1.stem.split("_")[1:-1])
phone2_name = " ".join(path_phone2.stem.split("_")[1:-1])
saveto_base = f"results/comparison_{phone1_name}_X_{phone2_name}"
print(f"Comparing data from {phone1_name} and {phone2_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
table_phone1 = hc.read_results(path_phone1)
table_phone2 = hc.read_results(path_phone2)

# Find matches
data_phone1, data_phone2 = [], []  # Lists to contain matching table entries
for row in table_phone1:  # Loop over the first table to look for matches
    # Find matches within a threshold
    time_differences = np.abs(table_phone2["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 100:  # Only consider it a match-up if it is within this many seconds
        continue
    phone1_time = datetime.fromtimestamp(row['UTC']).isoformat()
    phone2_time = datetime.fromtimestamp(table_phone2[closest]["UTC"]).isoformat()
    print(f"{phone1_name} time: {phone1_time} ; {phone2_name} time: {phone2_time} ; Difference: {time_diff:.1f} seconds")

    # Put the matching rows into the aforementioned lists
    data_phone1.append(row)
    data_phone2.append(table_phone2[closest])

# Make new tables from the match-up rows
data_phone1 = table.vstack(data_phone1)
data_phone2 = table.vstack(data_phone2)

# Correlation plot: Radiances and irradiance
parameters = ["Lu", "Lsky", "Ld", "Ed"]
labels = ["$L_u$", "$L_{sky}$", "$L_d$", "$E_d$"]
units = ["[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$]"]

for param, label, unit in zip(parameters, labels, units):
    hc.correlation_plot_RGB(data_phone1, data_phone2, param+" ({c})", param+" ({c})", xerrlabel=param+"_err ({c})", yerrlabel=param+"_err ({c})", xlabel=f"{phone1_name}\n{label} {unit}", ylabel=f"{phone2_name}\n{label} {unit}", regression="all", saveto=f"{saveto_base}_{param}.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = "$R_{rs}$"
unit_R_rs = "[sr$^{-1}$]"
hc.correlation_plot_RGB_equal(data_phone1, data_phone2, "R_rs ({c})", "R_rs ({c})", xerrlabel="R_rs_err ({c})", yerrlabel="R_rs_err ({c})", xlabel=f"{phone1_name} {label_R_rs} {unit_R_rs}", ylabel=f"{phone2_name}\n{label_R_rs} {unit_R_rs}", regression="all", saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
hc.correlation_plot_bands(data_phone1["R_rs (G/R)"], data_phone2["R_rs (G/R)"], data_phone1["R_rs (G/B)"], data_phone2["R_rs (G/B)"], x_err_GR=data_phone1["R_rs_err (G/R)"], y_err_GR=data_phone2["R_rs_err (G/R)"], x_err_GB=data_phone1["R_rs_err (G/B)"], y_err_GB=data_phone2["R_rs_err (G/B)"], quantity="$R_{rs}$", xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot for all radiances combined
def get_radiances(data):
    radiance_RGB = [np.ravel([data[f"{param} ({c})"].data for param in parameters[:3]]) for c in hc.colours]
    radiance_RGB_err = [np.ravel([data[f"{param}_err ({c})"].data for param in parameters[:3]]) for c in hc.colours]
    cols = [f"L ({c})" for c in hc.colours] + [f"L_err ({c})" for c in hc.colours]
    radiance = np.concatenate([radiance_RGB, radiance_RGB_err]).T

    radiance = table.Table(data=radiance, names=cols)
    return radiance

radiance_phone1 = get_radiances(data_phone1)
radiance_phone2 = get_radiances(data_phone2)

# Correlation plot: Radiance (all combined)
label = "$L$"
unit = "[ADU nm$^{-1}$ sr$^{-1}$]"
hc.correlation_plot_RGB(radiance_phone1, radiance_phone2, "L ({c})", "L ({c})", xerrlabel="L_err ({c})", yerrlabel="L_err ({c})", xlabel=f"{phone1_name} {label} {unit}", ylabel=f"{phone2_name} {label} {unit}", regression="all", saveto=f"{saveto_base}_L.pdf")

# Correlation plot: hue angle and Forel-Ule index
wa.correlation_plot_hue_angle_and_ForelUle(data_phone1["R_rs (hue)"], data_phone2["R_rs (hue)"], xlabel=phone1_name+" $R_{rs}$", ylabel=phone2_name+" $R_{rs}$", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
