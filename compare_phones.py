"""
Compare RGB RAW data between smartphones.

Command-line inputs:
    * path_phone1: Path to data file for first phone
    * path_phone2: Path to data file for second phone
"""

import numpy as np
from sys import argv
from spectacle import io
from astropy import table
from wk import hydrocolor as hc, plot

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)

# Names of the two phones being compared and some useful labels
phone1_name = " ".join(path_data1.stem.split("_")[1:-1])
phone2_name = " ".join(path_data2.stem.split("_")[1:-1])
saveto_base = f"results/comparison_{phone1_name}_X_{phone2_name}"
print(f"Comparing data from {phone1_name} and {phone2_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
table_phone1 = hc.read_results(path_data1)
table_phone2 = hc.read_results(path_data2)

# Find matches
data1, data2 = [], []  # Lists to contain matching table entries
for row in table_phone1:  # Loop over the first table to look for matches
    # Find matches within a threshold
    time_differences = np.abs(table_phone2["UTC"] - row["UTC"])
    closest = time_differences.argmin()
    time_diff = time_differences[closest]
    if time_diff > 100:  # Only consider it a match-up if it is within this many seconds
        continue
    phone1_time = hc.iso_timestamp(row['UTC'])
    phone2_time = hc.iso_timestamp(table_phone2[closest]["UTC"])
    print(f"{phone1_name} time: {phone1_time} ; {phone2_name} time: {phone2_time} ; Difference: {time_diff:.1f} seconds")

    # Put the matching rows into the aforementioned lists
    data1.append(row)
    data2.append(table_phone2[closest])

# Make new tables from the match-up rows
data1 = table.vstack(data1)
data2 = table.vstack(data2)

# Correlation plot: Radiances and irradiance
parameters = ["Lu", "Lsky", "Ld", "Ed"]
labels = ["$L_u$", "$L_{sky}$", "$L_d$", "$E_d$"]
units = ["[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$]"]

for param, label, unit in zip(parameters, labels, units):
    plot.correlation_plot_RGB(data1, data2, param+" ({c})", param+" ({c})", xerrlabel=param+"_err ({c})", yerrlabel=param+"_err ({c})", xlabel=f"{phone1_name}\n{label} {unit}", ylabel=f"{phone2_name}\n{label} {unit}", regression="all", saveto=f"{saveto_base}_{param}.pdf")

plot.correlation_plot_radiance(data1, data2, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_radiance.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = "$R_{rs}$"
unit_R_rs = "[sr$^{-1}$]"
plot.correlation_plot_RGB_equal(data1, data2, "R_rs ({c})", "R_rs ({c})", xerrlabel="R_rs_err ({c})", yerrlabel="R_rs_err ({c})", xlabel=f"{phone1_name} {label_R_rs} {unit_R_rs}", ylabel=f"{phone2_name}\n{label_R_rs} {unit_R_rs}", regression="all", saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1["R_rs (G/R)"], data2["R_rs (G/R)"], data1["R_rs (G/B)"], data2["R_rs (G/B)"], x_err_GR=data1["R_rs_err (G/R)"], y_err_GR=data2["R_rs_err (G/R)"], x_err_GB=data1["R_rs_err (G/B)"], y_err_GB=data2["R_rs_err (G/B)"], quantity="$R_{rs}$", xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot: Radiance (all combined)
radiance_phone1 = hc.get_radiances(data1)
radiance_phone2 = hc.get_radiances(data2)

label = "$L$"
unit = "[ADU nm$^{-1}$ sr$^{-1}$]"
plot.correlation_plot_RGB(radiance_phone1, radiance_phone2, "L ({c})", "L ({c})", xerrlabel="L_err ({c})", yerrlabel="L_err ({c})", xlabel=f"{phone1_name} {label} {unit}", ylabel=f"{phone2_name} {label} {unit}", regression="all", saveto=f"{saveto_base}_L.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=phone1_name+" $R_{rs}$", ylabel=phone2_name+" $R_{rs}$", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
