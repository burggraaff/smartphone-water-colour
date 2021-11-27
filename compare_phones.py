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
phone1_name = " ".join(path_data1.stem.split("_")[1:-2])
phone2_name = " ".join(path_data2.stem.split("_")[1:-2])
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

# Correlation plot: Radiances
plot.correlation_plot_radiance(data1, data2, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_radiance_simple.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_RGB_equal(data1, data2, "R_rs", errlabel="R_rs_err", xlabel=f"{phone1_name} {label_R_rs} {plot.persr}", ylabel=f"{phone2_name}\n{label_R_rs} {plot.persr}", regression="all", difference_unit=plot.persr, saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{phone1_name} {label_R_rs}", ylabel=f"{phone2_name} {label_R_rs}", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
