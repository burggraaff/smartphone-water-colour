"""
Compare RGB data from the same smartphone, for example RAW vs JPEG.

Command-line inputs:
    * path_data1: path to first table with data summary (e.g. RAW)
    * path_data2: path to second table with data summary (e.g. JPEG)
"""

import numpy as np
from sys import argv
from spectacle import io
from wk import hydrocolor as hc, wacodi as wa, plot

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)
phone_name = " ".join(path_data1.stem.split("_")[1:-1])

# Find out if we're doing JPEG or RAW and make appropriate labels
data_type1 = hc.data_type_RGB(path_data1)
data_type2 = hc.data_type_RGB(path_data2)
data_label1 = f"{phone_name} {data_type1}"
data_label2 = f"{phone_name} {data_type2}"
saveto_base = f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}"
print(f"Comparing {data_type1} and {data_type2} data from the {phone_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
data1 = hc.read_results(path_data1)
data2 = hc.read_results(path_data2)

# No need to find matches - the RAW and JPEG images were taken simultaneously

# Correlation plot: Radiances and irradiance
parameters = ["Lu", "Lsky", "Ld", "Ed"]
labels = ["$L_u$", "$L_{sky}$", "$L_d$", "$E_d$"]
units = ["[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$]"]

for param, label, unit in zip(parameters, labels, units):
    plot.correlation_plot_RGB(data1, data2, param+" ({c})", param+" ({c})", xerrlabel=param+"_err ({c})", yerrlabel=param+"_err ({c})", xlabel=f"{phone_name} {data_type1} {label} {unit}", ylabel=f"{phone_name} {data_type2} {label} {unit}", regression="rgb", saveto=f"{saveto_base}_{param}.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = "$R_{rs}$"
unit_R_rs = "[sr$^{-1}$]"
plot.correlation_plot_RGB_equal(data1, data2, "R_rs ({c})", "R_rs ({c})", xerrlabel="R_rs_err ({c})", yerrlabel="R_rs_err ({c})", xlabel=f"{phone_name} {data_type1} {label_R_rs} {unit_R_rs}", ylabel=f"{phone_name} {data_type2} {label_R_rs} {unit_R_rs}", regression="all", saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1["R_rs (G/R)"], data2["R_rs (G/R)"], data1["R_rs (G/B)"], data2["R_rs (G/B)"], x_err_GR=data1["R_rs_err (G/R)"], y_err_GR=data2["R_rs_err (G/R)"], x_err_GB=data1["R_rs_err (G/B)"], y_err_GB=data2["R_rs_err (G/B)"], quantity="$R_{rs}$", xlabel=f"{phone_name} {data_type1}", ylabel=f"{phone_name} {data_type2}", saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot: Radiance (all combined)
radiance1 = hc.get_radiances(data1)
radiance2 = hc.get_radiances(data2)

label = "$L$"
unit = "[ADU nm$^{-1}$ sr$^{-1}$]"
plot.correlation_plot_RGB(radiance1, radiance2, "L ({c})", "L ({c})", xerrlabel="L_err ({c})", yerrlabel="L_err ({c})", xlabel=f"{phone_name} {data_type1} {label} {unit}", ylabel=f"{phone_name} {data_type2} {label} {unit}", regression="rgb", saveto=f"{saveto_base}_L.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{phone_name} {data_type1}"+" $R_{rs}$", ylabel=f"{phone_name} {data_type2}"+" $R_{rs}$", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
