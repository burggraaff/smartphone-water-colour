"""
Compare RGB data from the same smartphone, for example RAW vs JPEG.

Command-line inputs:
    * path_data1: path to first table with data summary (e.g. RAW)
    * path_data2: path to second table with data summary (e.g. JPEG)
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io
from astropy import table
from wk import hydrocolor as hc, wacodi as wa

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)
phone_name = " ".join(path_data1.stem.split("_")[1:-1])

# Find out if we're doing JPEG or RAW
data_type1 = hc.data_type_RGB(path_data1)
data_type2 = hc.data_type_RGB(path_data2)

data1 = hc.read_results(path_data1)
data2 = hc.read_results(path_data2)

parameters = ["Lu", "Lsky", "Ld", "Ed"]
labels = ["$L_u$", "$L_{sky}$", "$L_d$", "$E_d$"]
units = ["[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$ sr$^{-1}$]", "[ADU nm$^{-1}$]"]

for param, label, unit in zip(parameters, labels, units):
    hc.correlation_plot_RGB(data1, data2, param+" ({c})", param+" ({c})", xerrlabel=param+"_err ({c})", yerrlabel=param+"_err ({c})", xlabel=f"{phone_name} {data_type1} {label} {unit}", ylabel=f"{phone_name} {data_type2} {label} {unit}", regression="rgb", saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_{param}.pdf")

    hc.comparison_histogram(data1, data2, param+" ({c})", xlabel=f"{phone_name} {data_type1}", ylabel=f"{phone_name} {data_type2}", quantity=label, saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_{param}_hist.pdf")

label_R_rs = "$R_{rs}$"
unit_R_rs = "[sr$^{-1}$]"
hc.correlation_plot_RGB_equal(data1, data2, "R_rs ({c})", "R_rs ({c})", xerrlabel="R_rs_err ({c})", yerrlabel="R_rs_err ({c})", xlabel=f"{phone_name} {data_type1} {label_R_rs} {unit_R_rs}", ylabel=f"{phone_name} {data_type2} {label_R_rs} {unit_R_rs}", regression="all", saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_R_rs.pdf")

hc.comparison_histogram(data1, data2, "R_rs ({c})", xlabel=f"{phone_name} {data_type1}", ylabel=f"{phone_name} {data_type2}", quantity=label, saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_R_rs_hist.pdf")

# Correlation plot: Band ratios
hc.correlation_plot_bands(data1["R_rs (G/R)"], data2["R_rs (G/R)"], data1["R_rs (G/B)"], data2["R_rs (G/B)"], x_err_GR=data1["R_rs_err (G/R)"], y_err_GR=data2["R_rs_err (G/R)"], x_err_GB=data1["R_rs_err (G/B)"], y_err_GB=data2["R_rs_err (G/B)"], quantity="$R_{rs}$", xlabel=f"{phone_name} {data_type1}", ylabel=f"{phone_name} {data_type2}", saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_band_ratio.pdf")

wa.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{phone_name} {data_type1}"+" $R_{rs}$", ylabel=f"{phone_name} {data_type2}"+" $R_{rs}$", saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_hueangle_ForelUle.pdf")

# Correlation plot for all radiances combined
def get_radiances(data):
    radiance_RGB = [np.ravel([data[f"{param} ({c})"].data for param in parameters[:3]]) for c in hc.colours[:3]]
    radiance_RGB_err = [np.ravel([data[f"{param}_err ({c})"].data for param in parameters[:3]]) for c in hc.colours[:3]]
    cols = [f"L ({c})" for c in hc.colours[:3]] + [f"L_err ({c})" for c in hc.colours[:3]]
    radiance = np.concatenate([radiance_RGB, radiance_RGB_err]).T

    radiance = table.Table(data=radiance, names=cols)
    return radiance

radiance1 = get_radiances(data1)
radiance2 = get_radiances(data2)

label = "$L$"
unit = "[ADU nm$^{-1}$ sr$^{-1}$]"
hc.correlation_plot_RGB(radiance1, radiance2, "L ({c})", "L ({c})", xerrlabel="L_err ({c})", yerrlabel="L_err ({c})", xlabel=f"{phone_name} {data_type1} {label} {unit}", ylabel=f"{phone_name} {data_type2} {label} {unit}", regression="rgb", saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_L.pdf")

hc.comparison_histogram(radiance1, radiance2, "L ({c})", xlabel=f"{phone_name} {data_type1}", ylabel=f"{phone_name} {data_type2}", quantity=label, saveto=f"results/comparison_{phone_name}_{data_type1}_X_{data_type2}_L_hist.pdf")
