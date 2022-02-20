"""
Compare RGB data from the same smartphone, for example RAW vs JPEG.

Command-line inputs:
    * path_data1: path to first table with data summary (e.g. RAW)
    * path_data2: path to second table with data summary (e.g. JPEG)
"""

from sys import argv
from spectacle import io
from wk import hydrocolor as hc, plot

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)
phone_name = " ".join(path_data1.stem.split("_")[1:-2])

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
plot.correlation_plot_radiance(data1, data2, xlabel=data_label1, ylabel=data_label2, regression="rgb", saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, xlabel=f"{data_label1} $L$ [a.u.]", ylabel=f"{data_label2} $L$ [a.u.]", regression="rgb", saveto=f"{saveto_base}_radiance_simple.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_RGB_equal(data1, data2, "R_rs", errlabel="R_rs_err", xlabel=f"{data_label1} {label_R_rs} {plot.persr}", ylabel=f"{data_label2} {label_R_rs} {plot.persr}", regression="all", difference_unit=plot.persr, saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=data_label1, ylabel=data_label2, saveto=f"{saveto_base}_band_ratio.pdf")
plot.correlation_plot_bandratios_combined(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=data_label1, ylabel=data_label2, saveto=f"{saveto_base}_band_ratio_combined.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{data_label1} {plot.keys_latex['R_rs']}", ylabel=f"{data_label2} {plot.keys_latex['R_rs']}", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
