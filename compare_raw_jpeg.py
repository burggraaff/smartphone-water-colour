"""
Compare data, including radiance, reflectance, and colour, from different data formats on the same smartphone.
Examples are RAW vs. JPEG, JPEG vs linearised JPEG, etc.

Command-line inputs:
    * path_data1: path to first table with data summary (e.g. RAW)
    * path_data2: path to second table with data summary (e.g. JPEG)

Example:
    %run compare_raw_jpeg.py water-colour-data/balaton_iPhone_SE_raw_18pct.csv water-colour-data/balaton_iPhone_SE_jpeg_18pct.csv
    %run compare_raw_jpeg.py water-colour-data/balaton_iPhone_SE_jpeg_18pct.csv water-colour-data/balaton_iPhone_SE_jpeg_linear_18pct.csv
    %run compare_raw_jpeg.py water-colour-data/balaton_iPhone_SE_raw_18pct.csv water-colour-data/balaton_iPhone_SE_jpeg_fromraw_18pct-camerawb.csv
    %run compare_raw_jpeg.py water-colour-data/balaton_iPhone_SE_jpeg_18pct.csv water-colour-data/balaton_iPhone_SE_jpeg_fromraw_18pct-camerawb.csv
    %run compare_raw_jpeg.py water-colour-data/balaton_Samsung_Galaxy_S8_raw_18pct.csv water-colour-data/balaton_Samsung_Galaxy_S8_jpeg_fromraw_18pct-camerawb.csv
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
plot.correlation_plot_radiance(data1, data2, xlabel=data_label1, ylabel=data_label2, title="RAW-JPEG radiance comparison", regression="rgb", saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, xlabel=f"{data_label1} $L$ [a.u.]", ylabel=f"{data_label2} $L$ [a.u.]", title="RAW-JPEG radiance comparison", regression="rgb", saveto=f"{saveto_base}_radiance_simple.pdf", saveto_stats=f"{saveto_base}_radiance_simple.dat")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_R_rs(data1, data2, xlabel=f"{data_label1} {label_R_rs} {plot.persr}", ylabel=f"{data_label2} {label_R_rs} {plot.persr}", title="RAW-JPEG reflectance comparison", saveto=f"{saveto_base}_R_rs.pdf", saveto_stats=f"{saveto_base}_R_rs.dat", compare_to_regression=True)

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=data_label1, ylabel=data_label2, title="RAW-JPEG\nband ratio comparison", saveto=f"{saveto_base}_band_ratio.pdf")
plot.correlation_plot_bandratios_combined(data1, data2, xlabel=f"{data_label1} {label_R_rs} band ratio", ylabel=f"{data_label2} {label_R_rs} band ratio", title="RAW-JPEG band ratio comparison", saveto=f"{saveto_base}_band_ratio_combined.pdf", saveto_stats=f"{saveto_base}_band_ratio_combined.dat")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{data_label1} {plot.keys_latex['R_rs']}", ylabel=f"{data_label2} {plot.keys_latex['R_rs']}", title="RAW-JPEG water color comparison", saveto=f"{saveto_base}_hueangle_ForelUle.pdf", saveto_stats=f"{saveto_base}_hueangle_ForelUle.dat")
