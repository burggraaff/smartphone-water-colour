"""
Compare data, including radiance, reflectance, and colour, from different smartphones.

Command-line inputs:
    * path_phone1: Path to data file for first phone
    * path_phone2: Path to data file for second phone

Example:
    %run compare_phones.py water-colour-data/balaton_iPhone_SE_raw_18pct.csv water-colour-data/balaton_Samsung_Galaxy_S8_raw_18pct.csv
    %run compare_phones.py water-colour-data/balaton_iPhone_SE_jpeg_fromraw_18pct-camerawb.csv water-colour-data/balaton_Samsung_Galaxy_S8_jpeg_fromraw_18pct-camerawb.csv
"""
from sys import argv
from spectacle import io
from astropy import table
from wk import hydrocolor as hc, hyperspectral as hy, plot

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)

# Names of the two phones being compared and some useful labels
phone1_name = hc.get_phone_name(path_data1)
phone2_name = hc.get_phone_name(path_data2)
saveto_base = f"results/comparison_{phone1_name}_X_{phone2_name}"
print(f"Comparing data from {phone1_name} and {phone2_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
table_phone1 = hc.read_results(path_data1)
table_phone2 = hc.read_results(path_data2)
print("Finished reading data")

# Find matches
data1, data2 = [], []  # Lists to contain matching table entries
for row_phone1 in table_phone1:  # Loop over the first table to look for matches
    # Find matches within a threshold
    nr_matches, close_enough, closest, min_time_diff = hy.find_elements_within_range(table_phone2, row_phone1["UTC"], maximum_difference=100)
    if nr_matches < 1:  # If no close enough matches are found, skip this observation
        continue

    phone1_time = hc.iso_timestamp(row_phone1["UTC"])
    phone2_time = hc.iso_timestamp(table_phone2[closest]["UTC"])
    print(f"{phone1_name} time: {phone1_time} ; {phone2_name} time: {phone2_time} ; Difference: {min_time_diff:.0f} seconds")

    # Put the matching rows into the aforementioned lists
    data1.append(row_phone1)
    data2.append(table_phone2[closest])

# Make new tables from the match-up rows
data1 = table.vstack(data1)
data2 = table.vstack(data2)

# Correlation plot: Radiances
plot.correlation_plot_radiance(data1, data2, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, xlabel=f"{phone1_name} $L$ [a.u.]", ylabel=f"{phone2_name} $L$ [a.u.]", saveto=f"{saveto_base}_radiance_simple.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_R_rs(data1, data2, xlabel=f"{phone1_name} {label_R_rs} {plot.persr}", ylabel=f"{phone2_name}\n{label_R_rs} {plot.persr}", saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=phone1_name, ylabel=phone2_name, saveto=f"{saveto_base}_band_ratio.pdf")
plot.correlation_plot_bandratios_combined(data1, data2, xlabel=f"{phone1_name} {label_R_rs} band ratio", ylabel=f"{phone2_name}\n{label_R_rs} band ratio", saveto=f"{saveto_base}_band_ratio_combined.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=f"{phone1_name} {label_R_rs}", ylabel=f"{phone2_name} {label_R_rs}", saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
