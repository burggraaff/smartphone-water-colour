"""
Compare two sets of hyperspectral data.

This script loops over the first data set (in our case, WISP-3) and finds unique timestamps (e.g. 2019-07-03T07:29:00, then 2019-07-03T07:31:00, etc.).
It then finds all data in both data sets taken within max_time_diff seconds of that time stamp.
These spectra are averaged and compared.

This script is somewhat hard-coded for our WISP-3 vs. SoRad comparison, and is probably not suitable for a general hyperspectral-hyperspectral data comparison.

Command-line inputs:
    * Hyperspectral reference data file 1
    * Hyperspectral reference data file 2

Example:
    %run compare_references.py water-colour-data/wisp_Balaton_20190703_20190705_table.csv water-colour-data/So-Rad_Balaton2019.csv
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, spectral
from astropy import table
from wk import hydrocolor as hc, hyperspectral as hy, plot

# Time limit for inclusion
max_time_diff = 60*3  # 3 minutes

# Get the data folder from the command line
path_data1, path_data2 = io.path_from_input(argv)

# Find out what reference sensor we're using
reference1, ref_small1 = hy.get_reference_name(path_data1)
reference2, ref_small2 = hy.get_reference_name(path_data2)

# Some useful labels
saveto_base = f"results/comparison_{reference1}_X_{reference2}"
print(f"Comparing data from {reference1} and {reference2}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
table_data1 = hy.read(path_data1)
table_data2 = hy.read(path_data2)
print("Finished reading data")

# Interpolate both data sets to 400-700 nm in 1 nm steps.
# Ignore covariance for now.
table_data1 = hy.interpolate_hyperspectral_table(table_data1)
table_data2 = hy.interpolate_hyperspectral_table(table_data2)
print("Interpolated data")

# Convolve to RGB for one phone, or just compare XYZ?

# Find unique time stamps in table 1
table_data1_timestamps, table_data1_timestamps_indices = np.unique(table_data1["timestamp"], return_index=True)

# Find matches
data1, data2 = [], []  # Lists to contain matching table entries
for timestamp, index_table1 in zip(table_data1_timestamps, table_data1_timestamps_indices):  # Loop over the unique time stamps
    reference_time = table_data1[index_table1]["UTC"]

    # Find matches in table 1
    nr_matches1, close_enough1, closest1, min_time_diff1 = hy.find_elements_within_range(table_data1, reference_time, maximum_difference=max_time_diff)

    # Find matches in table 2
    nr_matches2, close_enough2, closest2, min_time_diff2 = hy.find_elements_within_range(table_data2, reference_time, maximum_difference=max_time_diff)

    if nr_matches1 < 1 or nr_matches2 < 1:  # If no close enough matches are found, skip this observation
        continue

    # Calculate the median Lu/Lsky/Ed/R_rs within the matching observations, and uncertainty on this spectrum
    colour_keys = [*"XYZxy", *hc.bands_sRGB, "hue", "FU"]
    default_index1 = np.where(close_enough1 == closest1)[0][0]
    default_index2 = np.where(close_enough2 == closest2)[0][0]
    data1_averaged = hy.average_hyperspectral_data(table_data1[close_enough1], default_row=default_index1, colour_keys=colour_keys)
    data2_averaged = hy.average_hyperspectral_data(table_data2[close_enough2], default_row=default_index2, colour_keys=colour_keys)

    # Add some metadata that may be used to identify the quality of the match
    data1_averaged = hy.add_hyperspectral_matchup_metadata(data1_averaged, nr_matches1, min_time_diff1)
    data2_averaged = hy.add_hyperspectral_matchup_metadata(data2_averaged, nr_matches2, min_time_diff2)

    print("----")
    print(f"Time: {timestamp}")
    hy.print_matchup_metadata(reference1, nr_matches1, min_time_diff1)
    hy.print_matchup_metadata(reference2, nr_matches2, min_time_diff2)

    # Store matched rows in lists
    data1.append(data1_averaged)
    data2.append(data2_averaged)

    # Plot the spectra for comparison
    R_rs1 = hy.convert_columns_to_array(data1_averaged, hy.extend_keys_to_wavelengths("R_rs"))[0]
    R_rs1_uncertainty = hy.convert_columns_to_array(data1_averaged, hy.extend_keys_to_wavelengths("R_rs_err"))[0]
    spectrum1 = [hy.wavelengths_interpolation, R_rs1, R_rs1_uncertainty]

    R_rs2 = hy.convert_columns_to_array(data2_averaged, hy.extend_keys_to_wavelengths("R_rs"))[0]
    R_rs2_uncertainty = hy.convert_columns_to_array(data2_averaged, hy.extend_keys_to_wavelengths("R_rs_err"))[0]
    spectrum2 = [hy.wavelengths_interpolation, R_rs2, R_rs2_uncertainty]

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small1}_{ref_small2}_comparison/{reference1}_X_{reference2}_{timestamp}.pdf".replace(":", "-")
    plot.plot_R_rs_multi([spectrum1, spectrum2], labels=[reference1, reference2], title=timestamp, figsize=plot.smallpanel, saveto=saveto)

# Make new tables from the match-up rows
data1 = table.vstack(data1)
data2 = table.vstack(data2)

# Rename sRGB to RGB
parameters_RGB = hc.extend_keys_to_RGB(hy.parameters+hy.parameters_uncertainty)
parameters_sRGB = hc.extend_keys_to_RGB(hy.parameters+hy.parameters_uncertainty, hc.bands_sRGB)
data1.rename_columns(parameters_sRGB, parameters_RGB)
data2.rename_columns(parameters_sRGB, parameters_RGB)

# Add typical errors if only a single match was found
data1 = hy.fill_in_median_uncertainties(data1)
data2 = hy.fill_in_median_uncertainties(data2)

# Add band ratios to reference data
data1 = hy.add_bandratios_to_hyperspectral_data(data1)
data2 = hy.add_bandratios_to_hyperspectral_data(data2)

# Save the comparison table to file
saveto_data = f"{saveto_base}_data.csv"
table_combined = table.hstack([data1, data2], table_names=[ref_small1, ref_small2])
table_combined.remove_columns([key for key in table_combined.keys() if "cov_" in key])
table_combined.write(saveto_data, format="ascii.fast_csv", overwrite=True)
print(f"Saved comparison table to `{saveto_data}`.")

# Correlation plot: Radiances and irradiance
plot.correlation_plot_radiance(data1, data2, keys=["Lu", "Lsky"], xlabel=reference1, ylabel=reference2, saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, keys=["Lu", "Lsky"], xlabel=f"{reference1}\n$L$ {plot.Wnmsr}", ylabel=f"{reference2} $L$ {plot.Wnmsr}", saveto=f"{saveto_base}_radiance_simple.pdf")
plot.correlation_plot_RGB(data1, data2, "Ed ({c})", "Ed ({c})", xerrlabel="Ed_err ({c})", yerrlabel="Ed_err ({c})", xlabel=f"{reference1} {plot.keys_latex['Ed']} {plot.Wnm}", ylabel=f"{reference2} {plot.keys_latex['Ed']} {plot.Wnm}", regression="rgb", saveto=f"{saveto_base}_Ed.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_RGB_equal(data1, data2, "R_rs", errlabel="R_rs_err", xlabel=f"{reference1} {label_R_rs} {plot.persr}", ylabel=f"{reference2} {label_R_rs} {plot.persr}", regression="all", difference_unit=plot.persr, saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=reference1, ylabel=reference2, saveto=f"{saveto_base}_band_ratio.pdf")
plot.correlation_plot_bandratios_combined(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=reference1, ylabel=reference2, saveto=f"{saveto_base}_band_ratio_combined.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=reference1, ylabel=reference2, saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
