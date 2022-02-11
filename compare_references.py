"""
Compare two sets of hyperspectral data.

Command-line inputs:
    * Hyperspectral reference data file 1
    * Hyperspectral reference data file 2

Example:
    %run compare_references.py water-colour-data/wisp_Balaton_20190703_20190705_table.csv water-colour-data/So-Rad_Balaton2019.csv
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, spectral, load_camera
from astropy import table
from wk import hydrocolor as hc, hyperspectral as hy, plot

# Time limit for inclusion
max_time_diff = 60*10  # 10 minutes for everything except NZ data
# max_time_diff = 60*60  # 60 minutes for NZ data

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

# Parameters of interest
parameters = ["Ed", "Lsky", "Lu", "R_rs"]
wavelengths = np.arange(390, 701, 1)

# Interpolate both data sets to 390-700 nm in 1 nm steps.
# Ignore covariance for now.
for data in [table_data1, table_data2]:
    # Get the number of spectra in each parameter (should be equal)
    nr_spectra = len(data)

    # Extract the data for each parameter
    columns = [hy.get_keys_for_parameter(data, param) for param in parameters]
    wavelengths_old = hy.get_wavelengths_from_keys(columns[0], key=parameters[0])
    data_old = hy.convert_columns_to_array(data, columns[0])


# Convolve to RGB for one phone, or just compare XYZ?

raise Exception

# Find matches
data1, data2 = [], []  # Lists to contain matching table entries
for row in table_phone:  # Loop over the smartphone table to look for matches
    # Find matches within a threshold
    time_differences = np.abs(table_reference["UTC"] - row["UTC"])
    close_enough = np.where(time_differences <= max_time_diff)[0]
    closest = time_differences.argmin()
    min_time_diff = time_differences[closest]
    if min_time_diff > max_time_diff:  # If no close enough matches are found, skip this observation
        continue
    phone_time = hc.iso_timestamp(row['UTC'])
    reference_time = hc.iso_timestamp(table_reference[closest]["UTC"])

    # Calculate the median Lu/Lsky/Ed/R_rs within the matching observations, and uncertainty on this spectrum
    row_reference = table.Table(table_reference[closest])
    for key in parameters:
        # Average over the "close enough" rows
        keys = [f"{key}_{wvl:.1f}" for wvl in wavelengths] + hc.extend_keys_to_RGB(key)
        keys_err = [f"{key}_err_{wvl:.1f}" for wvl in wavelengths] + hc.extend_keys_to_RGB(key+"_err")

        row_reference[keys][0] = [np.nanmedian(table_reference[k][close_enough]) for k in keys]
        uncertainties = np.array([np.nanstd(table_reference[k][close_enough]) for k in keys])
        row_uncertainties = table.Table(data=uncertainties, names=keys_err)
        row_reference = table.hstack([row_reference, row_uncertainties])

    # If the uncertainties on the reference data are above a threshold, disregard this match-up
    # This may cause differences between RAW and JPEG matchup numbers for the same data set
    threshold = 0.1  # relative
    if any(row_reference[f"R_rs_err ({c})"]/row_reference[f"R_rs ({c})"] >= threshold for c in hc.colours):
        continue

    # Add some metadata that may be used to identify the quality of the match
    metadata = table.Table(names=["nr_matches", "closest_match"], data=np.array([len(close_enough), time_differences[closest]]))
    row_reference = table.hstack([metadata, row_reference])
    print("----")
    print(f"Phone time: {phone_time} ; {reference} time: {reference_time}")
    print(f"Number of matches: {row_reference['nr_matches'][0]:.0f}; Closest match: {row_reference['closest_match'][0]:.0f} s difference")

    # Store matched rows in lists
    data2.append(row)
    data1.append(row_reference)

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small}_comparison/{camera.name}_{data_type}_{phone_time}.pdf".replace(":", "-")

    # Plot the spectrum for comparison
    R_rs_reference = np.array([row_reference[f"R_rs_{wvl:.1f}"][0] for wvl in wavelengths])
    R_rs_reference_uncertainty = np.array([row_reference[f"R_rs_err_{wvl:.1f}"][0] for wvl in wavelengths])

    R_rs_phone = list(row[hc.extend_keys_to_RGB("R_rs")])
    R_rs_phone_err = list(row[hc.extend_keys_to_RGB("R_rs_err")])

    plot.plot_R_rs_RGB(RGB_wavelengths, R_rs_phone, effective_bandwidths, R_rs_phone_err, reference=[wavelengths, R_rs_reference, R_rs_reference_uncertainty], title=f"{cameralabel}\n{phone_time}", saveto=saveto)

# Make new tables from the match-up rows
data2 = table.vstack(data2)
data1 = table.vstack(data1)

# Add typical errors to R_rs (R, G, B) if only a single match was found
indices_single_match, indices_multiple_matches = np.where(data1["nr_matches"] == 1), np.where(data1["nr_matches"] > 1)
keys_uncertainties = hc.extend_keys_to_RGB([param+"_err" for param in parameters])
for key in keys_uncertainties:
    data1[key][indices_single_match] = np.nanmedian(data1[key][indices_multiple_matches])

# Add band ratios to reference data
bandratios = table.Table(data=hc.calculate_bandratios(data1["R_rs (R)"], data1["R_rs (G)"], data1["R_rs (B)"]).T, names=[f"R_rs ({label})" for label in hc.bandratio_labels])

bandratio_uncertainties = table.Table(data=[bandratios[col] * np.sqrt(data1[f"R_rs_err ({bands[0]})"]**2/data1[f"R_rs ({bands[0]})"]**2 + data1[f"R_rs_err ({bands[1]})"]**2/data1[f"R_rs ({bands[1]})"]**2) for col, bands in zip(bandratios.colnames, hc.bandratio_pairs)], names=[f"R_rs_err ({label})" for label in hc.bandratio_labels])

data1 = table.hstack([data1, bandratios, bandratio_uncertainties])

# Save the comparison table to file
saveto_data = f"{saveto_base}_data.csv"
table_combined = table.hstack([data1, data2], table_names=["reference", "phone"])
table_combined.remove_columns([key for key in table_combined.keys() if "cov_" in key])
table_combined.write(saveto_data, format="ascii.fast_csv", overwrite=True)
print(f"Saved comparison table to `{saveto_data}`.")

# Correlation plot: Radiances and irradiance
plot.correlation_plot_radiance(data1, data2, keys=["Lu", "Lsky"], xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data1, data2, keys=["Lu", "Lsky"], xlabel=f"{reference}\n$L$ {plot.Wnmsr}", ylabel=f"{cameralabel} $L$ [a.u.]", saveto=f"{saveto_base}_radiance_simple.pdf")
plot.correlation_plot_RGB(data1, data2, "Ed ({c})", "Ed ({c})", xerrlabel="Ed_err ({c})", yerrlabel="Ed_err ({c})", xlabel=f"{reference} {plot.keys_latex['Ed']} {plot.Wnm}", ylabel=f"{cameralabel} {plot.keys_latex['Ed']} {plot.ADUnm}", regression="rgb", saveto=f"{saveto_base}_Ed.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_RGB_equal(data1, data2, "R_rs", errlabel="R_rs_err", xlabel=f"{reference} {label_R_rs} {plot.persr}", ylabel=f"{cameralabel}\n{label_R_rs} {plot.persr}", regression="all", difference_unit=plot.persr, saveto=f"{saveto_base}_R_rs.pdf")

# Correlation plot: Band ratios
plot.correlation_plot_bands(data1, data2, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_band_ratio.pdf")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data1["R_rs (hue)"], data2["R_rs (hue)"], xlabel=reference, ylabel=cameralabel, saveto=f"{saveto_base}_hueangle_ForelUle.pdf")
