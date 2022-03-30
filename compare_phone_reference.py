"""
Compare data, including radiance, reflectance, and colour, from a smartphone and a hyperspectral sensor.

Command-line inputs:
    * SPECTACLE calibration folder
    * Smartphone data file
    * Hyperspectral reference data file

Example:
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/balaton_iPhone_SE_raw_18pct.csv water-colour-data/So-Rad_Balaton2019.csv
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/balaton_iPhone_SE_raw_18pct.csv water-colour-data/wisp_Balaton_20190703_20190705_table.csv
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/Samsung_Galaxy_S8/ water-colour-data/balaton_Samsung_Galaxy_S8_raw_18pct.csv water-colour-data/So-Rad_Balaton2019.csv
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/Samsung_Galaxy_S8/ water-colour-data/balaton_Samsung_Galaxy_S8_raw_18pct.csv water-colour-data/wisp_Balaton_20190703_20190705_table.csv
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_6S/ water-colour-data/balaton_iPhone_6S_raw_18pct.csv water-colour-data/Data_Monocle2019_L5All_TriOS_table.csv
    %run compare_phone_reference.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_6S/ water-colour-data/NZ_iPhone_6S_raw_18pct.csv water-colour-data/Trios_all_nz_south_L5statsPassed_TriOS_table.csv
"""
from sys import argv
import numpy as np
from astropy import table
from spectacle import io, spectral, load_camera
from wk import hydrocolor as hc, hyperspectral as hy, plot

# Get the data folder from the command line
path_calibration, path_phone, path_reference = io.path_from_input(argv)

# Time limit for inclusion
if "_nz_" in path_reference.stem:
    maximum_time_difference = 60*60  # 60 minutes for NZ data due to the campaign setup
else:
    maximum_time_difference = 60*10  # 10 minutes for everything except NZ data

# Find out if we're doing JPEG or RAW
data_type = hc.data_type_RGB(path_phone)

# Find out what reference sensor we're using
reference, ref_small = hy.get_reference_name(path_reference)

# Get Camera object
camera = load_camera(path_calibration)
print(f"Loaded Camera object:\n{camera}")

# Names of the data being compared and some useful labels
cameralabel = f"{camera.name} {data_type}"
saveto_base = f"results/comparison_{reference}_X_{camera.name}_{data_type}"
print(f"Comparing data from {reference} and {cameralabel}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Find the effective wavelength corresponding to the RGB bands
camera._load_spectral_response()
spectral_response = camera.spectral_response
wavelengths_phone = spectral_response[0]
RGB_responses = spectral_response[1:4]
RGB_wavelengths = spectral.effective_wavelengths(wavelengths_phone, RGB_responses)

# Effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands

# Read the data
table_phone = hc.read_results(path_phone)
table_reference = hy.read(path_reference)
print("Finished reading data")

# Parameters of interest
cols_example = hy.get_keys_for_parameter(table_reference, hy.parameters[0])
wavelengths = hy.get_wavelengths_from_keys(cols_example, key=hy.parameters[0])

# Spectral convolution of radiance and reflectance
table_reference = hy.convolve_radiance_to_camera_bands(table_reference, camera)
table_reference = hy.convolve_reflectance_to_camera_bands[ref_small](table_reference, camera)

# Find matches
data_phone, data_reference = [], []  # Lists to contain matching table entries
for row in table_phone:  # Loop over the smartphone table to look for matches
    # Find matches within a threshold
    nr_matches, close_enough, closest, min_time_diff = hy.find_elements_within_range(table_reference, row["UTC"], maximum_difference=maximum_time_difference)
    if nr_matches < 1:  # If no close enough matches are found, skip this observation
        continue

    phone_time = hc.iso_timestamp(row["UTC"])
    reference_time = hc.iso_timestamp(table_reference[closest]["UTC"])

    # Calculate the median Lu/Lsky/Ed/R_rs within the matching observations, and uncertainty on this spectrum
    default_index = np.where(close_enough == closest)[0][0]
    row_reference = hy.average_hyperspectral_data(table_reference[close_enough], default_row=default_index, wavelengths=wavelengths)

    # If the uncertainties on the reference data are above a threshold, disregard this match-up
    # This may cause differences between RAW and JPEG matchup numbers for the same data set
    threshold = 0.1  # relative
    if any(row_reference[f"R_rs_err ({c})"]/row_reference[f"R_rs ({c})"] >= threshold for c in hc.colours):
        continue

    # Add some metadata that may be used to identify the quality of the match
    row_reference = hy.add_hyperspectral_matchup_metadata(row_reference, nr_matches, min_time_diff)

    print("----")
    print(f"Phone time: {phone_time} ; {reference} time: {reference_time}")
    hy.print_matchup_metadata(reference, nr_matches, min_time_diff)

    # Store matched rows in lists
    data_phone.append(row)
    data_reference.append(row_reference)

    # Convert ":" to - in the filename when saving
    saveto = f"results/{ref_small}_comparison/{camera.name}_{data_type}_{phone_time}.pdf".replace(":", "-")

    # Plot the spectrum for comparison
    R_rs_reference = hy.convert_columns_to_array(row_reference, hy.extend_keys_to_wavelengths("R_rs", wavelengths))[0]
    R_rs_reference_uncertainty = hy.convert_columns_to_array(row_reference, hy.extend_keys_to_wavelengths("R_rs_err", wavelengths))[0]

    R_rs_phone = list(row[hc.extend_keys_to_RGB("R_rs")])
    R_rs_phone_err = list(row[hc.extend_keys_to_RGB("R_rs_err")])

    plot.plot_R_rs_RGB(RGB_wavelengths, R_rs_phone, effective_bandwidths, R_rs_phone_err, reference=[wavelengths, R_rs_reference, R_rs_reference_uncertainty], title=f"{cameralabel}\n{phone_time}", figsize=plot.smallpanel, saveto=saveto)

# Make new tables from the match-up rows
data_phone = table.vstack(data_phone)
data_reference = table.vstack(data_reference)

# Add typical errors if only a single match was found
data_reference = hy.fill_in_median_uncertainties(data_reference)

# Add band ratios to reference data
data_reference = hy.add_bandratios_to_hyperspectral_data(data_reference)

# Save the comparison table to file
saveto_data = f"{saveto_base}_data.csv"
table_combined = table.hstack([data_reference, data_phone], table_names=["reference", "phone"])
table_combined.remove_columns([key for key in table_combined.keys() if "cov_" in key])
table_combined.write(saveto_data, format="ascii.fast_csv", overwrite=True)
print(f"Saved comparison table to `{saveto_data}`.")

# Correlation plot: Radiances and irradiance
plot.correlation_plot_radiance(data_reference, data_phone, keys=["Lu", "Lsky"], xlabel=reference, ylabel=cameralabel, title="Smartphone-Reference\nradiance comparison", saveto=f"{saveto_base}_radiance.pdf")
plot.correlation_plot_radiance_combined(data_reference, data_phone, keys=["Lu", "Lsky"], xlabel=f"{reference} $L$ {plot.Wnmsr}", ylabel=f"{cameralabel} $L$ [a.u.]", title="Smartphone-Reference\nradiance comparison", saveto=f"{saveto_base}_radiance_simple.pdf", saveto_stats=f"{saveto_base}_radiance_simple.dat")
plot.correlation_plot_RGB(data_reference, data_phone, "Ed ({c})", "Ed ({c})", xerrlabel="Ed_err ({c})", yerrlabel="Ed_err ({c})", xlabel=f"{reference} {plot.keys_latex['Ed']} {plot.Wnm}", ylabel=f"{cameralabel} {plot.keys_latex['Ed']} {plot.ADUnm}", title="Smartphone-Reference\nirradiance comparison", regression="rgb", saveto=f"{saveto_base}_Ed.pdf")

# Correlation plot: Remote sensing reflectance
label_R_rs = plot.keys_latex["R_rs"]
plot.correlation_plot_R_rs(data_reference, data_phone, xlabel=f"{reference} {label_R_rs} {plot.persr}", ylabel=f"{cameralabel}\n{label_R_rs} {plot.persr}", title="Smartphone-Reference\nreflectance comparison", saveto=f"{saveto_base}_R_rs.pdf", saveto_stats=f"{saveto_base}_R_rs.dat", compare_to_regression=True)

# Correlation plot: Band ratios
plot.correlation_plot_bands(data_reference, data_phone, datalabel="R_rs", errlabel="R_rs_err", quantity=label_R_rs, xlabel=reference, ylabel=cameralabel, title="Smartphone-Reference\nband ratio comparison", saveto=f"{saveto_base}_band_ratio.pdf")
plot.correlation_plot_bandratios_combined(data_reference, data_phone, xlabel=f"{reference} {label_R_rs} band ratio", ylabel=f"{cameralabel} {label_R_rs}\nband ratio", title="Smartphone-Reference\nband ratio comparison", saveto=f"{saveto_base}_band_ratio_combined.pdf", saveto_stats=f"{saveto_base}_band_ratio_combined.dat")

# Correlation plot: hue angle and Forel-Ule index
plot.correlation_plot_hue_angle_and_ForelUle(data_reference["R_rs (hue)"], data_phone["R_rs (hue)"], xlabel=reference, ylabel=cameralabel, title="Smartphone-Reference\nwater color comparison", saveto=f"{saveto_base}_hueangle_ForelUle.pdf", saveto_stats=f"{saveto_base}_hueangle_ForelUle.dat")
