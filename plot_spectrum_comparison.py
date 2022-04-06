"""
Make a plot with 2x2 smartphone-reference spectrum comparisons.

Command-line inputs:
    * SPECTACLE calibration folder
    * Smartphone data file
    * Hyperspectral reference data file

Example:
    %run plot_spectrum_comparison.py C:/Users/Burggraaff/SPECTACLE_data/ "results/comparison_WISP-3_X_iPhone SE_RAW_data.csv" "results/comparison_WISP-3_X_Samsung Galaxy S8_RAW_data.csv" "results/comparison_So-Rad_X_iPhone SE_RAW_data.csv" "results/comparison_So-Rad_X_Samsung Galaxy S8_RAW_data.csv"
"""
from sys import argv
import numpy as np
from astropy import table
from matplotlib import pyplot as plt
from spectacle import io, spectral, load_camera
from wk import hydrocolor as hc, hyperspectral as hy, plot

# Indices we want to plot - arbitrary choices
indices = [19, 13, 1, 4]

# Get the data folder from the command line
path_calibration, *data_paths = io.path_from_input(argv)

# Get the correct labels
reference_names = [hy.get_reference_name(filename)[0] for filename in data_paths]
smartphone_names = [hc.get_phone_name(filename).split("X")[-1].strip() for filename in data_paths]

# Get the camera metadata
calibration_folders = [path_calibration/name.replace(" ", "_") for name in smartphone_names]
cameras = [load_camera(folder) for folder in calibration_folders]
for camera in cameras:
    camera._load_spectral_response()
    camera.load_spectral_bands()
RGB_wavelengths = [spectral.effective_wavelengths(camera.spectral_response[0], camera.spectral_response[1:4]) for camera in cameras]
effective_bandwidths = [camera.spectral_bands for camera in cameras]

# Very ugly way to change "Samsung Galaxy S8" to "Galaxy S8"
smartphone_names = [name.strip("Samsung ") for name in smartphone_names]

# Load the comparison data files
data_all = [hy.read(filename) for filename in data_paths]

# Create a figure
fig, axs = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True, figsize=(plot.col1, plot.col1), gridspec_kw={"hspace": 0.05, "wspace": 0.05})

# Plot everything
for ax, data, rownumber, phone_wavelengths, phone_bandwidths in zip(axs.ravel(), data_all, indices, RGB_wavelengths, effective_bandwidths):
    # Extract the reference data
    reference_wavelengths = hy.get_wavelengths_from_table(data, "R_rs")
    reference_keys = [hy.extend_keys_to_wavelengths(key, wavelengths=reference_wavelengths) for key in ["R_rs", "R_rs_err"]]
    reference_R_rs, reference_R_rs_err = [hy.convert_columns_to_array(data, keys)[rownumber] for keys in reference_keys]

    # Extract the smartphone data
    smartphone_keys = [hc.extend_keys_to_RGB(key) for key in ["R_rs", "R_rs_err"]]
    for key in np.ravel(smartphone_keys):
        data.rename_column(key+"_phone", key)
    smartphone_R_rs, smartphone_R_rs_err = [hy.convert_columns_to_array(data, keys)[rownumber] for keys in smartphone_keys]

    # Finally, plot everything
    plot.plot_R_rs_RGB(phone_wavelengths, smartphone_R_rs, phone_bandwidths, smartphone_R_rs_err, reference=[reference_wavelengths, reference_R_rs, reference_R_rs_err], ax=ax)

    # Print the time, for reference
    time = data["UTC (ISO)"][rownumber].replace("T", " ")[:-3]
    print(time)
    # ax.text(s=time, x=0.50, y=0.95, transform=ax.transAxes, bbox=plot.bbox_text, fontsize=9, horizontalalignment="center", verticalalignment="top")

# Adjust the axes
axs[0,0].set_xticks(np.arange(400, 1000, 200))
axs[0,0].set_yticks(np.arange(0.00, 0.08, 0.02))
axs[0,0].set_ylim(0, 0.065)

# Labels outside panels
for ax, label in zip(axs[0], smartphone_names):
    ax.set_xlabel(label)
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
for ax, label in zip(axs[:,1], reference_names[::2]):
    ax.set_ylabel(label)
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", left=False, labelleft=False)

# Final settings
fig.suptitle("Example match-up spectra")

# Save to file
plt.savefig("results/matchup_examples.pdf", bbox_inches="tight")
plt.show()
plt.close()
