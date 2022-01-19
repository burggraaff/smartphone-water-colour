"""
Investigate the angular response of the grey card's reflectance.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Bias
    - Flat-field

Command-line inputs:
    * SPECTACLE calibration folder
    * Folder containing data

Example:
    %run greycard.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/Greycard/
"""

import numpy as np

from sys import argv
from spectacle import io, load_camera
from spectacle.plot import _rgbplot
from matplotlib import pyplot as plt

from wk import hydrocolor as hc, wacodi as wa, plot, statistics as stats

# Get the data folder from the command line
calibration_folder, data_folder = io.path_from_input(argv)

# Get Camera object
camera = load_camera(calibration_folder)
print(f"Loaded Camera object:\n{camera}")

# Load data
central_slice = camera.central_slice(100, 100)
get_angle = lambda p: float(p.stem.split("_")[0][:-3])  # Get the angle from the filename
angles, means = io.load_means(data_folder/"stacks", retrieve_value=get_angle, selection=central_slice)
print("Loaded data")

# Bias correction
means_bias_corrected = camera.correct_bias(means, selection=central_slice)
print("Applied SPECTACLE corrections")

# Average per image
mean_values = means_bias_corrected.mean(axis=(1,2))
uncertainties = means_bias_corrected.std(axis=(1,2))

# Normalise by the values at 40 degrees
normalisation = mean_values[angles==40]
mean_values_normalised = mean_values / normalisation
uncertainties_normalised = uncertainties / normalisation

# Plot the result
plt.errorbar(angles, mean_values_normalised, yerr=uncertainties_normalised, fmt="o", c="k")

plt.xlabel("Grey card angle [$^\circ$]")
plt.ylabel("Mean value [ADU]")

plt.xlim(30, 50)

plt.show()

# Print the relevant values (within 5 degrees)
indices = np.where((angles >= 35) & (angles <= 45))[0]
for ind in indices:
    print(f"Angle {angles[ind]:.2f} degrees: Mean {mean_values_normalised[ind]:.2f} +- {uncertainties_normalised[ind]:.2f}")
