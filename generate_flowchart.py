"""
Process three images (water, sky, grey card), calibrated using SPECTACLE, to
calculate the remote sensing reflectance in the RGB channels, following the
HydroColor protocol.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Bias
    - Flat-field
    - Spectral response

Command-line inputs:
    * SPECTACLE calibration folder
    * Folder containing data

Example:
    %run generate_flowchart.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/Balaton_20190703/Ferry/utc07-45/
"""

import numpy as np
from sys import argv
from spectacle import io, load_camera

from wk import hydrocolor as hc, wacodi as wa, plot, statistics as stats

# Get the data folder from the command line
calibration_folder, folder = io.path_from_input(argv)
pattern = calibration_folder.stem

# Get Camera object
camera = load_camera(calibration_folder)
print(f"Loaded Camera object:\n{camera}")

# ISO speed and exposure time are assumed equal between all three images and
# thus can be ignored

# Load effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands

# Find the effective wavelength corresponding to the RGB bands
RGB_wavelengths = hc.effective_wavelength(calibration_folder)

# Get the correct data folder
data_path = list(hc.generate_folders([folder], pattern))[0]
saveto = data_path/"flow"

# Load data
image_paths = hc.generate_paths(data_path, camera.raw_extension)
images_raw = hc.load_raw_images(image_paths)
print("Loaded RAW data")

# Load and plot JPEG images
jpeg_paths = hc.generate_paths(data_path, ".JPG")
images_jpeg = hc.load_jpeg_images(jpeg_paths)
images_jpeg = np.moveaxis(images_jpeg, 1, -1)  # Move RGB axis to the end
images_jpeg = np.moveaxis(images_jpeg, 1, 2)  # Rotate images
plot.plot_three_images(images_jpeg, saveto=saveto/"images.pdf")
print("Plotted JPEG images")

# Plot the thumbnails

# Correct for bias
images_bias_corrected = camera.correct_bias(images_raw)
print("Corrected bias")

# Normalising for ISO speed is not necessary since this is a relative measurement

# Dark current is negligible

# Correct for flat-field
images_flatfield_corrected = camera.correct_flatfield(images_bias_corrected)
print("Corrected flat-field")

# Select the central pixels
central_slice = camera.central_slice(100, 100)
images_central_slices = images_flatfield_corrected[central_slice]

# Demosaick the data
images_RGBG = camera.demosaick(images_central_slices, selection=central_slice)
print("Demosaicked")

# Plot images of the data reduction so far
plot.plot_image_small(images_raw, saveto=saveto/"rawimage_{label}.pdf")
plot.plot_image_small(images_flatfield_corrected, saveto=saveto/"flatimage_{label}.pdf")
plot.plot_image_small(images_central_slices, saveto=saveto/"slice_{label}.pdf")  # Make this use the same vmin/vmax as above
plot.plot_image_small_RGBG2(images_RGBG, camera, equal_aspect=True, saveto=saveto/"RGBG2_{label}.pdf")

# Reshape the central images to lists
data_RGBG = images_RGBG.reshape((3, 4, -1))

# Divide by the spectral bandwidths to normalise to ADU nm^-1
data_RGBG /= effective_bandwidths[:, np.newaxis]

# Make a histogram
plot.histogram_small(data_RGBG, saveto=saveto/"histogram_{label}.pdf")
print("Finished making flow chart plots")

# Flatten the data into one long list
data_all = data_RGBG.reshape(12, -1)

# Calculate mean values
all_mean = data_all.mean(axis=-1)
print("Calculated mean values per image, per channel")

# Calculate covariance, correlation matrices for the combined radiances
all_covariance = np.cov(data_all)
max_correlation = stats.max_correlation_in_covariance_matrix(all_covariance)
print(f"Calculated covariance and correlation matrices. Maximum off-diagonal correlation r = {max_correlation:.2f}")

# Plot correlation coefficients
plot.plot_correlation_matrix_radiance(all_covariance, x1=data_all[4], y1=data_all[5], x1label=f"{plot.keys_latex['Lsky']} (R)", y1label=f"{plot.keys_latex['Lsky']} (G)", x2=data_all[1], y2=data_all[9], x2label=f"{plot.keys_latex['Lu']} (G)", y2label=f"{plot.keys_latex['Ld']} (G)", saveto=data_path/"correlation_raw.pdf")

# Average G and G2
all_mean_RGB = hc.convert_RGBG2_to_RGB(all_mean)
all_covariance_RGB = hc.convert_RGBG2_to_RGB_covariance(all_covariance)
data_RGB = hc.split_combined_radiances(all_mean_RGB)

# Convert to remote sensing reflectances
R_rs = hc.R_RS(*data_RGB)

# Save the resulting vectors to file
results = [*data_RGB, R_rs]
labels = ["Lu", "Lsky", "Ld", "R_rs"]
for vector_RGB, label in zip(results, labels):
    hc.output_latex_vector(vector_RGB, label=plot.keys_latex[label].strip("$"), saveto=saveto/f"vector_{label}.tex")
    print("Written to file:", label)

# Save the resulting covariance matrices to file
hc.output_latex_matrix(all_covariance_RGB, saveto=saveto/"matrix_L_RGB.tex")
