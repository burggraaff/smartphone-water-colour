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
    * Any number of folders containing data

Example:
    %run process_raw.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/Balaton_20190703/*
"""

import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from spectacle import io, load_camera
from spectacle.plot import _rgbplot
from matplotlib import pyplot as plt

from wk import hydrocolor as hc, wacodi as wa, plot, statistics as stats

# Get the data folder from the command line
calibration_folder, *folders = io.path_from_input(argv)
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

# Generate camera slices
boxsizes = np.arange(20, 201, 2)
default = 100
index_default = np.where(boxsizes == default)[0][0]
slices = [camera.central_slice(box, box) for box in boxsizes]

# Empty lists to store results
means_list = []
snrs_list = []

for data_path in hc.generate_folders(folders, pattern):
    print("\n  ", data_path)

    # Load data
    image_paths = hc.generate_paths(data_path, camera.raw_extension)
    images_raw = hc.load_raw_images(image_paths)
    print("Loaded RAW data")

    # Load EXIF data
    water_exif = hc.load_exif(image_paths[0])

    # Load thumbnails
    images_jpeg = hc.load_raw_thumbnails(image_paths)
    print("Created JPEG thumbnails")

    # Correct for bias
    images_bias_corrected = camera.correct_bias(images_raw)
    print("Corrected bias")

    # Normalising for ISO speed is not necessary since this is a relative measurement

    # Dark current is negligible

    # Correct for flat-field
    images_flatfield_corrected = camera.correct_flatfield(images_bias_corrected)
    print("Corrected flat-field")

    # Demosaick the data
    images_RGBG = [camera.demosaick(images_flatfield_corrected[central_slice], selection=central_slice) for central_slice in slices]
    print("Demosaicked")

    # Reshape the central images to lists
    data_RGBG = [img.reshape(3, 4, -1) for img in images_RGBG]

    # # Divide by the spectral bandwidths to normalise to ADU nm^-1
    # data_RGBG /= effective_bandwidths[:, np.newaxis]

    # Calculate mean values
    all_mean = np.array([data.mean(axis=-1) for data in data_RGBG])
    print("Calculated mean values per image, per channel")

    # Calculate standard deviations
    all_std = np.array([data.std(axis=-1) for data in data_RGBG])
    print("Calculated uncertainties per image, per channel")

    # Calculate signal-to-noise ratio
    all_snr = all_mean / all_std

    # Change the order of axes to iterate more easily
    all_mean_per_image = np.moveaxis(all_mean, 0, -1)
    all_snr_per_image = np.moveaxis(all_snr, 0, -1)  # New shape: [images, RGBG channels, box sizes]

    # Normalise the means
    all_mean_per_image /= all_mean_per_image[...,index_default,np.newaxis]

    # Plot the result
    # Plot Mean in one column, SNR in next
    labels = [plot.keys_latex[key] for key in ["Lu", "Lsky", "Ld"]]
    fig, axs = plt.subplots(ncols=2, nrows=3, sharex=True, sharey="col", figsize=(plot.col2, 6), tight_layout=True)
    for ax_row, means, snrs, label in zip(axs, all_mean_per_image, all_snr_per_image, labels):
        _rgbplot(boxsizes, means, func=ax_row[0].plot, lw=3)
        _rgbplot(boxsizes, snrs, func=ax_row[1].plot, lw=3)
        for ax in ax_row:
            plot._textbox(ax, label)
    for ax in axs[:,0]:
        ax.set_ylabel("Mean (normalised)")
        ax.set_ylim(0.85, 1.15)
    for ax in axs[:,1]:
        ax.set_ylabel("SNR")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_ylim(0, 60)
    for ax in axs.ravel():
        ax.grid(ls="--")
        ax.axvline(default, ls="--", c="k")
    for ax in axs[-1]:
        ax.set_xlabel("Box size [pixels]")
    fig.suptitle(data_path.parents[0].stem)
    plt.show()
    plt.close()

    # Attach results to previously made lists
    means_list.append(all_mean_per_image)
    snrs_list.append(all_snr_per_image)

# Combine the results into arrays
means_list = np.array(means_list)
snrs_list = np.array(snrs_list)

# Compare the means and SNRs at 50, 100, 150, 200 px box sizes
selection = np.searchsorted(boxsizes, [50, 100, 150, 200])

mean_median = np.median(means_list, axis=0)
mean_std = np.std(means_list, axis=0)

snr_median = np.median(snrs_list, axis=0)
snr_std = np.std(snrs_list, axis=0)

# Calculate the typical deviations from the mean value at the default box size, in percent
deviations_percent = 100*np.abs(1 - means_list)

dev_median = np.median(deviations_percent, axis=0)
dev_std = np.std(deviations_percent, axis=0)

threshold = 2.
fraction_below_threshold = 100*np.sum(deviations_percent <= threshold, axis=0) / deviations_percent.shape[0]
print(fraction_below_threshold[..., selection])