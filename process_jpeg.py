"""
Process three images (water, sky, grey card), in JPEG format, to
calculate the remote sensing reflectance in the RGB channels, following the
HydroColor protocol.

Requires the following SPECTACLE calibrations:
    - Metadata
    - Spectral response

Command-line inputs:
    * SPECTACLE calibration folder
    * Any number of folders containing data
"""

import numpy as np
from sys import argv
from spectacle import io, load_camera
from os import walk

from wk import hydrocolor as hc, wacodi as wa

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
effective_bandwidths = camera.spectral_bands[:3]  # No G2 band in JPEG data

# Find the effective wavelength corresponding to the RGB bands
RGB_wavelengths = hc.effective_wavelength(calibration_folder)

for folder_main in folders:
    for tup in walk(folder_main):
        folder = io.Path(tup[0])
        data_path = folder/pattern
        if not data_path.exists():
            continue

        # Load data
        water_path, sky_path, card_path = hc.generate_paths(data_path, ".JPG")
        water_jpeg, sky_jpeg, card_jpeg = hc.load_jpeg_images(water_path, sky_path, card_path)
        print("Loaded JPEG data")

        # Load EXIF data
        water_exif, sky_exif, card_exif = hc.load_exif(water_path, sky_path, card_path)

        # Select the central 100x100 pixels
        central_x, central_y = water_jpeg.shape[0]//2, water_jpeg.shape[1]//2
        box_size = 100
        central_slice = np.s_[central_x-box_size:central_x+box_size+1, central_y-box_size:central_y+box_size+1]
        water_cut = water_jpeg[central_slice]
        sky_cut = sky_jpeg[central_slice]
        card_cut = card_jpeg[central_slice]
        print(f"Selected central {2*box_size}x{2*box_size} pixels")

        # Combined histograms of different data reduction steps
        water_all = [water_jpeg, water_cut]
        sky_all = [sky_jpeg, sky_cut]
        card_all = [card_jpeg, card_cut]

        hc.histogram_jpeg(water_all, sky_all, card_all, saveto=data_path/"statistics_jpeg.pdf")

        # Convert to remote sensing reflectances
        water_mean = water_cut.mean(axis=(0,1))
        sky_mean = sky_cut.mean(axis=(0,1))
        card_mean = card_cut.mean(axis=(0,1))
        print("Calculated mean values per channel")

        water_std = water_cut.std(axis=(0,1))
        sky_std = sky_cut.std(axis=(0,1))
        card_std = card_cut.std(axis=(0,1))
        print("Calculated standard deviations per channel")

        R_rs = hc.R_RS(water_mean, sky_mean, card_mean)
        print("Calculated remote sensing reflectances")

        R_rs_err = hc.R_RS_error(water_mean, sky_mean, card_mean, water_std, sky_std, card_std)
        print("Calculated error in remote sensing reflectances")

        for R, R_err, c in zip(R_rs, R_rs_err, "RGB"):
            print(f"{c}: R_rs = {R:.3f} +- {R_err:.3f} sr^-1")

        # Plot the result
        hc.plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err)

        # Create a timestamp from EXIF (assume time zone UTC+2)
        UTC = hc.UTC_timestamp(water_exif)

        # Write the result to file
        hc.write_R_rs(UTC, R_rs, R_rs_err, saveto=data_path.parent / (data_path.stem + "_jpeg.csv"))
