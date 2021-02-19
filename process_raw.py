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
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, load_camera
from astropy import table
from datetime import datetime, timedelta
from os import walk
from scipy import stats

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
effective_bandwidths = camera.spectral_bands[:3]

# Find the effective wavelength corresponding to the RGB bands
RGB_wavelengths = hc.effective_wavelength(calibration_folder)

for folder_main in folders:
    for tup in walk(folder_main):
        folder = io.Path(tup[0])
        data_path = folder/pattern
        if not data_path.exists():
            continue

        # Load data
        water_path, sky_path, card_path = hc.generate_paths(data_path, camera.raw_extension)
        water_raw, sky_raw, card_raw = hc.load_raw_images(water_path, sky_path, card_path)
        print("Loaded RAW data")

        # Load EXIF data
        water_exif, sky_exif, card_exif = hc.load_exif(water_path, sky_path, card_path)

        # Load thumbnails
        water_jpeg, sky_jpeg, card_jpeg = hc.load_raw_thumbnails(water_path, sky_path, card_path)
        print("Created JPEG thumbnails")

        # Correct for bias
        water_bias, sky_bias, card_bias = camera.correct_bias(water_raw, sky_raw, card_raw)
        print("Corrected bias")

        # Normalising for ISO speed is not necessary since this is a relative measurement

        # Dark current is negligible

        # Correct for flat-field
        water_flat, sky_flat, card_flat = camera.correct_flatfield(water_bias, sky_bias, card_bias)
        print("Corrected flat-field")

        # Demosaick the data
        water_RGBG, sky_RGBG, card_RGBG = camera.demosaick(water_flat, sky_flat, card_flat)
        print("Demosaicked")

        # Select the central pixels
        water_cut, sky_cut, card_cut = hc.central_slice_raw(water_RGBG, sky_RGBG, card_RGBG)

        # Combined histograms of different data reduction steps
        water_all = [water_jpeg, water_raw, water_bias, water_flat, water_cut]
        sky_all = [sky_jpeg, sky_raw, sky_bias, sky_flat, sky_cut]
        card_all = [card_jpeg, card_raw, card_bias, card_flat, card_cut]

        hc.histogram_raw(water_all, sky_all, card_all, camera=camera, saveto=data_path/"statistics_raw.pdf")

        # Reshape the central images to lists
        water_RGBG = water_cut.reshape(4, -1)
        sky_RGBG = sky_cut.reshape(4, -1)
        card_RGBG = card_cut.reshape(4, -1)
        all_RGBG = np.concatenate([water_RGBG, sky_RGBG, card_RGBG])

        # Calculate mean values
        water_mean = water_RGBG.mean(axis=1)
        sky_mean = sky_RGBG.mean(axis=1)
        card_mean = card_RGBG.mean(axis=1)
        all_mean = all_RGBG.mean(axis=1)
        print("Calculated mean values per channel")

        water_std = water_RGBG.std(axis=1)
        sky_std = water_RGBG.std(axis=1)
        card_std = water_RGBG.std(axis=1)
        all_cov = np.cov(all_RGBG)
        all_cov_R = np.zeros((13,13)) ; all_cov_R[:12,:12] = all_cov ; all_cov_R[12,12] = 0.01**2
        print("Calculated standard deviations per channel")

        # Convert to remote sensing reflectances
        R_rs = hc.R_RS(water_mean, sky_mean, card_mean)
        print("Calculated remote sensing reflectances")

        # Covariances
        Rref = 0.18
        rho = 0.028
        J1 = 1/np.pi * Rref * np.eye(4) * (1/card_mean)
        J2 = -1/np.pi * Rref * rho * np.eye(4) * (1/card_mean)
        J3 = -1 * np.eye(4) * (R_rs / card_mean)
        JR = R_rs[:,np.newaxis] / Rref
        J = np.concatenate([J1, J2, J3, JR], axis=1)
        R_rs_cov = J @ all_cov_R @ J.T

        # HydroColor

        R_rs_err = hc.R_RS_error(water_mean, sky_mean, card_mean, water_std, sky_std, card_std)
        print("Calculated error in remote sensing reflectances")

        for R, R_err, c in zip(R_rs, R_rs_err, "RGB"):
            print(f"{c}: R_rs = {R:.3f} +- {R_err:.3f} sr^-1")

        # Plot the result
        hc.plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err)

        # # WACODI

        # # Convert RGBG2 to RGB
        # water_mean, sky_mean, card_mean = water_mean[:3], sky_mean[:3], card_mean[:3]
        # water_std, sky_std, card_std = water_std[:3], sky_std[:3], card_std[:3]

        # # Convert RGB to XYZ
        # water_XYZ, sky_XYZ, card_XYZ = camera.convert_to_XYZ(water_mean, sky_mean, card_mean)
        # water_XYZ_err, sky_XYZ_err, card_XYZ_err = wa.convert_errors_to_XYZ(camera.XYZ_matrix, water_std, sky_std, card_std)

        # # Calculate xy chromaticity
        # water_xy, sky_xy, card_xy = wa.convert_XYZ_to_xy(water_XYZ, sky_XYZ, card_XYZ)

        # # Calculate hue angle
        # water_hue, sky_hue, card_hue = wa.convert_xy_to_hue_angle(water_xy, sky_xy, card_xy)
        # water_hue_err, sky_hue_err, card_hue_err = [wa.convert_XYZ_error_to_hue_angle(XYZ_data, XYZ_error) for XYZ_data, XYZ_error in zip([water_XYZ, sky_XYZ, card_XYZ], [water_XYZ_err, sky_XYZ_err, card_XYZ_err])]

        # Create a timestamp from EXIF (assume time zone UTC+2)
        UTC = hc.UTC_timestamp(water_exif)

        # Write the result to file
        saveto = data_path.with_name(data_path.stem + "_raw.csv")
        hc.write_R_rs(UTC, R_rs, R_rs_err, saveto=saveto)
