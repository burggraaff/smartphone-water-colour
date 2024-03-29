"""
Process three RAW images (water, sky, grey card) to calculate the remote sensing reflectance R_rs.
This script analyses replicate images to determine the stability of smartphone observations.

Requires the following SPECTACLE calibrations:
    * Metadata
    * Bias
    * Flat-field
    * Spectral response

Command-line inputs:
    * SPECTACLE calibration folder
    * Any number of folders containing data

Example:
    %run process_replicates.py C:/Users/Burggraaff/SPECTACLE_data/Samsung_Galaxy_S8/ water-colour-data/Balaton*
"""
from sys import argv
import numpy as np
from spectacle import io, load_camera
from wk import hydrocolor as hc, wacodi as wa

# Number of replicates per image. If fewer are found in a folder, skip it
nr_replicates = 10

# Get the data folder from the command line
calibration_folder, *folders = io.path_from_input(argv)
pattern = calibration_folder.stem

# Get Camera object
camera = load_camera(calibration_folder)
print(f"Loaded Camera object:\n{camera}")

# ISO speed and exposure time are assumed equal between all three images and thus can be ignored

# Load effective spectral bandwidths
camera.load_spectral_bands()
effective_bandwidths = camera.spectral_bands

# Find the effective wavelength corresponding to the RGB bands
RGB_wavelengths = hc.effective_wavelength(calibration_folder)

for data_path in hc.generate_folders(folders, pattern):
    print("\n  ", data_path)

    # Load data
    image_paths = hc.generate_paths(data_path, camera.raw_extension, multiple=True)
    images_raw = hc.load_raw_images(image_paths)
    print("Loaded RAW data")

    # Check that 10 of each image were found - if not, skip ahead to the next folder
    if len(images_raw) < 3*nr_replicates:
        continue

    # Load EXIF data
    water_exif = hc.load_exif(image_paths[0])

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

    # Reshape the central images to lists
    data_RGBG = images_RGBG.reshape(3, nr_replicates, 4, -1)

    # Divide by the spectral bandwidths to normalise to ADU nm^-1
    data_RGBG /= effective_bandwidths[:, np.newaxis]

    # Calculate mean values
    mean_per_stack = data_RGBG.mean(axis=-1)

    # Flatten the data into one long list
    means_flattened = mean_per_stack.ravel()

    # Average G and G2
    means_flattened_RGB = hc.convert_RGBG2_to_RGB(means_flattened)
    mean_per_stack_RGB = means_flattened_RGB.reshape(3, nr_replicates, 3)

    # Calculate the mean and uncertainty in RGB
    all_mean = mean_per_stack_RGB.mean(axis=1)
    all_std = mean_per_stack_RGB.std(axis=1)
    uncertainty_relative = 100 * all_std/all_mean
    print("Calculated mean values and uncertainties per image, per channel")

    # Convert to remote sensing reflectances - use every combination of Lu, Lsky, Ld
    water, sky, card = mean_per_stack_RGB
    R_rs = hc.R_RS(water[:, np.newaxis, np.newaxis], sky[np.newaxis, :, np.newaxis], card)
    R_rs = R_rs.reshape(-1, 3)
    print("Calculated remote sensing reflectances")

    # Calculate mean values
    R_rs_mean = R_rs.mean(axis=0)
    R_rs_std = R_rs.std(axis=0)
    uncertainty_relative_R_rs = 100 * R_rs_std/R_rs_mean
    print("Calculated mean values and uncertainties per image, per channel, for R_rs")

    # Calculate band ratios
    R_rs_bandratio = hc.calculate_bandratios(*R_rs.T).T
    R_rs_bandratio_mean = R_rs_bandratio.mean(axis=0)
    R_rs_bandratio_std = R_rs_bandratio.std(axis=0)
    uncertainty_relative_R_rs_bandratio = 100 * R_rs_bandratio_std/R_rs_bandratio_mean
    print("Calculated mean values and uncertainties per image, per channel, for band ratios")

    # Convert to hue angle and FU colour
    R_rs_XYZ = camera.convert_to_XYZ(R_rs)
    R_rs_xy = wa.convert_XYZ_to_xy(R_rs_XYZ)
    R_rs_hue = wa.convert_xy_to_hue_angle(R_rs_xy)
    R_rs_FU = wa.convert_hue_angle_to_ForelUle(R_rs_hue)

    # Calculate statistics
    hue_std = R_rs_hue.std(axis=0)
    FU_std = R_rs_FU.std(axis=0)
    print("Calculated mean values and uncertainties per image, per channel, for hue angle/Forel-Ule")

    # Create a timestamp from EXIF (assume time zone UTC+2)
    UTC = hc.UTC_timestamp(water_exif, data_path)

    # Put all data into a single array
    data = [[UTC.timestamp(), UTC.isoformat(), *np.ravel(uncertainty_relative), *np.ravel(uncertainty_relative_R_rs), *np.ravel(uncertainty_relative_R_rs_bandratio), hue_std, FU_std]]
    header = ["UTC", "UTC (ISO)"] + hc.extend_keys_to_RGB(["Lu", "Lsky", "Ld", "R_rs"]) + [f"R_rs ({c})" for c in hc.bandratio_labels + ["hue", "FU"]]
    data_table = hc.table.Table(rows=data, names=header)

    # Write the result to file
    saveto = data_path.with_name(data_path.stem + "_replicates.csv")
    data_table.write(saveto, format="ascii.fast_csv")
