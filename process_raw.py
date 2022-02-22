"""
Process three RAW images (water, sky, grey card) to calculate the remote sensing reflectance R_rs.

Requires the following SPECTACLE calibrations:
    * Metadata
    * Bias
    * Flat-field
    * Spectral response

Command-line inputs:
    * SPECTACLE calibration folder
    * Any number of folders containing data

Example:
    %run process_raw.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ water-colour-data/Balaton*
    %run process_raw.py C:/Users/Burggraaff/SPECTACLE_data/Samsung_Galaxy_S8/ water-colour-data/Balaton*
    %run process_raw.py C:/Users/Burggraaff/SPECTACLE_data/iPhone_6S/ water-colour-data/Balaton* water-colour-data/NZ/* water-colour-data/Switzerland-Oli4/*
"""

import numpy as np
np.set_printoptions(precision=2)

from sys import argv
from spectacle import io, load_camera

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

    # Select the central pixels
    central_slice = camera.central_slice(100, 100)
    images_central_slices = images_flatfield_corrected[central_slice]

    # Demosaick the data
    images_RGBG = camera.demosaick(images_central_slices, selection=central_slice)
    print("Demosaicked")

    # Combined histograms of different data reduction steps
    all_data = [images_jpeg, images_raw, images_bias_corrected, images_flatfield_corrected, images_RGBG]
    water_all, sky_all, card_all = [[data_array[j] for data_array in all_data] for j in range(3)]

    plot.histogram_raw(water_all, sky_all, card_all, camera=camera, saveto=data_path/"statistics_raw.pdf")

    # Reshape the central images to lists
    data_RGBG = images_RGBG.reshape(3, 4, -1)

    # Divide by the spectral bandwidths to normalise to ADU nm^-1
    data_RGBG /= effective_bandwidths[:, np.newaxis]

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
    water_mean_RGB, sky_mean_RGB, card_mean_RGB = hc.split_combined_radiances(all_mean_RGB)

    # Add Rref to covariance matrix
    all_covariance_RGB_Rref = hc.add_Rref_to_covariance(all_covariance_RGB)

    # Calculate Ed from Ld
    Ld_covariance_RGB = all_covariance_RGB[-3:, -3:]  # Take only the Ld-Ld elements from the covariance matrix
    Ed = hc.convert_Ld_to_Ed(card_mean_RGB)
    Ed_covariance = hc.convert_Ld_to_Ed_covariance(Ld_covariance_RGB, Ed)

    # Convert to remote sensing reflectances
    R_rs = hc.R_RS(water_mean_RGB, sky_mean_RGB, card_mean_RGB)
    R_rs_covariance = hc.R_rs_covariance(all_covariance_RGB_Rref, R_rs, card_mean_RGB)
    print("Calculated remote sensing reflectances")

    # Derive naive uncertainty and correlation from covariance
    R_rs_uncertainty = stats.uncertainty_from_covariance(R_rs_covariance)  # Uncertainty per band, ignoring covariance


    # HydroColor
    for reflectance, reflectance_uncertainty, c in zip(R_rs, R_rs_uncertainty, "RGB"):
        print(f"R_rs({c}) = ({reflectance:.3f} +- {reflectance_uncertainty:.3f}) sr^-1   ({100*reflectance_uncertainty/reflectance:.0f}% uncertainty)")

    # Plot the result
    plot.plot_R_rs_RGB(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_uncertainty)

    # Calculate band ratios
    bandratios = hc.calculate_bandratios(*R_rs)
    bandratios_covariance = hc.calculate_bandratios_covariance(*R_rs, R_rs_covariance)
    hc.print_bandratios(bandratios, bandratios_covariance)

    # WACODI

    # Convert RGB to XYZ
    R_rs_XYZ = camera.convert_to_XYZ(R_rs)
    R_rs_XYZ_covariance = camera.XYZ_matrix @ R_rs_covariance @ camera.XYZ_matrix.T

    # Calculate xy chromaticity
    R_rs_xy = wa.convert_XYZ_to_xy(R_rs_XYZ)
    R_rs_xy_covariance = wa.convert_XYZ_to_xy_covariance(R_rs_XYZ_covariance, R_rs_XYZ)

    # Calculate correlation, uncertainty
    R_rs_xy_correlation = stats.correlation_from_covariance(R_rs_xy_covariance)
    R_rs_xy_uncertainty = stats.uncertainty_from_covariance(R_rs_xy_covariance)

    print("Converted to xy:", f"xy R_rs = {R_rs_xy} +- {R_rs_xy_uncertainty} (r = {R_rs_xy_correlation[0,1]:.2f})")

    # Plot chromaticity
    plot.plot_xy_on_gamut_covariance(R_rs_xy, R_rs_xy_covariance)

    # Calculate hue angle
    R_rs_hue = wa.convert_xy_to_hue_angle(R_rs_xy)
    R_rs_hue_uncertainty = wa.convert_xy_to_hue_angle_covariance(R_rs_xy_covariance, R_rs_xy)
    print("Calculated hue angles:", f"alpha R_rs = {R_rs_hue:.1f} +- {R_rs_hue_uncertainty:.1f} degrees")

    # Convert to Forel-Ule index
    R_rs_FU = wa.convert_hue_angle_to_ForelUle(R_rs_hue)
    R_rs_FU_range = wa.convert_hue_angle_to_ForelUle_uncertainty(R_rs_hue_uncertainty, R_rs_hue)
    print("Determined Forel-Ule indices:", f"FU R_rs = {R_rs_FU} [{R_rs_FU_range[0]}-{R_rs_FU_range[1]}]")


    # Create a timestamp from EXIF (assume time zone UTC+2)
    UTC = hc.UTC_timestamp(water_exif, data_path)

    # Write the result to file
    saveto = data_path.with_name(data_path.stem + "_raw.csv")
    hc.write_results(saveto, UTC, all_mean_RGB, all_covariance_RGB, Ed, Ed_covariance, R_rs, R_rs_covariance, bandratios, bandratios_covariance, R_rs_xy, R_rs_xy_covariance, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, R_rs_FU_range)
