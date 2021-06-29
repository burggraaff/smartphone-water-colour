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
np.set_printoptions(precision=2)

from sys import argv
from spectacle import io, load_camera
from os import walk

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
        image_paths = hc.generate_paths(data_path, ".JPG")
        images_jpeg = hc.load_jpeg_images(image_paths)
        print("Loaded JPEG data")

        # Load EXIF data
        water_exif = hc.load_exif(image_paths[0])

        # Select the central 100x100 pixels
        central_x, central_y = images_jpeg[0].shape[0]//2, images_jpeg[0].shape[1]//2
        box_size = 100
        central_slice = np.s_[central_x-box_size:central_x+box_size+1, central_y-box_size:central_y+box_size+1]
        images_central_slices = [image[central_slice] for image in images_jpeg]
        print(f"Selected central {2*box_size}x{2*box_size} pixels")

        # Combined histograms of different data reduction steps
        all_data = [images_jpeg, images_central_slices]
        water_all, sky_all, card_all = [[data_array[j] for data_array in all_data] for j in range(3)]

        plot.histogram_jpeg(water_all, sky_all, card_all, saveto=data_path/"statistics_jpeg.pdf")

        # Reshape the central images to lists
        # NB do not replace this with .reshape(3, -1) because that mixes channels
        data_RGB = np.array([image.reshape(-1, 3).T for image in images_central_slices])

        # Divide by the spectral bandwidths to normalise to ADU nm^-1
        data_RGB = data_RGB.astype(np.float64)
        data_RGB /= effective_bandwidths[:, np.newaxis]

        # Flatten the data into one long list
        data_all = data_RGB.reshape(9, -1)

        # Calculate mean values
        all_mean = data_all.mean(axis=-1)
        print("Calculated mean values per channel")

        # Calculate covariance, correlation matrices for the combined radiances
        all_covariance_RGB = np.cov(data_all)
        max_correlation = stats.max_correlation_in_covariance_matrix(all_covariance_RGB)
        print(f"Calculated covariance and correlation matrices. Maximum off-diagonal correlation r = {max_correlation:.2f}")

        # Plot correlation coefficients
        plot.plot_correlation_matrix_radiance(all_covariance_RGB, x1=data_all[3], y1=data_all[4], x1label="$L_s$ (R) [a.u.]", y1label="$L_s$ (G) [a.u.]", x2=data_all[1], y2=data_all[7], x2label="$L_u$ (G) [a.u.]", y2label="$L_d$ (G) [a.u.]", saveto=data_path/"correlation_jpeg.pdf")

        # Add Rref to covariance matrix
        all_covariance_RGB_Rref = hc.add_Rref_to_covariance(all_covariance_RGB)

        # Calculate Ed from Ld
        water_mean, sky_mean, card_mean = hc.split_combined_radiances(all_mean)
        Ld_covariance_RGB = all_covariance_RGB[-3:, -3:]  # Take only the Ld-Ld elements from the covariance matrix
        Ed = hc.convert_Ld_to_Ed(card_mean)
        Ed_covariance = hc.convert_Ld_to_Ed_covariance(Ld_covariance_RGB, Ed)

        # Convert to remote sensing reflectances
        R_rs = hc.R_RS(water_mean, sky_mean, card_mean)
        R_rs_covariance = hc.R_rs_covariance(all_covariance_RGB_Rref, R_rs, card_mean)
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

        bandratios_uncertainty = stats.uncertainty_from_covariance(bandratios_covariance)
        bandratios_correlation = stats.correlation_from_covariance(bandratios_covariance)
        print(f"Calculated average band ratios: R_rs (G/R) = {bandratios[0]:.2f} +- {bandratios_uncertainty[0]:.2f}    R_rs (G/B) = {bandratios[1]:.2f} +- {bandratios_uncertainty[1]:.2f}    (correlation r = {bandratios_correlation[0,1]:.2f})")


        # WACODI

        # Convert RGB to XYZ
        water_XYZ, sky_XYZ, card_XYZ, R_rs_XYZ = wa.convert_to_XYZ(wa.M_sRGB_to_XYZ_E, water_mean, sky_mean, card_mean, R_rs)
        R_rs_XYZ_covariance = wa.M_sRGB_to_XYZ_E @ R_rs_covariance @ wa.M_sRGB_to_XYZ_E.T

        radiance_RGB_to_XYZ = hc.block_diag(*[wa.M_sRGB_to_XYZ_E.T]*3)
        all_mean_XYZ = radiance_RGB_to_XYZ @ all_mean
        all_mean_XYZ_covariance = radiance_RGB_to_XYZ @ all_covariance_RGB @ radiance_RGB_to_XYZ.T

        # Calculate xy chromaticity
        water_xy, sky_xy, card_xy, R_rs_xy = wa.convert_XYZ_to_xy(water_XYZ, sky_XYZ, card_XYZ, R_rs_XYZ)
        water_xy_covariance = wa.convert_XYZ_to_xy_covariance(all_mean_XYZ_covariance[:3,:3], water_XYZ)
        R_rs_xy_covariance = wa.convert_XYZ_to_xy_covariance(R_rs_XYZ_covariance, R_rs_XYZ)

        # Calculate correlation, uncertainty
        R_rs_xy_correlation = stats.correlation_from_covariance(R_rs_xy_covariance)
        R_rs_xy_uncertainty = stats.uncertainty_from_covariance(R_rs_xy_covariance)
        water_xy_correlation = stats.correlation_from_covariance(water_xy_covariance)
        water_xy_uncertainty = stats.uncertainty_from_covariance(water_xy_covariance)

        print("Converted to xy:", f"xy R_rs = {R_rs_xy} +- {R_rs_xy_uncertainty} (r = {R_rs_xy_correlation[0,1]:.2f})", f"xy L_u  = {water_xy} +- {water_xy_uncertainty} (r = {water_xy_correlation[0,1]:.2f})", sep="\n")

        # Plot chromaticity
        plot.plot_xy_on_gamut_covariance(R_rs_xy, R_rs_xy_covariance)

        # Calculate hue angle
        water_hue, R_rs_hue = wa.convert_xy_to_hue_angle(water_xy, R_rs_xy)
        water_hue_uncertainty = wa.convert_xy_to_hue_angle_covariance(water_xy_covariance, water_xy)
        R_rs_hue_uncertainty = wa.convert_xy_to_hue_angle_covariance(R_rs_xy_covariance, R_rs_xy)
        print("Calculated hue angles:", f"alpha R_rs = {R_rs_hue:.1f} +- {R_rs_hue_uncertainty:.1f} degrees", f"alpha L_u  = {water_hue:.1f} +- {water_hue_uncertainty:.1f} degrees", sep="\n")

        # Convert to Forel-Ule index
        water_FU, R_rs_FU = wa.convert_hue_angle_to_ForelUle([water_hue, R_rs_hue])
        water_FU_range = wa.convert_hue_angle_to_ForelUle_uncertainty(water_hue_uncertainty, water_hue)
        R_rs_FU_range = wa.convert_hue_angle_to_ForelUle_uncertainty(R_rs_hue_uncertainty, R_rs_hue)
        print("Determined Forel-Ule indices:", f"FU R_rs = {R_rs_FU} [{R_rs_FU_range[0]}-{R_rs_FU_range[1]}]", f"FU L_u  = {water_FU} [{water_FU_range[0]}-{water_FU_range[1]}]", sep="\n")


        # Create a timestamp from EXIF (assume time zone UTC+2)
        # Time zone: UTC+2 for Balaton data, UTC for NZ data
        if folder_main.stem == "NZ":
            UTC = hc.UTC_timestamp(water_exif, conversion_to_utc=hc.timedelta(hours=0))
        else:
            UTC = hc.UTC_timestamp(water_exif)

        # Write the result to file
        saveto = data_path.with_name(data_path.stem + "_jpeg.csv")
        hc.write_results(saveto, UTC, all_mean, all_covariance_RGB, Ed, Ed_covariance, R_rs, R_rs_covariance, bandratios, bandratios_covariance, R_rs_xy, R_rs_xy_covariance, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, R_rs_FU_range)
