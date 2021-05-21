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
np.set_printoptions(precision=2)

from sys import argv
from spectacle import io, load_camera
from os import walk
from matplotlib import pyplot as plt

from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from mpl_toolkits.axes_grid1 import make_axes_locatable

def density_scatter(x, y, ax = None, sort = True, bins = 20, **kwargs):
    # https://stackoverflow.com/a/53865762
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    # norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    # cbar = fig.colorbar(cm.ScalarMappable(norm = norm), ax=ax)
    # cbar.ax.set_ylabel('Density')

    return ax

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
effective_bandwidths = camera.spectral_bands

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

        # Divide by the spectral bandwidths to normalise to ADU nm^-1
        water_RGBG /= effective_bandwidths[:, np.newaxis]
        sky_RGBG /= effective_bandwidths[:, np.newaxis]
        card_RGBG /= effective_bandwidths[:, np.newaxis]
        all_RGBG = np.concatenate([water_RGBG, sky_RGBG, card_RGBG])

        # Calculate mean values
        water_mean = water_RGBG.mean(axis=1)
        sky_mean = sky_RGBG.mean(axis=1)
        card_mean = card_RGBG.mean(axis=1)
        all_mean = all_RGBG.mean(axis=1)
        print("Calculated mean values per channel")

        # Calculate covariance, correlation matrices for the combined radiances
        all_covariance = np.cov(all_RGBG)
        all_correlation = np.corrcoef(all_RGBG)
        not_diagonal = ~np.eye(12, dtype=bool)  # Off-diagonal elements
        max_corr = np.nanmax(all_correlation[not_diagonal])
        print(f"Calculated covariance and correlation matrices. Maximum off-diagonal correlation r = {max_corr:.2f}")

        # Plot correlation coefficients
        kwargs = {"cmap": plt.cm.get_cmap("cividis", 10), "s": 5, "rasterized": True}

        fig, axs = plt.subplots(ncols=3, figsize=(7,3), dpi=600)

        divider = make_axes_locatable(axs[0])
        cax = divider.append_axes('bottom', size='10%', pad=0.3)
        im = axs[0].imshow(all_correlation, extent=(0,12,12,0), cmap=plt.cm.get_cmap("cividis", 10), vmin=0, vmax=1, origin="lower")
        fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.arange(0,1.1,0.2), label="Pearson $r$")

        axs[0].set_xticks(np.arange(0,13,4))
        xtick_offset = " "*10
        axs[0].set_xticklabels([f"{xtick_offset}$L_u$", f"{xtick_offset}$L_s$", f"{xtick_offset}$L_d$", ""])
        axs[0].set_yticks(np.arange(0,13,4))
        axs[0].set_yticklabels(["\n\n$L_d$", "\n\n$L_s$", "\n\n$L_u$", ""])

        # twin.set_xticks(np.arange(0.5,12))
        # twin.set_xticklabels(["R", "G", "B", "G$_2$", "R", "G", "B", "G$_2$", "R", "G", "B", "G$_2$"], fontsize="small")

        density_scatter(all_RGBG[4], all_RGBG[5], ax=axs[1], **kwargs)
        density_scatter(all_RGBG[1], all_RGBG[9], ax=axs[2], **kwargs)

        axs[1].set_xlabel("$L_s$ (R) [a.u.]")
        axs[1].set_ylabel("$L_s$ (G) [a.u.]")
        axs[1].set_title("$r =" + f"{all_correlation[4,5]:.2f}" + "$")

        axs[2].set_xlabel("$L_u$ (G) [a.u.]")
        axs[2].set_ylabel("$L_d$ (G) [a.u.]")
        axs[2].set_title("$r =" + f"{all_correlation[1,9]:.2f}" + "$")

        for ax in axs[1:]:
            ax.set_aspect("equal")
            ax.grid(ls="--", c="0.5", alpha=0.5)

        plt.subplots_adjust(wspace=0.5)
        plt.savefig("corr.pdf", bbox_inches="tight")
        plt.close()

        # Average G and G2
        M_RGBG2_to_RGB = np.array([[1, 0  , 0, 0  ],
                                   [0, 0.5, 0, 0.5],
                                   [0, 0  , 1, 0  ]])
        M_RGBG2_to_RGB_all_L = hc.block_diag(*[M_RGBG2_to_RGB]*3)  # Repeat three times along the diagonal, 0 elsewhere

        all_mean_RGB = M_RGBG2_to_RGB_all_L @ all_mean
        all_covariance_RGB = M_RGBG2_to_RGB_all_L @ all_covariance @ M_RGBG2_to_RGB_all_L.T
        all_correlation_RGB = hc.correlation_from_covariance(all_covariance_RGB)
        all_covariance_RGB_Rref = hc.add_Rref_to_covariance(all_covariance_RGB)

        water_mean_RGB, sky_mean_RGB, card_mean_RGB = hc.split_combined_radiances(all_mean_RGB)

        # Calculate Ed from Ld
        Ld_covariance_RGB = all_covariance_RGB[-3:, -3:]  # Take only the Ld-Ld elements from the covariance matrix
        Ed = hc.convert_Ld_to_Ed(card_mean_RGB)
        Ed_covariance = hc.convert_Ld_to_Ed_covariance(Ld_covariance_RGB, Ed)

        # Convert to remote sensing reflectances
        R_rs = hc.R_RS(water_mean_RGB, sky_mean_RGB, card_mean_RGB)
        R_rs_covariance = hc.R_rs_covariance(all_covariance_RGB_Rref, R_rs, card_mean_RGB)
        print("Calculated remote sensing reflectances")

        # Derive naive uncertainty and correlation from covariance
        R_rs_uncertainty = np.sqrt(np.diag(R_rs_covariance))  # Uncertainty per band, ignoring covariance
        R_rs_correlation = hc.correlation_from_covariance(R_rs_covariance)

        # HydroColor
        for reflectance, reflectance_uncertainty, c in zip(R_rs, R_rs_uncertainty, "RGB"):
            print(f"R_rs({c}) = ({reflectance:.3f} +- {reflectance_uncertainty:.3f}) sr^-1   ({100*reflectance_uncertainty/reflectance:.0f}% uncertainty)")

        # Plot the result
        hc.plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_uncertainty)

        # Calculate band ratios
        beta = R_rs[1] / R_rs[2]  # G/B
        rho = R_rs[1] / R_rs[0]  # G/R
        bandratios = np.array([rho, beta])

        bandratios_J = np.array([[-rho/R_rs[0], 1/R_rs[0], 0            ],
                                 [0           , 1/R_rs[2], -beta/R_rs[2]]])

        bandratios_covariance = bandratios_J @ R_rs_covariance @ bandratios_J.T
        bandratios_uncertainty = np.sqrt(np.diag(bandratios_covariance))
        bandratios_correlation = hc.correlation_from_covariance(bandratios_covariance)
        print(f"Calculated average band ratios: R_rs(G)/R_rs(R) = {bandratios[0]:.2f} +- {bandratios_uncertainty[0]:.2f}    R_rs(G)/R_rs(B) = {bandratios[1]:.2f} +- {bandratios_uncertainty[1]:.2f}    (correlation r = {bandratios_correlation[0,1]:.2f})")

        # WACODI

        # Convert RGB to XYZ
        water_XYZ, sky_XYZ, card_XYZ, R_rs_XYZ = camera.convert_to_XYZ(water_mean_RGB, sky_mean_RGB, card_mean_RGB, R_rs)
        R_rs_XYZ_covariance = camera.XYZ_matrix @ R_rs_covariance @ camera.XYZ_matrix.T

        radiance_RGB_to_XYZ = hc.block_diag(*[camera.XYZ_matrix]*3)
        all_mean_XYZ = radiance_RGB_to_XYZ @ all_mean_RGB
        all_mean_XYZ_covariance = radiance_RGB_to_XYZ @ all_covariance_RGB @ radiance_RGB_to_XYZ.T

        # Calculate xy chromaticity
        water_xy, sky_xy, card_xy, R_rs_xy = wa.convert_XYZ_to_xy(water_XYZ, sky_XYZ, card_XYZ, R_rs_XYZ)
        water_xy_covariance = wa.convert_XYZ_to_xy_covariance(all_mean_XYZ_covariance[:3,:3], water_XYZ)
        R_rs_xy_covariance = wa.convert_XYZ_to_xy_covariance(R_rs_XYZ_covariance, R_rs_XYZ)

        # Calculate correlation
        R_rs_xy_correlation = hc.correlation_from_covariance(R_rs_xy_covariance)
        water_xy_correlation = hc.correlation_from_covariance(water_xy_covariance)

        print("Converted to xy:", f"xy R_rs = {R_rs_xy} +- {np.sqrt(np.diag(R_rs_xy_covariance))} (r = {R_rs_xy_correlation[0,1]:.2f})", f"xy  L_u = {water_xy} +- {np.sqrt(np.diag(water_xy_covariance))} (r = {water_xy_correlation[0,1]:.2f})", sep="\n")

        # Plot chromaticity
        wa.plot_xy_on_gamut_covariance(R_rs_xy, R_rs_xy_covariance)

        # Calculate hue angle
        water_hue, R_rs_hue = wa.convert_xy_to_hue_angle(water_xy, R_rs_xy)
        water_hue_uncertainty = wa.convert_xy_to_hue_angle_covariance(water_xy_covariance, water_xy)
        R_rs_hue_uncertainty = wa.convert_xy_to_hue_angle_covariance(R_rs_xy_covariance, R_rs_xy)
        print("Calculated hue angles:", f"alpha R_rs = {R_rs_hue:.1f} +- {R_rs_hue_uncertainty:.1f} degrees", f"alpha  L_u = {water_hue:.1f} +- {water_hue_uncertainty:.1f} degrees", sep="\n")

        # Create a timestamp from EXIF (assume time zone UTC+2)
        # Time zone: UTC+2 for Balaton data, UTC for NZ data
        if folder_main.stem == "NZ":
            UTC = hc.UTC_timestamp(water_exif, conversion_to_utc=hc.timedelta(hours=0))
        else:
            UTC = hc.UTC_timestamp(water_exif)

        # Write the result to file
        saveto = data_path.with_name(data_path.stem + "_raw.csv")
        hc.write_results(saveto, UTC, all_mean_RGB, all_covariance_RGB, Ed, Ed_covariance, R_rs, R_rs_covariance)
