"""
Module with functions etc for HydroColor
"""

from spectacle import io, analyse, calibrate, spectral
from spectacle.general import RMS
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from astropy import table
from scipy.stats import linregress

colours = ["R", "G", "B", "G2"]  # Smartphone bands
plot_colours = [[213/255,94/255,0], [0,158/255,115/255], [0/255,114/255,178/255], [0,158/255,115/255]]  # Plot colours from Okabe-Ito

def R_RS(L_u, L_s, L_d, rho=0.028, R_ref=0.18):
    """
    Calculate the remote sensing reflectance (R_rs) from upwelling radiance L_u,
    sky radiance L_s, downwelling radiance L_d.
    Additional parameters are surface reflectivity rho (default 0.028), grey card
    reflectance R_ref (0.18).
    L_u, L_s, L_d can be NumPy arrays.
    """
    return (L_u - rho * L_s) / ((np.pi / R_ref) * L_d)


def R_RS_error(L_u, L_s, L_d, L_u_err, L_s_err, L_d_err, rho=0.028, R_ref=0.18):
    """
    Calculate the uncertainty in R_rs from the uncertainty in L_u, L_s, L_d.
    Note this does NOT account for uncertainty in R_ref nor covariance.
    """
    # Calculate squared errors individually
    R_rs_err_water = L_u_err**2 * ((0.18/np.pi) * L_d**-1)**2
    R_rs_err_sky = L_s_err**2 * ((0.18/np.pi) * 0.028 * L_d**-1)**2
    R_rs_err_card = L_d_err**2 * ((0.18/np.pi) * (L_u - 0.028 * L_s) * L_d**-2)**2

    R_rs_err = np.sqrt(R_rs_err_water + R_rs_err_sky + R_rs_err_card)
    return R_rs_err


def R_rs_covariance(L_Rref_covariance, R_rs, L_d, rho=0.028, R_ref=0.18):
    """
    Propagate the covariance in radiance and R_ref into a covariance matrix
    for R_rs. Automatically determine the number of bands and return an
    appropriately sized matrix.
    """
    # Determine the number of bands and create an appropriate identity matrix
    nr_bands = len(R_rs)
    I = np.eye(nr_bands)

    # Calculate the four parts of the Jacobian matrix
    J1 = 1/np.pi * R_ref * I * (1/L_d)
    J2 = -1/np.pi * R_ref * rho * I * (1/L_d)
    J3 = -1 * I * (R_rs / L_d)
    JR = R_rs[:,np.newaxis] / R_ref

    # Combine the parts of the Jacobian
    J = np.concatenate([J1, J2, J3, JR], axis=1)

    # Propagate the individual covariances
    R_rs_cov = J @ L_Rref_covariance @ J.T

    return R_rs_cov


def data_type_RGB(filename):
    """
    Find out if a given filename has RAW, JPEG, or linearised JPEG data.
    """
    name = filename.stem
    if "raw" in name:
        return "RAW"
    elif "jpeg" in name:
        if "linear" in name:
            return "JPEG (Linear)"
        else:
            return "JPEG"
    else:
        raise ValueError(f"File `{filename}` does not match known patterns ('raw', 'jpeg', 'jpeg_linear').")


def generate_paths(data_path, extension=".dng"):
    """
    Generate the paths to the water, sky, and greycard images
    """
    paths = [data_path/(photo + extension) for photo in ("water", "sky", "greycard")]
    return paths


def load_raw_images(*filenames):
    raw_images = [io.load_raw_image(filename) for filename in filenames]
    return raw_images


def load_jpeg_images(*filenames):
    jpg_images = [io.load_jpg_image(filename) for filename in filenames]
    return jpg_images


def load_exif(*filenames):
    exif = [io.load_exif(filename) for filename in filenames]
    return exif


def load_raw_thumbnails(*filenames):
    thumbnails = [io.load_raw_image_postprocessed(filename, half_size=True, user_flip=0) for filename in filenames]
    return thumbnails


box_size = 100
def central_slice_jpeg(*images, size=box_size):
    central_x, central_y = images[0].shape[0]//2, images[0].shape[1]//2
    central_slice = np.s_[central_x-size:central_x+size+1, central_y-size:central_y+size+1, :]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {2*size}x{2*size} pixels in the JPEG data")

    return images_cut


def central_slice_raw(*images, size=box_size):
    half_size = size//2

    central_x, central_y = images[0].shape[1]//2, images[0].shape[2]//2
    central_slice = np.s_[:, central_x-half_size:central_x+half_size+1, central_y-half_size:central_y+half_size+1]

    images_cut = [image[central_slice] for image in images]
    print(f"Selected central {size}x{size} pixels in the RAW data")

    return images_cut


def histogram_raw(water_data, sky_data, card_data, saveto, camera=None):
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    for ax_col, water, sky, card in zip(axs[:,1:].T, water_data[1:], sky_data[1:], card_data[1:]):
        data_combined = np.ravel([water, sky, card])
        xmin, xmax = analyse.symmetric_percentiles(data_combined, percent=0.001)
        bins = np.linspace(xmin, xmax, 150)

        for ax, data in zip(ax_col, [water, sky, card]):
            ax.hist(data.ravel(), bins=bins, color="k")

            if camera is not None:
                # If the data are already in RGBG format, use them
                if len(data.shape) == 3:
                    data_RGBG = data
                # If not, then demosaick the data
                else:
                    data_RGBG = camera.demosaick(data)

                data_RGB = RGBG2_to_RGB(data_RGBG)[0]
                for j, c in enumerate(plot_colours[:3]):
                    ax.hist(data_RGB[j].ravel(), bins=bins, color=c, histtype="step")

            ax.set_xlim(xmin, xmax)
            ax.grid(True, ls="--", alpha=0.7)

    for ax, img in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(img)
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axs.ravel():
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():
        ax.tick_params(bottom=False, labelbottom=False)
    for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
        ax.set_ylabel(label)
    for ax, title in zip(axs[0], ["Image", "Raw", "Bias-corrected", "Flat-fielded", "Central slice"]):
        ax.set_title(title)

    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def histogram_jpeg(water_data, sky_data, card_data, saveto, normalisation=255):
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    for ax_col, water, sky, card in zip(axs.T[1:], water_data, sky_data, card_data):
        bins = np.linspace(0, normalisation, 50)

        for ax, data in zip(ax_col, [water, sky, card]):
            ax.hist(data.ravel(), bins=bins, color="k")
            for j, c in enumerate(plot_colours[:3]):
                ax.hist(data[...,j].ravel(), bins=bins, color=c, histtype="step")
            ax.set_xlim(0, normalisation)
            ax.grid(True, ls="--", alpha=0.7)

    for ax, image in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(image)
        ax.tick_params(bottom=False, labelbottom=False)

    for ax in axs.ravel():
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():
        ax.tick_params(bottom=False, labelbottom=False)
    for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):
        ax.set_ylabel(label)
    for ax, title in zip(axs[0], ["Image", "JPEG (full)", "Central slice"]):
        ax.set_title(title)

    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def RGBG2_to_RGB(*arrays):
    RGB_lists = [[array[0].ravel(), array[1::2].ravel(), array[2].ravel()] for array in arrays]
    return RGB_lists


def effective_wavelength(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]
    RGB_wavelengths = spectral.effective_wavelengths(wavelengths, RGB_responses)

    return RGB_wavelengths


def effective_bandwidth(calibration_folder):
    spectral_response = calibrate.load_spectral_response(calibration_folder)
    wavelengths = spectral_response[0]
    RGB_responses = spectral_response[1:4]

    RGB_responses_normalised = RGB_responses / RGB_responses.max(axis=1)[:,np.newaxis]
    effective_bandwidths = np.trapz(RGB_responses_normalised, x=wavelengths, axis=1)

    return effective_bandwidths


def _loop_RGBG2_or_RGB(data1, data2, param):
    """
    Return RGBG2 if each data table has keys corresponding to each of those bands.
    Otherwise, return only RGB.
    """
    loop_colours = colours
    # We can't do `data1.keys() + data2.keys()` because we must check each individually
    if all("G2" not in key for key in data1.keys()) or all("G2" not in key for key in data2.keys()):
        loop_colours = colours[:3]

    return loop_colours


# Pearson r correlation coefficient
correlation = lambda x, y: np.corrcoef(x, y)[0, 1]


def MAD(x, y):
    """
    Median absolute deviation (sometimes MAE) between data sets x and y.
    """
    return np.nanmedian(np.abs(y-x))


def MAPD(x, y):
    """
    Median absolute percentage deviation (sometimes MAPE) between data sets x and y.
    Normalised relative to (x+y)/2 rather than just x or y.
    Expressed in %, so already multiplied by 100.
    """
    normalisation = (x+y)/2  # Normalisation factors
    MAD_relative = np.nanmedian(np.abs((y-x)/normalisation))
    MAD_percent = MAD_relative * 100.
    return MAD_percent


def ravel_table(data, key, loop_keys):
    """
    Apply np.ravel to a number of columns, e.g. to combine Rrs R, Rrs G, Rrs B
    into one array for all Rrs.
    data is the input table.
    key is the fixed key, e.g. "Rrs".
    loop_keys is an interable list of keys to loop over, e.g. "RGB"
    """
    return np.ravel([data[key.format(c=c)] for c in loop_keys])


def statistic_RGB(func, data1, data2, xdatalabel, ydatalabel):
    """
    Calculate a statistic (e.g. MAD, MAPD, RMSE) in a given parameter `param`,
    e.g. Rrs, between two Astropy data tables. Assumes the same key structure
    in each table, namely `{param} {c}` where c is R, G, B, and optionally G2.

    Returns the statistic overall and per band.
    """
    loop_colours = _loop_RGBG2_or_RGB(data1, data2, xdatalabel)

    stat_RGB = np.array([func(data1[xdatalabel.format(c=c)], data2[ydatalabel.format(c=c)]) for c in loop_colours])
    data1_combined = ravel_table(data1, xdatalabel, loop_colours)
    data2_combined = ravel_table(data2, ydatalabel, loop_colours)
    stat_all = func(data1_combined, data2_combined)

    return stat_all, stat_RGB


def plot_R_rs(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err, saveto=None):
    plt.figure(figsize=(4,3))
    for j, c in enumerate(plot_colours[:3]):
        plt.errorbar(RGB_wavelengths[j], R_rs[j], xerr=effective_bandwidths[j]/2, yerr=R_rs_err[j], c=c, fmt="o")
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
    plt.xlim(390, 700)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(ls="--")
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def residual_table(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None):
    """
    Calculate the column-wise RGB residuals between two tables for given labels
    `xdatalabel` and `ydatalabel` (e.g. "Rrs {c}"). If errors are included,
    propagate them (sum of squares).
    Returns a new table with differences.
    """
    # Use a copy of x to store the residuals in
    result = x.copy()

    # Remove columns that are in x but not in y
    # For example, G2 if you are comparing RAW and JPEG
    keys_not_overlapping = [key for key in result.keys() if key not in y.keys()]
    result.remove_columns(keys_not_overlapping)

    # Loop over the keys and update them to include x-y instead of just x
    for c in _loop_RGBG2_or_RGB(x, y, xdatalabel):
        result[xdatalabel.format(c=c)] = y[ydatalabel.format(c=c)] - x[xdatalabel.format(c=c)]
        if xerrlabel and yerrlabel:
            result[xerrlabel.format(c=c)] = np.sqrt(x[xerrlabel.format(c=c)]**2 + y[yerrlabel.format(c=c)]**2)

    return result


def _correlation_plot_errorbars(ax, x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, setmax=True, equal_aspect=False):
    """
    Plot data into a correlation plot.
    Helper function.
    """
    xmax = 0.  # Maximum on x axis
    ymax = 0.  # Maximum on y axis

    # Loop over the RGBG2 bands and plot the relevant data points
    for c, pc in zip(colours, plot_colours):
        try:
            xdata = x[xdatalabel.format(c=c)]
            ydata = y[ydatalabel.format(c=c)]
        except KeyError as e:
            # If a key was not found, print which key it was, and continue
            # For JPEG data, we want to skip the non-existent key
            # If something else is wrong, the user is alerted
            print("The following KeyError was raised but will be ignored:", e)
            continue
        try:
            xerr = x[xerrlabel.format(c=c)]
        except (KeyError, AttributeError):
            xerr = None
        try:
            yerr = y[yerrlabel.format(c=c)]
        except (KeyError, AttributeError):
            yerr = None
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, color=pc, fmt="o")
        xmax = max(xmax, np.nanmax(xdata))
        ymax = max(ymax, np.nanmax(ydata))

    if setmax:
        if equal_aspect:
            xmax = ymax = max(xmax, ymax)
        ax.set_xlim(0, 1.05*xmax)
        ax.set_ylim(0, 1.05*ymax)


def correlation_plot_RGB(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB/RGBG2 versions.
    For example, if `xlabel` == `f"Rrs {c}"` then the columns "Rrs R", "RRs G",
    "Rrs B", and "Rrs G2" (if available) will be used.
    """
    # Create figure
    plt.figure(figsize=(4,4), tight_layout=True)

    # Plot in the one panel
    _correlation_plot_errorbars(plt.gca(), x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel)

    # Plot the x=y line
    plt.plot([-1e6, 1e6], [-1e6, 1e6], c='k', ls="--")

    # Plot settings
    plt.grid(True, ls="--")

    # Get statistics for title
    r_all, r_RGB = statistic_RGB(correlation, x, y, xdatalabel, ydatalabel)
    title = f"$r$ = {r_all:.2f}"
    plt.title(title)

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save, show, close plot
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def correlation_plot_RGB_equal(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB/RGBG2 versions.
    For example, if `xlabel` == `f"Rrs {c}"` then the columns "Rrs R", "Rrs G",
    "Rrs B", and "Rrs G2" (if available) will be used.
    """
    # Calculate residuals
    residuals = residual_table(x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel)

    # Create figure to hold plot
    fig, axs = plt.subplots(figsize=(4,5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3,1]})

    # Plot in both panels
    _correlation_plot_errorbars(axs[0], x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, equal_aspect=True)
    _correlation_plot_errorbars(axs[1], x, residuals, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, setmax=False)

    # Plot the x=y line (top) and horizontal (bottom)
    axs[0].plot([-1e6, 1e6], [-1e6, 1e6], c='k', ls="--")
    axs[1].axhline(0, c='k', ls="--")

    # Plot settings
    for ax in axs:
        ax.grid(True, ls="--")

    # Get statistics for title
    MAD_all, MAD_RGB = statistic_RGB(MAD, x, y, xdatalabel, ydatalabel)
    MAPD_all, MAPD_RGB = statistic_RGB(MAPD, x, y, xdatalabel, ydatalabel)
    r_all, r_RGB = statistic_RGB(correlation, x, y, xdatalabel, ydatalabel)

    title_r = f"$r$ = {r_all:.2f}"
    title_MAD = f"    MAD = {MAD_all:.3f} sr$" + "^{-1}$" + f" ({MAPD_all:.0f}%)"
    title = f"{title_r} {title_MAD}"
    axs[0].set_title(title)

    # Labels
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Difference")
    axs[0].set_ylabel(ylabel)

    # Save, show, close plot
    fig.subplots_adjust(hspace=0.1)
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def correlation_plot_bands(x, y, datalabel="Rrs", quantity="$R_{rs}$", unit="sr$^{-1}$", xlabel="", ylabel="", saveto=None):
    """
    Make a correlation plot between the band ratios/differences for G-B and G-R.
    """
    max_ratio = 0.
    max_diff = 0.
    min_diff = 0.
    fig, axs = plt.subplots(2, 2, sharex="col", sharey="col", figsize=(5,5), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    for j, c in enumerate("BR"):
        label_G = f"{datalabel} G"
        label_G_err = f"{datalabel}_err G"
        label_c = f"{datalabel} {c}"
        label_c_err = f"{datalabel}_err {c}"

        ratio_x = x[label_G]/x[label_c]
        ratio_y = y[label_G]/y[label_c]
        ratio_err_x = ratio_x * np.sqrt(x[label_G_err]**2/x[label_G]**2 + x[label_c_err]**2/x[label_c]**2)
        ratio_err_y = ratio_y * np.sqrt(y[label_G_err]**2/y[label_G]**2 + y[label_c_err]**2/y[label_c]**2)

        diff_x = x[label_G] - x[label_c]
        diff_y = y[label_G] - y[label_c]
        diff_err_x = np.sqrt(x[label_G_err]**2 + x[label_c_err]**2)
        diff_err_y = np.sqrt(y[label_G_err]**2 + y[label_c_err]**2)

        axs[j,0].errorbar(ratio_x, ratio_y, xerr=ratio_err_x, yerr=ratio_err_y, c="k", fmt="o")
        axs[j,1].errorbar(diff_x, diff_y, xerr=diff_err_x, yerr=diff_err_y, c="k", fmt="o")

        max_ratio = max(np.nanmax(ratio_x), np.nanmax(ratio_y), max_ratio)
        max_diff = max(np.nanmax(diff_x), np.nanmax(diff_y), max_diff)
        min_diff = min(np.nanmin(diff_x), np.nanmin(diff_y), min_diff)

        axs[j,0].set_xlim(0, 1.1*max_ratio)
        axs[j,0].set_ylim(0, 1.1*max_ratio)
        axs[j,1].set_xlim(1.1*min_diff, 1.1*max_diff)
        axs[j,1].set_ylim(1.1*min_diff, 1.1*max_diff)

    for ax in axs.ravel():
        ax.grid(True, ls="--")
        ax.plot([-1, 5], [-1, 5], c='k', ls="--")
        ax.set_aspect("equal")
        ax.locator_params(axis="both", nbins=4)

    for ax in axs[:,1]:
        ax.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
        ax.yaxis.set_label_position("right")

    for ax in axs[0]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
        ax.xaxis.set_label_position("top")

    axs[0,0].set_xlabel(f"{xlabel} {quantity} $G / B$")
    axs[0,0].set_ylabel(f"{ylabel}\n{quantity} $G / B$")
    axs[1,0].set_xlabel(f"{xlabel} {quantity} $G / R$")
    axs[1,0].set_ylabel(f"{ylabel}\n{quantity} $G / R$")

    axs[0,1].set_xlabel(f"{xlabel} {quantity} $G - B$ [{unit}]")
    axs[0,1].set_ylabel(f"{ylabel}\n{quantity} $G - B$ [{unit}]")
    axs[1,1].set_xlabel(f"{xlabel} {quantity} $G - R$ [{unit}]")
    axs[1,1].set_ylabel(f"{ylabel}\n{quantity} $G - R$ [{unit}]")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def comparison_histogram(x_table, y_table, param="Rrs {c}", xlabel="", ylabel="", quantity="", saveto=None):
    """
    Make a histogram of the ratio and difference in a given `param` for `x` and `y`
    """
    loop_colours = _loop_RGBG2_or_RGB(x_table, y_table, param)
    x = ravel_table(x_table, param, loop_colours)
    y = ravel_table(y_table, param, loop_colours)

    ratio = y/x
    diff = y-x

    fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(5,2), gridspec_kw={"hspace": 0.1, "wspace": 0.1}, constrained_layout=True)
    axs[0].hist(ratio, bins=10, facecolor="k")
    axs[1].hist(diff, bins=10, facecolor="k")

    fig.suptitle(f"{quantity}: {xlabel} vs {ylabel}")
    axs[0].set_xlabel("Ratio")
    axs[1].set_xlabel("Difference")
    axs[0].set_ylabel("Frequency")

    for ax, q in zip(axs, [ratio, diff]):
        ax.set_title(f"$\mu$ = {np.nanmean(q):.3f}   $\sigma$ = {np.nanstd(q):.3f}")

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def UTC_timestamp(water_exif, conversion_to_utc=timedelta(hours=2)):
    try:
        timestamp = water_exif["EXIF DateTimeOriginal"].values
    except KeyError:
        timestamp = water_exif["Image DateTimeOriginal"].values
    # Convert to ISO format
    timestamp_ISO = timestamp[:4] + "-" + timestamp[5:7] + "-" + timestamp[8:10] + "T" + timestamp[11:]
    UTC = datetime.fromisoformat(timestamp_ISO)
    UTC = UTC - conversion_to_utc

    return UTC


def write_results(saveto, timestamp, water, water_err, sky, sky_err, grey, grey_err, Rrs, Rrs_err, Rref=0.18):
    assert len(water) == len(water_err) == len(sky) == len(sky_err) == len(grey) == len(grey_err) == len(Rrs) == len(Rrs_err), "Not all input arrays have the same length"
    header = ["Lu {c}", "Lu_err {c}", "Lsky {c}", "Lsky_err {c}", "Ld {c}", "Ld_err {c}", "Ed {c}", "Ed_err {c}", "Rrs {c}", "Rrs_err {c}"]
    colours_here = "RGB" if len(water) == 3 else colours
    header_full = [[s.format(c=c) for c in colours_here] for s in header]
    header = ["UTC", "UTC (ISO)"] + [item for sublist in header_full for item in sublist]

    Ed = np.pi / Rref * grey
    Ed_err = np.pi / Rref * grey_err
    data = [[timestamp.timestamp(), timestamp.isoformat(), *water, *water_err, *sky, *sky_err, *grey, *grey_err, *Ed, *Ed_err, *Rrs, *Rrs_err]]

    result = table.Table(rows=data, names=header)

    result.write(saveto, format="ascii.fast_csv")
    print(f"Saved results to `{saveto}`")
