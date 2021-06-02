"""
Module with functions etc for HydroColor
"""

from spectacle import io, analyse, calibrate, spectral
from spectacle.general import RMS
import numpy as np
from matplotlib import pyplot as plt, cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime, timedelta
from astropy import table
from scipy.linalg import block_diag
from scipy.interpolate import interpn

from . import colours
from .statistics import correlation, MAD, MAPD, zeta, SSPB, ravel_table, statistic_RGB, linear_regression, full_statistics_for_title

plot_colours = [[213/255,94/255,0], [0,158/255,115/255], [0/255,114/255,178/255]]  # Plot colours from Okabe-Ito


def correlation_from_covariance(covariance):
    """
    Convert a covariance matrix into a correlation matrix
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    v = np.sqrt(np.diag(covariance))
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def add_Rref_to_covariance(covariance, Rref_uncertainty=0.01):
    """
    Add a column and row for R_ref to a covariance matrix.
    The input Rref_uncertainty is assumed fully uncorrelated
    to the other elements.
    """
    covariance_with_Rref = block_diag(covariance, [Rref_uncertainty**2])

    return covariance_with_Rref


def convert_Ld_to_Ed(Ld, R_ref=0.18):
    """
    Convert downwelling radiance from a grey card (Ld) to downwelling
    irradiance (Ed) using the reference reflectance (R_ref).
    """
    Ed = Ld / R_ref
    return Ed


def convert_Ld_to_Ed_covariance(Ld_covariance, Ed, R_ref=0.18, R_ref_uncertainty=0.01):
    """
    Convert the covariance in downwelling radiance (Ld) and the
    reference reflectance (R_ref) to a covariance in downwelling
    irradiance (Ed).
    """
    nr_bands = len(Ed)  # Number of bands - 3 for RGB, 4 for RGBG2
    total_covariance = add_Rref_to_covariance(Ld_covariance, R_ref_uncertainty)
    J = np.block([np.eye(nr_bands)/R_ref, (-Ed/R_ref)[:,np.newaxis]])  # Jacobian matrix
    Ed_covariance = J @ total_covariance @ J.T

    return Ed_covariance


def split_combined_radiances(radiances):
    """
    For a combined radiance array, e.g. [Lu(R), Lu(G), Lu(B), Ls(R), ..., Ld(G), Ld(B)],
    split it into three separate arrays: [Lu(R), Lu(G), Lu(B)], [Ls(R), ...], ...
    """
    n = len(radiances)//3
    Lu, Ls, Ld = radiances[:n], radiances[n:2*n], radiances[2*n:]
    return Lu, Ls, Ld


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
    J = np.block([J1, J2, J3, JR])

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
    keys_not_overlapping = [key for key in result.keys() if key not in y.keys()]
    result.remove_columns(keys_not_overlapping)

    # Loop over the keys and update them to include x-y instead of just x
    for c in colours:
        result[xdatalabel.format(c=c)] = y[ydatalabel.format(c=c)] - x[xdatalabel.format(c=c)]
        if xerrlabel and yerrlabel:
            result[xerrlabel.format(c=c)] = np.sqrt(x[xerrlabel.format(c=c)]**2 + y[yerrlabel.format(c=c)]**2)

    return result


def _correlation_plot_gridlines(ax=None):
    """
    Add grid lines and the y=x line to a plot.
    """
    # Get the active Axes object
    if ax is None:
        ax = plt.gca()

    ax.plot([-1e6, 1e6], [-1e6, 1e6], c='k', ls="--")  # Diagonal
    ax.grid(True, ls="--")  # Grid lines


def correlation_plot_simple(x, y, xerr=None, yerr=None, xlabel="", ylabel="", ax=None, equal_aspect=False, minzero=False, setmax=True, regression=False, saveto=None):
    """
    Simple correlation plot, no RGB stuff.
    """
    # If no Axes object was given, make a new one
    if ax is None:
        newaxes = True
        plt.figure(figsize=(3,3))
        ax = plt.gca()
    else:
        newaxes = False

    # Plot the data
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, color="k", fmt="o")

    # Set the origin to (0, 0)
    if minzero:
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0)

    # Set the top right corner to include all data
    if setmax:
        xmax = 1.05*np.nanmax(x)
        ymax = 1.05*np.nanmax(y)
        if equal_aspect:
            xmax = ymax = max(xmax, ymax)
        ax.set_xlim(xmax=xmax)
        ax.set_ylim(ymax=ymax)

    # Grid lines and y=x diagonal
    _correlation_plot_gridlines(ax)

    # If wanted, perform a linear regression and plot the result
    if regression:
        params, params_cov, func = linear_regression(x, y, xerr, yerr)
        x_plot = np.array([-1000., 1000.])
        y_plot = func(x_plot)
        ax.plot(x_plot, y_plot, color="k")

    # Get statistics for title
    r = correlation(x, y)
    title = f"$r$ = {r:.2f}"
    ax.set_title(title)

    # Labels for x and y axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save, show, close plot (if a new plot was made)
    if newaxes:
        if saveto:
            plt.savefig(saveto, bbox_inches="tight")
        plt.show()
        plt.close()


def _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, setmax=True, equal_aspect=False):
    """
    Plot data into a correlation plot.
    Helper function.
    """
    xmax = 0.  # Maximum on x axis
    ymax = 0.  # Maximum on y axis

    # Loop over the colour bands and plot the relevant data points
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
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Create figure
    plt.figure(figsize=(4,4), tight_layout=True)

    # Plot in the one panel
    _correlation_plot_errorbars_RGB(plt.gca(), x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel)

    # y=x line and grid lines
    _correlation_plot_gridlines()

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
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Calculate residuals
    residuals = residual_table(x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel)

    # Create figure to hold plot
    fig, axs = plt.subplots(figsize=(4,5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3,1]})

    # Plot in both panels
    _correlation_plot_errorbars_RGB(axs[0], x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, equal_aspect=True)
    _correlation_plot_errorbars_RGB(axs[1], x, residuals, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, setmax=False)

    # Plot the x=y line (top) and horizontal (bottom)
    _correlation_plot_gridlines(axs[0])
    axs[1].axhline(0, c='k', ls="--")
    axs[1].grid(True, ls="--")

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


def correlation_plot_bands(x_GR, y_GR, x_GB, y_GB, x_err_GR=None, y_err_GR=None, x_err_GB=None, y_err_GB=None, quantity="$R_{rs}$", xlabel="", ylabel="", saveto=None):
    """
    Make a correlation plot between the band ratios G/R and G/B.
    """
    fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(4,2), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    correlation_plot_simple(x_GR, y_GR, xerr=x_err_GR, yerr=y_err_GR, ax=axs[0], xlabel=f"{quantity} G/R\n({xlabel})", ylabel=f"{quantity} G/R\n({ylabel})", equal_aspect=True, minzero=False, setmax=True)
    correlation_plot_simple(x_GB, y_GB, xerr=x_err_GB, yerr=y_err_GB, ax=axs[1], xlabel=f"{quantity} G/B\n({xlabel})", ylabel=f"{quantity} G/B\n({ylabel})", equal_aspect=True, minzero=False, setmax=True)

    # Switch ytick labels on the right plot to the right
    axs[1].tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    axs[1].yaxis.set_label_position("right")

    # Calculate statistics
    for ax, x, y in zip (axs, [x_GR, x_GB], [y_GR, y_GB]):
        stats, statistic_text = full_statistics_for_title(x, y)
        ax.set_title(statistic_text, fontdict={"fontsize": "small"})  # Replace old title

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def comparison_histogram(x_table, y_table, param="Rrs {c}", xlabel="", ylabel="", quantity="", saveto=None):
    """
    Make a histogram of the ratio and difference in a given `param` for `x` and `y`
    """
    x = ravel_table(x_table, param, colours)
    y = ravel_table(y_table, param, colours)

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

    return ax


def plot_correlation_matrix_radiance(correlation_matrix, x1, y1, x2, y2, x1label="[a.u.]", y1label="[a.u.]", x2label="[a.u.]", y2label="[a.u.]", saveto=None):
    """
    Plot a given correlation matrix consisting of RGB or RGBG2 radiances.
    """
    # Plot correlation coefficients
    kwargs = {"cmap": plt.cm.get_cmap("cividis", 10), "s": 5, "rasterized": True}

    fig, axs = plt.subplots(ncols=3, figsize=(7,3), dpi=600)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('bottom', size='10%', pad=0.3)
    im = axs[0].imshow(correlation_matrix, extent=(0,1,1,0), cmap=plt.cm.get_cmap("cividis", 5), vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.arange(0,1.1,0.2), label="Pearson $r$")

    ticks = np.linspace(0,1,4)
    axs[0].set_xticks(ticks)
    xtick_offset = " "*10
    axs[0].set_xticklabels([f"{xtick_offset}$L_u$", f"{xtick_offset}$L_s$", f"{xtick_offset}$L_d$", ""])
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels(["\n\n$L_d$", "\n\n$L_s$", "\n\n$L_u$", ""])

    for ax, x, y, xlabel, ylabel in zip(axs[1:], [x1, x2], [y1, y2], [x1label, x2label], [y1label, y2label]):
        density_scatter(x, y, ax=ax, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("$r =" + f"{correlation(x,y):.2f}" + "$")

        ax.set_aspect("equal")
        ax.grid(ls="--", c="0.5", alpha=0.5)

    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()

    plt.subplots_adjust(wspace=0.5)
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
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


def _convert_symmetric_matrix_to_list(sym):
    """
    Convert a symmetric matrix `sym` to a list that contains its
    upper-triangular (including diagonal) elements.
    """
    return sym[np.triu_indices_from(sym)]


def _convert_list_to_symmetric_matrix(symlist):
    """
    Convert a list containing elemens of a symmetric matrix
    (e.g. generated using _convert_symmetric_matrix_to_list) back
    into a matrix.
    """
    # Number of independent elements in symmetric matrix of size nxn is
    # L = n*(n+1)/2
    # Inverted gives n = -0.5 + 0.5*sqrt(1 + 8L)
    nr_columns = int(-0.5 + 0.5*np.sqrt(1 + 8*len(symlist)))

    # Create the array
    arr = np.zeros((nr_columns, nr_columns))
    arr[np.triu_indices(nr_columns)] = symlist  # Add the upper-triangular elements
    arr = arr + arr.T - np.diag(np.diag(arr))  # Add the lower-triangular elements without doubling the diagonal

    return arr


def _generic_header(elements, prefix=""):
    """
    Generate a generic header (list of names) for a list `elements`.
    Optionally use a prefix to identify them.
    """
    header = [f"{prefix}_{j:04d}" for j, ele in enumerate(elements)]
    return header


def write_results(saveto, timestamp, radiances, radiances_covariance, Ed, Ed_covariance, R_rs, R_rs_covariance, band_ratios, band_ratios_covariance, R_rs_xy, R_rs_xy_covariance, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, R_rs_FU_range):
    # assert len(water) == len(water_err) == len(sky) == len(sky_err) == len(grey) == len(grey_err) == len(Rrs) == len(Rrs_err), "Not all input arrays have the same length"

    # Split the covariance matrices out
    radiances_covariance_list = _convert_symmetric_matrix_to_list(radiances_covariance)
    Ed_covariance_list = _convert_symmetric_matrix_to_list(Ed_covariance)
    R_rs_covariance_list = _convert_symmetric_matrix_to_list(R_rs_covariance)
    band_ratios_covariance_list = _convert_symmetric_matrix_to_list(band_ratios_covariance)
    R_rs_xy_covariance_list = _convert_symmetric_matrix_to_list(R_rs_xy_covariance)

    # Headers for the covariance matrices
    radiances_covariance_header = _generic_header(radiances_covariance_list, "cov_L")
    Ed_covariance_header = _generic_header(Ed_covariance_list, "cov_Ed")
    R_rs_covariance_header = _generic_header(R_rs_covariance_list, "cov_R_rs_RGB")
    band_ratios_covariance_header = _generic_header(band_ratios_covariance_list, "cov_band_ratio")
    R_rs_xy_covariance_header = _generic_header(R_rs_xy_covariance_list, "cov_R_rs_xy")

    # Make a header with the relevant items
    header_RGB = ["Lu ({c})", "Lsky ({c})", "Ld ({c})", "Ed ({c})", "R_rs ({c})"]
    bands = "RGB"
    header_RGB_full = [[s.format(c=c) for c in bands] for s in header_RGB]
    header_hue = ["R_rs (G/R)", "R_rs (G/B)", "R_rs (x)", "R_rs (y)", "R_rs (hue)", "R_rs_err (hue)", "R_rs (FU)", "R_rs_min (FU)", "R_rs_max (FU)"]
    header = ["UTC", "UTC (ISO)"] + [item for sublist in header_RGB_full for item in sublist] + header_hue + radiances_covariance_header + Ed_covariance_header + R_rs_covariance_header + band_ratios_covariance_header + R_rs_xy_covariance_header

    # Add the data to a row, and that row to a table
    data = [[timestamp.timestamp(), timestamp.isoformat(), *radiances, *Ed, *R_rs, *band_ratios, *R_rs_xy, R_rs_hue, R_rs_hue_uncertainty, R_rs_FU, *R_rs_FU_range, *radiances_covariance_list, *Ed_covariance_list, *R_rs_covariance_list, *band_ratios_covariance_list, *R_rs_xy_covariance_list]]
    result = table.Table(rows=data, names=header)

    # Write the result to file
    result.write(saveto, format="ascii.fast_csv")
    print(f"Saved results to `{saveto}`")


def _convert_matrix_to_uncertainties_column(covariance_matrices, labels):
    """
    Take a column containing covariance matrices and generate a number of columns
    containing the uncertainties on its diagonal.
    """
    assert len(labels) == len(covariance_matrices[0]), f"Number of labels (len{labels}) does not match matrix dimensionality ({len(covariance_matrices[0])})."
    diagonals = np.array([np.diag(matrix) for matrix in covariance_matrices])
    uncertainties = np.sqrt(diagonals)
    columns = [table.Column(name=label, data=uncertainties[:,j]) for j, label in enumerate(labels)]
    return columns


def read_results(filename):
    """
    Read a results file generated with write_results.
    """
    # Read the file
    data = table.Table.read(filename)

    # Iterate over the different covariance columns and make them into arrays again
    covariance_keys = ["cov_L", "cov_Ed", "cov_R_rs_RGB", "cov_band_ratio", "cov_R_rs_xy"]
    for key_cov in covariance_keys:
        keys = sorted([key for key in data.keys() if key_cov in key])
        # [*row] puts the row data into a list; otherwise the iteration does not work
        covariance_matrices = [_convert_list_to_symmetric_matrix([*row]) for row in data[keys]]

        # Add a new column with these matrices and remove the raw data columns
        data.add_column(table.Column(name=key_cov, data=covariance_matrices))
        data.remove_columns(keys)

    # Iterate over the covariance matrices and calculate simple uncertainties from them
    covariance_keys_split = [np.ravel([[f"L{sub}_err ({c})" for c in colours] for sub in ["u", "sky", "d"]]),
                             [f"Ed_err ({c})" for c in colours],
                             [f"R_rs_err ({c})" for c in colours],
                             [f"R_rs_err ({ratio})" for ratio in ["G/R", "G/B"]],
                             [f"R_rs_err ({c})" for c in "xy"]]
    for key, keys_split in zip(covariance_keys, covariance_keys_split):
        uncertainties = _convert_matrix_to_uncertainties_column(data[key], keys_split)
        data.add_columns(uncertainties)

    return data
