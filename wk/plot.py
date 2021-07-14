"""
Module with common plotting functions
"""
from matplotlib import pyplot as plt, transforms, patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse

import numpy as np
from colorio._tools import plot_flat_gamut

from . import statistics as stats, colours, hydrocolor as hc
from .wacodi import FU_hueangles, compare_FU_matches_from_hue_angle

from spectacle.plot import RGB_OkabeIto, _saveshow, cmaps


# Commonly used unit strings
ADUnmsr = "[ADU nm$^{-1}$ sr$^{-1}$]"
ADUnm = "[ADU nm$^{-1}$]"
Wnmsr = "[W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]"
Wnm = "[W m$^{-2}$ nm$^{-1}$]"
persr = "[sr$^{-1}$]"

# Dictionary mapping keys to LaTeX strings
keys_latex = {"Lu": "$L_u$", "Lsky": "$L_{sky}$", "Ld": "$L_d$", "Ed": "$E_d$", "L": "$L$", "R_rs": "$R_{rs}$"}

# Frontiers column widths
col1 = 85/25.4
col2 = 180/25.4
smallpanel = (2, 1.5)


def _histogram_axis_settings(axs, column_labels):
    """
    Helper function.
    Adjust the x- and y-axis labels on histogram panels.
    """
    for ax in axs.ravel():  # No ticks on the left in any panels
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():  # No ticks on the bottom for the top 2 rows
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axs[:,1:].ravel():  # Grid
        ax.grid(True, ls="--", alpha=0.7)
    for ax, label in zip(axs[:,0], ["Water", "Sky", "Grey card"]):  # Labels on the y-axes
        ax.set_ylabel(label)
    for ax, title in zip(axs[0], column_labels):  # Titles for the columns
        ax.set_title(title)


def _histogram_RGB(data_RGB, ax, **kwargs):
    """
    Helper function.
    Draw line-type histograms of RGB data on an Axes object.
    """
    # Loop over the colour channels
    for j, c in enumerate(RGB_OkabeIto[:3]):
        ax.hist(data_RGB[j].ravel(), color=c, histtype="step", **kwargs)


def plot_three_images(images, axs=None, saveto=None):
    """
    Plot the three images (water, sky, grey card) nicely.
    If `axs` are provided, plot them in those. Otherwise, create
    a new figure.
    """
    # Create a new figure if necessary
    if axs is None:
        newaxes = True
        fig, axs = plt.subplots(ncols=3, figsize=(col1, 4), gridspec_kw={"hspace": 0.01, "wspace": 0.1}, sharex=True, sharey=True)
    else:
        newaxes = False

    # Plot the images in the respective panels
    for ax, img in zip(axs, images):
        ax.imshow(img.astype(np.uint8))
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # If this is a stand-alone figure, add labels
    if newaxes:
        labels = [f"{keys_latex['Lu']}\nWater", f"{keys_latex['Lsky']}\nSky", f"{keys_latex['Ld']}\nGrey card"]
        for ax, label in zip(axs, labels):
            ax.set_title(label)

    # If desired, save the result
    if newaxes:
        _saveshow(saveto)


def _plot_triple(func):
    """
    Decorator to do `func` three times. Used to repeat the functions below
    for water, sky, and grey card images.
    """
    def newfunc(images, *args, saveto=None, **kwargs):
        # Determine saveto names
        format_saveto = lambda p, l: p.parent / p.name.format(label=l)
        if saveto is not None:
            saveto_filenames = [format_saveto(saveto, label) for label in ["water", "sky", "greycard"]]
        else:
            saveto_filenames = [None]*3

        # Get the vmin and vmax
        vmin, vmax = stats.symmetric_percentiles(np.ravel(images))

        # Make the plots
        for image, filename in zip(images, saveto_filenames):
            func(image, *args, saveto=filename, vmin=vmin, vmax=vmax, **kwargs)

    return newfunc


@_plot_triple
def plot_image_small(image, vmin=0, vmax=None, saveto=None):
    """
    Plot a small version of an image.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=smallpanel)

    # Get the vmin and vmax if none were given
    if vmax is None:
        vmin, vmax = stats.symmetric_percentiles(image)

    # Plot the image
    ax.imshow(image, cmap=plt.cm.cividis, vmin=vmin, vmax=vmax)
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Show
    _saveshow(saveto)


@_plot_triple
def plot_image_small_RGBG2(images_RGBG2, camera, vmin=0, vmax=None, saveto=None):
    """
    Plot RGBG2 demosaicked images.
    """
    # Create a figure
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=smallpanel, sharex=True, sharey=True)
    axs = np.ravel(axs)

    # Get the vmin and vmax if none were given
    if vmax is None:
        vmin, vmax = stats.symmetric_percentiles(images_RGBG2)

    # Find the order to make the plots in
    bayer_order = np.ravel(camera.bayer_pattern)

    # Make the plots
    for ax, b in zip(axs, bayer_order):
        colour = "RGBG"[b]
        image = images_RGBG2[b]
        ax.imshow(image, vmin=vmin, vmax=vmax, cmap=cmaps[colour+"r"])

    # Plot parameters
    for ax in axs:
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # Show
    fig.subplots_adjust(wspace=0, hspace=0)
    _saveshow(saveto)


@_plot_triple
def histogram_small(image_RGBG2, vmin=0, vmax=None, nrbins=101, saveto=None):
    """
    Plot a small black-and-RGB histogram of an image.
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=smallpanel)

    # Get the vmin and vmax if none were given
    if vmax is None:
        vmin, vmax = stats.symmetric_percentiles(image_RGBG2)

    # Plot the overall histogram
    bins = np.linspace(vmin, vmax, nrbins)
    ax.hist(image_RGBG2.ravel(), bins=bins, color='k')

    # Plot the RGB histograms
    data_RGB = hc.convert_RGBG2_to_RGB_without_average(image_RGBG2)[0]
    _histogram_RGB(data_RGB, ax, bins=bins, lw=3)

    # Plot settings
    diff = bins[1] - bins[0]
    ax.set_xlim(vmin, vmax+diff)
    ax.tick_params(left=False, labelleft=False)
    ax.grid(ls="--", c="0.5")

    # Show
    _saveshow(saveto)


def histogram_raw(water_data, sky_data, card_data, saveto=None, camera=None):
    """
    Draw histograms of RAW water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    images = [water_data[0], sky_data[0], card_data[0]]
    plot_three_images(images, axs=axs[:,0])

    # Loop over the columns representing processing steps
    for ax_col, water, sky, card in zip(axs[:,1:].T, water_data[1:], sky_data[1:], card_data[1:]):
        # Determine histogram bins that suit all data in this column
        data_combined = np.ravel([water, sky, card])
        xmin, xmax = stats.symmetric_percentiles(data_combined, percent=0.001)
        bins = np.linspace(xmin, xmax, 150)

        # Loop over the three data sets (water/sky/grey card) and plot them in the three rows
        for ax, data in zip(ax_col, [water, sky, card]):
            # Draw the histogram
            ax.hist(data.ravel(), bins=bins, color="k")

            # If camera information was provided, also include RGB(G2) histograms
            if camera is not None:
                # If the data are already in RGBG format, use them
                if len(data.shape) == 3:
                    data_RGBG = data
                # If not, then demosaick the data
                else:
                    data_RGBG = camera.demosaick(data)

                # Combine the G and G2 channels
                data_RGB = hc.convert_RGBG2_to_RGB_without_average(data_RGBG)[0]

                # Plot the RGB histograms as lines on top of the black overall histograms
                _histogram_RGB(data_RGB, ax, bins=bins)

            # Plot settings
            ax.set_xlim(xmin, xmax)

    # Adjust the x- and y-axis ticks on the different panels
    _histogram_axis_settings(axs, ["Image", "Raw", "Bias-corrected", "Flat-fielded", "Central slice"])

    # Save the result
    _saveshow(saveto)
    if saveto is not None:
        print(f"Saved statistics plot to `{saveto}`")


def histogram_jpeg(water_data, sky_data, card_data, saveto=None, normalisation=255):
    """
    Draw histograms of RAW water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    images = [np.moveaxis(data[0], 0, -1) for data in (water_data, sky_data, card_data)]  # Move the colour channel back to the end for the plot
    plot_three_images(images, axs=axs[:,0])

    # Loop over the columns representing processing steps
    for ax_col, water, sky, card in zip(axs.T[1:], water_data, sky_data, card_data):
        # Always use the same bins, spanning the whole 8-bit range
        bins = np.linspace(0, normalisation, 50)

        # Loop over the three data sets (water/sky/grey card) and plot them in the three rows
        for ax, data in zip(ax_col, [water, sky, card]):
            # Draw the histogram
            ax.hist(data.ravel(), bins=bins, color="k")

            # Plot the RGB histograms as lines on top of the black overall histograms
            _histogram_RGB(data, ax, bins=bins)

            # Plot settings
            ax.set_xlim(0, normalisation)

    # Adjust the x- and y-axis ticks on the different panels
    _histogram_axis_settings(axs, ["Image", "JPEG (full)", "Central slice"])

    # Save the result
    _saveshow(saveto)
    if saveto is not None:
        print(f"Saved statistics plot to `{saveto}`")


def plot_R_rs_RGB(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err, saveto=None):
    """
    Plot RGB R_rs data.
    """
    # Create the figure
    plt.figure(figsize=(col1,2))

    # Plot the RGB bands
    for j, c in enumerate(RGB_OkabeIto[:3]):
        plt.errorbar(RGB_wavelengths[j], R_rs[j], xerr=effective_bandwidths[j]/2, yerr=R_rs_err[j], c=c, fmt="o")

    # Plot settings
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("R$_{rs}$ [sr$^{-1}$]")
    plt.xlim(390, 700)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(ls="--")

    # Save the result
    _saveshow(saveto)


def _correlation_plot_gridlines(ax=None):
    """
    Add grid lines to a plot.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    ax.grid(True, ls="--")


def _plot_diagonal(ax=None, **kwargs):
    """
    Add the y=x line to a plot.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    ax.plot([-1e6, 1e6], [-1e6, 1e6], c='k', zorder=10, **kwargs)


def _plot_linear_regression(func, ax=None, color="k", x=np.array([-1000., 1000.]), **kwargs):
    """
    Helper function to plot linear regression lines consistently.
    `func` is a function describing how to map `x` to y.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    y = func(x)
    ax.plot(x, y, color=color, ls="--", zorder=10, **kwargs)


def _plot_linear_regression_RGB(funcs, *args, **kwargs):
    """
    Helper function to plot linear regression lines consistently.
    This plots separate lines as included in a list `funcs`.
    Simply loops over the regression functions and colours, and passes
    everything else to the main function _plot_linear_regression.
    """
    # Path effects for RGB linear regression lines
    path_effects_RGBlines = [pe.Stroke(linewidth=3, foreground="k"), pe.Normal()]
    for func, c in zip(funcs, RGB_OkabeIto):
        _plot_linear_regression(func, *args, color=c, path_effects=path_effects_RGBlines, **kwargs)


def _plot_statistics(x, y, ax=None, xerr=None, yerr=None, **kwargs):
    """
    Plot statistics about the data into a text box in the top right corner.
    Statistics: r, MAD, zeta, SSPB.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate the statistics
    statistics, text = stats.full_statistics_for_title(x, y, xerr, yerr)

    # Plot the text box
    bbox = {"boxstyle": "round", "facecolor": "white"}
    ax.text(0.05, 0.95, text, transform=ax.transAxes, verticalalignment="top", multialignment="right", bbox=bbox, **kwargs)



def correlation_plot_simple(x, y, xerr=None, yerr=None, xlabel="", ylabel="", ax=None, equal_aspect=False, minzero=False, setmax=True, regression=False, saveto=None):
    """
    Simple correlation plot, no RGB stuff.
    """
    # If no Axes object was given, make a new one
    if ax is None:
        newaxes = True
        plt.figure(figsize=(col1,col1))
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
    _plot_diagonal(ax)

    # If wanted, perform a linear regression and plot the result
    if regression:
        params, params_cov, func = stats.linear_regression(x, y, xerr, yerr)
        _plot_linear_regression(func, ax)

    # Get statistics for title
    weights = None if xerr is None else 1/xerr**2
    r = stats.correlation(x, y, w=weights)
    title = f"$r$ = {r:.2g}"
    ax.set_title(title)

    # Labels for x and y axes
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the result (if a new plot was made)
    if newaxes:
        _saveshow(saveto)


def _axis_limit_RGB(data, key):
    """
    Get axis limits for RGB data, based on the maximum value in those data.
    `data` is a table, `key` is an RGB-aware key like "L ({c})" for radiance.
    """
    xmin = 0
    xmax = 1.05*np.nanmax(stats.ravel_table(data, key))
    return (xmin, xmax)


def force_equal_ticks(ax):
    """
    Force an Axes object's x and y axes to have the same ticks.
    """
    ax.set_yticks(ax.get_xticks())
    ax.set_xticks(ax.get_yticks())


def _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, setmax=True, equal_aspect=False, regression="none"):
    """
    Plot data into a correlation plot.
    Helper function.

    If `regression` is "none", don't do any linear regression.
    If "all", do one on all data combined.
    If "rgb", do a separate regression for each band.
    """
    regression_functions = []
    regression = regression.lower()

    # Loop over the colour bands and plot the relevant data points
    for c, pc in zip(colours, RGB_OkabeIto):
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

        # If wanted, perform a linear regression
        if regression == "rgb":
            regression_functions.append(stats.linear_regression(xdata, ydata, xerr, yerr)[2])

    # If wanted, perform a linear regression and plot the result
    if regression == "all":
        xdata, ydata = stats.ravel_table(x, xdatalabel), stats.ravel_table(y, ydatalabel)
        try:
            xerr = stats.ravel_table(x, xerrlabel)
        except (KeyError, AttributeError):
            xerr = None
        try:
            yerr = stats.ravel_table(y, yerrlabel)
        except (KeyError, AttributeError):
            yerr = None

        func = stats.linear_regression(xdata, ydata, xerr, yerr)[2]
        _plot_linear_regression(func, ax)

    elif regression == "rgb":
        _plot_linear_regression_RGB(regression_functions, ax)

    if setmax:
        xmax = _axis_limit_RGB(x, xdatalabel)[1]
        ymax = _axis_limit_RGB(y, ydatalabel)[1]
        if equal_aspect:
            xmax = ymax = max(xmax, ymax)
        ax.set_xlim(0, 1.05*xmax)
        ax.set_ylim(0, 1.05*ymax)


def correlation_plot_RGB(x, y, xdatalabel, ydatalabel, ax=None, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", regression="none", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Create figure if none was given
    if ax is None:
        newfig = True
        plt.figure(figsize=(col1,col1), tight_layout=True)
        ax = plt.gca()
    else:
        newfig = False

    # Plot in the one panel
    _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, regression=regression)

    # y=x line and grid lines
    _correlation_plot_gridlines(ax)

    # Get statistics for title
    r_all, r_RGB = stats.statistic_RGB(stats.correlation, x, y, xdatalabel, ydatalabel)
    if regression == "rgb":
        title = "   ".join(f"$r_{c}$ = {r_RGB[j]:.2g}" for j, c in enumerate(colours))
    else:
        title = f"$r$ = {r_all:.2g}"
    ax.set_title(title)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Save the result (if a new plot was made)
    if newfig:
        _saveshow(saveto)



def correlation_plot_RGB_equal(x, y, datalabel, errlabel=None, xlabel="x", ylabel="y", regression="none", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Convert the data and error labels to RGB-aware ones
    datalabel, errlabel = [label + " ({c})" for label in (datalabel, errlabel)]

    # Calculate residuals
    residuals = stats.residual_table(x, y, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel)

    # Create figure to hold plot
    fig, axs = plt.subplots(figsize=(col1,5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3,1]})

    # Plot in both panels
    _correlation_plot_errorbars_RGB(axs[0], x, y, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel, equal_aspect=True, regression=regression)
    _correlation_plot_errorbars_RGB(axs[1], x, residuals, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel, setmax=False)

    # Plot the x=y line (top) and horizontal (bottom)
    for ax in axs:
        _correlation_plot_gridlines(ax)
    _plot_diagonal(axs[0])
    axs[1].axhline(0, c='k')

    # Add statistics in a text box
    x_all, y_all = stats.ravel_table(x, datalabel), stats.ravel_table(y, datalabel)
    x_err_all, y_err_all = stats.ravel_table(x, errlabel), stats.ravel_table(y, errlabel)
    _plot_statistics(x_all, y_all, axs[0], xerr=x_err_all, yerr=y_err_all)

    # Labels
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel("Difference")
    axs[0].set_ylabel(ylabel)

    # Save the result
    fig.subplots_adjust(hspace=0.1)
    _saveshow(saveto)


def correlation_plot_radiance(x, y, keys=["Lu", "Lsky", "Ld"], combine=True, xlabel="x", ylabel="y", xunit=ADUnmsr, yunit=ADUnmsr, regression="all",  saveto=None):
    """
    Make a multi-panel plot comparing radiances.
    Each panel represents one of the keys, for example upwelling, sky, and downwelling radiance.
    If `combine` is True then also add a panel combining all the keys.
    Do a combined linear regression and plot the result in each figure.
    """
    # How many panels to plot: one for each key, and one if `combine` is True.
    nr_panels = len(keys) + combine

    # Create the figure and panels
    fig, axs = plt.subplots(nrows=nr_panels, figsize=(col1, nr_panels*2.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.05, "wspace": 0})

    # Plot each key
    for ax, key in zip(axs, keys):
        key_c = key + " ({c})"  # RGB-aware key
        key_c_err = key + "_err ({c})"
        correlation_plot_RGB(x, y, key_c, key_c, ax=ax, xerrlabel=key_c_err, yerrlabel=key_c_err, xlabel=None, ylabel=ylabel)

    # Generate a combined radiance table (always) and plot it (if `combine`)
    x_radiance, y_radiance = hc.get_radiances(x, keys), hc.get_radiances(y, keys)
    if combine:
        key_c = "L ({c})"
        key_c_err = "L_err ({c})"
        correlation_plot_RGB(x_radiance, y_radiance, key_c, key_c, ax=axs[-1], xerrlabel=key_c_err, yerrlabel=key_c_err, xlabel=None, ylabel=ylabel)

    # Combined linear regression
    if regression == "all":
        xdata, ydata = stats.ravel_table(x_radiance, "L ({c})"), stats.ravel_table(y_radiance, "L ({c})")
        xerr, yerr = stats.ravel_table(x_radiance, "L_err ({c})"), stats.ravel_table(y_radiance, "L_err ({c})")

        func_linear = stats.linear_regression(xdata, ydata, xerr, yerr)[2]
        for ax in axs:
            _plot_linear_regression(func_linear, ax)

    # If RGB regression, do each band separately
    elif regression == "rgb":
        funcs_linear = [stats.linear_regression(x_radiance[f"L ({c})"], y_radiance[f"L ({c})"], x_radiance[f"L_err ({c})"], y_radiance[f"L_err ({c})"])[2] for c in colours]
        for ax in axs:
            _plot_linear_regression_RGB(funcs_linear, ax)

    # Plot settings
    axs[0].set_xlim(_axis_limit_RGB(x_radiance, "L ({c})"))
    axs[0].set_ylim(_axis_limit_RGB(y_radiance, "L ({c})"))

    # Labels
    if combine:
        keys = keys + ["L"]  # Don't use append because it changes the original object
    axs[-1].set_xlabel(xlabel)
    for ax, key in zip(axs, keys):
        ax.set_title(None)  # Remove default titles
        ax.text(0.05, 0.95, keys_latex[key], transform=ax.transAxes, fontsize=14, verticalalignment="top")

    # Save the result
    _saveshow(saveto)


def correlation_plot_bands(x, y, datalabel="R_rs", errlabel=None, quantity="$R_{rs}$", xlabel="", ylabel="", saveto=None):
    """
    Make a correlation plot between the band ratios G/R and G/B.
    """
    # Get the data out of the input tables
    GB_label, GR_label, RB_label = [f"{datalabel} ({c})" for c in hc.bandratio_labels]
    x_GB, y_GB = x[GB_label], y[GB_label]
    x_GR, y_GR = x[GR_label], y[GR_label]
    x_RB, y_RB = x[RB_label], y[RB_label]

    # Get the uncertainties out of the input tables if available
    if errlabel is not None:
        GB_err_label, GR_err_label, RB_err_label = [f"{errlabel} ({c})" for c in hc.bandratio_labels]
        x_err_GB, y_err_GB = x[GB_err_label], y[GB_err_label]
        x_err_GR, y_err_GR = x[GR_err_label], y[GR_err_label]
        x_err_RB, y_err_RB = x[RB_err_label], y[RB_err_label]
    else:
        x_err_GB = y_err_GB = x_err_GR = y_err_GR = x_err_RB = y_err_RB = None

    # Plot the data
    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(col1, 6), gridspec_kw={"hspace": 0.1, "wspace": 0.1})
    correlation_plot_simple(x_GB, y_GB, xerr=x_err_GB, yerr=y_err_GB, ax=axs[0], xlabel=f"{xlabel}\n{quantity} (G/B)", ylabel=f"{quantity} (G/B)", equal_aspect=True)
    correlation_plot_simple(x_GR, y_GR, xerr=x_err_GR, yerr=y_err_GR, ax=axs[1], xlabel="", ylabel=f"{ylabel}\n{quantity} (G/R)", equal_aspect=True)
    correlation_plot_simple(x_RB, y_RB, xerr=x_err_RB, yerr=y_err_RB, ax=axs[2], xlabel=f"{quantity} (R/B)\n{xlabel}", ylabel=f"{quantity} (R/B)", equal_aspect=True)

    # Axis settings
    data_combined = [x_GR, x_GB, x_RB, y_GR, y_GB, y_RB]
    axmin, axmax = np.nanmin(data_combined)-0.05, np.nanmax(data_combined)+0.05
    axs[0].set_xlim(axmin, axmax)
    axs[0].set_ylim(axmin, axmax)
    for ax in axs:
        ax.set_aspect("equal")
        ax.locator_params(nbins=4)

    # Switch xtick labels on the bottom plot to the top
    axs[0].tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    axs[0].xaxis.set_label_position("top")

    # Calculate statistics
    for ax, x, y, xerr, yerr in zip(axs, [x_GR, x_GB, x_RB], [y_GR, y_GB, y_RB], [x_err_GR, x_err_GB, x_err_RB], [y_err_GR, y_err_GB, y_err_RB]):
        ax.set_title("")
        _plot_statistics(x, y, ax, xerr=xerr, yerr=yerr, fontsize=8)

    # Save the result
    _saveshow(saveto)


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    # https://stackoverflow.com/a/53865762
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True )
    z = stats.interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    return ax


def plot_correlation_matrix_radiance(covariance_matrix, x1, y1, x2, y2, x1label="[a.u.]", y1label="[a.u.]", x2label="[a.u.]", y2label="[a.u.]", saveto=None):
    """
    Plot a given correlation matrix consisting of RGB or RGBG2 radiances.
    """
    # Plot correlation coefficients
    kwargs = {"cmap": plt.cm.get_cmap("cividis", 10), "s": 5, "rasterized": True}

    # Calculate the correlation matrix
    correlation_matrix = stats.correlation_from_covariance(covariance_matrix)

    fig, axs = plt.subplots(ncols=3, figsize=(col2,3), dpi=600)

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('bottom', size='10%', pad=0.3)
    im = axs[0].imshow(correlation_matrix, extent=(0,1,1,0), cmap=plt.cm.get_cmap("cividis", 5), vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, cax=cax, orientation='horizontal', ticks=np.arange(0,1.1,0.2), label="Pearson $r$")

    ticks = np.linspace(0,1,4)
    axs[0].set_xticks(ticks)
    xtick_offset = " "*10
    axs[0].set_xticklabels([f"{xtick_offset}{keys_latex['Lu']}", f"{xtick_offset}{keys_latex['Lsky']}", f"{xtick_offset}{keys_latex['Ld']}", ""])
    axs[0].set_yticks(ticks)
    axs[0].set_yticklabels([f"\n\n{keys_latex['Ld']}", f"\n\n{keys_latex['Lsky']}", f"\n\n{keys_latex['Lu']}", ""])

    for ax, x, y, xlabel, ylabel in zip(axs[1:], [x1, x2], [y1, y2], [x1label, x2label], [y1label, y2label]):
        density_scatter(x, y, ax=ax, **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title("$r =" + f"{stats.correlation(x,y):.2g}" + "$")

        ax.set_aspect("equal")
        ax.grid(ls="--", c="0.5", alpha=0.5)

    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()

    plt.subplots_adjust(wspace=0.5)
    # Save the result
    _saveshow(saveto)


def _confidence_ellipse(center, covariance, ax, covariance_scale=1, **kwargs):
    """
    Plot a confidence ellipse from a given (2x2) covariance matrix.
    https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    correlation = stats.correlation_from_covariance(covariance)[0,1]
    ell_radius_x = np.sqrt(1 + correlation)
    ell_radius_y = np.sqrt(1 - correlation)
    ellipse = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, **kwargs)

    scale_x = np.sqrt(covariance[0,0])*covariance_scale
    scale_y = np.sqrt(covariance[1,1])*covariance_scale

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(*center)
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


def plot_xy_on_gamut_covariance(xy, xy_covariance, covariance_scale=1, saveto=None):
    """
    Plot xy coordinates on the gamut including their covariance ellipse.
    """
    fig = plt.figure(figsize=(col1,col1))
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    _confidence_ellipse(xy, xy_covariance, plt.gca(), covariance_scale=covariance_scale, edgecolor="k", fill=False, linestyle="--")
    plt.scatter(*xy, c="k", s=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")

    # Save the result
    _saveshow(saveto)


def correlation_plot_hue_angle_and_ForelUle(x, y, xerr=None, yerr=None, xlabel="", ylabel="", saveto=None):
    """
    Make a correlation plot of hue angles (x and y).
    Draw the equivalent Forel-Ule indices on the grid for reference.
    """
    # Generate labels for the x and y axes
    xlabel_hue = f"{xlabel}\nHue angle $\\alpha$ [degrees]"
    ylabel_hue = f"{ylabel}\nHue angle $\\alpha$ [degrees]"
    xlabel_FU = f"{xlabel}\nForel-Ule index"
    ylabel_FU = f"{ylabel}\nForel-Ule index"

    # Create figure
    fig, ax = plt.subplots(figsize=(col1,col1))

    # Plot the data
    correlation_plot_simple(x, y, xerr=xerr, yerr=yerr, ax=ax, equal_aspect=True)

    # Set the x and y labels for hue angle
    ax.set_xlabel(xlabel_hue)
    ax.set_ylabel(ylabel_hue)
    ax.grid(False)

    # Plot lines correspnding to the FU colour limits, and
    # colour the squares along the x=y line.
    line_kwargs = {"c": "k", "lw": 0.5}
    for fu,angle in enumerate(FU_hueangles):
        ax.axvline(angle, **line_kwargs)
        ax.axhline(angle, **line_kwargs)
        square = FU_hueangles[fu:fu+2]
        square_next = FU_hueangles[fu:fu+3:2]
        ax.fill_between(square_next, *square_next, facecolor="0.7")
        ax.fill_between(square, *square, facecolor="0.5")

    # Same ticks on x and y.
    # Weird quirk in matplotlib: you need to do it twice (x->y then y->x)
    # to actually get the same ticks on both axes.
    force_equal_ticks(ax)

    # Set the axis minima and maxima
    minangle = np.nanmin([x, y])-5
    maxangle = np.nanmax([x, y])+5
    ax.set_xlim(minangle, maxangle)
    ax.set_ylim(minangle, maxangle)

    # Labels for FU colours: every odd colour, in the middle of the range
    FU_middles = np.array([(a + b)/2 for a, b in zip(FU_hueangles, FU_hueangles[1:])])[::2]
    FU_labels = np.arange(1,21)[::2]

    # Add a new x axis at the top with FU colours
    ax2 = ax.twinx()
    ax2.set_yticks(FU_middles)
    ax2.set_yticklabels(FU_labels)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_ylabel(ylabel_FU)
    ax2.tick_params(axis="y", length=0)

    # Add a new y axis on the right with FU colours
    ax3 = ax2.twiny()
    ax3.set_xticks(FU_middles)
    ax3.set_xticklabels(FU_labels)
    ax3.set_xlim(ax.get_xlim())
    ax3.set_xlabel(xlabel_FU)
    ax3.tick_params(axis="x", length=0)

    # Calculate some statistics to compare the data
    # Median absolute deviation and number of FU matches
    mad_hueangle = stats.MAD(x, y)
    FU_matches, FU_near_matches, mad_FU = compare_FU_matches_from_hue_angle(x, y)
    title = f"MAD = ${mad_hueangle:.1f} \\degree$ ({mad_FU:.0f} FU)\n{FU_matches:.0f}% $\Delta$FU$= 0$   {FU_near_matches:.0f}% $\Delta$FU$\leq 1$"
    ax.set_title(title)

    # Save the result
    _saveshow(saveto)
