"""
Functions and variables used for plotting data and results.
Some of these will be moved to SPECTACLE in the near future.
"""
import functools
from pathlib import Path
from sys import stdout
from matplotlib import pyplot as plt, transforms, patheffects as pe, rcParams
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerTuple
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interpn
import numpy as np
from colorio._tools import plot_flat_gamut
from spectacle.plot import RGB_OkabeIto, save_or_show, cmaps
from . import statistics as stats, colours, hydrocolor as hc, hyperspectral as hy
from .wacodi import FU_hueangles, compare_hue_angles

# Legend settings
rcParams["legend.loc"] = "lower right"
rcParams["legend.framealpha"] = 1
rcParams["legend.edgecolor"] = "k"
rcParams["legend.handletextpad"] = 0.5
rcParams["grid.linestyle"] = "--"

# Commonly used unit strings
ADUnmsr = "[ADU nm$^{-1}$ sr$^{-1}$]"
ADUnm = "[ADU nm$^{-1}$]"
Wnmsr = "[W m$^{-2}$ nm$^{-1}$ sr$^{-1}$]"
Wnm = "[W m$^{-2}$ nm$^{-1}$]"
persr = "[sr$^{-1}$]"

# Dictionary mapping keys to LaTeX strings
keys_latex = {"Lu": "$L_u$", "Lsky": "$L_{sky}$", "Ld": "$L_d$", "Ed": "$E_d$", "L": "$L$", "R_rs": "$R_{rs}$"}

# Dictionary mapping keys to marker symbols
markers = {"Lu": "o", "Lsky": "v", "Ld": "s"}

# Colours for band ratios
bandratio_plotcolours = [RGB_OkabeIto[i[0]] for i in hc.bandratio_indices]

# bbox for text
bbox_text = {"boxstyle": "round", "facecolor": "white"}

# Frontiers column widths - Applied Optics's are very similar
col1 = 85/25.4
col2 = 180/25.4
smallpanel = (2, 1.5)


def _textbox(ax, text, x=0.05, y=0.95, fontsize=10, verticalalignment="top", horizontalalignment="left", bbox_kwargs={}, zorder=15, **kwargs):
    """
    Add a text box with given text and some standard parameters.
    """
    bbox_new = {**bbox_text, **bbox_kwargs}  # Merge the kwarg lists
    return ax.text(x, y, text, transform=ax.transAxes, verticalalignment=verticalalignment, horizontalalignment=horizontalalignment, bbox=bbox_new, zorder=zorder, fontsize=fontsize, **kwargs)


def new_or_existing_figure(func):
    """
    Decorator that handles the choice between creating a new figure or plotting in an existing one.
    Checks if an Axes object was given - if yes, use that - if no, create a new one.
    In the "no" case, save/show the resulting plot at the end.
    """
    @functools.wraps(func)
    def newfunc(*args, ax=None, title=None, figsize=(col1, col1), figure_kwargs={}, saveto=None, dpi=300, bbox_inches="tight", **kwargs):
        # If no Axes object was given, make a new one
        if ax is None:
            newaxes = True
            plt.figure(figsize=figsize, **figure_kwargs)
            ax = plt.gca()
        else:
            newaxes = False

        # Plot everything as normal
        func(*args, ax=ax, **kwargs)

        # If this is a new plot, add a title and save/show the result
        if newaxes:
            ax.set_title(title)
            save_or_show(saveto, dpi=dpi, bbox_inches=bbox_inches)

    return newfunc


def _histogram_axis_settings(axs, column_labels):
    """
    Helper function.
    Adjust the x- and y-axis labels on histogram panels.
    """
    for ax in axs.ravel():  # No ticks on the left in any panels
        ax.tick_params(left=False, labelleft=False)
    for ax in axs[:2].ravel():  # No ticks on the bottom for the top 2 rows
        ax.tick_params(bottom=False, labelbottom=False)
    for ax in axs[:, 1:].ravel():  # Grid
        ax.grid(alpha=0.7)
    for ax, label in zip(axs[:, 0], ["Water", "Sky", "Grey card"]):  # Labels on the y-axes
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
    If `axs` are provided, plot them in those.
    Otherwise, create a new figure.
    """
    # Create a new figure if necessary
    if axs is None:
        newaxes = True
        fig, axs = plt.subplots(ncols=3, figsize=(col1, col1), gridspec_kw={"hspace": 0.01, "wspace": 0.1}, sharex=True, sharey=True)
    else:
        newaxes = False

    # Plot the images in the respective panels
    for ax, img in zip(axs, images):
        try:
            ax.imshow(img.astype(np.uint8))
        except TypeError:  # If the image data are in the wrong order, e.g. (c, x, y) instead of (x, y, c)
            ax.imshow(np.moveaxis(img.astype(np.uint8), 0, -1))
        ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

    # If this is a stand-alone figure, add labels
    if newaxes:
        labels = [f"{keys_latex['Lu']}\nWater", f"{keys_latex['Lsky']}\nSky", f"{keys_latex['Ld']}\nGrey card"]
        for ax, label in zip(axs, labels):
            ax.set_title(label)

    # If desired, save the result
    if newaxes:
        save_or_show(saveto)


def _plot_triple(func):
    """
    Decorator to do `func` three times.
    Used to repeat the functions below for water, sky, and grey card images.
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
    save_or_show(saveto, dpi=300)


@_plot_triple
def plot_image_small_RGBG2(images_RGBG2, camera, vmin=0, vmax=None, equal_aspect=False, saveto=None):
    """
    Plot RGBG2 demosaicked images.
    """
    # Create a figure
    if equal_aspect:
        figsize = (smallpanel[0], smallpanel[0])
    else:
        figsize = smallpanel
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=figsize, sharex=True, sharey=True)
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
    save_or_show(saveto, dpi=300)


@_plot_triple
def histogram_small(image_RGBG2, vmin=0, vmax=None, nrbins=51, saveto=None):
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

    # Show
    save_or_show(saveto)


def histogram_raw(water_data, sky_data, card_data, saveto=None, camera=None):
    """
    Draw histograms of RAW water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11, 4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    images = [water_data[0], sky_data[0], card_data[0]]
    plot_three_images(images, axs=axs[:, 0])

    # Loop over the columns representing processing steps
    for ax_col, water, sky, card in zip(axs[:, 1:].T, water_data[1:], sky_data[1:], card_data[1:]):
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
    save_or_show(saveto)
    if saveto is not None:
        print(f"Saved statistics plot to `{saveto}`")


def histogram_jpeg(water_data, sky_data, card_data, saveto=None, normalisation=255):
    """
    Draw histograms of JPEG water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9, 4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    images = [np.moveaxis(data[0], 0, -1) for data in (water_data, sky_data, card_data)]  # Move the colour channel back to the end for the plot
    plot_three_images(images, axs=axs[:, 0])

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
    save_or_show(saveto)
    if saveto is not None:
        print(f"Saved statistics plot to `{saveto}`")


def _plot_settings_R_rs(ax, title=None):
    """
    Apply some default settings to an R_rs plot.
    """
    # Labels
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel(f"{keys_latex['R_rs']} {persr}")
    ax.set_title(title)

    # Axis limits and ticks
    ax.set_yticks(np.arange(0, 0.1, 0.01))
    ax.set_ylim(0, 0.07)
    ax.set_xticks(np.arange(300, 1000, 100))
    ax.set_xlim(350, 850)
    ax.grid(True)


@new_or_existing_figure
def plot_reference_spectrum(wavelengths, spectrum, uncertainty=None, *, facecolor="k", title=None, ax=None, saveto=None, **kwargs):
    """
    Plot a hyperspectral reference spectrum, with uncertainties if available.
    """
    # Plot the spectrum
    ax.plot(wavelengths, spectrum, c=facecolor, **kwargs)

    # If uncertainties were given, plot these too
    if uncertainty is not None:
        ax.fill_between(wavelengths, spectrum-uncertainty, spectrum+uncertainty, facecolor=facecolor, alpha=0.5)


@new_or_existing_figure
def plot_R_rs_multi(spectrums, labels=None, title=None, ax=None, saveto=None, **kwargs):
    """
    Plot multiple hyperspectral reference spectra in one panel.
    """
    # If no labels were given, make a decoy list
    if labels is None:
        labels = [None]*len(spectrums)

    # Plot the spectra
    linecolours = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for spectrum, label, colour in zip(spectrums, labels, linecolours):
        plot_reference_spectrum(*spectrum, ax=ax, label=label, facecolor=colour, **kwargs)

    # Panel properties
    _plot_settings_R_rs(ax, title=title)
    ax.legend(loc="best")


@new_or_existing_figure
def plot_hyperspectral_dataset(data, parameter="R_rs", *, title=None, facecolor="k", alpha=0.05, ax=None, saveto=None, **kwargs):
    """
    Plot an entire hyperspectral dataset into one panel, using transparency.
    """
    # Get the wavelengths and spectra
    column_names = hy.get_keys_for_parameter(data, parameter, exclude_err=True)
    wavelengths = hy.get_wavelengths_from_keys(column_names, parameter)
    spectra = hy.convert_columns_to_array(data, column_names)

    # Plot the spectra
    ax.plot(wavelengths, spectra.T, c=facecolor, alpha=alpha, rasterized=True, **kwargs)

    # Plot settings
    _plot_settings_R_rs(ax)


@new_or_existing_figure
def plot_R_rs_RGB(RGB_wavelengths, R_rs, effective_bandwidths=None, R_rs_err=None, reference=None, title=None, ax=None, saveto=None, **kwargs):
    """
    Plot RGB R_rs data, with an optional hyperspectral reference.
    `reference` must contain 2 or 3 elements: [wavelengths, R_rs, R_rs_uncertainty (optional)]
    """
    # Turn the effective bandwidth into an xerr by halving it, if values were given
    try:
        xerr = effective_bandwidths/2
    except TypeError:
        xerr = [None]*3

    # Add decoy elements to R_rs_err if necessary
    if R_rs_err is None:
        R_rs_err = [None]*3

    # Plot the RGB bands
    for j, c in enumerate(RGB_OkabeIto[:3]):
        ax.errorbar(RGB_wavelengths[j], R_rs[j], xerr=xerr[j], yerr=R_rs_err[j], c=c, fmt="o", **kwargs)

    # Plot the reference line if one was given
    if reference:
        plot_reference_spectrum(*reference, ax=ax)

    # Plot settings
    _plot_settings_R_rs(ax, title=title)


def _plot_diagonal(ax=None, **kwargs):
    """
    Add the y=x line to a plot.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    ax.plot([-1e6, 1e6], [-1e6, 1e6], c='k', zorder=10, label="1:1", **kwargs)


def _plot_linear_regression(func, ax=None, color="k", ls="--", x=np.linspace(0, 50, 1000), **kwargs):
    """
    Helper function to plot linear regression lines consistently.
    `func` is a function describing how to map `x` to y.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    y = func(x)
    return ax.plot(x, y, color=color, ls=ls, zorder=10, **kwargs)


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
        _plot_linear_regression(func, *args, color=c, ls="-", path_effects=path_effects_RGBlines, **kwargs)


def _plot_statistics(x, y, ax=None, xerr=None, yerr=None, fontsize=9, **kwargs):
    """
    Plot statistics about the data into a text box in the top right corner.
    Statistics: r, MAD, zeta, SSPB.
    """
    if ax is None:
        ax = plt.gca()

    # Calculate the statistics
    statistics, text = stats.full_statistics_for_title(x, y, xerr, yerr)

    # Plot the text box
    return _textbox(ax, text, fontsize=fontsize, **kwargs)


@new_or_existing_figure
def correlation_plot_simple(x, y, xerr=None, yerr=None, xlabel="", ylabel="", equal_aspect=False, minzero=False, setmax=True, regression=False, ax=None, saveto=None):
    """
    Simple correlation plot between iterables `x` and `y`.
    """
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
            ax.set_aspect("equal")
        ax.set_xlim(xmax=xmax)
        ax.set_ylim(ymax=ymax)

    # Grid lines and y=x diagonal
    ax.grid(True)
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


def _axis_limit_RGB(data, key, loop_keys=colours):
    """
    Get axis limits for RGB data, based on the maximum value in those data.
    `data` is a table, `key` is an RGB-aware key like "L ({c})" for radiance.
    """
    data_all = stats.ravel_table(data, key, loop_keys=loop_keys)
    xmin = 0.95*np.nanmin(data_all)
    xmax = 1.05*np.nanmax(data_all)
    return (xmin, xmax)


def force_equal_ticks(ax):
    """
    Force an Axes object's x and y axes to have the same ticks.
    """
    ax.set_yticks(ax.get_xticks())
    ax.set_xticks(ax.get_yticks())


def _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, loop_keys=colours, plot_colours=RGB_OkabeIto, setmax=True, equal_aspect=False, regression="none", markers="ooo", **kwargs):
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
    for c, pc, marker in zip(loop_keys, plot_colours, markers):
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
        ax.errorbar(xdata, ydata, xerr=xerr, yerr=yerr, color=pc, marker=marker, linestyle="", label=c, **kwargs)

        # If wanted, perform a linear regression
        if regression == "rgb":
            regression_functions.append(stats.linear_regression(xdata, ydata, xerr, yerr)[2])

    # If wanted, perform a linear regression and plot the result
    if regression == "all":
        xdata, ydata = stats.ravel_table(x, xdatalabel, loop_keys=loop_keys), stats.ravel_table(y, ydatalabel, loop_keys=loop_keys)
        try:
            xerr = stats.ravel_table(x, xerrlabel, loop_keys=loop_keys)
        except (KeyError, AttributeError):
            xerr = None
        try:
            yerr = stats.ravel_table(y, yerrlabel, loop_keys=loop_keys)
        except (KeyError, AttributeError):
            yerr = None

        func = stats.linear_regression(xdata, ydata, xerr, yerr)[2]
        _plot_linear_regression(func, ax, label="Best fit")

    elif regression == "rgb":
        _plot_linear_regression_RGB(regression_functions, ax)

    if setmax:
        xmin, xmax = _axis_limit_RGB(x, xdatalabel, loop_keys=loop_keys)
        ymin, ymax = _axis_limit_RGB(y, ydatalabel, loop_keys=loop_keys)
        if equal_aspect:
            xmin = ymin = min(xmin, ymin)
            xmax = ymax = max(xmax, ymax)
            ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)


@new_or_existing_figure
def correlation_plot_RGB(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", regression="none", ax=None, saveto=None, **kwargs):
    """
    Make a correlation plot between two tables `x` and `y`.
    Use the labels `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)", "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Plot the data
    _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, regression=regression, **kwargs)

    # y=x line and grid lines
    ax.grid(True)

    # Labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def correlation_plot_RGB_equal(x, y, datalabel, errlabel=None, xlabel="x", ylabel="y", title=None, regression="none", difference_unit="", legend=False, loop_keys=colours, through_origin=True, compare_to_regression=False, saveto=None, saveto_stats=stdout, **kwargs):
    """
    Make a correlation plot between two tables `x` and `y`.
    Use the labels `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    `loop_keys` can be used to iterate over something other than RGB, for example band ratio labels.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)", "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Convert the data and error labels to RGB-aware ones
    datalabel, errlabel = [label + " ({c})" for label in (datalabel, errlabel)]

    # Calculate residuals
    residuals = stats.residual_table(x, y, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel, loop_keys=loop_keys)

    # Create figure to hold plot
    fig, axs = plt.subplots(figsize=(col1, 5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3, 1]})

    # Plot in both panels
    _correlation_plot_errorbars_RGB(axs[0], x, y, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel, equal_aspect=True, regression=regression, loop_keys=loop_keys, **kwargs)
    _correlation_plot_errorbars_RGB(axs[1], x, residuals, datalabel, datalabel, xerrlabel=errlabel, yerrlabel=errlabel, setmax=False, loop_keys=loop_keys, **kwargs)

    # Plot the x=y line (top) and horizontal (bottom)
    for ax in axs:
        ax.grid(True)
    _plot_diagonal(axs[0])
    axs[1].axhline(0, c='k', zorder=15)

    # If desired, change the lower limit on the x and y axes to go through the origin
    if through_origin:
        axs[0].set_xlim(xmin=0)
        axs[0].set_ylim(ymin=0)

    # Add statistics in a text box
    x_all, y_all = stats.ravel_table(x, datalabel, loop_keys=loop_keys), stats.ravel_table(y, datalabel, loop_keys=loop_keys)
    x_err_all, y_err_all = stats.ravel_table(x, errlabel, loop_keys=loop_keys), stats.ravel_table(y, errlabel, loop_keys=loop_keys)
    stats_textbox = _plot_statistics(x_all, y_all, axs[0], xerr=x_err_all, yerr=y_err_all)
    stats.save_statistics_to_file(x_all, y_all, x_err_all, y_err_all, saveto=saveto_stats)

    # If desired, add statistics comparing the data to the linear regression
    if compare_to_regression:
        # First, perform a linear regression of y to x - this lets us rescale y (e.g. smartphone) to the scale of x (e.g. WISP-3)
        params, _, func = stats.linear_regression(y_all, x_all, y_err_all, x_err_all)
        y_rescaled = func(y_all)
        y_err_rescaled = params[0]*y_err_all

        # Calculate and print the results
        stats_text = stats.full_statistics_for_title(x_all, y_rescaled, xerr=x_err_all, yerr=y_err_rescaled)[1]
        stats_text = "\n".join(stats_text.split("\n")[2:])  # Remove the first two lines
        _textbox(axs[0], stats_text, x=0.40, fontsize=9, bbox_kwargs={"linestyle": "--"})
        if saveto_stats != stdout:
            saveto_stats = Path(saveto_stats)
        stats.save_statistics_to_file(x_all, y_rescaled, x_err_all, y_err_rescaled, saveto=saveto_stats.with_name(saveto_stats.stem+"_linear_regression.dat"))


    # Add a legend if desired
    if legend:
        axs[0].legend()

    # Labels
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(f"Difference {difference_unit}")
    axs[0].set_ylabel(ylabel)
    axs[0].set_title(title)
    force_equal_ticks(axs[0])
    fig.align_ylabels(axs)

    # Save the result
    fig.subplots_adjust(hspace=0.1)
    save_or_show(saveto)


# Shortcuts for RGB and band ratio R_rs plots
correlation_plot_R_rs = functools.partial(correlation_plot_RGB_equal, datalabel="R_rs", errlabel="R_rs_err", regression="all", difference_unit=persr, legend=False, through_origin=True)
correlation_plot_bandratios_combined = functools.partial(correlation_plot_RGB_equal, datalabel="R_rs", errlabel="R_rs_err", regression="all", difference_unit="", legend=True, loop_keys=hc.bandratio_labels, plot_colours=bandratio_plotcolours, through_origin=False, markers="ovs")


def correlation_plot_radiance(x, y, keys=["Lu", "Lsky", "Ld"], combine=True, xlabel="x", ylabel="y", title=None, regression="all", saveto=None, saveto_stats=stdout):
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
        y_fitted = func_linear(xdata)
        _plot_statistics(ydata, y_fitted, axs[-1])
        stats.save_statistics_to_file(ydata, y_fitted, saveto=saveto_stats)
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
    for ax in axs[:-1]:
        ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Add labels to the corners of each plot to indicate which radiance they show
    if combine:
        keys = keys + ["L"]  # Don't use append because it changes the original object
    axs[-1].set_xlabel(xlabel)
    for ax, key in zip(axs, keys):
        ax.set_title(None)  # Remove default titles
        _textbox(ax, keys_latex[key], x=0.85, y=0.15, multialignment="right")
    axs[0].set_title(title)

    # Save the result
    save_or_show(saveto)


@new_or_existing_figure
def correlation_plot_radiance_combined(x, y, keys=["Lu", "Lsky", "Ld"], xlabel="x", ylabel="y", regression="all", compare_directly=False, ax=None, saveto=None, saveto_stats=stdout):
    """
    Make a single-panel plot comparing the combined radiances from x and y.
    Do a combined linear regression and plot the result.
    """
    # Bit of a hack - plot each key into the panel separately
    for key in keys:
        key_c = key + " ({c})"
        key_c_err = key + "_err ({c})"
        correlation_plot_RGB(x, y, key_c, key_c, ax=ax, xerrlabel=key_c_err, yerrlabel=key_c_err, xlabel=xlabel, ylabel=ylabel, markers=markers[key]*3)

    # Generate a combined radiance table
    x_radiance, y_radiance = hc.get_radiances(x, keys), hc.get_radiances(y, keys)
    key_c = "L ({c})"
    key_c_err = "L_err ({c})"

    # Combined linear regression
    if regression == "all":
        xdata, ydata = stats.ravel_table(x_radiance, "L ({c})"), stats.ravel_table(y_radiance, "L ({c})")
        xerr, yerr = stats.ravel_table(x_radiance, "L_err ({c})"), stats.ravel_table(y_radiance, "L_err ({c})")

        # Directly compare x and y - for when comparing two references (x, y)
        if compare_directly:
            _plot_statistics(xdata, ydata, ax, xerr=xerr, yerr=yerr)
            stats.save_statistics_to_file(x, y, xerr, yerr, saveto=saveto_stats)
        # Fit x to y so the MAD is in x units - for when comparing a reference (x) to a smartphone (y)
        else:
            *_, func_y_to_x = stats.linear_regression(ydata, xdata, yerr, xerr)
            x_fitted = func_y_to_x(ydata)
            _plot_statistics(xdata, x_fitted, ax)
            stats.save_statistics_to_file(xdata, x_fitted, saveto=saveto_stats)

        # Fit y to x as normal and plot this
        params, _, func_linear = stats.linear_regression(xdata, ydata, xerr, yerr)
        regression_line, = _plot_linear_regression(func_linear, ax)
        regression_label = "Best\nfit"# f"$y =$\n${params[1]:.3g} + {params[0]:.3g} x$"

    # If RGB regression (JPEG-RAW), fit a power law to each channel separately
    elif regression == "rgb":
        # Fit power laws and plot them
        params, param_covs, funcs = zip(*[stats.powerlaw_regression(x_radiance[f"L ({c})"], y_radiance[f"L ({c})"], x_radiance[f"L_err ({c})"], y_radiance[f"L_err ({c})"]) for c in colours])
        _plot_linear_regression_RGB(funcs, ax)

        # Compare x to the regressed y data and compare them. This means MAD is in y units.
        for c, func, p, cov in zip(colours, funcs, params, param_covs):
            if saveto_stats != stdout:
                saveto_stats = Path(saveto_stats)
                saveto = saveto_stats.with_name(saveto_stats.stem+f"_powerlaw_regression_{c}.dat")
            else:
                saveto = stdout
            y_fitted = func(x_radiance[f"L ({c})"])
            stats.save_statistics_to_file(y_fitted, y_radiance[f"L ({c})"], saveto=saveto)
            param_uncertainties = np.sqrt(np.diag(cov))
            param_correlation = stats.correlation_from_covariance(cov)[0, 1]
            print(f"Radiance power law parameters for {c}: {p} +- {param_uncertainties} (r = {param_correlation:.2f})")

    # Plot settings
    ax.set_xlim(0, _axis_limit_RGB(x_radiance, "L ({c})")[1])
    ax.set_ylim(0, _axis_limit_RGB(y_radiance, "L ({c})")[1])
    ax.set_title(None)  # Remove default titles
    if compare_directly:  # Plot the y=x diagonal if x and y are to be compared directly
        _plot_diagonal(ax)

    # Generate a legend where each symbol has an entry, with coloured markers
    labels = [keys_latex[key] for key in keys]
    scatters = [tuple([ax.scatter([-1], [-1], marker=markers[key], label=keys_latex[key], color=c) for c in RGB_OkabeIto]) for key in keys]
    # If a regression on all data was done, add its line to the legend
    if regression == "all":
        scatters.append(regression_line)
        labels.append(regression_label)
    ax.legend(scatters, labels, numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})


def correlation_plot_bands(x, y, datalabel="R_rs", errlabel=None, quantity=keys_latex["R_rs"], xlabel="", ylabel="", title=None, saveto=None, saveto_stats=stdout):
    """
    Make a correlation plot for each of the RGB band ratios.
    """
    # Get the data out of the input tables
    datalabel_bandratio = hc.extend_keys_to_RGB(datalabel, hc.bandratio_labels)
    xy_pairs = [(x[label], y[label]) for label in datalabel_bandratio]

    # Get the uncertainties out of the input tables if available
    if errlabel is not None:
        errlabel_bandratio = hc.extend_keys_to_RGB(errlabel, hc.bandratio_labels)
        xy_err_pairs = [(x[label], y[label]) for label in errlabel_bandratio]
    else:
        xy_err_pairs = [(None, None)] * 3

    # Prepare x- and y-labels for the panels
    xlabels = [f"{xlabel}\n{quantity} ({hc.bandratio_labels[0]})", "", f"{quantity} ({hc.bandratio_labels[2]})\n{xlabel}"]
    ylabels = [f"{quantity} ({hc.bandratio_labels[0]})", f"{ylabel}\n{quantity} ({hc.bandratio_labels[1]})", f"{quantity} ({hc.bandratio_labels[2]})"]

    # Plot the data
    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(col1, 7.5), gridspec_kw={"hspace": 0.05, "wspace": 0.05})
    for xy, xy_err, ax, xlabel, ylabel in zip(xy_pairs, xy_err_pairs, axs, xlabels, ylabels):
        correlation_plot_simple(*xy, xerr=xy_err[0], yerr=xy_err[1], ax=ax, xlabel=xlabel, ylabel=ylabel, equal_aspect=True)

    # Axis settings
    data_combined = [element for sublist in xy_pairs for element in sublist]
    axmin, axmax = np.nanmin(data_combined)-0.05, np.nanmax(data_combined)+0.05
    axs[0].set_xlim(axmin, axmax)
    axs[0].set_ylim(axmin, axmax)
    for ax in axs:
        ax.set_aspect("equal")
        ax.locator_params(nbins=4)

    # Switch xtick labels on the bottom plot to the top
    axs[0].tick_params(axis="x", bottom=False, labelbottom=False, top=True, labeltop=True)
    axs[0].xaxis.set_label_position("top")
    axs[1].tick_params(axis="x", bottom=False, labelbottom=False)

    # Calculate statistics
    for ax, xy, xy_err in zip(axs, xy_pairs, xy_err_pairs):
        ax.set_title("")
        _plot_statistics(*xy, ax, xerr=xy_err[0], yerr=xy_err[1])
        stats.save_statistics_to_file(*xy, xy_err[0], xy_err[1], saveto=saveto_stats)

    # Set the title if wanted
    axs[0].set_title(title)

    # Save the result
    save_or_show(saveto)


def density_scatter(x, y, ax=None, sort=True, bins=20, **kwargs):
    """
    Plot points in a scatter plot, with colours corresponding to the density of points.
    This is useful for plotting large data sets with many overlapping points.

    Original: https://stackoverflow.com/a/53865762
    """
    if ax is None:
        fig, ax = plt.subplots()
    data, x_e, y_e = np.histogram2d(x, y, bins=bins, density=True)
    z = interpn((0.5*(x_e[1:] + x_e[:-1]), 0.5*(y_e[1:]+y_e[:-1])), data, np.vstack([x, y]).T, method="splinef2d", bounds_error=False)

    #To be sure to plot all data
    z[np.where(np.isnan(z))] = 0.0

    # Sort the points by density, so that the densest points are plotted last
    if sort:
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter(x, y, c=z, **kwargs)

    return ax


def plot_correlation_matrix_radiance(covariance_matrix, x1, y1, x2, y2, x1label="[a.u.]", y1label="[a.u.]", x2label="[a.u.]", y2label="[a.u.]", saveto=None):
    """
    Plot a given correlation matrix consisting of RGB or RGBG2 radiances.
    """
    # Calculate the correlation matrix
    correlation_matrix = stats.correlation_from_covariance(covariance_matrix)

    fig = plt.figure(figsize=(col1, col1), constrained_layout=True)
    gs = GridSpec(nrows=2, ncols=2, figure=fig)
    ax_imshow = fig.add_subplot(gs[0, :])
    axs_scatter = [fig.add_subplot(gs[1, j]) for j in (0, 1)]
    axs = [ax_imshow, *axs_scatter]

    divider = make_axes_locatable(axs[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = axs[0].imshow(correlation_matrix, extent=(0, 1, 0, 1), cmap=plt.cm.get_cmap("cividis", 5), vmin=0, vmax=1, origin="lower")
    fig.colorbar(im, cax=cax, orientation='vertical', ticks=np.arange(0, 1.1, 0.2), label="Pearson $r$")

    # Put the different radiances on the matrix plot ticks
    majorticks = np.linspace(0, 1, 4)
    minorticks = majorticks[:-1] + 1/6
    for axis in [axs[0].xaxis, axs[0].yaxis]:
        axis.set_ticks(majorticks)
        axis.set_ticks(minorticks, minor=True)
        axis.set_tick_params(which="major", labelleft=False, labelbottom=False)
        axis.set_tick_params(which="minor", left=False, bottom=False)
        axis.set_ticklabels([keys_latex[key] for key in ["Lu", "Lsky", "Ld"]], minor=True)

    # Parameters for density scatter plot
    kwargs = {"cmap": plt.cm.get_cmap("magma", 10), "s": 5, "rasterized": True}

    # Plot the density scatter plots
    for ax, x, y, xlabel, ylabel in zip(axs[1:], [x1, x2], [y1, y2], [x1label, x2label], [y1label, y2label]):
        density_scatter(x, y, ax=ax, **kwargs)

        # Plot parameters
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _textbox(ax, f"$r = {stats.correlation(x,y):.2g}$")

        ax.set_aspect("equal")
        ax.grid(alpha=0.5)

    # Move the y-axis label to the right on the second scatter plot
    axs[-1].yaxis.tick_right()
    axs[-1].yaxis.set_label_position("right")
    fig.align_xlabels(axs[1:])
    fig.align_ylabels(axs[1:])

    # Save the result
    save_or_show(saveto, dpi=600)


def _confidence_ellipse(center, covariance, ax, covariance_scale=1, **kwargs):
    """
    Plot a confidence ellipse from a given (2x2) covariance matrix.
    Original: https://matplotlib.org/devdocs/gallery/statistics/confidence_ellipse.html
    """
    correlation = stats.correlation_from_covariance(covariance)[0, 1]
    ell_radius_x = np.sqrt(1 + correlation)
    ell_radius_y = np.sqrt(1 - correlation)
    ellipse = Ellipse((0, 0), width=ell_radius_x*2, height=ell_radius_y*2, **kwargs)

    scale_x = np.sqrt(covariance[0, 0])*covariance_scale
    scale_y = np.sqrt(covariance[1, 1])*covariance_scale

    transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(*center)
    ellipse.set_transform(transf + ax.transData)

    return ax.add_patch(ellipse)


@new_or_existing_figure
def plot_xy_on_gamut_covariance(xy, xy_covariance, covariance_scale=1, ax=None, saveto=None):
    """
    Plot xy coordinates on the gamut including their covariance ellipse.
    """
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    _confidence_ellipse(xy, xy_covariance, ax, covariance_scale=covariance_scale, edgecolor="k", fill=False, linestyle="--")
    plt.scatter(*xy, c="k", s=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")


@new_or_existing_figure
def correlation_plot_hue_angle_and_ForelUle(x, y, xerr=None, yerr=None, xlabel="", ylabel="", ax=None, saveto=None, saveto_stats=stdout):
    """
    Make a correlation plot of hue angles (x and y).
    Draw the equivalent Forel-Ule indices on the grid for reference.
    """
    # Generate labels for the x and y axes
    xlabel_hue = f"{xlabel} $\\alpha$ [$^\\circ$]"
    ylabel_hue = f"{ylabel} $\\alpha$ [$^\\circ$]"
    xlabel_FU = f"{xlabel} FU index"
    ylabel_FU = f"{ylabel} FU index"

    # Plot the data
    correlation_plot_simple(x, y, xerr=xerr, yerr=yerr, ax=ax, equal_aspect=True)

    # Set the x and y labels for hue angle
    ax.set_xlabel(xlabel_hue)
    ax.set_ylabel(ylabel_hue)
    ax.grid(False)

    # Plot lines correspnding to the FU colour limits, and
    # colour the squares along the x=y line.
    line_kwargs = {"c": "k", "lw": 0.5}
    for fu, angle in enumerate(FU_hueangles):
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
    fu_start, fu_step = 2, 3
    FU_middles = np.array([(a + b)/2 for a, b in zip(FU_hueangles, FU_hueangles[1:])])[fu_start::fu_step]
    FU_labels = np.arange(1, 21)[fu_start::fu_step]

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
    N = len(x)
    mad_hue_angle, mad_FU, matches_percent, near_matches_percent = compare_hue_angles(x, y)
    stats_text = f"$N$ = {N}\n{stats.mad_symbol} = ${mad_hue_angle[0]:.1f} \\degree$\n{stats.mad_symbol} = {mad_FU[0]:.0f} FU\n{matches_percent[0]:.0f}% $\Delta$FU$= 0$\n{near_matches_percent[0]:.0f}% $\Delta$FU$\leq 1$"
    _textbox(ax, stats_text)

    # Put everything into a table
    labels = ["MAD (alpha)", "MAD (FU)", "FU matches", "FU near-matches"]
    stats_combined = np.array([mad_hue_angle, mad_FU, matches_percent, near_matches_percent])
    stats_combined = [labels, stats_combined[:,1], stats_combined[:,0], stats_combined[:,2]]
    stats_table = stats.table.Table(data=stats_combined, names=["Key", "P5", "Median", "P95"])
    stats_table.add_row(["N", N, N, N])

    # Save the table to file
    stats_table.write(saveto_stats, format="ascii.fixed_width", overwrite=True)



def compare_hyperspectral_datasets(datasets, parameter="R_rs", labels=None, saveto=None):
    """
    Make a plot comparing two hyperspectral data sets.
    Plot all spectra with the given parameter (R_rs by default) in adjacent panels, one panel per data set.
    """
    # If no labels were given, create an empty list
    if labels is None:
        labels = [None] * len(datasets)

    # Create a new figure
    fig, axs = plt.subplots(ncols=len(datasets), figsize=(col1, col1/2), sharex=True, sharey=True)

    # Plot the spectra
    for data, label, ax in zip(datasets, labels, axs):
        plot_hyperspectral_dataset(data, parameter=parameter, ax=ax)
        ax.set_title(f"{label} ({len(data)})")

    # Adjust the axis labels
    axs[0].set_xticks(axs[0].get_xticks()[1::2])
    axs[0].set_yticks(axs[0].get_yticks()[::2])
    for ax in axs[1:]:
        ax.set_ylabel(None)
        axs[1].tick_params(axis="y", left=False, labelleft=False)

    # Save/show the result
    save_or_show(saveto, dpi=600)
