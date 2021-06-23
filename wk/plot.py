"""
Module with common plotting functions
"""
from matplotlib import pyplot as plt, transforms, patheffects as pe
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse

import numpy as np
from colorio._tools import plot_flat_gamut

from . import statistics as stats, colours
from .hydrocolor import RGBG2_to_RGB
from .wacodi import FU_hueangles, compare_FU_matches_from_hue_angle

from spectacle.plot import RGB_OkabeIto


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


def histogram_raw(water_data, sky_data, card_data, saveto, camera=None):
    """
    Draw histograms of RAW water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(11,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    for ax, img in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(img)
        ax.tick_params(bottom=False, labelbottom=False)

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
                data_RGB = RGBG2_to_RGB(data_RGBG)[0]

                # Plot the RGB histograms as lines on top of the black overall histograms
                _histogram_RGB(data_RGB, ax, bins=bins)

            # Plot settings
            ax.set_xlim(xmin, xmax)

    # Adjust the x- and y-axis ticks on the different panels
    _histogram_axis_settings(axs, ["Image", "Raw", "Bias-corrected", "Flat-fielded", "Central slice"])

    # Save the result
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def histogram_jpeg(water_data, sky_data, card_data, saveto, normalisation=255):
    """
    Draw histograms of RAW water/sky/grey card data at various steps of processing.
    """
    # Create the figure
    fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(9,4), gridspec_kw={"hspace": 0.04, "wspace": 0.04}, sharex="col", sharey="col")

    # Plot the original images in the left-most column
    for ax, image in zip(axs[:,0], [water_data[0], sky_data[0], card_data[0]]):
        ax.imshow(image)
        ax.tick_params(bottom=False, labelbottom=False)

    # Loop over the columns representing processing steps
    for ax_col, water, sky, card in zip(axs.T[1:], water_data, sky_data, card_data):
        # Always use the same bins, spanning the whole 8-bit range
        bins = np.linspace(0, normalisation, 50)

        # Loop over the three data sets (water/sky/grey card) and plot them in the three rows
        for ax, data in zip(ax_col, [water, sky, card]):
            # Draw the histogram
            ax.hist(data.ravel(), bins=bins, color="k")

            # Plot the RGB histograms as lines on top of the black overall histograms
            # Data array needs to be transposed so the colour axis is at the start
            _histogram_RGB(data.T, ax, bins=bins)

            # Plot settings
            ax.set_xlim(0, normalisation)

    # Adjust the x- and y-axis ticks on the different panels
    _histogram_axis_settings(axs, ["Image", "JPEG (full)", "Central slice"])

    # Save the result
    plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

    print(f"Saved statistics plot to `{saveto}`")


def plot_R_rs_RGB(RGB_wavelengths, R_rs, effective_bandwidths, R_rs_err, saveto=None):
    """
    Plot RGB R_rs data.
    """
    # Create the figure
    plt.figure(figsize=(3,2))

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
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


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

    ax.plot([-1e6, 1e6], [-1e6, 1e6], c='k', zorder=10)


def _plot_linear_regression(x, y, ax=None, color="k", **kwargs):
    """
    Helper function to plot linear regression lines consistently.
    """
    # Get the active Axes object if none was given
    if ax is None:
        ax = plt.gca()

    ax.plot(x, y, color=color, ls="--", zorder=10, **kwargs)


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
    _plot_diagonal(ax)

    # If wanted, perform a linear regression and plot the result
    if regression:
        params, params_cov, func = stats.linear_regression(x, y, xerr, yerr)
        x_plot = np.array([-1000., 1000.])
        y_plot = func(x_plot)
        _plot_linear_regression(x_plot, y_plot, ax)

    # Get statistics for title
    weights = None if xerr is None else 1/xerr**2
    r = stats.correlation(x, y, w=weights)
    title = f"$r$ = {r:.2g}"
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


def _correlation_plot_errorbars_RGB(ax, x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, setmax=True, equal_aspect=False, regression="none"):
    """
    Plot data into a correlation plot.
    Helper function.

    If `regression` is "none", don't do any linear regression.
    If "all", do one on all data combined.
    If "rgb", do a separate regression for each band.
    """
    xmax = 0.  # Maximum on x axis
    ymax = 0.  # Maximum on y axis

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
        xmax = max(xmax, np.nanmax(xdata))
        ymax = max(ymax, np.nanmax(ydata))

        # If wanted, perform a linear regression
        if regression == "rgb":
            regression_functions.append(stats.linear_regression(xdata, ydata, xerr, yerr)[2])

    # If wanted, perform a linear regression and plot the result
    x_plot = np.array([-1000., 1000.])
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

        y_plot = func(x_plot)
        _plot_linear_regression(x_plot, y_plot, ax)

    elif regression == "rgb":
        for func, c in zip(regression_functions, RGB_OkabeIto):
            y_plot = func(x_plot)
            _plot_linear_regression(x_plot, y_plot, ax, color=c, path_effects=[pe.Stroke(linewidth=3, foreground="k"), pe.Normal()])

    if setmax:
        if equal_aspect:
            xmax = ymax = max(xmax, ymax)
        ax.set_xlim(0, 1.05*xmax)
        ax.set_ylim(0, 1.05*ymax)


def correlation_plot_RGB(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", regression="none", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Create figure
    plt.figure(figsize=(4,4), tight_layout=True)

    # Plot in the one panel
    _correlation_plot_errorbars_RGB(plt.gca(), x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, regression=regression)

    # y=x line and grid lines
    _correlation_plot_gridlines()
    _plot_diagonal()

    # Get statistics for title
    r_all, r_RGB = stats.statistic_RGB(stats.correlation, x, y, xdatalabel, ydatalabel)
    if regression == "rgb":
        title = "   ".join(f"$r_{c}$ = {r_RGB[j]:.2g}" for j, c in enumerate(colours))
    else:
        title = f"$r$ = {r_all:.2g}"
    plt.title(title)

    # Labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Save, show, close plot
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def correlation_plot_RGB_equal(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, xlabel="x", ylabel="y", regression="none", saveto=None):
    """
    Make a correlation plot between two tables `x` and `y`. Use the labels
    `xdatalabel` and `ydatalabel`, which are assumed to have RGB versions.
    For example, if `xlabel` == `f"R_rs ({c})"` then the columns "R_rs (R)",
    "R_rs (G)", and "R_rs (B)" will be used.
    """
    # Calculate residuals
    residuals = stats.residual_table(x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel)

    # Create figure to hold plot
    fig, axs = plt.subplots(figsize=(4,5), nrows=2, sharex=True, gridspec_kw={"height_ratios": [3,1]})

    # Plot in both panels
    _correlation_plot_errorbars_RGB(axs[0], x, y, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, equal_aspect=True, regression=regression)
    _correlation_plot_errorbars_RGB(axs[1], x, residuals, xdatalabel, ydatalabel, xerrlabel=xerrlabel, yerrlabel=yerrlabel, setmax=False)

    # Plot the x=y line (top) and horizontal (bottom)
    for ax in axs:
        _correlation_plot_gridlines(ax)
    _plot_diagonal(axs[0])
    axs[1].axhline(0, c='k', ls="--")

    # Get statistics for title
    MAD_all, MAD_RGB = stats.statistic_RGB(stats.MAD, x, y, xdatalabel, ydatalabel)
    MAPD_all, MAPD_RGB = stats.statistic_RGB(stats.MAPD, x, y, xdatalabel, ydatalabel)
    r_all, r_RGB = stats.statistic_RGB(stats.correlation, x, y, xdatalabel, ydatalabel)

    x_all, y_all = stats.ravel_table(x, "R_rs ({c})"), stats.ravel_table(y, "R_rs ({c})")
    statistics_all, title = stats.full_statistics_for_title(x_all, y_all)
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
        _, statistic_text = stats.full_statistics_for_title(x, y)
        ax.set_title(statistic_text, fontdict={"fontsize": "small"})  # Replace old title

    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()


def comparison_histogram(x_table, y_table, param="Rrs {c}", xlabel="", ylabel="", quantity="", saveto=None):
    """
    Make a histogram of the ratio and difference in a given `param` for `x` and `y`
    """
    x = stats.ravel_table(x_table, param, colours)
    y = stats.ravel_table(y_table, param, colours)

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
        ax.set_title(f"$\mu$ = {np.nanmean(q):.3g}   $\sigma$ = {np.nanstd(q):.3g}")

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
    z = stats.interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False)

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
        ax.set_title("$r =" + f"{stats.correlation(x,y):.2g}" + "$")

        ax.set_aspect("equal")
        ax.grid(ls="--", c="0.5", alpha=0.5)

    axs[2].yaxis.set_label_position("right")
    axs[2].yaxis.tick_right()

    plt.subplots_adjust(wspace=0.5)
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.close()


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


def plot_xy_on_gamut_covariance(xy, xy_covariance, covariance_scale=1):
    """
    Plot xy coordinates on the gamut including their covariance ellipse.
    """
    fig = plt.figure(figsize=(3,3))
    plot_flat_gamut(plot_planckian_locus=False, axes_labels=("", ""))
    _confidence_ellipse(xy, xy_covariance, plt.gca(), covariance_scale=covariance_scale, edgecolor="k", fill=False, linestyle="--")
    plt.scatter(*xy, c="k", s=5)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.show()
    plt.close()


def correlation_plot_hue_angle_and_ForelUle(x, y, xerr=None, yerr=None, xlabel="", ylabel="", saveto=None):
    """
    Make a correlation plot of hue angles (x and y).
    Draw the equivalent Forel-Ule indices on the grid for reference.
    """
    # Generate labels for the x and y axes
    xlabel_hue = f"{xlabel}\nHue angle $\\alpha$ (degrees)"
    ylabel_hue = f"{ylabel}\nHue angle $\\alpha$ (degrees)"
    xlabel_FU = f"{xlabel}\nForel-Ule index"
    ylabel_FU = f"{ylabel}\nForel-Ule index"

    # Create figure
    fig, ax = plt.subplots(figsize=(4,4))

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
        ax.fill_between(square, *square, facecolor="0.5")

    # Same ticks on x and y.
    # Weird quirk in matplotlib: you need to do it twice (x->y then y->x)
    # to actually get the same ticks on both axes.
    ax.set_yticks(ax.get_xticks())
    ax.set_xticks(ax.get_yticks())

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

    mad_hueangle = stats.MAD(x, y)
    FU_matches, FU_near_matches, mad_FU = compare_FU_matches_from_hue_angle(x, y)

    title = f"MAD = ${mad_hueangle:.1f} \\degree$ ({mad_FU:.0f} FU)\n{FU_matches:.0f}% $\Delta$FU$= 0$   {FU_near_matches:.0f}% $\Delta$FU$\leq 1$"
    ax.set_title(title)

    # Number of matches (Delta 0)
    # Number of near-matches (Delta 1)

    # Save and close the plot
    if saveto is not None:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()
