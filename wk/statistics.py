"""
Functions and variables used for statistical analysis.
Many of these will be moved to SPECTACLE in the near future.
"""
from functools import partial
import numpy as np
from scipy import odr, stats
from wquantiles import median as weighted_median_wq
from spectacle.general import symmetric_percentiles, weighted_mean, uncertainty_from_covariance, correlation_from_covariance
from . import colours

mad_symbol = r"$\mathcal{M}$"
sspb_symbol = r"$\mathcal{B}$"


def weighted_median(x, w=None):
    """
    Weighted median.
    If no weights are supplied, normal median.
    """
    if w is None:
        return np.nanmedian(x)
    else:
        return weighted_median_wq(x, w)


def correlation(x, y, w=None):
    """
    Calculate the Pearson r correlation coefficient between data sets x and y.
    Optional parameter w for weights.
    """
    return correlation_from_covariance(np.cov(x, y, aweights=w))[0, 1]


def correlation_with_confidence_interval(x, y, w=None, alpha=0.05):
    """
    Calculate the sample correlation coefficient between data sets x and y with the confidence interval.
    Optional parameter w for weights (currently not used for the confidence interval).
    """
    # Calculate the correlation coefficient itself first
    r = correlation(x, y, w=w)

    # Calculate the confidence interval
    n = len(x)
    z_half_alpha = stats.norm.ppf(1-alpha/2)
    theta = 0.5*(np.log(1+r) - np.log(1-r))
    a, b = theta - z_half_alpha/np.sqrt(n-3), theta + z_half_alpha/np.sqrt(n-3)
    rmin = (np.exp(2*a) - 1) / (np.exp(2*a) + 1)
    rmax = (np.exp(2*b) - 1) / (np.exp(2*b) + 1)

    return r, rmin, rmax


def max_correlation_in_covariance_matrix(covariance):
    """
    Find the highest correlation in a covariance matrix, excluding the diagonals.
    """
    correlation_matrix = correlation_from_covariance(covariance)
    not_diagonal = ~np.eye(len(covariance), dtype=bool)  # Off-diagonal elements
    max_correlation = np.nanmax(correlation_matrix[not_diagonal])

    return max_correlation


def MAD(x, y, w=None):
    """
    Median absolute deviation (sometimes MAE) between data sets x and y.
    """
    return weighted_median(np.abs(y-x), w=w)


def MAPD(x, y):
    """
    Symmetric median absolute percentage deviation (sometimes sMAPE) between data sets x and y.
    Normalised relative to (x+y)/2 rather than just x or y.
    Expressed in %, so already multiplied by 100.
    """
    normalisation = (x+y)/2  # Normalisation factors
    MAD_relative = np.nanmedian(np.abs((y-x)/normalisation))
    MAD_percent = MAD_relative * 100.
    return MAD_percent


def zeta(x, y):
    """
    Log accuracy ratio between data sets x and y.
    """
    logR = np.abs(np.log(y/x))
    zeta = 100 * (np.exp(np.nanmedian(logR)) - 1)
    return zeta


def SSPB(x, y):
    """
    Symmetric signed percentage bias between data sets x and y.
    """
    MlogQ = np.nanmedian(np.log(y/x))
    SSPB = 100 * np.sign(MlogQ) * (np.exp(np.abs(MlogQ)) - 1)
    return SSPB


def statistic_with_bootstrap(data, statistic: callable, vectorized=False, paired=True, **kwargs):
    """
    Calculate a statistic with confidence intervals from bootstrapping.
    `data` must contain all data in one iterable, e.g. data=(x,y).
    """
    statistic_mean = statistic(*data)
    statistic_bootstrap = stats.bootstrap(data, statistic, vectorized=vectorized, paired=paired, **kwargs)
    statistic_low, statistic_high = statistic_bootstrap.confidence_interval.low, statistic_bootstrap.confidence_interval.high

    return statistic_mean, statistic_low, statistic_high


def ravel_table(data, key, loop_keys=colours):
    """
    Apply np.ravel to a number of columns, e.g. to combine R_rs R, R_rs G, R_rs B into one array for all R_rs.
    data is the input table.
    key is the fixed key, e.g. "R_rs".
    loop_keys is an iterable containing the keys to loop over, default "RGB".
    """
    return np.ravel([data[key.format(c=c)] for c in loop_keys])


def statistic_RGB(func, data1, data2, xdatalabel, ydatalabel, loop_keys=colours):
    """
    Calculate a statistic (e.g. MAD, MAPD, RMSE) in a given parameter `param`, e.g. Rrs, between two Astropy data tables.
    Assumes the same key structure in each table, namely `{param} ({c})` where c is R, G, or B.

    Returns the statistic overall and per band.
    """
    stat_RGB = np.array([func(data1[xdatalabel.format(c=c)], data2[ydatalabel.format(c=c)]) for c in loop_keys])
    data1_combined = ravel_table(data1, xdatalabel, loop_keys)
    data2_combined = ravel_table(data2, ydatalabel, loop_keys)
    stat_all = func(data1_combined, data2_combined)

    return stat_all, stat_RGB


def residual_table(x, y, xdatalabel, ydatalabel, xerrlabel=None, yerrlabel=None, loop_keys=colours):
    """
    Calculate the column-wise RGB residuals between two tables for given labels `xdatalabel` and `ydatalabel` (e.g. "R_rs ({c})").
    {c} is filled in with every value of `loop_keys` - by default this is `colours` (RGB) but it can also be something else, e.g. the band ratios.
    If uncertainties are included, propagate them (sum of squares).
    Returns a new table with differences.
    """
    # Use a copy of x to store the residuals in
    result = x.copy()

    # Remove columns that are in x but not in y
    keys_not_overlapping = [key for key in result.keys() if key not in y.keys()]
    result.remove_columns(keys_not_overlapping)

    # Loop over the keys and update them to include x-y instead of just x
    for c in loop_keys:
        result[xdatalabel.format(c=c)] = y[ydatalabel.format(c=c)] - x[xdatalabel.format(c=c)]
        if xerrlabel and yerrlabel:
            result[xerrlabel.format(c=c)] = np.sqrt(x[xerrlabel.format(c=c)]**2 + y[yerrlabel.format(c=c)]**2)

    return result


def linear_regression(x, y, xerr=0, yerr=0):
    """
    Linear regression, using uncertainties in x and y.
    https://docs.scipy.org/doc/scipy/reference/odr.html
    """
    def linear(params, x):
        return params[0]*x + params[1]

    data = odr.RealData(x, y, sx=xerr, sy=yerr)
    model = odr.Model(linear)  # Ignore Jacobians for now

    odr_holder = odr.ODR(data, model, beta0=[1., 0.])
    output = odr_holder.run()

    params = output.beta
    params_cov = output.cov_beta
    output_function = lambda x: linear(params, x)

    return params, params_cov, output_function


def full_statistics_for_title(x, y, xerr=None, yerr=None):
    """
    Calculate the Pearson correlation r, median absolute deviation (MAD), zeta, and SSPB between x and y, and format them nicely.
    Returns the calculated values (in a list) and a formatted string.
    """
    # Calculate weights if desired
    if xerr is not None and yerr is not None:
        weights = 1/(xerr**2 + yerr**2)
        weights_relative = 1/(xerr**2/x**2 + yerr**2/y**2)
    else:
        weights = weights_relative = None

    # Calculate the statistics
    N = len(x)
    if weights is not None:
        r, mad = [statistic_with_bootstrap((x,y,weights), func) for func in (correlation, MAD)]
    else:
        r, mad = [statistic_with_bootstrap((x,y), func) for func in (correlation, MAD)]
    z, sspb = [statistic_with_bootstrap((x,y), func) for func in (zeta, SSPB)]

    # Put everything into a nice string and return it
    parts = [f"$N$ = {N}", f"$r$ = {r[0]:.2g}", f"({r[1]:.2g}$-${r[2]:.2g})", f"{mad_symbol} = {mad[0]:.2g}", f"({mad[1]:.2g}$-$\n{mad[2]:.2g})", f"$\zeta$ = {z[0]:.2g}%", f"({z[1]:.2g}$-${z[2]:.2g})%", f"{sspb_symbol} = {sspb[0]:+.2g}%", f"({sspb[1]:.2g} $-$ {sspb[2]:.2g})%"]
    statistic_text = "\n".join(parts)
    return [r, mad, z, sspb], statistic_text
