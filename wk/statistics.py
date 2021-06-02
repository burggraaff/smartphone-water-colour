"""
Module with some statistics used in the analysis
"""
import numpy as np
from scipy import odr

from . import colours

# Pearson r correlation coefficient
correlation = lambda x, y: np.corrcoef(x, y)[0, 1]


def MAD(x, y):
    """
    Median absolute deviation (sometimes MAE) between data sets x and y.
    """
    return np.nanmedian(np.abs(y-x))


def MAPD(x, y):
    """
    Symmetric median absolute percentage deviation (sometimes sMAPE) between
    data sets x and y.
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


def ravel_table(data, key, loop_keys="RGB"):
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
    in each table, namely `{param} {c}` where c is R, G, or B.

    Returns the statistic overall and per band.
    """
    stat_RGB = np.array([func(data1[xdatalabel.format(c=c)], data2[ydatalabel.format(c=c)]) for c in colours])
    data1_combined = ravel_table(data1, xdatalabel, colours)
    data2_combined = ravel_table(data2, ydatalabel, colours)
    stat_all = func(data1_combined, data2_combined)

    return stat_all, stat_RGB


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


def full_statistics_for_title(x, y):
    """
    Calculate the Pearson correlation r, median absolute deviation (MAD),
    zeta, and SSPB between x and y, and format them nicely.
    """
    r, mad, z, sspb = correlation(x, y), MAD(x, y), zeta(x, y), SSPB(x, y)
    statistic_text = f"r = {r:.2g}     MAD = {mad:.2g}   \n$\zeta$ = {z:.2g}%   SSPB = {sspb:+.2g}%"
    return [r, mad, z, sspb], statistic_text
