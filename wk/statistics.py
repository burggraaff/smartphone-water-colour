"""
Module with some statistics used in the analysis
"""
import numpy as np
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
    in each table, namely `{param} {c}` where c is R, G, or B.

    Returns the statistic overall and per band.
    """
    stat_RGB = np.array([func(data1[xdatalabel.format(c=c)], data2[ydatalabel.format(c=c)]) for c in colours])
    data1_combined = ravel_table(data1, xdatalabel, colours)
    data2_combined = ravel_table(data2, ydatalabel, colours)
    stat_all = func(data1_combined, data2_combined)

    return stat_all, stat_RGB
