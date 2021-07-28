"""
Empirically determine the grey card reflectance from smartphone-reference comparisons.
Only Lu is used because Lsky tends to be very noisy.
While Lu and Ld also have noise, because they are affected by the whole sky and not just a small section of it, they are less time-sensitive.

Command-line inputs:
    * Any number of smartphone-reference comparison tables (compiled with compare_phone_reference.py)

Example:
    %run compare_phone_reference.py "results/comparison_WISP-3_X_iPhone SE_RAW_data.csv" "results/comparison_WISP-3_X_Samsung Galaxy S8_RAW_data.csv"
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, spectral, load_camera
from astropy import table
from wk import hydrocolor as hc, plot, statistics as stats

# Get the data folder from the command line
paths = io.path_from_input(argv)

# If a single path was given, make a list anyway
if not isinstance(paths, list):
    paths = [paths]

# Load the data
data = [table.Table.read(p) for p in paths]
print(f"Loaded {len(data)} data table(s)")

# Remove the hyperspectral columns
for d in data:
    d.remove_columns([key for key in d.keys() if key[-2:] == ".0"])
print("Removed hyperspectral columns")

# Get all data corresponding to a key from all tables
def ravel_all(tables, key):
    """
    Apply stats.ravel_table to a list of tables and concatenate the result.
    """
    data_all = np.concatenate([stats.ravel_table(t, key) for t in tables])
    return data_all

# Get the (ir)radiance data
Lu_ref, Ed_ref, Lu_err_ref, Ed_err_ref = [ravel_all(data, key + " ({c})_reference") for key in ["Lu", "Ed", "Lu_err", "Ed_err"]]
Lu_phone, Lu_err_phone  = [ravel_all(data, key + " ({c})_phone") for key in ["Lu", "Lu_err"]]
Ld_phone, Ld_err_phone = [ravel_all(data, key + " ({c})") for key in ["Ld", "Ld_err"]]

# Calculate Rref
Rref = np.pi * (Ld_phone / Ed_ref) * (Lu_ref / Lu_phone)
Rref_err = Rref * np.sqrt( (Ld_err_phone/Ld_phone)**2 + (Ed_err_ref/Ed_ref)**2 + (Lu_err_ref/Lu_ref)**2 + (Lu_err_phone/Lu_phone)**2)

# Weighted mean
Rref_mean, Rref_uncertainty = stats.weighted_mean(Rref, 1/Rref_err**2)

print(f"{Rref_mean}, {Rref_uncertainty}")

print(f"Estimated grey card reflectance: R_ref = {Rref_mean:.2g} +- {Rref_uncertainty:.1g}")
