"""
Compare RGB RAW and JPEG data from the same smartphone.

Command-line inputs:
    * path_raw: path to table with RAW data summary
    * path_jpeg: path to table with JPEG data summary
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io
from astropy import table
from datetime import datetime
from os import walk

# Get the data folder from the command line
path_raw, path_jpeg = io.path_from_input(argv)

table_raw = table.Table.read(path_raw)
table_jpeg = table.Table.read(path_jpeg)

max_val = 0

plt.figure(figsize=(5,5), tight_layout=True)
for c in "RGB":
    plt.errorbar(table_raw[f"R_rs ({c})"], table_jpeg[f"R_rs ({c})"], xerr=table_raw[f"R_rs_err ({c})"], yerr=table_jpeg[f"R_rs_err ({c})"], color=c, fmt="o")
    max_val = max(max_val, table_raw[f"R_rs ({c})"].max(), table_jpeg[f"R_rs ({c})"].max())
plt.plot([-1, 1], [-1, 1], c='k', ls="--")
plt.xlim(0, 1.05*max_val)
plt.ylim(0, 1.05*max_val)
plt.grid(True, ls="--")
plt.xlabel("RAW $R_{rs}$ [sr$^{-1}$]")
plt.ylabel("JPEG $R_{rs}$ [sr$^{-1}$]")
plt.savefig(f"comparison_RAW_X_JPEG.pdf")
plt.show()
