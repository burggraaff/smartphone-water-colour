"""
Process data from the PML So-Rad-mounted TriOS.

Command line inputs:
    * Folder containing the So-Rad data file.
        Example: "water-colour-data\Balaton_20190703\SoRad"

Outputs:
    * None

To do:
    * Subtract offset
    * Import Ed, calculate Lw
    * Export results
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path

# Get filenames
folder = Path(argv[1])
print("Input folder:", folder.absolute())

filename = folder/"So-Rad_Rrs_Balaton2019.csv"

# Function that converts a row of text to an array
def convert_row(row):
    row_split = row.split(";")
    start = row_split[:-1]
    end = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = start + end
    return row_final

# Read data
wavelengths = np.arange(320, 955, 3.3)
print("Now reading data from", filename)
with open(filename) as file:
    data = file.readlines()
    header = data[0]
    data = data[1:]
    cols = header.split(";")[:-1] + [f"Rrs_{wvl:.1f}" for wvl in wavelengths]

    rows = [convert_row(row) for row in data]
    dtypes = ["S30" for h in header.split(";")[:-1]] + [np.float32 for wvl in wavelengths]

    data = table.Table(rows=rows, names=cols, dtype=dtypes)

print("Finished reading data")

# Plot sample of data
Rrs = np.array([[row[f"Rrs_{wvl:.1f}"] for wvl in wavelengths] for row in data[::100]]).T

plt.figure(figsize=(6,3), tight_layout=True)
plt.plot(wavelengths, Rrs, c="k", alpha=0.1)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
plt.xlim(320, 955)
plt.ylim(0, 0.1)
plt.yticks(np.arange(0, 0.12, 0.02))
plt.grid(ls="--")
plt.title("Example R_rs spectra")
plt.show()
plt.close()
