"""
Empirically determine the grey card reflectance from comparison measurements.

Command-line inputs:
    * Data file with header

Example:
    %run calculate_Rref_from_Ed_data.py water-colour-data/greycard_data_Maine.csv
"""
import numpy as np
from sys import argv
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams["lines.linewidth"] = 3
from spectacle import io, spectral, load_camera
from astropy import table
from wk import hydrocolor as hc, plot, statistics as stats

# Get the data folder from the command line
path = io.path_from_input(argv)

# Load the data
data = table.Table.read(path)
print("Loaded data table")

# Plot the data
plt.figure(figsize=(plot.col1, plot.col1))
plt.plot(data["wavelength"], data["Ed"], label="Cosine collector", c='k')
plt.plot(data["wavelength"], data["Es_white"], label="White card")
plt.plot(data["wavelength"], data["Es_grey"], label="Grey card")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Irradiance [W ...]")
plt.xlim(data["wavelength"].min(), data["wavelength"].max())
plt.ylim(ymin=0)
plt.title("Irradiance")
plt.legend(loc="best")
plt.grid(ls="--")
plt.show()
plt.close()

# Convert back to the original data
data["Es_white"] *= 0.99
data["Es_grey"] *= 0.18

# Calculate Rref per wavelength
Rref_white = data["Es_white"] / data["Ed"]
Rref_grey = data["Es_grey"] / data["Ed"]

# Plot the data
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(plot.col1, plot.col1), gridspec_kw={"hspace": 0.5})
for ax, y, nominal, label in zip(axs, [Rref_white, Rref_grey], [100, 18], ["White card", "Grey card"]):
    ax.plot(data["wavelength"], y*100)
    ax.axhline(nominal, c='k', ls="--", alpha=0.8)
    ax.set_title(label)
for ax in axs:
    ax.set_ylabel("Reflectance [%]")
    ax.grid(ls="--")
axs[0].set_xlim(data["wavelength"].min(), data["wavelength"].max())
axs[1].set_xlabel("Wavelength [nm]")
axs[0].tick_params(bottom=False, labelbottom=False)
plt.savefig("results/greycard_reflectance.pdf", bbox_inches="tight")
plt.show()
plt.close()
