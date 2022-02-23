"""
Empirically determine the grey card reflectance from hyperspectral comparison measurements.

Command-line inputs:
    * Data file with header
    * Any number of folders with SPECTACLE data for smartphone bands

Example:
    %run calculate_Rref_from_Ed_data.py water-colour-data/greycard_data_Maine.csv C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/ C:/Users/Burggraaff/SPECTACLE_data/Samsung_Galaxy_S8/
"""
from sys import argv
from matplotlib import pyplot as plt, rcParams
from spectacle import io
from astropy import table
from wk import plot

# Default plot settings
rcParams["lines.linewidth"] = 3

# Get the data folder from the command line
path_data, *paths_smartphones = io.path_from_input(argv)

# Load the data
data = table.Table.read(path_data)
print("Loaded data table")

# Plot the data
plt.figure(figsize=(plot.col1, plot.col1))
plt.plot(data["wavelength"], data["Ed"], label="Cosine collector", c='k')
plt.plot(data["wavelength"], data["Es_white"], label="White card")
plt.plot(data["wavelength"], data["Es_grey"], label="Grey card")
plt.xlabel("Wavelength [nm]")
plt.ylabel(f"Irradiance {plot.Wnm}")
plt.xlim(data["wavelength"].min(), data["wavelength"].max())
plt.ylim(ymin=0)
plt.title("Irradiance")
plt.legend(loc="best")
plt.grid(ls="--")
plt.show()
plt.close()

# Calculate Rref per wavelength
Rref_white = 0.99*data["Es_white"] / data["Ed"]
Rref_grey = 0.18*data["Es_grey"] / data["Ed"]

# Plot the data
fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(plot.col1, plot.col1), gridspec_kw={"hspace": 0.5})
for ax, y, nominal, label in zip(axs, [Rref_white, Rref_grey], [99, 18], ["White card", "Grey card"]):
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

# Calculate Rref in the smartphone bands
cameras = [io.load_camera(path) for path in paths_smartphones]
Rref_grey_RGB = [camera.convolve(data["wavelength"], Rref_grey)[:3] for camera in cameras]

print("Grey card reflectance in RGB bands:")
for camera, Rref in zip(cameras, Rref_grey_RGB):
    Rref_strings = [f"Rref ({band}) = {100*Rref_band:.1f}%" for band, Rref_band in zip("RGB", Rref)]
    print(camera.name)
    print(*Rref_strings, sep="\n")
    print("----")
