"""
Analyse replicate smartphone images that have been pre-processed using process_replicates.py

Command-line inputs:
    * path_data: Path to the combined data file.
"""

import numpy as np
from sys import argv
from spectacle import io, symmetric_percentiles
from astropy import table
from wk import hydrocolor as hc, plot
from matplotlib import pyplot as plt

# Get the data folder from the command line
path_data = io.path_from_input(argv)

# Names of the two phones being compared and some useful labels
phone_name = " ".join(path_data.stem.split("_")[1:-2])
saveto_base = f"results/replicates_{phone_name}"
print(f"Analysing replicate data from {phone_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
data = hc.table.Table.read(path_data)

# Turn the data into an array matplotlib can understand
keys = ["Lu", "Lsky", "Ld", "R_rs"]
nr_keys = len(keys)
keys_RGB = hc.extend_keys_to_RGB(keys)
data_RGB = np.stack([data[key_RGB] for key_RGB in keys_RGB])
data_max = data_RGB.max()

# Plot parameters
# We want the boxes for the same parameter to be close together
positions = np.ravel(np.array([0, 0.6, 1.2]) + 2.5*np.arange(nr_keys)[:,np.newaxis])
labels = np.ravel([["", plot.keys_latex[key], ""] for key in keys])
colours = plot.RGB_OkabeIto * nr_keys

# Make a box-plot of the relative uncertainties
fig = plt.figure(figsize=(plot.col1, plot.col1))

# Plot the data
bplot = plt.boxplot(data_RGB.T, positions=positions, labels=labels, patch_artist=True)

# Adjust the colours
for patch, colour in zip(bplot["boxes"], colours):
    patch.set_facecolor(colour)

# Plot settings
plt.yticks(np.arange(0, data_max+5, 5))
plt.ylim(0, data_max*1.05)
plt.ylabel("Relative uncertainty [%]")
plt.tick_params(axis="x", bottom=False)
plt.grid(axis="y", ls="--")

# Save/show the result
plot._saveshow(f"{saveto_base}_relative_uncertainty.pdf")

# Calculate and print statistics
pct5, pct95 = symmetric_percentiles(data_RGB, percent=5, axis=1)
medians = np.nanmedian(data_RGB, axis=1)
stats_table = table.Table(data=[keys_RGB, pct5, medians, pct95], names=["Key", "5%", "Median", "95%"])
with open(f"{saveto_base}_statistics.dat", "w") as file:
    print(stats_table, file=file)
