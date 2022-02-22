"""
Analyse replicate smartphone images that have been pre-processed using process_replicates.py

Command-line inputs:
    * path_data: Path to the combined data file.

Example:
    %run compare_replicates.py water-colour-data/balaton_Samsung_Galaxy_S8_raw_replicates.csv
"""

import numpy as np
from sys import argv
from spectacle import io, symmetric_percentiles
from astropy import table
from wk import hydrocolor as hc, plot, hyperspectral as hy
from matplotlib import pyplot as plt

# Get the data folder from the command line
path_data = io.path_from_input(argv)

# Name of the phone and some useful labels
phone_name = " ".join(path_data.stem.split("_")[1:-2])
saveto_base = f"results/replicates_{phone_name}"
print(f"Analysing replicate data from {phone_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
data = hc.table.Table.read(path_data)

# Multiply the FU index values by a factor so they fit in the plot
FU_scale_factor = 5.
data["R_rs (FU)"] *= FU_scale_factor

# Change the order of the band ratio labels for the plot
roll_labels = lambda labels: list(np.roll(labels, 1))
bandratio_labels = roll_labels(hc.bandratio_labels)
bandratio_labels_latex = roll_labels(hc.bandratio_labels_latex)

# Turn the data into an array matplotlib can understand
keys = ["Lu", "Lsky", "Ld", "R_rs"]
keys_RGB = hc.extend_keys_to_RGB(keys) + hc.extend_keys_to_RGB("R_rs", bandratio_labels+["hue", "FU"])
data_RGB = hy.convert_columns_to_array(data, keys_RGB)
data_max = data_RGB.max()
ymax = 20.  # %/degree for all except FU
ymax_FU = ymax/FU_scale_factor  # Scale the ymax by a factor for FU

# Plot parameters
# We want the boxes for the same parameter to be close together
positions = np.ravel(np.array([0, 0.6, 1.2]) + 2.5*np.arange(5)[:,np.newaxis])
positions = np.append(positions, positions[-1]+np.array([0.8, 1.6]))  # Positions for hue angle/FU
labels = sum([["$R$", "$G$\n"+plot.keys_latex[key], "$B$"] for key in keys], start=[]) + bandratio_labels_latex + [r"$\alpha$", "FU"]
labels[-4] += "\n" + plot.keys_latex["R_rs"]
colours = plot.RGB_OkabeIto * 4 + 5*["k"]

# Make a box-plot of the relative uncertainties
fig = plt.figure(figsize=(plot.col1, 0.5*plot.col1))

# Plot the data
bplot = plt.boxplot(data_RGB, positions=positions, labels=labels, sym=".", patch_artist=True)

# Adjust the colours
for patch, colour in zip(bplot["boxes"], colours):
    patch.set_facecolor(colour)

# Plot settings
plt.yticks(np.arange(0, ymax+5, 5))
plt.ylim(0, ymax)
plt.ylabel("Uncertainty [%, $^\circ$]")
plt.tick_params(axis="x", bottom=False)
plt.grid(axis="y", ls="--")
plt.title("Variations between replicate images")

# Add a second y-axis for FU
ax1 = plt.gca()
ax2 = ax1.twinx()
ax2.set_yticks(ax1.get_yticks()/5)
ax2.set_ylim(np.array(ax1.get_ylim())/5)
ax2.set_ylabel("Uncertainty [FU]")

# Save/show the result
plot._saveshow(f"{saveto_base}_relative_uncertainty.pdf")

# Calculate and print statistics
pct5, pct95 = symmetric_percentiles(data_RGB, percent=5, axis=0)
medians = np.nanmedian(data_RGB, axis=0)
nr_above_ymax = (data_RGB > ymax).sum(axis=0)
nr_above_ymax[-1] = (data_RGB[-1] > ymax_FU).sum()
stats_table = table.Table(data=[keys_RGB, pct5, medians, pct95, nr_above_ymax], names=["Key", "5%", "Median", "95%", "# out of bounds"])
stats_table.write(f"{saveto_base}_statistics.dat", format="ascii.fixed_width", overwrite=True)
