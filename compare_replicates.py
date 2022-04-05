"""
Analyse replicate smartphone images that have been pre-processed using process_replicates.py

Command-line inputs:
    * path_data: Path to the combined data file.

Example:
    %run compare_replicates.py water-colour-data/balaton_Samsung_Galaxy_S8_raw_replicates.csv
"""
from sys import argv
import numpy as np
from spectacle import io, symmetric_percentiles
from astropy import table
from matplotlib import pyplot as plt
from wk import hydrocolor as hc, plot, hyperspectral as hy

# Get the data folder from the command line
path_data = io.path_from_input(argv)

# Name of the phone and some useful labels
phone_name = " ".join(path_data.stem.split("_")[1:-2])
saveto_base = f"results/replicates_{phone_name}"
print(f"Analysing replicate data from {phone_name}. Results will be saved to '{saveto_base}_XXX.pdf'.")

# Read the data
data = hc.table.Table.read(path_data)

# Multiply the FU index values by a factor so they fit in the plot
FU_scale_factor = 10.
data["R_rs (FU)"] *= FU_scale_factor

# Turn the data into an array matplotlib can understand
keys = ["Lu", "Lsky", "Ld", "R_rs"]
keys_RGB = hc.extend_keys_to_RGB(keys) + hc.extend_keys_to_RGB("R_rs", hc.bandratio_labels+["hue", "FU"])
data_RGB = hy.convert_columns_to_array(data, keys_RGB)
data_max = data_RGB.max()
ymax = 20.  # %/degree for all except FU

# Plot parameters
# We want the boxes for the same parameter to be close together
positions = np.ravel(np.array([0, 0.6, 1.2]) + 2.5*np.arange(5)[:, np.newaxis])
positions = np.append(positions, positions[-1]+np.array([0.8, 1.6]))  # Positions for hue angle/FU
labels = sum([["$R$", "$G$\n"+plot.keys_latex[key], "$B$"] for key in keys], start=[]) + hc.bandratio_labels_latex + [r"$\alpha$", "FU"]
# labels[-4] += "\n" + plot.keys_latex["R_rs"]
colours = plot.RGB_OkabeIto * 4 + 5*["k"]

# Make a box-plot of the relative uncertainties
fig, axs = plt.subplots(ncols=2, figsize=(plot.col2, 0.25*plot.col2), gridspec_kw={"width_ratios": [10, 2], "wspace": 0.25})

# Plot the data
bplot_kwargs = dict(sym=".", patch_artist=True)
bplot = axs[0].boxplot(data_RGB[:,:-2], positions=positions[:-2], labels=labels[:-2], **bplot_kwargs)
axs[1].boxplot(data_RGB[:,-2:], positions=positions[-2:], labels=labels[-2:], boxprops={"facecolor": "k"}, widths=0.4, **bplot_kwargs)

# Adjust the colours
for patch, colour in zip(bplot["boxes"], colours):
    patch.set_facecolor(colour)

# Plot settings
axs[0].set_yticks(np.arange(0, ymax+5, 5))
axs[0].set_ylim(0, ymax)
axs[0].set_ylabel(r"Uncertainty [%]")
for item in axs[0].get_xticklabels()[-3:]:
    item.set_fontsize(14)

axs[1].tick_params(axis="y", left=True, labelleft=True)
axs[1].set_ylabel(r"Uncertainty [$^\circ$]")
axs[1].set_ylim(ymin=0)
axs[1].axvline(np.mean(positions[-2:]), c='k', lw=plot.rcParams["axes.linewidth"])  # Vertical line in the second panel

for ax in axs:
    ax.tick_params(axis="x", bottom=False)
    ax.grid(axis="y", ls="--")

# Add a second y-axis for FU
ax2 = axs[1].twinx()
ax2.set_yticks(axs[1].get_yticks()/FU_scale_factor)
ax2.set_ylim(np.array(axs[1].get_ylim())/FU_scale_factor)
ax2.set_ylabel("Uncertainty [FU]")
ymax_FU = ax2.get_ylim()[1]  # Scale the ymax by a factor for FU

fig.suptitle("Variations between replicate images")

# Save/show the result
plot.save_or_show(f"{saveto_base}_relative_uncertainty.pdf")

# Divide the FU index values by a factor so we get correct statistics
data["R_rs (FU)"] /= FU_scale_factor
data_RGB[:,-1] = data["R_rs (FU)"]

# Get combined RGB statistics
data_RGB_combined = np.array([np.ravel(data_RGB[:,3*i:3*(i+1)]) for i in range(5)]).T  # Very ugly but .reshape screws with the order of elements
q1_combined, q3_combined = symmetric_percentiles(data_RGB_combined, percent=25, axis=0)
medians_combined = np.nanmedian(data_RGB_combined, axis=0)
nr_above_ymax_combined = (data_RGB_combined > ymax).sum(axis=0)
keys_combined = hc.extend_keys_to_RGB(keys+["B.R."], ["all"])

# Calculate and print statistics
q1, q3 = symmetric_percentiles(data_RGB, percent=25, axis=0)  # Outer limits of the first and third quartiles
medians = np.nanmedian(data_RGB, axis=0)
nr_above_ymax = (data_RGB > ymax).sum(axis=0)
nr_above_ymax[-1] = (data_RGB[-1] > ymax_FU).sum()
stats_table = table.Table(data=[keys_RGB, q1, medians, q3, nr_above_ymax], names=["Key", "Q1", "Median", "Q3", "# out of bounds"])
stats_combined_table = table.Table(data=[keys_combined, q1_combined, medians_combined, q3_combined, nr_above_ymax_combined], names=["Key", "Q1", "Median", "Q3", "# out of bounds"])
stats_table = table.vstack([stats_table, stats_combined_table])
stats_table.write(f"{saveto_base}_statistics.dat", format="ascii.fixed_width", overwrite=True)
