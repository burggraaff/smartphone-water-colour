"""
Process data from the Stirling TriOS.

Command line inputs:
    * Folder containing the TriOS data file.
        Example: "water-colour-data\Data_Monocle2019_L5All.csv"

Outputs:
    * File containing Ed, Ls, Lt, R_rs
        Example: "water-colour-data\Data_Monocle2019_L5All_table.csv"
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path
from datetime import datetime

from wk import hydrocolor as hc, wacodi as wa

# Get filenames
filename = Path(argv[1])
print("Input file:", filename.absolute())
saveto = filename.parent / (filename.stem + "_TriOS_table.csv")


# Label that matches column header
def label(text, wvl):
    return f"{text}_{wvl:.1f}"

# Read data
data = table.Table.read(filename)
print("Finished reading data")

# Add UTC timestamps
data_datetime = [datetime.fromisoformat(DT) for DT in data["DateTime"]]
data_timestamps = [dt.timestamp() for dt in data_datetime]
data.add_column(table.Column(data=data_timestamps, name="UTC"))
data.sort("UTC")

# Remove unused columns
data.remove_columns(["EdPAR", "EdDevice"])

# Rename columns from Rrs to R_rs
R_rs_columns = [key for key in data.keys() if "Rrs_" in key]
wavelengths = np.array([float(key[4:]) for key in R_rs_columns])
R_rs_columns_new = [key.replace("Rrs", "R_rs") for key in R_rs_columns]
data.rename_columns(R_rs_columns, R_rs_columns_new)
print("Renamed columns from Rrs to R_rs")

# Add dummy Ed, Lu, Lsky columns
data = hc.add_dummy_columns(data)
print("Added dummy radiance columns")

# Add WACODI data - XYZ, xy, hue angle, Forel-Ule
data = wa.add_colour_data_to_table(data)

# Write data to file
data.write(saveto, format="ascii.fast_csv")

# Plot sample of data
def plot_sample(data_plot, sample_quantity, ylabel="", saveto=None):
    sample_cols = [label(sample_quantity, wvl) for wvl in wavelengths]
    data_sub = data_plot[sample_cols]
    data_sub = np.array([data_sub[col].data for col in data_sub.colnames])  # Iteration over data_sub.columns does not work

    plt.figure(figsize=(6,3), tight_layout=True)
    plt.plot(wavelengths, data_sub, c="k", alpha=0.1)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel(ylabel)
    plt.xlim(320, 955)
    plt.ylim(ymin=0)
    plt.grid(ls="--")
    plt.title(f"Example {sample_quantity} spectra ({data_sub.shape[1]}/{len(data_plot)})")
    if saveto:
        plt.savefig(saveto, bbox_inches="tight")
    plt.show()
    plt.close()

# Plot R_rs
filename_R_rs = f"results/{filename.stem}.pdf"
plot_sample(data, "R_rs", ylabel="$R_{rs}$ [sr$^{-1}$]", saveto=filename_R_rs)
