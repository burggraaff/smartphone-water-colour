"""
Process data from the PML So-Rad-mounted TriOS.

Command line inputs:
    * Folder containing the So-Rad data file.
        Example: "water-colour-data\Balaton_20190703\SoRad"

Outputs:
    * None

To do:
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

filename_Ed = folder/"So-Rad_Ed_Balaton2019.csv"
filename_Ls = folder/"So-Rad_Ls_Balaton2019.csv"
filename_Lt = folder/"So-Rad_Lt_Balaton2019.csv"
filename_Rrs = folder/"So-Rad_Rrs_Balaton2019.csv"

wavelengths = np.arange(320, 955, 3.3)

# Function that converts a row of text to an array
def convert_row(row):
    row_split = row.split(";")
    metadata = row_split[:-1]  # GPS data, alt/az, rho, offset
    radiometry = np.array(row_split[-1].split(","), dtype=np.float32).tolist()
    row_final = metadata + radiometry
    return row_final

# Label that matches column header
def label(text, wvl):
    return f"{text}_{wvl:.1f}"

# Function to read data
def read_data(filename, nr_columns_as_float=0):
    """
    Read a So-Rad data file from `filename`

    nr_columns_as_float is an integer that controls how many columns in the header
    (i.e. not the spectrum itself) should be cast to floats, not strings.
    Example: for Rrs, this should be 2: rho and offset
    """
    datatype = filename.stem.split("_")[1]  # Ed, Rrs, etc.
    data_columns = [label(datatype, wvl) for wvl in wavelengths]
    print("Now reading data from", filename)
    with open(filename) as file:
        data = file.readlines()
        header, data = data[0], data[1:]
        header_columns = header.split(";")[:-1]
        columns = header_columns + data_columns

        rows = [convert_row(row) for row in data]
        # Convert header columns, except the last `nr_columns_as_float`, to strings
        # Convert everything else (spectrum + last header columns) to floats
        dtypes = ["S30"] * (len(header_columns) - nr_columns_as_float) + [np.float32] * (len(wavelengths) + nr_columns_as_float)

        data = table.Table(rows=rows, names=columns, dtype=dtypes)

    return data

# Read data
data_ed = read_data(filename_Ed)
data_ls = read_data(filename_Ls)
data_lt = read_data(filename_Lt)
data_rrs = read_data(filename_Rrs, nr_columns_as_float=2)

# Join tables into one
data = table.join(data_ed, data_ls)
data = table.join(data, data_lt)
data = table.join(data, data_rrs)

Rrs_columns = [label("Rrs", wvl) for wvl in wavelengths]

print("Finished reading data")

# Plot histograms at multiple wavelengths
def plot_histograms(data_plot, wavelengths_hist=[353.0, 402.5, 501.5, 600.5, 702.8, 801.8, 900.8], bins=np.linspace(-0.02, 0.10, 75)):
    fig, axs = plt.subplots(ncols=len(wavelengths_hist), sharex=True, sharey=True, figsize=(10, 2))
    for wvl, ax in zip(wavelengths_hist, axs):
        wvl_label = label("Rrs", wvl)
        data_wvl = data_plot[wvl_label]
        mean, std = np.nanmean(data_wvl), np.nanstd(data_wvl)
        ax.hist(data_wvl, bins=bins)
        ax.set_title(label("Rrs", wvl))
        print(f"Wavelength: {wvl:.1f} nm  ;  Mean: {mean:.3f} +- {std:.3f}")
    plt.show()
    plt.close()

# Plot sample of data
def plot_sample(data_plot, sample_cols=Rrs_columns, sample_rows=100):
    data_sub = data_plot[sample_cols][::sample_rows]
    data_sub = np.array([data_sub[col].data for col in data_sub.colnames])  # Iteration over data_sub.columns does not work

    plt.figure(figsize=(6,3), tight_layout=True)
    plt.plot(wavelengths, data_sub, c="k", alpha=0.1)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
    plt.xlim(320, 955)
    plt.ylim(0, 0.1)
    plt.yticks(np.arange(0, 0.12, 0.02))
    plt.grid(ls="--")
    plt.title("Example R_rs spectra")
    plt.show()
    plt.close()

print("Before offset subtraction:")
plot_histograms(data)
plot_sample(data)

for col in Rrs_columns:
    data[col] -= data["offset"]

print("After offset subtraction:")
plot_histograms(data)
plot_sample(data)
