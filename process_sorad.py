"""
Process data from the PML So-Rad-mounted TriOS.

Command line inputs:
    * Folder containing the So-Rad data file.
        Example: "water-colour-data\Balaton_20190703\SoRad"

Outputs:
    * File containing Ed, Ls, Lt, Rrs
        Example: "water-colour-data\Balaton_20190703\SoRad\So-Rad_Balaton2019.csv"
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path

# Get filenames
folder = Path(argv[1])
print("Input folder:", folder.absolute())

filename_Ed = folder/"Ed_2019-07-03.csv"
filename_Ls = folder/"Ls_2019-07-03.csv"
filename_Lt = folder/"Lt_2019-07-03.csv"
filename_Rrs = folder/"Rrs_finger_2019-07-03.csv"
filename_meta = folder/"Meta_2019-07-03.csv"
filename_qc = folder/"QCmask_2019-07-03.csv"

wavelengths = np.arange(320, 955, 3.3)

# Label that matches column header
def label(text, wvl):
    return f"{text}_{wvl:.1f}"

# Function to read data
def read_data(filename, rename=None):
    """
    Read a So-Rad data file from `filename`
    If `rename` is given, rename columns to that, e.g. from `Ls` to `Lsky`
    """
    datatype = rename if rename else filename.stem.split("_")[0]  # Ed, Rrs, etc.
    data_columns = [label(datatype, wvl) for wvl in wavelengths]
    data = np.loadtxt(filename, delimiter=",")
    data /= 1000.  # Conversion from mW to W
    data = table.Table(data=data, names=data_columns)

    return data

# Read data
data_ed = read_data(filename_Ed)
data_ls = read_data(filename_Ls, rename="Lsky")
data_lt = read_data(filename_Lt, rename="Lu")
data_rrs = read_data(filename_Rrs)
data_meta = table.Table.read(filename_meta)
data_qc = table.Table.read(filename_qc)
print("Finished reading data")

# Join tables into one
data = table.hstack([data_meta, data_qc, data_ed, data_ls, data_lt, data_rrs])
print("Joined data tables")

# Remove invalid data
remove = np.where(data["Q_rad_finger"] == 0.0)
len_orig = len(data)
data.remove_rows(remove)
print(f"Removed {len_orig - len(data)}/{len_orig} rows flagged as invalid.")

# Write data to file
filename_result = folder/"So-Rad_Balaton2019.csv"
data.write(filename_result, format="ascii.fast_csv")

# Plot histograms at multiple wavelengths
def plot_histograms(data_plot, wavelengths_hist=[353.0, 402.5, 501.5, 600.5, 702.8, 801.8, 900.8], bins=np.linspace(-1e-6, 5e-5, 15)):
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
def plot_sample(data_plot, sample_quantity, sample_rows=10, ylabel="", saveto=None):
    sample_cols = [label(sample_quantity, wvl) for wvl in wavelengths]
    data_sub = data_plot[sample_cols][::sample_rows]
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

# Plot Ed, Lt, Ls
plot_sample(data, "Ed", ylabel="$E_d$ [W nm$^{-1}$ m$^{-2}$]", saveto=filename_Ed.with_suffix(".pdf"))
plot_sample(data, "Lu", ylabel="$L_u$ [W nm$^{-1}$ m$^{-2}$ sr$^{-1}$]", saveto=filename_Lt.with_suffix(".pdf"))
plot_sample(data, "Lsky", ylabel="$L_{sky}$ [W nm$^{-1}$ m$^{-2}$ sr$^{-1}$]", saveto=filename_Ls.with_suffix(".pdf"))

# Plot Rrs
# print("Before offset subtraction:")
# plot_histograms(data)
# plot_sample(data, "Rrs", ylabel="$R_{rs}$ [sr$^{-1}$]")

# Rrs_columns = [label("Rrs", wvl) for wvl in wavelengths]
# for col in Rrs_columns:
#     data[col] -= data["offset"]

# print("After offset subtraction:")
plot_histograms(data)
plot_sample(data, "Rrs", ylabel="$R_{rs}$ [sr$^{-1}$]", saveto=filename_Rrs.with_suffix(".pdf"))
