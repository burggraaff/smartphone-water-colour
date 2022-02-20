"""
Process data from the PML So-Rad-mounted TriOS.

Command line inputs:
    * Folder containing the So-Rad data file.
        Example: %run process_sorad.py "water-colour-data\Balaton_20190703\SoRad"

Outputs:
    * File containing Ed, Ls, Lt, R_rs
        Example: "water-colour-data\Balaton_20190703\SoRad\So-Rad_Balaton2019.csv"
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path
from datetime import datetime

from wk import hyperspectral as hy, plot

# Get filenames
folder = Path(argv[1])
print("Input folder:", folder.absolute())

filename_Ed = folder/"Ed_2019-07-03.csv"
filename_Ls = folder/"Ls_2019-07-03.csv"
filename_Lt = folder/"Lt_2019-07-03.csv"
filename_R_rs = folder/"Rrs_3C_2019-07-03.csv"
filename_meta = folder/"metadata_2019-07-03.csv"
filename_qc = folder/"QCmask_2019-07-03.csv"

wavelengths = np.arange(320, 955, 3.3)

# Function to read data
def read_data(filename, rename=None, normalise=True):
    """
    Read a So-Rad data file from `filename`
    If `rename` is given, rename columns to that, e.g. from `Ls` to `Lsky`
    """
    datatype = rename if rename else filename.stem.split("_")[0]  # Ed, R_rs, etc.
    data_columns = hy.extend_keys_to_wavelengths(datatype, wavelengths)
    data = np.loadtxt(filename, delimiter=",")
    if normalise:
        data /= 1000.  # Conversion from mW to W
    data = table.Table(data=data, names=data_columns)

    return data

# Read data
data_ed = read_data(filename_Ed)
data_ls = read_data(filename_Ls, rename="Lsky")
data_lt = read_data(filename_Lt, rename="Lu")
data_rrs = read_data(filename_R_rs, rename="R_rs", normalise=False)
data_meta = table.Table.read(filename_meta)
print("Finished reading data")

# Adjust header for metadata
data_meta.rename_column("col0", "index")

# Join tables into one
data = table.hstack([data_meta, data_ed, data_ls, data_lt, data_rrs])
print("Joined data tables")

# Remove invalid data
remove = np.where(data["q"] < 1)
len_orig = len(data)
data.remove_rows(remove)
print(f"Removed {len_orig - len(data)}/{len_orig} rows flagged as invalid.")

# Remove data with bad solar zenith angles
remove = np.where(data["theta_s"] < 30)
len_orig = len(data)
data.remove_rows(remove)
print(f"Removed {len_orig - len(data)}/{len_orig} rows with solar zenith angles < 30 degrees.")

# Add UTC timestamps
sorad_datetime = [datetime.fromisoformat(DT) for DT in data["timestamp"]]
sorad_timestamps = [dt.timestamp() for dt in sorad_datetime]
data.add_column(table.Column(data=sorad_timestamps, name="UTC"))
data.sort("UTC")

# Remove data from after we moved to the front deck - between 09:35 and 10:17 UTC
length_original = len(data)
switch_time = "2019-07-03 10:00:00"
switch_to_front_deck = datetime.fromisoformat(switch_time).timestamp()
data.remove_rows(data["UTC"] > switch_to_front_deck)
length_after_removal = len(data)
print(f"Removed {length_original-length_after_removal} data points ({length_after_removal} remaining) taken after {switch_time}.")

# Add WACODI data - XYZ, xy, hue angle, Forel-Ule
data = hy.add_colour_data_to_hyperspectral_data_multiple_keys(data)

# Write data to file
filename_result = folder/"So-Rad_Balaton2019.csv"
data.write(filename_result, format="ascii.fast_csv")
print("Output file:", filename_result.absolute())

# Plot histograms at multiple wavelengths
def plot_histograms(data_plot, wavelengths_hist=[353.0, 402.5, 501.5, 600.5, 702.8, 801.8, 900.8], bins=np.linspace(-0.01, 0.05, 15)):
    fig, axs = plt.subplots(ncols=len(wavelengths_hist), sharex=True, sharey=True, figsize=(10, 2))
    for wvl, ax in zip(wavelengths_hist, axs):
        wvl_label = f"R_rs_{wvl:.1f}"
        data_wvl = data_plot[wvl_label]
        mean, std = np.nanmean(data_wvl), np.nanstd(data_wvl)
        ax.hist(data_wvl, bins=bins)
        ax.set_title(wvl_label)
        print(f"Wavelength: {wvl:.1f} nm  ;  Mean: {mean:.3f} +- {std:.3f}")
    plt.show()
    plt.close()

# Plot the resulting data
filename_figure = lambda filename_data: f"results/SoRad-{filename_data.stem}.pdf"
plot_histograms(data)

# plot.plot_hyperspectral_dataset(data, title=f"SoRad spectra ($N$ = {len(data)})", saveto=filename_figure(filename_Ed))
# plot.plot_hyperspectral_dataset(data, title=f"SoRad spectra ($N$ = {len(data)})", saveto=filename_figure(filename_Ls))
# plot.plot_hyperspectral_dataset(data, title=f"SoRad spectra ($N$ = {len(data)})", saveto=filename_figure(filename_Lt))
plot.plot_hyperspectral_dataset(data, title=f"SoRad spectra ($N$ = {len(data)})", saveto=filename_figure(filename_R_rs))
print(f"Saved plot to {filename_R_rs}")
