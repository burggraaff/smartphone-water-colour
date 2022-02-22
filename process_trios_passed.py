"""
Process data from the Stirling TriOS, that have been pre-filtered.

Command line inputs:
    * Path to the TriOS data file.

Example:
    %run process_trios_passed.py water-colour-data/NZ/Trios_rrs/Trios_all_nz_south_L5statsPassed.csv
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path
from datetime import datetime

from wk import hyperspectral as hy, plot

# Get filenames
filename = Path(argv[1])
print("Input file:", filename.absolute())
saveto = filename.parent / (filename.stem + "_TriOS_table.csv")

# Read data
data = table.Table.read(filename)
print("Finished reading data")

# Add UTC timestamps
data_datetime = [datetime.fromisoformat(DT) for DT in data["DateTime"]]
data_timestamps = [dt.timestamp() for dt in data_datetime]
data.add_column(table.Column(data=data_timestamps, name="UTC"))
data.sort("UTC")

# Rename columns
data.rename_columns(["Latitude_x", "Longitude_x"], ["Latitude", "Longitude"])
R_rs_columns = [key for key in data.keys() if "med" in key and "Rrs" in key]
R_rs_columns_new = [key[:-3].replace("Rrs", "R_rs") for key in R_rs_columns]
data.rename_columns(R_rs_columns, R_rs_columns_new)

# Remove columns
data.remove_columns([key for key in data.keys() if "Device" in key or "_std" in key])

# Add dummy Ed, Lu, Lsky columns
data = hy.add_dummy_columns(data)
print("Added dummy radiance columns")

# Add colour data (XYZ, xy, hue angle, FU, sRGB)
data = hy.add_colour_data_to_hyperspectral_data_multiple_keys(data)

# Write data to file
data.write(saveto, format="ascii.fast_csv")
print("Output file:", saveto.absolute())

# Plot R_rs
saveto_R_rs = f"results/{filename.stem}.pdf"
plot.plot_hyperspectral_dataset(data, title=f"TriOS RAMSES spectra ($N$ = {len(data)})", saveto=saveto_R_rs)
print(f"Saved plot to {saveto_R_rs}")
