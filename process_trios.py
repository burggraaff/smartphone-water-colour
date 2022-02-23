"""
Process data from the Stirling TriOS.

Command line inputs:
    * Path to the TriOS data file.

Example:
    %run process_trios.py water-colour-data/Data_Monocle2019_L5All.csv
    %run process_trios.py water-colour-data/NZ/Trios_rrs/Trios_all_nz_south_L5All.csv
"""
from sys import argv
from pathlib import Path
from datetime import datetime
import numpy as np
from astropy import table
from wk import hyperspectral as hy, plot

# Get filenames
filename = Path(argv[1])
print("Input file:", filename.absolute())
filename_result = filename.parent / (filename.stem + "_TriOS_table.csv")

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
data = hy.add_dummy_columns(data)
print("Added dummy radiance columns")

# Add colour data (XYZ, xy, hue angle, FU, sRGB)
data = hy.add_colour_data_to_hyperspectral_data_multiple_keys(data)

# Write data to file
data.write(filename_result, format="ascii.fast_csv")
print("Output file:", filename_result.absolute())

# Plot R_rs
saveto_R_rs = f"results/{filename.stem}.pdf"
plot.plot_hyperspectral_dataset(data, title=f"TriOS RAMSES spectra ($N$ = {len(data)})", saveto=saveto_R_rs)
print(f"Saved plot to {saveto_R_rs}")
