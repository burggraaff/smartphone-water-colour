"""
Process TriOS data from the Switzerland campaign.

Command line inputs:
    * Folder containing the TriOS data files.
        Example: "water-colour-data\Switzerland_Oli4\Reference"

Outputs:
    * File containing Ed, Ls, Lt, R_rs
        Example: "water-colour-data\data_switzerland_table.csv"
"""
import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv
from pathlib import Path
from datetime import datetime

from wk import wacodi as wa, hydrocolor as hc

# Get filenames
folder = Path(argv[1])
print("Input folder:", folder.absolute())
saveto = Path("water-colour-data/trios_switzerland_table.csv")
trios_filename = "Rrs_output_simis.csv"

# Loop over the data folders for each lake
data_folders = hc.generate_folders([folder], "RAMSES_calibrated")
data_tables = []

for folder in data_folders:
    # Filename in this folder
    Rrs_filename = folder / trios_filename

    # Load the data
    header = np.genfromtxt(Rrs_filename, delimiter=",", dtype=str, max_rows=1)  # Contains timestamps
    data = np.genfromtxt(Rrs_filename, delimiter=",", dtype=float, skip_header=1)  # Contains radiometry

    # Convert header into timestamps and UTC format
    only_radiometry = np.s_[1:-3]
    header = header[only_radiometry]  # Remove wavelength and percentile columns
    timestamps = [h.replace(" ", "T") for h in header]
    UTC = [datetime.fromisoformat(t).timestamp() for t in timestamps]

    # Convert the data to the right format
    wavelengths = data[:,0]
    wavelength_header = [f"R_rs_{wvl:.1f}" for wvl in wavelengths]
    data = data.T[only_radiometry]

    # Transpose the data and convert everything to astropy tables
    timestamps = table.Column(name="DateTime", data=timestamps)
    UTC = table.Column(name="UTC", data=UTC)
    radiometry = table.Table(names=wavelength_header, data=data)

    # Combine everythign into one table and append that to the list
    radiometry.add_columns([timestamps, UTC], indexes=[0,0])
    data_tables.append(radiometry)

# Combine the individual data tables
data = table.vstack(data_tables)

# Add WACODI data - XYZ, xy, hue angle, Forel-Ule
data = wa.add_colour_data_to_table(data)
