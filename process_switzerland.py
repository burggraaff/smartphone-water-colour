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
saveto = Path("water-colour-data/data_switzerland_table.csv")
trios_filename = "Rrs_output_simis.csv"

# Loop over the data folders for each lake
data_folders = hc.generate_folders([folder], "RAMSES_calibrated")

for folder in data_folders:
    # Filename in this folder
    Rrs_filename = folder / trios_filename

    # Load the data
    header = np.genfromtxt(Rrs_filename, delimiter=",", dtype=str, max_rows=1)  # Contains timestamps
    data = np.genfromtxt(Rrs_filename, delimiter=",", dtype=float)  # Contains radiometry

    # Convert header into timestamps and UTC format
    header = header[1:-3]  # Remove wavelength and percentile columns
    timestamps = [h.replace(" ", "T") for h in header]
    UTC = [datetime.fromisoformat(t).timestamp() for t in timestamps]

    # Transpose the data and convert everything to astropy tables
    timestamps = table.Column(name="DateTime", data=timestamps)
    UTC = table.Column(name="UTC", data=UTC)
