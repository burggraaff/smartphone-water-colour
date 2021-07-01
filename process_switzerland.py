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


# Loop over the data folders for each lake
data_folders = hc.generate_folders([folder], "RAMSES_calibrated")

for folder in data_folders:
    pass
