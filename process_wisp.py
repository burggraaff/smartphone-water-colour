"""
Process data from the WISP-3.

Command line inputs:
    * Path to the WISP-3 data file.
        Example: "water-colour-data\wisp_Balaton_20190703_20190705.csv"

Outputs:
    * File containing Ed, Ls, Lt, Rrs in columns
        Example: "water-colour-data\wisp_Balaton_20190703_20190705_columns.csv"
"""

import numpy as np
from astropy import table
from sys import argv
from pathlib import Path
from datetime import datetime
from wk import wacodi as wa

# Label that matches column header
def label(text, wvl):
    return f"{text}_{wvl:.1f}"

def _split_line(line):
    # Strip white space, remove quotation marks, split on commas
    line_split = line.strip().replace('\"', '').split(",")
    return line_split

def _convert_wisp_block(block):
    block_split = [_split_line(line) for line in block]
    block_arr = np.array(block_split, dtype=np.float64)
    wavelengths = block_arr[:,0]
    data = block_arr[:,1:]
    return wavelengths, data

# Convert WISP timestamps to datetime objects
def _convert_wisp_timestamp(timestamp):
    day, month, year, hour, minute = int(timestamp[:2]), int(timestamp[3:5]), int(timestamp[6:11]), int(timestamp[11:13]), int(timestamp[14:16])
    dt = datetime(year, month, day, hour, minute)
    return dt

def load_wisp_data(wisp_filename, rho=0.028):
    """
    Get (ir)radiance and reflectance data from a WISPweb output file
    """
    with open(wisp_filename, "r") as file:
        lines = file.readlines()

    dates = np.array(_split_line(lines[13])[1:])
    dates_datetime = [_convert_wisp_timestamp(d) for d in dates]
    timestamps = [date.isoformat() for date in dates_datetime]
    UTC = [date.timestamp() for date in dates_datetime]

    latitudes = np.array(_split_line(lines[14])[1:], dtype=np.float64)
    longitudes = np.array(_split_line(lines[15])[1:], dtype=np.float64)

    # Lines are blocked per wavelength, so each block always starts at the same index
    block_Lsky = lines[425:826]
    block_Lu = lines[827:1228]
    block_Ed = lines[1229:]

    wavelengths, Lsky = _convert_wisp_block(block_Lsky)
    wavelengths, Lu = _convert_wisp_block(block_Lu)
    wavelengths, Ed = _convert_wisp_block(block_Ed)

    Rrs = (Lu - rho*Lsky) / Ed

    header_meta = ["timestamp", "UTC", "latitude", "longitude"]
    header_Lsky = [label("Lsky", wvl) for wvl in wavelengths]
    header_Lu = [label("Lu", wvl) for wvl in wavelengths]
    header_Ed = [label("Ed", wvl) for wvl in wavelengths]
    header_Rrs = [label("R_rs", wvl) for wvl in wavelengths]

    data_table = table.Table(data=[timestamps, UTC, latitudes, longitudes, *Lsky, *Lu, *Ed, *Rrs], names=[*header_meta, *header_Lsky, *header_Lu, *header_Ed, *header_Rrs])

    data_table.sort("UTC")

    # Remove duplicate entries (from pressing "save" multiple times)
    data_table = table.unique(data_table)

    return data_table


# Get filenames
filename = Path(argv[1])
print("Input file:", filename.absolute())

# Convert to table
data = load_wisp_data(filename)

# Manually remove data from 2019-07-05 10:35:51 because these look like noise
bad_row = np.where(data["timestamp"] == "2019-07-05T10:36:00")[0][0]
data.remove_row(bad_row)
print(f"Removed row {bad_row}.")

# Add WACODI data - XYZ, xy, hue angle, Forel-Ule
data = wa.add_colour_data_to_table(data)

# Write to file
filename_result = filename.with_name(filename.stem + "_table.csv")
data.write(filename_result, format="ascii.fast_csv")
