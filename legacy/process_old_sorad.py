# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:59:21 2021

@author: Burggraaff
"""

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