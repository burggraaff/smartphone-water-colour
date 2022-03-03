"""
Combine multiple separate data table files, matching a filename pattern, into a single file.

Command-line inputs:
    * pattern: Path pattern to look for, e.g. "iPhone_SE_raw.csv"
    * folders: Any number of paths to folders containing data

Example:
    %run combined_table_smartphone.py "iPhone_SE_raw.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "iPhone_SE_jpeg.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "iPhone_SE_jpeg_linear.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "iPhone_SE_jpeg_fromraw.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "Samsung_Galaxy_S8_raw.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "Samsung_Galaxy_S8_jpeg_fromraw.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "Samsung_Galaxy_S8_replicates.csv" water-colour-data/Balaton*
    %run combined_table_smartphone.py "iPhone_6S_raw.csv" water-colour-data/Balaton* water-colour-data/NZ* water-colour-data/Switzerland_Oli4*
    %run combined_table_smartphone.py "iPhone_6S_jpeg.csv" water-colour-data/Balaton* water-colour-data/NZ* water-colour-data/Switzerland_Oli4*
"""

from sys import argv
from os import walk
from spectacle import io
from astropy import table

# Get the data folder from the command line
pattern, *folders = io.path_from_input(argv)

all_data = []
for folder in folders:
    for tup in walk(folder):
        folder_here = io.Path(tup[0])
        file_path = folder_here / pattern

        if file_path.exists():
            print(file_path)
            data_here = table.Table.read(file_path)
            all_data.append(data_here)

        else:
            continue

data_combined = table.vstack(all_data)
data_combined.sort("UTC")

print(data_combined)

save_to = folders[0].parent / f"combined_{pattern}"

data_combined.write(save_to, format="ascii.fast_csv")
print(f"Saved results to {save_to}")
