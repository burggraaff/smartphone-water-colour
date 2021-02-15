"""
Process TriOS data.
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy import table
from sys import argv

filename = argv[1]

data = table.Table.read(filename, format="ascii.csv")

wavelengths = np.arange(323.3, 953.7, 3.3)
Rrs = np.array([data[f"Rrs_{wvl:.1f}"][1236] for wvl in wavelengths])

plt.figure(figsize=(3,3), tight_layout=True)
plt.plot(wavelengths, Rrs)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance [sr$^{-1}$]")
plt.xlim(390, 700)
plt.ylim(0, 0.05)
plt.yticks([0, 0.01, 0.02, 0.03, 0.04, 0.05])
plt.grid(ls="--")
plt.show()