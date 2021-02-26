"""
Process data taken with a Lee-mode HyperPro.

To do:
    * Convert Lu to Lw
"""

import numpy as np
from matplotlib import pyplot as plt
from sys import argv

Es_file, Lu_file, output_file = argv[1:]

wvl_Es, Es = np.loadtxt(Es_file, delimiter=",", skiprows=1, unpack=True)
wvl_Lu, Lu = np.loadtxt(Lu_file, delimiter=",", skiprows=1, unpack=True)

plt.plot(wvl_Es, Es)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Surface irradiance")
plt.show()
plt.close()

plt.plot(wvl_Lu, Lu)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Upwelling radiance")
plt.show()
plt.close()

R_rs = Lu/Es
plt.plot(wvl_Lu, R_rs)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Remote sensing reflectance")
plt.show()
plt.close()

R_rs_stack = np.stack([wvl_Lu, Es, Lu, R_rs])
np.savetxt(output_file, R_rs_stack, delimiter=",", fmt="%.10f")
