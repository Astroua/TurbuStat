# Licensed under an MIT open source license - see LICENSE


'''
Load in data sets for tests. 32^2 pixels portions of two data sets
are loaded in (one Design and one fiducial run).
Only the channels with signal were kept. Additional channels the match the
original spectral axis are added on and filled with noise centered around the
limit.
'''

# Need to create the property arrays
from ..data_reduction import property_arrays

import os
import warnings
import numpy as np
import numpy.random as ra
from astropy.io.fits import getheader

# Set seed for adding noise.
ra.seed(121212)

turb_path = os.path.dirname(__file__)

# Open header for both
hdr_path = os.path.join(turb_path, "data/header.fits")
header = getheader(hdr_path)


keywords = ["centroid", "centroid_error", "integrated_intensity",
            "integrated_intensity_error", "linewidth",
            "linewidth_error", "moment0", "moment0_error", "cube"]

path1 = os.path.join(turb_path, "data/dataset1.npz")

dataset1 = np.load(path1)

cube1 = np.empty((500, 32, 32))

count = 0
for posn, kept in zip(*dataset1["channels"]):
    posn = int(posn)
    if kept:
        cube1[posn, :, :] = dataset1["cube"][count, :, :]
        count += 1
    else:
        cube1[posn, :, :] = ra.normal(0.005, 0.005, (32, 32))

props1 = property_arrays((cube1, header), rms_noise=0.001)
props1.return_all(save=False)

dataset1 = props1.dataset

##############################################################################

path2 = os.path.join(turb_path, "data/dataset2.npz")

dataset2 = np.load(path2)

cube2 = np.empty((500, 32, 32))

count = 0
for posn, kept in zip(*dataset2["channels"]):
    posn = int(posn)
    if kept:
        cube2[posn, :, :] = dataset2["cube"][count, :, :]
        count += 1
    else:
        cube2[posn, :, :] = ra.normal(0.005, 0.005, (32, 32))

props2 = property_arrays((cube2, header), rms_noise=0.001)
props2.return_all(save=False)

dataset2 = dataset

##############################################################################

# Load in saved comparison data.
try:
    computed_data = np.load(os.path.join(turb_path, "data/checkVals.npz"))

    computed_distances = np.load(os.path.join(turb_path,
                                              "data/computed_distances.npz"))
except IOError:
    warnings.warn("No checkVals or computed_distances files.")
