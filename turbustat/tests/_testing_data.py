# Licensed under an MIT open source license - see LICENSE


'''
Load in data sets for tests. 32^2 pixels portions of two data sets
are loaded in (one Design and one fiducial run).
Only the channels with signal were kept. Additional channels the match the
original spectral axis are added on and filled with noise centered around the
limit.
'''

# Need to create the property arrays
from ..moments import Moments

from spectral_cube import SpectralCube, LazyMask, Projection

import os
import warnings
import numpy as np
import numpy.random as ra
from astropy.io.fits import getheader
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

# Set seed for adding noise.
ra.seed(121212)

turb_path = os.path.dirname(__file__)

# Open header for both
hdr_path = os.path.join(turb_path, "data/header.fits")
header = getheader(hdr_path)


keywords = ["centroid", "centroid_error", "linewidth",
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

sc1 = SpectralCube(data=cube1, wcs=WCS(header))
mask = LazyMask(np.isfinite, sc1)
sc1 = sc1.with_mask(mask)
# Set the scale for the purposes of the tests
props1 = Moments(sc1, scale=0.003031065017916262 * u.Unit(""))
# props1.make_mask(mask=mask)
props1.make_moments()
props1.make_moment_errors()

dataset1 = props1.to_dict()

moment0_hdu1 = fits.PrimaryHDU(dataset1["moment0"][0],
                               header=dataset1["moment0"][1])

moment0_proj = Projection.from_hdu(moment0_hdu1)

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

sc2 = SpectralCube(data=cube2, wcs=WCS(header))
mask = LazyMask(np.isfinite, sc2)
sc2 = sc2.with_mask(mask)
# Set the scale for the purposes of the tests
props2 = Moments(sc2, scale=0.003029658694658428 * u.Unit(""))
props2.make_moments()
props2.make_moment_errors()

dataset2 = props2.to_dict()

##############################################################################

# Load in saved comparison data.
try:
    computed_data = np.load(os.path.join(turb_path, "data/checkVals.npz"),
                            encoding='latin1', allow_pickle=True)

    computed_distances = np.load(os.path.join(turb_path,
                                              "data/computed_distances.npz"),
                                 encoding='latin1', allow_pickle=True)
except IOError:
    warnings.warn("No checkVals or computed_distances files.")
