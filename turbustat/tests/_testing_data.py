# Licensed under an MIT open source license - see LICENSE


'''
Load in data sets for tests. 32^2 pixels portions of two data sets
are loaded in (one Design and one fiducial run).
Only the channels with signal were kept. Additional channels the match the
original spectral axis are added on and filled with noise centered around the
limit.
'''

# Need to create the property arrays
from ..data_reduction import Mask_and_Moments

from spectral_cube import SpectralCube, LazyMask

import os
import warnings
import numpy as np
import numpy.random as ra
from astropy.io.fits import getheader
from astropy.wcs import WCS
from astropy.io import fits

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

sc1 = SpectralCube(data=cube1, wcs=WCS(header))
mask = LazyMask(np.isfinite, sc1)
sc1 = sc1.with_mask(mask)
props1 = Mask_and_Moments(sc1)
props1.make_mask(mask=mask)
props1.make_moments()
props1.make_moment_errors()

dataset1 = props1.to_dict()

moment0_hdu1 = fits.PrimaryHDU(dataset1["moment0"][0],
                               header=dataset1["moment0"][1])

# Save moments for testing i/o loading
try:
    props1.to_fits(save_name="dataset1")
except IOError:
    warnings.warn("It looks like the moment arrays are already saved...")

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
props2 = Mask_and_Moments(sc2)
props2.make_moments()
props2.make_moment_errors()

dataset2 = props2.to_dict()

##############################################################################

# Load in saved comparison data.
try:
    computed_data = np.load(os.path.join(turb_path, "data/checkVals.npz"))

    computed_distances = np.load(os.path.join(turb_path,
                                              "data/computed_distances.npz"))
except IOError:
    warnings.warn("No checkVals or computed_distances files.")


# Define ways to make 1D and 2D profiles (typically Gaussian for ease)
def twoD_gaussian(shape=(201, 201), x_std=10., y_std=10., theta=0, amp=1.,
                  bkg=0.):

    from astropy.modeling.models import Gaussian2D, Const2D

    centre = tuple([val // 2 for val in shape])

    mod = Gaussian2D(x_mean=centre[0], y_mean=centre[1],
                     x_stddev=x_std, y_stddev=y_std,
                     amplitude=amp, theta=theta) + \
        Const2D(amplitude=bkg)

    return mod


def generate_2D_array(shape=(201, 201), curve_type='gaussian', **kwargs):

    if curve_type == "gaussian":

        mod = twoD_gaussian(shape=shape, **kwargs)

    else:
        raise ValueError("curve_type must be 'gaussian'.")

    ygrid, xgrid = np.mgrid[:shape[0], :shape[1]]

    return mod(xgrid, ygrid)


def oneD_gaussian(shape=200, mean=0., std=10., amp=1., bkg=0.):

    from astropy.modeling.models import Gaussian1D, Const1D

    mod = Gaussian1D(mean=mean, stddev=std, amplitude=amp) + \
        Const1D(amplitude=bkg)

    return mod


def generate_1D_array(shape=200, curve_type='gaussian', **kwargs):

    if curve_type == "gaussian":

        mod = oneD_gaussian(shape=shape, **kwargs)

    else:
        raise ValueError("curve_type must be 'gaussian'.")

    xgrid = np.arange(shape)

    return mod(xgrid)


def assert_between(value, lower, upper):
    '''
    Check if a value is between two values.
    '''

    within_lower = value >= lower
    within_upper = value <= upper

    if within_lower and within_upper:
        return
    else:
        raise AssertionError("{0} not within {1} and {2}".format(value, lower,
                                                                 upper))
