# Licensed under an MIT open source license - see LICENSE

import numpy as np
from astropy.modeling.models import Gaussian2D, Const2D, Gaussian1D, Const1D
from astropy.utils import NumpyRNGContext


# Define ways to make 1D and 2D profiles (typically Gaussian for ease)
def twoD_gaussian(shape=(201, 201), x_std=10., y_std=10., theta=0, amp=1.,
                  bkg=0.):

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
