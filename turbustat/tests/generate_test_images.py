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


def make_extended(imsize, imsize2=None, powerlaw=2.0, theta=0., ellip=1.,
                  return_psd=False, randomseed=32768324):
    '''
    Adapted from https://github.com/keflavich/image_registration. Added ability
    to make the power spectra elliptical.

    Parameters
    ----------
    imsize : int
        Array size.
    imsize2 : int, optional
        Array size in 2nd dimension.
    powerlaw : float, optional
        Powerlaw index.
    theta : float, optional
        Position angle of major axis in radians. Has no effect when ellip==1.
    ellip : float, optional
        Ratio of the minor to major axis. Must be > 0 and <= 1. Defaults to
        the circular case (ellip=1).
    return_psd : bool, optional
        Return the power-map instead of the image.

    Returns
    -------
    newmap : np.ndarray
        Two-dimensional array with the given power-law properties.
    '''
    imsize = int(imsize)
    if imsize2 is None:
        imsize2 = imsize

    if ellip > 1 or ellip <= 0:
        raise ValueError("ellip must be > 0 and <= 1.")

    yy, xx = np.indices((imsize2, imsize), dtype='float')
    xcen = imsize / 2. - (1. - imsize % 2)
    ycen = imsize2 / 2. - (1. - imsize2 % 2)
    yy -= ycen
    xx -= xcen

    if ellip < 1:
        # Apply a rotation and scale the x-axis (ellip).
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        xprime = ellip * (xx * costheta - yy * sintheta)
        yprime = xx * sintheta + yy * costheta

        rr2 = xprime**2 + yprime**2

        rr = rr2**0.5
    else:
        # Circular whenever ellip == 1
        rr = (xx**2 + yy**2)**0.5

    # flag out the bad point to avoid warnings
    rr[rr == 0] = np.nan

    with NumpyRNGContext(randomseed):
        powermap = rr**(-powerlaw / 2.) * \
            np.exp(1j * np.random.uniform(0, 2 * np.pi,
                                          size=(imsize2, imsize)))

    powermap[powermap != powermap] = 0

    if return_psd:
        return powermap

    newmap = np.abs(np.fft.fftshift(np.fft.fft2(powermap)))

    return newmap
