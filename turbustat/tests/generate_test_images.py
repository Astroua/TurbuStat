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


def make_extended(imsize, powerlaw=2.0, theta=0., ellip=1.,
                  return_psd=False, randomseed=32768324):
    '''

    Generate a power-law image with a specified index and random phases.

    Adapted from https://github.com/keflavich/image_registration. Added ability
    to make the power spectra elliptical. Also changed the random sampling so
    the random phases are Hermitian (and the inverse FFT gives a real-valued
    image).

    Parameters
    ----------
    imsize : int
        Array size.
    powerlaw : float, optional
        Powerlaw index.
    theta : float, optional
        Position angle of major axis in radians. Has no effect when ellip==1.
    ellip : float, optional
        Ratio of the minor to major axis. Must be > 0 and <= 1. Defaults to
        the circular case (ellip=1).
    return_psd : bool, optional
        Return the power-map instead of the image. The full powermap is
        returned, including the redundant negative phase phases for the RFFT.
    randomseed: int, optional
        Seed for random number generator.

    Returns
    -------
    newmap : np.ndarray
        Two-dimensional array with the given power-law properties.
    full_powermap : np.ndarray
        The 2D array in Fourier space.
    '''
    imsize = int(imsize)

    if ellip > 1 or ellip <= 0:
        raise ValueError("ellip must be > 0 and <= 1.")

    yy, xx = np.meshgrid(np.fft.fftfreq(imsize),
                         np.fft.rfftfreq(imsize), indexing="ij")

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

        Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=(imsize, Np1 + 1))

        phases = np.cos(angles) + 1j * np.sin(angles)

        # Impose symmetry
        # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
        if imsize % 2 == 0:
            phases[1:Np1, 0] = np.conj(phases[imsize:Np1:-1, 0])
            phases[1:Np1, -1] = np.conj(phases[imsize:Np1:-1, -1])
        else:
            phases[1:Np1, 0] = np.conj(phases[imsize:Np1 + 1:-1, 0])
            phases[1:Np1, -1] = np.conj(phases[imsize:Np1 + 1:-1, -1])

        powermap = (rr**(-powerlaw / 2.)).astype('complex') * phases

    powermap[powermap != powermap] = 0

    if return_psd:

        # Create the full power map, with the symmetric conjugate component
        if imsize % 2 == 0:
            power_map_symm = np.conj(powermap[:, -2:0:-1])
        else:
            power_map_symm = np.conj(powermap[:, -1:0:-1])

        power_map_symm[1::, :] = power_map_symm[:0:-1, :]

        full_powermap = np.concatenate((powermap, power_map_symm), axis=1)

        if not full_powermap.shape[1] == imsize:
            raise ValueError("The full powermap should have a square shape."
                             " Instead has {}".format(full_powermap.shape))

        return full_powermap

    newmap = np.fft.irfft2(powermap)

    return newmap
