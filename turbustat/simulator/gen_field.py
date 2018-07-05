
import numpy as np
from astropy.utils import NumpyRNGContext


def make_3dfield(imsize, powerlaw=2.0, amp=1.0,
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
        The 2D array in Fourier space. The zero-frequency is shifted to
        the centre.
    '''
    imsize = int(imsize)

    yy, xx, zz = np.meshgrid(np.fft.fftfreq(imsize),
                             np.fft.fftfreq(imsize),
                             np.fft.rfftfreq(imsize), indexing="ij")

    rr = (xx**2 + yy**2 + zz**2)**0.5

    # flag out the bad point to avoid warnings
    rr[rr == 0] = np.nan

    with NumpyRNGContext(randomseed):

        Np1 = (imsize - 1) // 2 if imsize % 2 != 0 else imsize // 2

        angles = np.random.uniform(0, 2 * np.pi,
                                   size=(imsize, imsize, Np1 + 1))

        phases = np.cos(angles) + 1j * np.sin(angles)

        # Impose symmetry
        # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
        if imsize % 2 == 0:
            phases[1:Np1, 0, 0] = np.conj(phases[imsize:Np1:-1, 0, 0])
            phases[1:Np1, -1, -1] = np.conj(phases[imsize:Np1:-1, -1, -1])
            phases[0, 1:Np1, 0] = np.conj(phases[0, imsize:Np1:-1, 0])
            phases[-1, 1:Np1, -1] = np.conj(phases[-1, imsize:Np1:-1, -1])
        else:
            phases[1:Np1, 0, 0] = np.conj(phases[imsize:Np1 + 1:-1, 0, 0])
            phases[1:Np1, -1, -1] = np.conj(phases[imsize:Np1 + 1:-1, -1, -1])
            phases[0, 1:Np1, 0] = np.conj(phases[0, imsize:Np1 + 1:-1, 0])
            phases[-1, 1:Np1, -1] = np.conj(phases[-1, imsize:Np1 + 1:-1, -1])

        powermap = amp * (rr**(-powerlaw / 2.)).astype('complex') * phases

    powermap[powermap != powermap] = 0

    if return_psd:

        return powermap

    newmap = np.fft.irfftn(powermap)

    return newmap
