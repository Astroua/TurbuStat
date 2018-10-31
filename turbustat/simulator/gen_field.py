
import numpy as np
from astropy.utils import NumpyRNGContext


def make_3dfield(imsize, powerlaw=2.0, amp=1.0,
                 return_fft=False, randomseed=32768324):
    '''

    Generate a 3D power-law field with a specified index and random phases.

    Heavily adapted from https://github.com/keflavich/image_registration.

    Parameters
    ----------
    imsize : int
        Array size.
    powerlaw : float, optional
        Powerlaw index.
    return_fft : bool, optional
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

    output = (rr**(-powerlaw / 2.)).astype('complex') * phases

    output[np.isnan(output)] = 0. + 1j * 0.0

    # Impose symmetry
    # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
    if imsize % 2 == 0:

        # Think of this like a 2x2 block matrix.
        # First element along diagonal equals the conjugate of the second:
        output[1:Np1, 1:Np1, 0] = \
            np.conj(output[imsize:Np1:-1, imsize:Np1:-1, 0])
        output[1:Np1, 1:Np1, -1] = \
            np.conj(output[imsize:Np1:-1, imsize:Np1:-1, -1])

        # And the conjugates off the diagonal must also equal:
        output[1:Np1, imsize:Np1:-1, 0] = \
            np.conj(output[imsize:Np1:-1, 1:Np1, 0])
        output[1:Np1, imsize:Np1:-1, -1] = \
            np.conj(output[imsize:Np1:-1, 1:Np1, -1])
        output[1:Np1, imsize:Np1:-1, 0] = \
            np.conj(output[imsize:Np1:-1, 1:Np1, 0])
        output[1:Np1, imsize:Np1:-1, -1] = \
            np.conj(output[imsize:Np1:-1, 1:Np1, -1])

        # The 2D equivalents along the 0 and N/2 frequencies
        output[1:Np1, Np1, 0] = np.conj(output[imsize:Np1:-1, Np1, 0])
        output[1:Np1, Np1, -1] = np.conj(output[imsize:Np1:-1, Np1, -1])
        output[Np1, 1:Np1, 0] = np.conj(output[Np1, imsize:Np1:-1, 0])
        output[Np1, 1:Np1, -1] = np.conj(output[Np1, imsize:Np1:-1, -1])

        output[1:Np1, 0, 0] = np.conj(output[imsize:Np1:-1, 0, 0])
        output[1:Np1, 0, -1] = np.conj(output[imsize:Np1:-1, 0, -1])
        output[0, 1:Np1, 0] = np.conj(output[0, imsize:Np1:-1, 0])
        output[0, 1:Np1, -1] = np.conj(output[0, imsize:Np1:-1, -1])

        # Even shapes need the N/2 imaginary part to be zero
        output[Np1, 0, 0] = output[Np1, 0, 0].real + 1j * 0.0
        output[Np1, 0, -1] = output[Np1, 0, -1].real + 1j * 0.0
        output[0, Np1, 0] = output[0, Np1, 0].real + 1j * 0.0
        output[0, Np1, -1] = output[0, Np1, -1].real + 1j * 0.0
        output[Np1, Np1, 0] = output[Np1, Np1, 0].real + 1j * 0.0
        output[Np1, Np1, -1] = output[Np1, Np1, -1].real + 1j * 0.0

    else:

        output[1:Np1 + 1, 1:Np1 + 1, 0] = \
            np.conj(output[imsize:Np1:-1, imsize:Np1:-1, 0])
        output[1:Np1 + 1, 1:Np1 + 1, -1] = \
            np.conj(output[imsize:Np1:-1, imsize:Np1:-1, -1])

        # And the conjugates off the diagonal must also equal:
        output[1:Np1 + 1, imsize:Np1:-1, 0] = \
            np.conj(output[imsize:Np1:-1, 1:Np1 + 1, 0])
        output[1:Np1 + 1, imsize:Np1:-1, -1] = \
            np.conj(output[imsize:Np1:-1, 1:Np1 + 1, -1])
        output[1:Np1 + 1, imsize:Np1:-1, 0] = \
            np.conj(output[imsize:Np1:-1, 1:Np1 + 1, 0])
        output[1:Np1 + 1, imsize:Np1:-1, -1] = \
            np.conj(output[imsize:Np1:-1, 1:Np1 + 1, -1])

        # The 2D equivalents along the 0 frequency
        output[1:Np1 + 1, 0, 0] = np.conj(output[imsize:Np1:-1, 0, 0])
        output[1:Np1 + 1, 0, -1] = np.conj(output[imsize:Np1:-1, 0, -1])
        output[0, 1:Np1 + 1, 0] = np.conj(output[0, imsize:Np1:-1, 0])
        output[0, 1:Np1 + 1, -1] = np.conj(output[0, imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    output[0, 0, 0] = output[0, 0, 0].real + 1j * 0.0
    output[0, 0, -1] = output[0, 0, -1].real + 1j * 0.0

    newmap = np.fft.irfftn(output)

    # Normalize to the correct amplitude.
    newmap /= (np.sqrt(np.sum(newmap**2)) / np.sqrt(newmap.size)) / amp

    if return_fft:

        return np.fft.rfftn(newmap)

    return newmap


def make_extended(imsize, powerlaw=2.0, theta=0., ellip=1.,
                  return_fft=False, full_fft=True, randomseed=32768324):
    '''

    Generate a 2D power-law image with a specified index and random phases.

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
    return_fft : bool, optional
        Return the FFT instead of the image. The full FFT is
        returned, including the redundant negative phase phases for the RFFT.
    full_fft : bool, optional
        When `return_fft=True`, the full FFT, with negative frequencies, will
        be returned. If `full_fft=False`, the RFFT is returned.
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

    # Rescale phases to an amplitude of unity
    phases /= np.sqrt(np.sum(phases**2) / float(phases.size))

    output = (rr**(-powerlaw / 2.)).astype('complex') * phases

    output[np.isnan(output)] = 0.

    # Impose symmetry
    # From https://dsp.stackexchange.com/questions/26312/numpys-real-fft-rfft-losing-power
    if imsize % 2 == 0:
        output[1:Np1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1, -1] = np.conj(output[imsize:Np1:-1, -1])
        output[Np1, 0] = output[Np1, 0].real + 1j * 0.0
        output[Np1, -1] = output[Np1, -1].real + 1j * 0.0

    else:
        output[1:Np1 + 1, 0] = np.conj(output[imsize:Np1:-1, 0])
        output[1:Np1 + 1, -1] = np.conj(output[imsize:Np1:-1, -1])

    # Zero freq components must have no imaginary part to be own conjugate
    output[0, -1] = output[0, -1].real + 1j * 0.0
    output[0, 0] = output[0, 0].real + 1j * 0.0

    if return_fft:

        if not full_fft:
            return output

        # Create the full power map, with the symmetric conjugate component
        if imsize % 2 == 0:
            power_map_symm = np.conj(output[:, -2:0:-1])
        else:
            power_map_symm = np.conj(output[:, -1:0:-1])

        power_map_symm[1::, :] = power_map_symm[:0:-1, :]

        full_powermap = np.concatenate((output, power_map_symm), axis=1)

        if not full_powermap.shape[1] == imsize:
            raise ValueError("The full output should have a square shape."
                             " Instead has {}".format(full_powermap.shape))

        return np.fft.fftshift(full_powermap)

    newmap = np.fft.irfft2(output)

    return newmap
