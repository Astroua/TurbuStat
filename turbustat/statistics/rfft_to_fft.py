# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from warnings import warn

try:
    from pyfftw.interfaces.numpy_fft import rfftn
    PYFFTW_FLAG = True
except ImportError:
    PYFFTW_FLAG = False


'''
Reconstruct FFT output from RFFT in order to save memory
Largely follows the solution from:
http://stackoverflow.com/questions/25147045/full-frequency-array-reconstruction-after-numpy-fft-rfftn
'''


def rfft_to_fft(image, keep_rfft=False, use_pyfftw=False,
                threads=1, **pyfftw_kwargs):
    '''
    Perform a RFFT on the image (2 or 3D) and return the absolute value in
    the same format as you would get with the fft (negative frequencies).
    This avoids ever having to have the full complex cube in memory.

    Inputs
    ------
    image : numpy.ndarray
        2 or 3D array.
    keep_rfft : bool, optional
        Return the rfft output instead of expanding to the
        negative frequencies of the full FFT.
    use_pyfftw : bool, optional
        Try using pyfftw for the FFT.
    threads : int, optional
        Number of threads to use when using pyfftw. Default is 1.
    pyfftw_kwargs : Passed to `~pyfftw.interfaces.numpy_fft.rfftn`.

    Outputs
    -------
    fft_abs : absolute value of the fft.
    '''

    ndim = len(image.shape)

    if ndim < 2 or ndim > 3:
        raise TypeError("Dimension of image must be 2D or 3D.")

    last_dim = image.shape[-1]

    if use_pyfftw:
        if PYFFTW_FLAG:
            fft_abs = np.abs(rfftn(image, **pyfftw_kwargs))
        else:
            use_pyfftw = False
            warn("pyfftw is not installed")

    if not use_pyfftw:
        fft_abs = np.abs(np.fft.rfftn(image))

    if keep_rfft:
        return fft_abs

    if ndim == 2:
        if last_dim % 2 == 0:
            fftstar_abs = fft_abs.copy()[:, -2:0:-1]
        else:
            fftstar_abs = fft_abs.copy()[:, -1:0:-1]

        fftstar_abs[1::, :] = fftstar_abs[:0:-1, :]

        return np.concatenate((fft_abs, fftstar_abs), axis=1)

    elif ndim == 3:
        if last_dim % 2 == 0:
            fftstar_abs = fft_abs.copy()[:, :, -2:0:-1]
        else:
            fftstar_abs = fft_abs.copy()[:, :, -1:0:-1]

        fftstar_abs[1::, :, :] = fftstar_abs[:0:-1, :, :]
        fftstar_abs[:, 1::, :] = fftstar_abs[:, :0:-1, :]

        return np.concatenate((fft_abs, fftstar_abs), axis=2)
