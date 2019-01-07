# Licensed under an MIT open source license - see LICENSE
from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
    PYFFTW_FLAG = True
except ImportError:
    PYFFTW_FLAG = False

from astropy.convolution import convolve_fft
from astropy.version import version as astro_version
import numpy as np
from warnings import warn


def convolution_wrapper(img, kernel, use_pyfftw=False, threads=1,
                        pyfftw_kwargs={}, **kwargs):
    '''
    Adjust parameter setting to be consistent with astropy <2 and >=2.

    Also allow the FFT to be performed with pyfftw.

    Parameters
    ----------
    img : numpy.ndarray
        Image.
    use_pyfftw : bool, optional
        Enable to use pyfftw, if it is installed.
    threads : int, optional
        Number of threads to use in FFT when using pyfftw.
    pyfftw_kwargs : dict, optional
        Passed to `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
        `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
        for a list of accepted kwargs.
    kwargs : Passed to `~astropy.convolution.convolve_fft`.

    Returns
    -------
    conv_img : `~numpy.ndarray`
        Convolved image.
    '''

    if use_pyfftw:
        if PYFFTW_FLAG:
            use_fftn = fftn
            use_ifftn = ifftn
        else:
            warn("pyfftw not installed. Using numpy.fft functions.")
            use_fftn = np.fft.fftn
            use_ifftn = np.fft.ifftn
    else:
        use_fftn = np.fft.fftn
        use_ifftn = np.fft.ifftn

    if int(astro_version[0]) >= 2:

        conv_img = convolve_fft(img, kernel, normalize_kernel=True,
                                fftn=use_fftn,
                                ifftn=use_ifftn,
                                **kwargs)

    else:
        raise Exception("Delta-variance requires astropy version >2.")

        # in astropy >= v2, fill_value can be a NaN. ignore_edge_zeros gives
        # the same behaviour in older versions.
        # if kwargs.get('fill_value'):
        #     kwargs.pop('fill_value')
        # conv_img = convolve_fft(img, kernel, normalize_kernel=True,
        #                         ignore_edge_zeros=True,
        #                         fftn=use_fftn,
        #                         ifftn=use_ifftn,
        #                         **kwargs)

    return conv_img
