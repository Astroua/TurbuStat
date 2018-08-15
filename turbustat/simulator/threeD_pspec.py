
import numpy as np


def threeD_pspec(arr):
    '''
    Return a 1D power spectrum from a 3D array.

    Parameters
    ----------
    arr : `~numpy.ndarray`
        Three dimensional array.

    Returns
    -------
    freq_bins : `~numpy.ndarray`
        Radial frequency bins.
    ps1D : `~numpy.ndarray`
        One-dimensional azimuthally-averaged power spectrum.
    ps1D_stderr : `~numpy.ndarray`
        Standard deviation of `ps1D`.
    '''

    if arr.ndim != 3:
        raise ValueError("arr must have three dimensions.")

    ps3D = np.abs(np.fft.fftn(arr))**2

    xfreq = np.fft.fftfreq(arr.shape[0])
    yfreq = np.fft.fftfreq(arr.shape[1])
    zfreq = np.fft.fftfreq(arr.shape[2])

    xx, yy, zz = np.meshgrid(xfreq, yfreq, zfreq, indexing='ij')

    rr = np.sqrt(xx**2 + yy**2 + zz**2)

    freq_min = 1 / float(max(arr.shape))
    freq_max = 1 / 2.

    freq_bins = np.arange(freq_min, freq_max, freq_min)

    whichbin = np.digitize(rr.flat, freq_bins)
    ncount = np.bincount(whichbin)

    ps1D = np.zeros(len(ncount) - 1)
    ps1D_stderr = np.zeros(len(ncount) - 1)

    for n in range(1, len(ncount)):
        ps1D[n - 1] = np.mean(ps3D.flat[whichbin == n])
        ps1D_stderr[n - 1] = np.std(ps3D.flat[whichbin == n])

    return freq_bins, ps1D, ps1D_stderr
