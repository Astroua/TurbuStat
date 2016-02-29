"""
From Adam Ginsburg's AG_fft_tools:

Copyright (c) 2009 Adam Ginsburg

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
try:
    import matplotlib.pyplot as pyplot
    pyplotOK = True
except ImportError:
    pyplotOK = False
from radialProfile import azimuthalAverageBins, radialAverageBins
from scipy.stats import binned_statistic


def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.
    See numpy.hanning for the 1d description
    """

    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        # scalar unity; don't window if dims are too small
        return np.hanning(N)
    else:
        return np.outer(np.hanning(M), np.hanning(N))


def power_spectrum(*args, **kwargs):
    """
    Thin wrapper of PSD2. Returns the 1D power spectrum in stead of the 2D
    Power Spectral Density
    """
    kwargs['oned'] = True
    return PSD2(*args, **kwargs)


def PSD2(image, image2=None, oned=False,
         fft_pad=False, real=False, imag=False,
         binsize=1.0, radbins=1, azbins=1, radial=False, hanning=False,
         wavnum_scale=False, twopi_scale=False, **kwargs):
    """
    Two-dimensional Power Spectral Density.
    NAN values are treated as zero.

    image2 - can specify a second image if you want to see the
        cross-power-spectrum instead of the power spectrum.
    oned - return radial profile of 2D PSD (i.e. mean power as a function of
        spatial frequency)
    freq, zz = PSD2(image); plot(freq,zz) is a power spectrum
    fft_pad - Add zeros to the edge of the image before FFTing for a speed
    boost? (the edge padding will be removed afterwards)
    real - Only compute the real part of the PSD (Default is absolute value)
    imag - Only compute the complex part of the PSD (Default is absolute value)
    hanning - Multiply the image to be PSD'd by a 2D Hanning window before
        performing the FTs. Reduces edge effects. This idea courtesy
        Paul Ricchiazzia (May 1993), author of the IDL astrolib psd.pro
    wavnum_scale - multiply the FFT^2 by the wavenumber when computing the PSD?
    twopi_scale - multiply the FFT^2 by 2pi?
    azbins - Number of azimuthal (angular) bins to include. Default is 1, or
    all 360 degrees. If azbins>1, the data will be split into [azbins]
    equally sized pie pieces. Azbins can also be a np array. See
    AG_image_tools.azimuthalAverageBins for details
    radial - An option to return the *azimuthal* power spectrum (i.e. the
        spectral power as a function of angle). Not commonly used.
    radbins - number of radial bins (you can compute the azimuthal power
        spectrum in different annuli)
    """

    # prevent modification of input image (i.e., the next two lines of active
    # code)
    image = image.copy()

    # remove NANs (but not inf's)
    image[image != image] = 0

    if hanning:
        image = hanning2d(*image.shape) * image

    if image2 is None:
        image2 = image
    else:
        image2 = image2.copy()
        image2[image2 != image2] = 0
        if hanning:
            image2 = hanning2d(*image2.shape) * image2

    if real:
        psd2 = np.real(
            correlate2d(image, image2, return_fft=True, fft_pad=fft_pad))
    elif imag:
        psd2 = np.imag(
            correlate2d(image, image2, return_fft=True, fft_pad=fft_pad))
    else:  # default is absolute value
        psd2 = np.abs(
            correlate2d(image, image2, return_fft=True, fft_pad=fft_pad))
    # normalization is approximately
    # (np.abs(image).sum()*np.abs(image2).sum())

    if wavnum_scale:
        wx = np.concatenate([np.arange(image.shape[0] / 2, dtype='float'),
                             image.shape[0] / 2 - np.arange(image.shape[0] / 2, dtype='float') - 1]) / (image.shape[0] / 2.)
        wy = np.concatenate([np.arange(image.shape[1] / 2, dtype='float'), image.shape[
                               1] / 2 - np.arange(image.shape[1] / 2, dtype='float') - 1]) / (image.shape[1] / 2.)
        wx /= wx.max()
        wy /= wy.max()
        wavnum = np.sqrt(np.outer(
            wx, np.ones(wx.shape))**2 + np.outer(np.ones(wy.shape), wx)**2)
        psd2 *= wavnum

    if twopi_scale:
        psd2 *= np.pi * 2

    if radial:
        azbins, az, zz = radialAverageBins(
            psd2, radbins=radbins, interpnan=True, binsize=binsize, **kwargs)
        if len(zz) == 1:
            return az, zz[0]
        else:
            return az, zz

    if oned:
        return pspec(psd2, azbins=azbins, binsize=binsize, **kwargs)

    # else...
    return psd2


def pspec(psd2, return_index=True, wavenumber=True, return_stddev=False,
          azbins=1, binsize=1.0, view=False, **kwargs):
    """
    Create a Power Spectrum (radial profile of a PSD) from a Power Spectral
    Density image

    return_index - if true, the first return item will be the indexes
    wavenumber - if one dimensional and return_index set, will return a
        normalized wavenumber instead
    view - Plot the PSD (in logspace)?
    """

    azbins, (freq, zz) = azimuthalAverageBins(
        psd2, azbins=azbins, interpnan=True, binsize=binsize, **kwargs)
    if len(zz) == 1:
        zz = zz[0]
    # the "Frequency" is the spatial frequency f = 1/x for the standard np
    # fft, which follows the convention
    # A_k = \sum_{m=0}^{n-1} a_m \exp\left\{-2\pi i{mk \over n}\right\}
    # or
    # F_f = Sum( a_m e^(-2 pi i f x_m) over the range m,m_max where a_m are
    # the values of the pixels, x_m are the
    # indices of the pixels, and f is the spatial frequency
    # there was a +1.0 here before, presumably to deal with div-by-0, but that
    # shouldn't happen and shouldn't have been "accounted for" anyway
    freq = freq.astype('float')

    if return_index:
        if wavenumber:
            fftwavenum = np.fft.rfftfreq(zz.size)
            return_vals = list((fftwavenum, zz))
        else:
            return_vals = list((freq, zz))
    else:
        return_vals = list(zz)
    if return_stddev:
        zzstd = \
            azimuthalAverageBins(psd2, azbins=azbins, stddev=True,
                                 interpnan=True,
                                 binsize=binsize, **kwargs)[1][1]
        return_vals.append(zzstd)

    if view and pyplotOK:
        pyplot.loglog(freq, zz)
        pyplot.xlabel("Spatial Frequency")
        pyplot.ylabel("Spectral Power")

    return return_vals

##########################################################################


def correlate2d(im1, im2, boundary='wrap', **kwargs):
    """
    Cross-correlation of two images of arbitrary size. Returns an image
    cropped to the largest of each dimension of the input images

    Options
    -------
    return_fft - if true, return fft(im1)*fft(im2[::-1,::-1]), which is the power
    spectral density
    fftshift - if true, return the shifted psd so that the DC component is in
    the center of the image
    pad - Default on. Zero-pad image to the nearest 2^n
    crop - Default on. Return an image of the size of the largest input image.
    If the images are asymmetric in opposite directions, will return the largest
    image in both directions.
    boundary: str, optional
    A flag indicating how to handle boundaries:
    * 'fill' : set values outside the array boundary to fill_value
    (default)
    * 'wrap' : periodic boundary

    WARNING: Normalization may be arbitrary if you use the PSD
    """

    from astropy.convolve import convolve

    return convolve(np.conjugate(im1), im2[::-1, ::-1], normalize_kernel=False,
                    boundary=boundary, ignore_edge_zeros=False, **kwargs)


def new_pspec(psd2, bins=10, return_stddev=False):
    '''
    Calculate the radial profile using scipy.stats.binned_statistic.
    '''

    yfreqs = np.fft.fftshift(np.abs(np.fft.fftfreq(psd2.shape[0])))
    xfreqs = np.fft.fftshift(np.abs(np.fft.fftfreq(psd2.shape[1])))

    yy, xx = np.meshgrid(yfreqs, xfreqs)

    freqs_dist = np.sqrt(yy**2 + xx**2)

    # nbins = int(np.round(freqs_dist.max()) + 1)
    # bins = np.logspace(np.log10(freqs_dist.min()), 0., nbins+1)

    ps1D, bin_edge, cts = binned_statistic(freqs_dist.ravel(),
                                           psd2.ravel(),
                                           bins=bins)#,
                                           # statistic=np.nanmean)

    bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.

    if not return_stddev:
        return bin_cents, ps1D
    else:
        ps1D_stddev, _, _ = binned_statistic(freqs_dist.ravel(),
                                             psd2.ravel(),
                                             bins=bins,
                                             statistic=np.nanstd)
        return bin_cents, ps1D, ps1D_stddev
