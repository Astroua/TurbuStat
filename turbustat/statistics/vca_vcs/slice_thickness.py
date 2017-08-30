# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy import units as u
from spectral_cube.spectral_cube import BaseSpectralCube
from astropy.convolution import Gaussian1DKernel
from warnings import warn


def spectral_regrid_cube(cube, channel_width):
    '''
    Spectrally regrid a SpectralCube to a given channel width. The data are
    first spectrally smoothed with a Gaussian kernel. The kernel width is
    the target channel width deconvolved from the current channel width
    (http://spectral-cube.readthedocs.io/en/latest/smoothing.html#spectral-smoothing).

    Note that, in order to ensure the regridded cube covers the same spectral
    range, the smoothed cube may have channels that differ slightly from the
    given width. The number of channels is chosen to be the next lowest
    integer when dividing the original number of channels by the ratio of the
    new and old channel widths.

    Parameters
    ----------
    cube : `~spectral_cube.SpectralCube`
        Spectral cube to regrid.
    channel_width : `~astropy.units.Quantity`
        The width of the new channels. This should be given in equivalent
        spectral units used in the cube, or in pixel units. For example,
        downsampling by a factor of 2 for a cube with a channel width of
        0.1 km/s can be achieved by setting `channel_width` to `2 * u.pix`
        or `0.2 km /s`.

    Returns
    -------
    regridded_cube : `spectral_cube.SpectralCube`
        The smoothed and regridded cube.
    '''

    if not isinstance(cube, BaseSpectralCube):
        raise TypeError("`cube` must be a SpectralCube object.")

    if not isinstance(channel_width, u.Quantity):
        raise TypeError("channel_width must be an astropy.units.Quantity.")

    fwhm_factor = np.sqrt(8 * np.log(2))

    pix_unit = channel_width.unit.is_equivalent(u.pix)

    spec_width = np.diff(cube.spectral_axis[:2])[0]

    current_resolution = np.diff(cube.spectral_axis[:2])[0]

    if pix_unit:
        target_resolution = channel_width.value * spec_width
    else:
        target_resolution = channel_width.to(current_resolution.unit)

    diff_factor = np.abs(target_resolution / current_resolution).value

    if diff_factor == 1:
        warn("The requested channel width match the original channel width. "
             "The original cube is returned.")
        return cube

    if diff_factor < 1:
        raise ValueError("Only down-sampling the spectral grid is supported. "
                         "The requested channel width of {0} is a factor {1} "
                         "smaller than the original channel width."
                         .format(target_resolution, diff_factor))

    pixel_scale = np.abs(current_resolution)

    gaussian_width = ((target_resolution**2 - current_resolution**2)**0.5 /
                      pixel_scale / fwhm_factor)
    kernel = Gaussian1DKernel(gaussian_width)
    new_cube = cube.spectral_smooth(kernel)

    # Now define the new spectral axis at the new resolution
    num_chan = int(np.floor_divide(cube.shape[0], diff_factor))
    new_specaxis = np.linspace(cube.spectral_axis.min().value,
                               cube.spectral_axis.max().value,
                               num_chan) * current_resolution.unit
    # Keep the same order (max to min or min to max)
    if current_resolution.value < 0:
        new_specaxis = new_specaxis[::-1]

    return new_cube.spectral_interpolate(new_specaxis,
                                         suppress_smooth_warning=True)
