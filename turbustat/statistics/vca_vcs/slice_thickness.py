# Licensed under an MIT open source license - see LICENSE


import numpy as np
from astropy import units as u
from spectral_cube import SpectralCube
from astropy.convolution import Gaussian1DKernel


def spectral_regrid_cube(cube, channel_width):

    fwhm_factor = np.sqrt(8 * np.log(2))
    current_resolution = np.diff(cube.spectral_axis[:2])[0]
    target_resolution = channel_width.to(current_resolution.unit)
    diff_factor = np.abs(target_resolution / current_resolution).value

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
