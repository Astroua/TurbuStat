# Licensed under an MIT open source license - see LICENSE
from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

from astropy.convolution import convolve_fft
from astropy.version import version as astro_version


def convolution_wrapper(img, kernel, **kwargs):
    '''
    Adjust parameter setting to be consistent with astropy <2 and >=2.
    '''

    if int(astro_version[0]) >= 2:
        if kwargs.get("nan_interpolate"):
            if kwargs['nan_interpolate']:
                nan_treatment = 'interpolate'
            else:
                nan_treatment = 'fill'
        else:
            # Default to not nan interpolating
            nan_treatment = 'fill'
        kwargs.pop('nan_interpolate')

        conv_img = convolve_fft(img, kernel, normalize_kernel=True,
                                nan_treatment=nan_treatment,
                                preserve_nan=False,
                                **kwargs)
    else:
        # in astropy >= v2, fill_value can be a NaN. ignore_edge_zeros gives
        # the same behaviour in older versions.
        if kwargs.get('fill_value'):
            kwargs.pop('fill_value')
        conv_img = convolve_fft(img, kernel, normalize_kernel=True,
                                ignore_edge_zeros=True, **kwargs)

    return conv_img
