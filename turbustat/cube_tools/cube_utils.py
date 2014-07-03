# Licensed under an MIT open source license - see LICENSE


'''
Utilities for cube_tools
'''

import numpy as np
import warnings

import spectral_cube as sc
from astropy.convolution import Kernel2D

try:
    from radio_beam import Beam
except ImportError:
    warnings.warn("radio_beam is not available")
    Beam = type(None)


def _check_mask(mask):
    '''
    Checks to make sure mask is of an appropriate form.
    '''

    if isinstance(mask, np.ndarray):
        pass
    elif isinstance(mask, sc.LazyMask):
        pass
    elif isinstance(mask, sc.BooleanArrayMask):
        pass
    elif isinstance(mask, sc.CompositeMask):
        pass
    elif isinstance(mask, sc.FunctionMask):
        pass
    elif isinstance(mask, sc.InvertedMask):
        pass
    else:
        raise TypeError("Inputted mask of type %s is not an accepted type."
                        % (type(mask)))


def _check_beam(beam):
    '''
    '''
    if isinstance(beam, Beam):
        pass
    elif isinstance(beam, Kernel2D):
        pass
    else:
        raise TypeError("beam of type %s is not an accepted type" %
                        (type(beam)))


def _get_int_intensity(cube_class):
    '''
    Get an integrated intensity image of the cube.

    Parameters
    ----------

    cube_class - SimCube or ObsCube class
    '''

    good_channels = cube_class.noise.spectral_norm > cube_class.noise.scale

    channel_range = cube_class.cube.spectral_axis[good_channels][[0, -1]]

    channel_size = np.abs(cube_class.cube.spectral_axis[1] -
                          cube_class.cube.spectral_axis[0])

    slab = cube_class.cube.spectral_slab(*channel_range).filled_data[:]

    cube_class._intint = np.nansum(slab, axis=0) * channel_size

    return cube_class
