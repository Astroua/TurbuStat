
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

    ndim = len(mask.shape)

    if ndim < 2 or ndim > 4:
        raise ValueError("mask must have dimensions between 2 and 4.")

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
