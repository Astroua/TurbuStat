
'''
Wrapper class for handling observational datasets
'''

import numpy as np

import spectral_cube as sc

try:
    from signal_id import Mask, Noise
except:
    print("No signal_id package!")

#  from cleaning_algs import *
from cube_utils import _check_mask, _check_beam # , _update


class ObsCube(object):
    """
    A wrapping class of SpectralCube which prepares observational data cubes
    to be compared to any other data cube.

    Parameters
    ----------

    cube : str
        Path to file.
    mask : numpy.ndarray, any mask class from spectral_cube, optional
        Mask to be applied to the cube.
    algorithm : {NAME HERE}, optional
        Name of the cleaning algorithm to use.

    Example
    -------
    ```
    from turbustat.cube_tools import ObsCube

    cube = ObsCube("data.fits")
    cube.apply_cleaning(algorithm="SUPERAWESOMECLEANING")

    ```
    """
    def __init__(self, cube, mask=None, algorithm=None, beam=None):
        super(ObsCube, self).__init__()
        self.cube = sc.SpectralCube.read(cube)

        self.algorithm = algorithm

        # Make sure mask is an accepted type
        if mask is not None:
            _check_mask(mask)
        self.mask = mask

        if beam is not None:
            _check_beam(beam)
        self.noise = Noise(self.cube, beam=beam)

    def apply_cleaning(self):

        # self = _update(data=data, wcs=wcs)

        return self

    def apply_mask(self, mask=None):
        # self.cube = Mask(self.cube)
        if mask is not None:
            _check_mask(mask)
            self.mask = mask

        default_mask = np.isfinite(self.cube)
        ## ADD a snr check to the default mask
        default_mask *= self.noise.spectral_norm > self.noise.scale
        if self.mask is None:
            self.mask = default_mask
        else:
            self.mask *= default_mask

        self.cube.apply_mask(self.mask)

        return self

    def compute_properties(self):
        '''
        Compute the properties of the cube.
        '''

        self._moment0 = self.cube.moment0()

        self._moment1 = self.cube.moment1()

        self._moment2 = self.cube.moment2()

        self._intint = self.get_int_intensity()

        return self

    @property
    def moment0(self):
        return self._moment0

    @property
    def moment1(self):
        return self._moment1

    @property
    def moment2(self):
        return self._moment2

    @property
    def intint(self):
        return self._intint

    def get_int_intensity(self):
        '''
        Get an integrated intensity image of the cube.
        '''

        good_channels = self.noise.spectral_norm > self.noise.scale

        channel_range = self.cube.spectral_axis[good_channels][[0, -1]]

        channel_size = np.abs(self.cube.spectral_axis[1] -
                              self.cube.spectral_axis[0])

        slab = self.cube.spectral_slab(*channel_range).filled_data[:]

        self._intint = np.nansum(slab, axis=0) * channel_size

        return self
