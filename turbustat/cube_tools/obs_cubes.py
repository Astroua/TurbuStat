# Licensed under an MIT open source license - see LICENSE


'''
Wrapper class for handling observational datasets
'''

import numpy as np

from spectral_cube import SpectralCube, CompositeMask

try:
    from signal_id import Noise
except:
    print("No signal_id package!")
    pass
    prefix = "/srv/astro/erickoch/"  # Adjust if you're not me!
    execfile(prefix + "Dropbox/code_development/signal-id/signal_id/noise.py")


from cube_utils import _check_mask, _check_beam, _get_int_intensity
#  from cleaning_algs import *


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

    def clean_cube(self, algorithm=None):
        raise NotImplementedError("")

    def apply_mask(self, mask=None):
        '''
        Check if the given mask is acceptable abd apply to
        SpectralCube.
        '''

        # Update mask
        if mask is not None:
            _check_mask(mask)
            self.mask = mask

        # Create the mask, auto masking nan values
        default_mask = np.isfinite(self.cube.filled_data[:])
        if self.mask is not None:
            self.mask = CompositeMask(default_mask, self.mask)
        else:
            self.mask = default_mask

        # Apply mask to spectral cube object
        self.cube = self.cube.with_mask(mask)

        return self

    def _update(self, data=None, wcs=None, beam=None, method="MAD"):
        '''
        Helper function to update classes.
        '''

        # Check if we need a new SpectralCube
        if data is None and wcs is None:
            pass
        else:
            if data is None:
                data = self.cube.unmasked_data[:]
            if wcs is None:
                wcs = self.cube.wcs
            # Make new SpectralCube object
            self.cube = SpectralCube(data=data, wcs=wcs)

        if beam is not None:
            _check_beam(beam)
            self.noise = Noise(self.cube, beam=beam, method=method)

    def compute_properties(self):
        '''
        Use SpectralCube to compute the moments. Also compute the integrated
        intensity based on the noise properties from Noise.
        '''

        self._moment0 = self.cube.moment0().value

        self._moment1 = self.cube.moment1().value

        self._moment2 = self.cube.moment2().value

        _get_int_intensity(self)

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

    def prep(self, mask=None, algorithm=None):
        '''
        Prepares the cube to be compared to another cube.
        '''

        if not mask is None:
            self.apply_mask()

        self.clean_cube()
        self.compute_properties()

        return self
