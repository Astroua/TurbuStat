# Licensed under an MIT open source license - see LICENSE


'''
Wrapper on spectral_cube for simulated datasets
'''

import numpy as np

from spectral_cube import SpectralCube, CompositeMask

try:
    from signal_id import Noise
except ImportError:
    pass
    prefix = "/srv/astro/erickoch/"  # Adjust if you're not me!
    execfile(prefix + "Dropbox/code_development/signal-id/signal_id/noise.py")

from cube_utils import _check_mask, _check_beam, _get_int_intensity


class SimCube(object):
    '''
    A wrapping class to prepare a simulated spectral data cube for
    comparison with another cube.
    '''

    def __init__(self, cube, beam=None, mask=None, method="MAD", compute=True):

        # Initialize cube object
        self.cube = SpectralCube.read(cube)

        if mask is not None:
            _check_mask(mask)
        self.mask = mask

        if beam is not None:
            _check_beam(mask)

        # Initialize noise object
        self.noise = Noise(self.cube, beam=beam, method=method)

    def add_noise(self):
        '''
        Use Noise to add synthetic noise to the data. Then update
        SpectralCube.
        '''

        # Create the noisy cube
        self.noise.get_noise_cube()
        noise_data = self.noise.noise_cube +\
            self.cube.filled_data[:]

        # Update SpectralCube object
        self._update(data=noise_data)

        return self

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

    def sim_prep(self, mask=None):
        '''
        Prepares the cube when being compared to another simulation.
        This entails:
            * Optionally applying a mask to the data.
            * Computing the cube's property arrays
        '''

        if not mask is None:
            self.apply_mask()

        self.compute_properties()

        return self

    def obs_prep(self, mask=None):
        '''
        Prepares the cube when being compared to observational data.
        This entails:
            * Optionally applying a mask to the data.
            * Add synthetic noise based on the cube's properties.
            * Computing the cube's property arrays
        '''

        if not mask is None:
            self.apply_mask()

        self.add_noise()
        self.compute_properties()

        return self
