
'''
Wrapper on spectral_cube for simulated datasets
'''

import numpy as np

import spectral_cube as SpectralCube

try:
    from signal_id import Noise
except ImportError:
    prefix = "/srv/astro/erickoch/"  # Adjust if you're not me!
    execfile(prefix + "Dropbox/code_development/signal-id/noise.py")


class SimCube(object):

    def __init__(self, cube, beam=None, mask=None, method="MAD", compute=True):

        # Initialize cube object
        self.cube = SpectralCube.read(cube)

        # Initialize noise object
        self.noise = Noise(self.cube, beam=beam, method=method)

        self.mask = mask

    def add_noise(self):

        # Create the noisy cube
        self.noise.get_noise_cube()
        self._noise_cube = self.noise.noise_cube +\
            self.cube.filled_data[:]

        # Update SpectralCube object
        self._update(data=self.noise_cube)

        return self

    def apply_mask(self, mask=None):

        # Update mask
        if mask is not None:
            self.mask = mask

        # Create the mask, auto masking nan values
        default_mask = np.isfinite(self.cube)
        if self.mask is not None:
            self.mask *= default_mask
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
        if data is None & wcs is None:
            pass
        else:
            if data is None:
                data = self.cube.unmasked_data[:]
            if wcs is None:
                wcs = self.cube.wcs
            # Make new SpectralCube object
            self.cube = SpectralCube(data=data, wcs=wcs)

        if beam is not None:
            self.noise = Noise(self.cube, beam=beam, method=method)

    def compute_properties(self):

        self._moment0 = self.cube.moment0().value

        self._moment1 = self.cube.moment1().value

        self._moment2 = self.cube.moment2().value

        self.get_int_intensity()

        return self

    @property
    def noise_cube(self):
        return self._noise_cube

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
