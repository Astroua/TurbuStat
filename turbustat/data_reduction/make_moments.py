
from spectral_cube import SpectralCube, LazyMask
from spectral_cube.wcs_utils import drop_axis
from signal_id import Noise, RadioMask
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve
from scipy import ndimage as nd
import itertools as it
import operator as op
import os

from _moment_errs import _slice0, _slice1, _slice2, _cube0, _cube1, _cube2


class Mask_and_Moments(object):
    """
    A unified approach to deriving the noise level in a cube, applying a
    mask, and deriving moments along with their errors. All the heavy lifting
    is done with `spectral_-ube <http://spectral-cube.readthedocs.org/en/latest/>`_.

    Parameters
    ----------
    cube : SpectralCube or str
        Either a SpectralCube object, or the filename of a cube readable
        by spectral-cube.
    noise_type : {'constant'}, optional
        *NO CURRENT FUNCTION* Once implemented, it will set parameters
        for deriving the noise level.
    clip : float, optional
        Sigma level to clip data at.
    scale : float, optional
        The noise level in the cube. Overrides estimation using
        `signal_id <https://github.com/radio-astro-tools/signal-id>`_
    moment_method : {'slice', 'cube', 'ray'}, optional
        The method to use for creating the moments. See the spectral-cube
        docs for an explanation of the differences.
    """
    def __init__(self, cube, noise_type='constant', clip=3, scale=None,
                 moment_method='slice'):
        super(Mask_and_Moments, self).__init__()

        if isinstance(cube, SpectralCube):
            self.cube = cube
            self.save_name = None
        else:
            self.cube = SpectralCube.read(cube)
            # Default save name to the cube name without the suffix.
            self.save_name = ".".join(cube.split(".")[:-1])

        self.noise_type = noise_type
        self.clip = clip

        if moment_method not in ['slice', 'cube', 'ray']:
            raise TypeError("Moment method must be 'slice', 'cube', or 'ray'.")
        self.moment_method = moment_method

        if scale is None:
            self.scale = Noise(self.cube).scale
        else:
            self.scale = scale

        self.prop_headers = None
        self.prop_err_headers = None

    def find_noise(self, return_obj=False):

        noise = Noise(self.cube)

        self.scale = noise.scale

        if return_obj:
            return noise

        return noise.scale

    def make_mask(self, mask=None):

        if mask is None:
            rad_mask = RadioMask(self.cube)
            mask = rad_mask.to_mask()

        self.cube = self.cube.with_mask(mask)

        return self

    def make_moments(self, axis=0, units=False):

        self._moment0 = self.cube.moment0(axis=axis, how=self.moment_method)
        self._moment1 = self.cube.moment1(axis=axis, how=self.moment_method)
        self._moment2 = self.cube.moment2(axis=axis, how=self.moment_method)

        # The 'how' is set directly in the int intensity function.
        self._intint = self._get_int_intensity(axis=axis)

        if not units:
            self._moment0 = self._moment0.value
            self._moment1 = self._moment1.value
            self._moment2 = self._moment2.value
            self._intint = self._intint.value
        return self

    def make_moment_errors(self):

        self._moment0_err = self._get_moment0_err()
        self._moment1_err = self._get_moment1_err()
        self._moment2_err = self._get_moment2_err()
        self._intint_err = self._get_int_intensity_err()

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
    def linewidth(self):
        return np.sqrt(self.moment2)

    @property
    def intint(self):
        return self._intint

    @property
    def moment0_err(self):
        return self._moment0_err

    @property
    def moment1_err(self):
        return self._moment1_err

    @property
    def moment2_err(self):
        return self._moment2_err

    @property
    def linewidth_err(self):
        return self.moment2_err / (2 * np.sqrt(self.moment2))

    @property
    def intint_err(self):
        return self._intint_err

    def all_moments(self):
        return [self._moment0, self._moment1, self.linewidth, self._intint]

    def all_moment_errs(self):
        return [self._moment0_err, self._moment1_err, self.linewidth_err,
                self._intint_err]

    def to_dict(self):
        '''
        Returns a dictionary form containing the cube and the property arrays.
        This is the expected form for the wrapper scripts and methods in
        TurbuStat.
        '''

        self.get_prop_hdrs()

        prop_dict = {}

        if _try_remove_unit(self.cube.filled_data[:]):
            prop_dict['cube'] = [self.cube.filled_data[:].value,
                                 self.cube.header]
        else:
            prop_dict['cube'] = [self.cube.filled_data[:], self.cube.header]

        if _try_remove_unit(self.moment0):
            prop_dict['moment0'] = [self.moment0.value, self.prop_headers[0]]
        else:
            prop_dict['moment0'] = [self.moment0, self.prop_headers[0]]

        if _try_remove_unit(self.moment0_err):
            prop_dict['moment0_error'] = [self.moment0_err.value,
                                          self.prop_err_headers[0]]
        else:
            prop_dict['moment0_error'] = [self.moment0_err,
                                          self.prop_err_headers[0]]

        if _try_remove_unit(self.moment1):
            prop_dict['centroid'] = [self.moment1.value, self.prop_headers[1]]
        else:
            prop_dict['centroid'] = [self.moment1, self.prop_headers[1]]

        if _try_remove_unit(self.moment1_err):
            prop_dict['centroid_error'] = [self.moment1_err.value,
                                           self.prop_err_headers[1]]
        else:
            prop_dict['centroid_error'] = [self.moment1_err,
                                           self.prop_err_headers[1]]

        if _try_remove_unit(self.linewidth):
            prop_dict['linewidth'] = [self.linewidth.value,
                                      self.prop_headers[2]]
        else:
            prop_dict['linewidth'] = [self.linewidth,
                                      self.prop_headers[2]]

        if _try_remove_unit(self.linewidth_err):
            prop_dict['linewidth_error'] = [self.linewidth_err.value,
                                            self.prop_err_headers[2]]
        else:
            prop_dict['linewidth_error'] = [self.linewidth_err,
                                            self.prop_err_headers[2]]

        if _try_remove_unit(self.intint):
            prop_dict['integrated_intensity'] = [self.intint.value,
                                                 self.prop_headers[3]]
        else:
            prop_dict['integrated_intensity'] = [self.intint,
                                                 self.prop_headers[3]]

        if _try_remove_unit(self.intint_err):
            prop_dict['integrated_intensity_error'] = \
                [self.intint_err.value, self.prop_err_headers[3]]
        else:
            prop_dict['integrated_intensity_error'] = \
                [self.intint_err, self.prop_err_headers[3]]

        return prop_dict

    def get_prop_hdrs(self):
        '''
        '''

        bunits = [self.cube.unit, self.cube.spectral_axis.unit,
                  self.cube.spectral_axis.unit,
                  self.cube.unit*self.cube.spectral_axis.unit]

        comments = ["Image of the Zeroth Moment",
                    "Image of the First Moment",
                    "Image of the Second Moment",
                    "Image of the Integrated Intensity"]

        self.prop_headers = []
        self.prop_err_headers = []

        for i in range(len(bunits)):

            wcs = self.cube.wcs.copy()
            new_wcs = drop_axis(wcs, -1)

            hdr = new_wcs.to_header()
            hdr_err = new_wcs.to_header()
            hdr["BUNIT"] = bunits[i].to_string()
            hdr_err["BUNIT"] = bunits[i].to_string()
            hdr["COMMENT"] = comments[i]
            hdr_err["COMMENT"] = comments[i] + " Error."

            self.prop_headers.append(hdr)
            self.prop_err_headers.append(hdr_err)

        return self

    def to_fits(self, save_name=None):
        '''
        Save the property arrays as fits files.
        '''

        if self.prop_headers is None:
            self.get_prop_hdrs()

        if save_name is None:
            if self.save_name is None:
                Warning("No save_name has been specified, using 'default'")
                self.save_name = 'default'
        else:
            self.save_name = save_name

        labels = ["_moment0", "_centroid", "_linewidth", "_intint"]

        for i, (arr, err, hdr, hdr_err) in \
          enumerate(zip(self.all_moments(), self.all_moment_errs(),
                        self.prop_headers, self.prop_err_headers)):

            hdu = fits.HDUList([fits.PrimaryHDU(arr, header=hdr),
                                fits.ImageHDU(err, header=hdr_err)])

            hdu.writeto(self.save_name+labels[i]+".fits")

    @staticmethod
    def from_fits(fits_name, moments_path=None, mask_name=None,
                  moment0=None, centroid=None, linewidth=None,
                  intint=None):
        '''
        Load pre-made moment arrays given a cube name. Saved moments must
        match the naming of the cube for the automatic loading to work
        (e.g. a cube called test.fits will have a moment 0 array solved
        test_moment0.fits). Otherwise, specify a path to one of the keyword
        arguments.

        Parameters
        ----------
        fits_name : str
            Filename of the cube. Is also used as the prefix to the saved
            moment files.
        moments_path : str, optional
            Path to where the moments are saved.
        mask_name : str, optional
            Filename of a saved mask to be applied to the data.
        moment0 : str, optional
            Filename of the moment0 array. Use if naming scheme is not valid
            for automatic loading.
        centroid : str, optional
            Filename of the centroid array. Use if naming scheme is not valid
            for automatic loading.
        linewidth : str, optional
            Filename of the linewidth array. Use if naming scheme is not valid
            for automatic loading.
        intint : str, optional
            Filename of the integrated intensity array. Use if naming scheme
            is not valid for automatic loading.
        '''

        if moments_path is None:
            moments_path = ""

        root_name = fits_name[:-4]

        self = Mask_and_Moments(fits_name)

        if mask_name is not None:
            mask = fits.getdata(mask_name)
            self.with_mask(mask=mask)

        # Moment 0
        if moment0 is not None:
            moment0 = fits.open(moment0)
            self._moment0 = moment0[0].data
            self._moment0_err = moment0[1].data
        else:
            try:
                moment0 = fits.open(os.path.join(moments_path,
                                                 root_name+"_moment0.fits"))
                self._moment0 = moment0[0].data
                self._moment0_err = moment0[1].data
            except IOError as e:
                self._moment0 = None
                self._moment0_err = None
                print e
                print("Moment 0 fits file not found.")

        if centroid is not None:
            moment1 = fits.open(centroid)
            self._moment1 = moment1[0].data
            self._moment1_err = moment1[1].data
        else:
            try:
                moment1 = fits.open(os.path.join(moments_path,
                                                 root_name+"_centroid.fits"))
                self._moment1 = moment1[0].data
                self._moment1_err = moment1[1].data
            except IOError as e:
                self._moment1 = None
                self._moment1_err = None
                print e
                print("Centroid fits file not found.")

        if linewidth is not None:
            linewidth = fits.open(linewidth)

            self._moment2 = np.power(linewidth[0].data, 2.)
            self._moment2_err = linewidth[1].data * (2 * np.sqrt(self.moment2))
        else:
            try:
                linewidth = \
                    fits.open(os.path.join(moments_path,
                                           root_name+"_linewidth.fits"))

                self._moment2 = np.power(linewidth[0].data, 2.)
                self._moment2_err = linewidth[1].data * (2 * np.sqrt(self.moment2))
            except IOError as e:
                self._moment2 = None
                self._moment2_err = None
                print e
                print("Linewidth fits file not found.")

        if intint is not None:
            intint = fits.open(intint)
            self._intint = intint[0].data
            self._intint_err = intint[1].data
        else:
            try:
                intint = fits.open(os.path.join(moments_path,
                                                 root_name+"_intint.fits"))
                self._intint = intint[0].data
                self._intint_err = intint[1].data
            except IOError as e:
                self._intint = None
                self._intint_err = None
                print e
                print("Integrated intensity fits file not found.")

        return self

    def _get_int_intensity(self, axis=0):
        '''
        Get an integrated intensity image of the cube.

        Parameters
        ----------

        '''

        shape = self.cube.shape
        view = [slice(None)] * 3

        if self.moment_method is 'cube':
            channel_max = \
                np.nanmax(self.cube.filled_data[:].reshape(-1, shape[1]*shape[2]),
                          axis=1).value
        else:
            channel_max = np.empty((shape[axis]))
            for i in range(shape[axis]):
                view[axis] = i
                plane = self.cube._get_filled_data(fill=0, view=view)

                channel_max[i] = np.nanmax(plane)

        good_channels = np.where(channel_max > self.clip*self.scale)[0]

        # Get the longest sequence
        good_channels = longestSequence(good_channels)

        if not np.any(good_channels):
            raise ValueError("Cannot find any channels with signal.")

        self.channel_range = self.cube.spectral_axis[good_channels][[0, -1]]

        slab = self.cube.spectral_slab(*self.channel_range)

        return slab.moment0(axis=axis, how=self.moment_method)

    def _get_int_intensity_err(self, axis=0):
        '''
        '''
        slab = self.cube.spectral_slab(*self.channel_range)

        if self.moment_method is 'cube':
            return _cube0(slab, axis, self.scale)
        elif self.moment_method is 'slice':
            return _slice0(slab, axis, self.scale)
        elif self.moment_method is 'ray':
            raise NotImplementedError

    def _get_moment0_err(self, axis=0):
        '''
        '''

        if self.moment_method is 'cube':
            return _cube0(self.cube, axis, self.scale)
        elif self.moment_method is 'slice':
            return _slice0(self.cube, axis, self.scale)
        elif self.moment_method is 'ray':
            raise NotImplementedError

    def _get_moment1_err(self, axis=0):
        '''
        '''

        if self.moment_method is 'cube':
            return _cube1(self.cube, axis, self.scale, self.moment0,
                          self.moment1)
        elif self.moment_method is 'slice':
            return _slice1(self.cube, axis, self.scale, self.moment0,
                           self.moment1)
        elif self.moment_method is 'ray':
            raise NotImplementedError

    def _get_moment2_err(self, axis=0):
        '''
        '''

        if self.moment_method is 'cube':
            return _cube2(self.cube, axis, self.scale, self.moment0,
                          self.moment1, self.moment2, self.moment1_err)
        elif self.moment_method is 'slice':
            return _slice2(self.cube, axis, self.scale, self.moment0,
                           self.moment1, self.moment2, self.moment1_err)
        elif self.moment_method is 'ray':
            raise NotImplementedError


def moment_masking(cube, kernel_size, clip=5, dilations=1):
    '''
    '''

    smooth_data = convolve(cube.filled_data[:], gauss_kern(kernel_size))

    fake_mask = LazyMask(np.isfinite, cube=cube)

    smooth_cube = SpectralCube(data=smooth_data, wcs=cube.wcs, mask=fake_mask)

    smooth_scale = Noise(smooth_cube).scale

    mask = (smooth_cube > (clip * smooth_scale)).include()

    # Now dilate the mask once

    dilate_struct = nd.generate_binary_structure(3, 3)
    mask = nd.binary_dilation(mask, structure=dilate_struct,
                              iterations=dilations)

    return mask


def gauss_kern(size, ysize=None, zsize=None):
    """ Returns a normalized 3D gauss kernel array for convolutions """
    size = int(size)
    if not ysize:
        ysize = size
    else:
        ysize = int(ysize)
    if not zsize:
        zsize = size
    else:
        zsize = int(zsize)

    x, y, z = np.mgrid[-size:size + 1, -ysize:ysize + 1, -zsize:zsize + 1]
    g = np.exp(-(x ** 2 / float(size) + y **
                 2 / float(ysize) + z ** 2 / float(zsize)))
    return g / g.sum()


def _try_remove_unit(arr):
    try:
        unit = arr.unit
        return True
    except AttributeError:
        return False


def longestSequence(data):

    longest = []

    sequences = []
    for k, g in it.groupby(enumerate(data), lambda(i, y): i-y):
        sequences.append(map(op.itemgetter(1), g))

    longest = max(sequences, key=len)

    return longest
