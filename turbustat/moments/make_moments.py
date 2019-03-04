# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.io import fits
from astropy.convolution import convolve
import astropy.units as u
from scipy import ndimage as nd
import itertools as it
import operator as op
import os
from warnings import warn

try:
    from spectral_cube import SpectralCube, LazyMask
    from spectral_cube.wcs_utils import drop_axis
    spectral_cube_flag = True
except ImportError:
    warn("spectral-cube is not installed. Using Moments requires"
         " spectral-cube to be installed.")
    spectral_cube_flag = False

# try:
#     from signal_id import Noise
#     signal_id_flag = True
# except ImportError:
#     warn("signal-id is not installed. Disabling associated functionality.")
#     signal_id_flag = False

from ._moment_errs import (moment0_error, moment1_error, linewidth_sigma_err)


class Moments(object):
    """
    A unified approach to deriving the noise level in a cube, applying a
    mask, and deriving moments along with their errors. All the heavy lifting
    is done with
    `spectral_cube <http://spectral-cube.readthedocs.org/en/latest/>`_.

    Parameters
    ----------
    cube : SpectralCube or str
        Either a SpectralCube object, or the filename of a cube readable
        by spectral-cube.
    scale : `~astropy.units.Quantity`, optional
        The noise level in the cube. Used to estimate uncertainties of the
        moment maps.
    moment_method : {'slice', 'cube', 'ray'}, optional
        The method to use for creating the moments. See the spectral-cube
        docs for an explanation of the differences.
    """
    def __init__(self, cube, scale=None, moment_method='slice'):
        super(Moments, self).__init__()

        if not spectral_cube_flag:
            raise ImportError("Moments requires the spectral-cube "
                              " to be installed: https://github.com/"
                              "radio-astro-tools/spectral-cube")

        if isinstance(cube, SpectralCube):
            self.cube = cube
            self.save_name = None
        else:
            self.cube = SpectralCube.read(cube)
            # Default save name to the cube name without the suffix.
            self.save_name = ".".join(cube.split(".")[:-1])

        if moment_method not in ['slice', 'cube', 'ray']:
            raise TypeError("Moment method must be 'slice', 'cube', or 'ray'.")
        self.moment_how = moment_method

        self.scale = scale

        self.prop_headers = None
        self.prop_err_headers = None

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, value):

        if value is None:
            self._scale = value

        else:
            if not hasattr(value, 'unit'):
                raise TypeError("Given scale must be an `astropy.Quantity`"
                                " with units matching the units of the cube.")

            if not value.unit.is_equivalent(self.cube.unit):
                raise u.UnitsError("Given scale must have units equivalent"
                                   " to the units of the cube.")

            if value.value < 0:
                raise ValueError("Noise level is set to negative. The noise"
                                 " must be zero (noiseless) or positive.")

            self._scale = value

    def apply_mask(self, mask):
        '''
        Apply a mask to the cube.

        Parameters
        ----------
        mask : spectral-cube Mask or numpy.ndarray, optional
            The mask to be applied to the data. If None is given, RadioMask
            is used with its default settings.
        '''

        # if mask is None:
        #     rad_mask = RadioMask(self.cube)
        #     mask = rad_mask.to_mask()

        self.cube = self.cube.with_mask(mask)

    def make_moments(self, axis=0, units=True):
        '''
        Calculate the moments.

        Parameters
        ----------
        axis : int, optional
            The axis to calculate the moments along.
        units : bool, optional
            If enabled, the units of the arrays are kept.
        '''

        self._moment0 = self.cube.moment0(axis=axis, how=self.moment_how)
        self._moment1 = self.cube.moment1(axis=axis, how=self.moment_how)
        self._linewidth = \
            self.cube.linewidth_sigma(how=self.moment_how)

        if not units:
            self._moment0 = self._moment0.value
            self._moment1 = self._moment1.value
            self._moment2 = self._moment2.value

    def make_moment_errors(self, axis=0, scale=None):
        '''
        Calculate the errors in the moments.

        Parameters
        ----------
        axis : int, optional
            The axis to calculate the moments along.
        '''

        if not hasattr(self, "_moment0"):
            raise ValueError("Run Moments.make_moments first.")

        if self.scale is None and scale is None:
            warn("scale not set to the rms noise and will not be used in "
                 "error calculations.")
            scale = 0.0 * self.cube.unit

            self._moment0_err = np.zeros_like(self.moment0)
            self._moment1_err = np.zeros_like(self.moment1)
            self._linewidth_err = np.zeros_like(self.linewidth)

        elif self.scale is not None:
            scale = self.scale

            self._moment0_err = moment0_error(self.cube, scale,
                                              how=self.moment_how, axis=axis)
            self._moment1_err = moment1_error(self.cube, scale,
                                              how=self.moment_how,
                                              axis=axis, moment0=self.moment0,
                                              moment1=self.moment1)
            self._linewidth_err = \
                linewidth_sigma_err(self.cube, scale,
                                    how=self.moment_how, moment0=self.moment0,
                                    moment1=self.moment1,
                                    moment1_err=self.moment1_err)

    @property
    def moment0(self):
        return self._moment0

    @property
    def moment1(self):
        return self._moment1

    @property
    def linewidth(self):
        return self._linewidth

    @property
    def moment0_err(self):
        return self._moment0_err

    @property
    def moment1_err(self):
        return self._moment1_err

    @property
    def linewidth_err(self):
        return self._linewidth_err

    def all_moments(self):
        return [self._moment0, self._moment1, self.linewidth]

    def all_moment_errs(self):
        return [self._moment0_err, self._moment1_err, self.linewidth_err]

    def to_dict(self):
        '''
        Returns a dictionary with the cube and the property arrays.
        '''

        self.get_prop_hdrs()

        prop_dict = {}

        # Avoid reading in the whole cube when it is big, unless
        # you set cube.allow_huge_operations=True
        if self.cube._is_huge and not self.cube.allow_huge_operations:
            raise ValueError("This will load the whole cube into memory. Set "
                             "``cube.allow_huge_operations=True`` to"
                             " allow this")

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

        return prop_dict

    def get_prop_hdrs(self):
        '''
        Generate headers for the moments.
        '''

        bunits = [self.cube.unit, self.cube.spectral_axis.unit,
                  self.cube.spectral_axis.unit,
                  self.cube.unit * self.cube.spectral_axis.unit]

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

    def to_fits(self, save_name=None):
        '''
        Save the property arrays as fits files.

        Parameters
        ----------
        save_name : str, optional
            Prefix to use when saving the moment arrays.
            If None is given, 'default' is used.
        '''

        if self.prop_headers is None:
            self.get_prop_hdrs()

        if save_name is None:
            if self.save_name is None:
                Warning("No save_name has been specified, using 'default'")
                self.save_name = 'default'
        else:
            self.save_name = save_name

        labels = ["_moment0", "_centroid", "_linewidth"]

        for i, (arr, err, hdr, hdr_err) in \
          enumerate(zip(self.all_moments(), self.all_moment_errs(),
                        self.prop_headers, self.prop_err_headers)):

            # Can't write quantities.
            if _try_remove_unit(arr):
                arr = arr.value

            if _try_remove_unit(err):
                err = err.value

            hdu = fits.HDUList([fits.PrimaryHDU(arr, header=hdr),
                                fits.ImageHDU(err, header=hdr_err)])

            hdu.writeto(self.save_name + labels[i] + ".fits")

    @staticmethod
    def from_fits(fits_name, moments_prefix=None, moments_path=None,
                  mask_name=None, moment0=None, centroid=None, linewidth=None,
                  scale=None):
        '''
        Load pre-made moment arrays given a cube name. Saved moments must
        match the naming of the cube for the automatic loading to work
        (e.g. a cube called test.fits will have a moment 0 array with the name
        test_moment0.fits). Otherwise, specify a path to one of the keyword
        arguments.

        Parameters
        ----------
        fits_name : str
            Filename of the cube or a SpectralCube object. If a filename is
            given, it is also used as the prefix to the saved moment files.
        moments_prefix : str, optional
            If a SpectralCube object is given in ``fits_name``, the prefix
            for the saved files can be provided here.
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
        scale : `~astropy.units.Quantity`, optional
            The noise level in the cube. Overrides estimation using
            `signal_id <https://github.com/radio-astro-tools/signal-id>`_
        '''

        if not spectral_cube_flag:
            raise ImportError("Moments requires the spectral-cube "
                              " to be installed: https://github.com/"
                              "radio-astro-tools/spectral-cube")

        if moments_path is None:
            moments_path = ""

        if not isinstance(fits_name, SpectralCube):
            root_name = os.path.basename(fits_name[:-5])
        else:
            root_name = moments_prefix

        self = Moments(fits_name, scale=scale)

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
                                                 root_name + "_moment0.fits"))
                self._moment0 = moment0[0].data
                self._moment0_err = moment0[1].data
            except IOError as e:
                self._moment0 = None
                self._moment0_err = None
                print(e)
                print("Moment 0 fits file not found.")

        if centroid is not None:
            moment1 = fits.open(centroid)
            self._moment1 = moment1[0].data
            self._moment1_err = moment1[1].data
        else:
            try:
                moment1 = fits.open(os.path.join(moments_path,
                                                 root_name + "_centroid.fits"))
                self._moment1 = moment1[0].data
                self._moment1_err = moment1[1].data
            except IOError as e:
                self._moment1 = None
                self._moment1_err = None
                print(e)
                print("Centroid fits file not found.")

        if linewidth is not None:
            linewidth = fits.open(linewidth)

            self._linewidth = linewidth[0].data
            self._linewidth_err = linewidth[1].data
        else:
            try:
                linewidth = \
                    fits.open(os.path.join(moments_path,
                                           root_name + "_linewidth.fits"))

                self._linewidth = linewidth[0].data
                self._linewidth_err = linewidth[1].data
            except IOError as e:
                self._linewidth = None
                self._linewidth_err = None
                print(e)
                print("Linewidth fits file not found.")

        return self


def moment_masking(cube, kernel_size, clip=5, dilations=1):
    '''
    '''

    if not signal_id_flag:
        raise ImportError("signal-id is not installed."
                          " This function is not available.")

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

# class Mask_and_Moments(object):
#     """
#     A unified approach to deriving the noise level in a cube, applying a
#     mask, and deriving moments along with their errors. All the heavy lifting
#     is done with
#     `spectral_-ube <http://spectral-cube.readthedocs.org/en/latest/>`_.
#     Parameters
#     ----------
#     cube : SpectralCube or str
#         Either a SpectralCube object, or the filename of a cube readable
#         by spectral-cube.
#     noise_type : {'constant'}, optional
#         *NO CURRENT FUNCTION* Once implemented, it will set parameters
#         for deriving the noise level.
#     clip : float, optional
#         Sigma level to clip data at.
#     scale : `~astropy.units.Quantity`, optional
#         The noise level in the cube. Overrides estimation using
#         `signal_id <https://github.com/radio-astro-tools/signal-id>`_
#     moment_method : {'slice', 'cube', 'ray'}, optional
#         The method to use for creating the moments. See the spectral-cube
#         docs for an explanation of the differences.
#     """
#     def __init__(self, cube, noise_type='constant', clip=3, scale=None,
#                  moment_method='slice'):
#         super(Mask_and_Moments, self).__init__()

#         if not spectral_cube_flag:
#             raise ImportError("Mask_and_Moments requires the spectral-cube "
#                               " to be installed: https://github.com/"
#                               "radio-astro-tools/spectral-cube")

#         if isinstance(cube, SpectralCube):
#             self.cube = cube
#             self.save_name = None
#         else:
#             self.cube = SpectralCube.read(cube)
#             # Default save name to the cube name without the suffix.
#             self.save_name = ".".join(cube.split(".")[:-1])

#         self.noise_type = noise_type
#         self.clip = clip

#         if moment_method not in ['slice', 'cube', 'ray']:
#             raise TypeError("Moment method must be 'slice', 'cube', or 'ray'.")
#         self.moment_how = moment_method

#         if scale is None:
#             if not signal_id_flag:
#                 raise ImportError("signal-id is not installed and error"
#                                   " estimation is not available. You must "
#                                   "provide the noise scale.")

#             self.scale = Noise(self.cube).scale * self.cube.unit
#         else:
#             if not isinstance(scale, u.Quantity):
#                 raise TypeError("scale must be a Quantity with the same units"
#                                 " as the given cube.")
#             if not scale.unit == self.cube.unit:
#                 raise u.UnitsError("scale must have the same units"
#                                    " as the given cube.")
#             self.scale = scale

#         self.prop_headers = None
#         self.prop_err_headers = None

#     def find_noise(self, return_obj=False):
#         '''
#         Returns noise estimate, or the whole Noise object.
#         Parameters
#         ----------
#         return_obj : bool, optional
#             If True, returns the Noise object. Otherwise returns the estimated
#             noise level.
#         '''

#         if not signal_id_flag:
#             raise ImportError("signal-id is not installed."
#                               " This function is not available.")

#         noise = Noise(self.cube)

#         self.scale = noise.scale

#         if return_obj:
#             return noise

#         return noise.scale

#     def make_mask(self, mask):
#         '''
#         Apply a mask to the cube.
#         Parameters
#         ----------
#         mask : spectral-cube Mask or numpy.ndarray, optional
#             The mask to be applied to the data. If None is given, RadioMask
#             is used with its default settings.
#         '''

#         # if mask is None:
#         #     rad_mask = RadioMask(self.cube)
#         #     mask = rad_mask.to_mask()

#         self.cube = self.cube.with_mask(mask)

#     def make_moments(self, axis=0, units=True):
#         '''
#         Calculate the moments.
#         Parameters
#         ----------
#         axis : int, optional
#             The axis to calculate the moments along.
#         units : bool, optional
#             If enabled, the units of the arrays are kept.
#         '''

#         self._moment0 = self.cube.moment0(axis=axis, how=self.moment_how)
#         self._moment1 = self.cube.moment1(axis=axis, how=self.moment_how)
#         self._linewidth = \
#             self.cube.linewidth_sigma(how=self.moment_how)

#         if not units:
#             self._moment0 = self._moment0.value
#             self._moment1 = self._moment1.value
#             self._moment2 = self._moment2.value

#     def make_moment_errors(self, axis=0):
#         '''
#         Calculate the errors in the moments.
#         Parameters
#         ----------
#         axis : int, optional
#             The axis to calculate the moments along.
#         '''

#         self._moment0_err = moment0_error(self.cube, self.scale,
#                                           how=self.moment_how, axis=axis)
#         self._moment1_err = moment1_error(self.cube, self.scale,
#                                           how=self.moment_how,
#                                           axis=axis, moment0=self.moment0,
#                                           moment1=self.moment1)
#         self._linewidth_err = \
#             linewidth_sigma_err(self.cube, self.scale,
#                                 how=self.moment_how, moment0=self.moment0,
#                                 moment1=self.moment1,
#                                 moment1_err=self.moment1_err)

#     @property
#     def moment0(self):
#         return self._moment0

#     @property
#     def moment1(self):
#         return self._moment1

#     @property
#     def linewidth(self):
#         return self._linewidth

#     @property
#     def moment0_err(self):
#         return self._moment0_err

#     @property
#     def moment1_err(self):
#         return self._moment1_err

#     @property
#     def linewidth_err(self):
#         return self._linewidth_err

#     def all_moments(self):
#         return [self._moment0, self._moment1, self.linewidth]

#     def all_moment_errs(self):
#         return [self._moment0_err, self._moment1_err, self.linewidth_err]

#     def to_dict(self):
#         '''
#         Returns a dictionary form containing the cube and the property arrays.
#         This is the expected form for the wrapper scripts and methods in
#         TurbuStat.
#         '''

#         self.get_prop_hdrs()

#         prop_dict = {}

#         if _try_remove_unit(self.cube.filled_data[:]):
#             prop_dict['cube'] = [self.cube.filled_data[:].value,
#                                  self.cube.header]
#         else:
#             prop_dict['cube'] = [self.cube.filled_data[:], self.cube.header]

#         if _try_remove_unit(self.moment0):
#             prop_dict['moment0'] = [self.moment0.value, self.prop_headers[0]]
#         else:
#             prop_dict['moment0'] = [self.moment0, self.prop_headers[0]]

#         if _try_remove_unit(self.moment0_err):
#             prop_dict['moment0_error'] = [self.moment0_err.value,
#                                           self.prop_err_headers[0]]
#         else:
#             prop_dict['moment0_error'] = [self.moment0_err,
#                                           self.prop_err_headers[0]]

#         if _try_remove_unit(self.moment1):
#             prop_dict['centroid'] = [self.moment1.value, self.prop_headers[1]]
#         else:
#             prop_dict['centroid'] = [self.moment1, self.prop_headers[1]]

#         if _try_remove_unit(self.moment1_err):
#             prop_dict['centroid_error'] = [self.moment1_err.value,
#                                            self.prop_err_headers[1]]
#         else:
#             prop_dict['centroid_error'] = [self.moment1_err,
#                                            self.prop_err_headers[1]]

#         if _try_remove_unit(self.linewidth):
#             prop_dict['linewidth'] = [self.linewidth.value,
#                                       self.prop_headers[2]]
#         else:
#             prop_dict['linewidth'] = [self.linewidth,
#                                       self.prop_headers[2]]

#         if _try_remove_unit(self.linewidth_err):
#             prop_dict['linewidth_error'] = [self.linewidth_err.value,
#                                             self.prop_err_headers[2]]
#         else:
#             prop_dict['linewidth_error'] = [self.linewidth_err,
#                                             self.prop_err_headers[2]]

#         return prop_dict

#     def get_prop_hdrs(self):
#         '''
#         Generate headers for the moments.
#         '''

#         bunits = [self.cube.unit, self.cube.spectral_axis.unit,
#                   self.cube.spectral_axis.unit,
#                   self.cube.unit * self.cube.spectral_axis.unit]

#         comments = ["Image of the Zeroth Moment",
#                     "Image of the First Moment",
#                     "Image of the Second Moment",
#                     "Image of the Integrated Intensity"]

#         self.prop_headers = []
#         self.prop_err_headers = []

#         for i in range(len(bunits)):

#             wcs = self.cube.wcs.copy()
#             new_wcs = drop_axis(wcs, -1)

#             hdr = new_wcs.to_header()
#             hdr_err = new_wcs.to_header()
#             hdr["BUNIT"] = bunits[i].to_string()
#             hdr_err["BUNIT"] = bunits[i].to_string()
#             hdr["COMMENT"] = comments[i]
#             hdr_err["COMMENT"] = comments[i] + " Error."

#             self.prop_headers.append(hdr)
#             self.prop_err_headers.append(hdr_err)

#     def to_fits(self, save_name=None):
#         '''
#         Save the property arrays as fits files.
#         Parameters
#         ----------
#         save_name : str, optional
#             Prefix to use when saving the moment arrays.
#             If None is given, 'default' is used.
#         '''

#         if self.prop_headers is None:
#             self.get_prop_hdrs()

#         if save_name is None:
#             if self.save_name is None:
#                 Warning("No save_name has been specified, using 'default'")
#                 self.save_name = 'default'
#         else:
#             self.save_name = save_name

#         labels = ["_moment0", "_centroid", "_linewidth"]

#         for i, (arr, err, hdr, hdr_err) in \
#           enumerate(zip(self.all_moments(), self.all_moment_errs(),
#                         self.prop_headers, self.prop_err_headers)):

#             # Can't write quantities.
#             if _try_remove_unit(arr):
#                 arr = arr.value

#             if _try_remove_unit(err):
#                 err = err.value

#             hdu = fits.HDUList([fits.PrimaryHDU(arr, header=hdr),
#                                 fits.ImageHDU(err, header=hdr_err)])

#             hdu.writeto(self.save_name + labels[i] + ".fits")

#     @staticmethod
#     def from_fits(fits_name, moments_prefix=None, moments_path=None,
#                   mask_name=None, moment0=None, centroid=None, linewidth=None,
#                   scale=None):
#         '''
#         Load pre-made moment arrays given a cube name. Saved moments must
#         match the naming of the cube for the automatic loading to work
#         (e.g. a cube called test.fits will have a moment 0 array solved
#         test_moment0.fits). Otherwise, specify a path to one of the keyword
#         arguments.
#         Parameters
#         ----------
#         fits_name : str
#             Filename of the cube or a SpectralCube object. If a filename is
#             given, it is also used as the prefix to the saved moment files.
#         moments_prefix : str, optional
#             If a SpectralCube object is given in ``fits_name``, the prefix
#             for the saved files can be provided here.
#         moments_path : str, optional
#             Path to where the moments are saved.
#         mask_name : str, optional
#             Filename of a saved mask to be applied to the data.
#         moment0 : str, optional
#             Filename of the moment0 array. Use if naming scheme is not valid
#             for automatic loading.
#         centroid : str, optional
#             Filename of the centroid array. Use if naming scheme is not valid
#             for automatic loading.
#         linewidth : str, optional
#             Filename of the linewidth array. Use if naming scheme is not valid
#             for automatic loading.
#         scale : `~astropy.units.Quantity`, optional
#             The noise level in the cube. Overrides estimation using
#             `signal_id <https://github.com/radio-astro-tools/signal-id>`_
#         '''

#         if not spectral_cube_flag:
#             raise ImportError("Mask_and_Moments requires the spectral-cube "
#                               " to be installed: https://github.com/"
#                               "radio-astro-tools/spectral-cube")

#         if moments_path is None:
#             moments_path = ""

#         if not isinstance(fits_name, SpectralCube):
#             root_name = os.path.basename(fits_name[:-5])
#         else:
#             root_name = moments_prefix

#         self = Mask_and_Moments(fits_name, scale=scale)

#         if mask_name is not None:
#             mask = fits.getdata(mask_name)
#             self.with_mask(mask=mask)

#         # Moment 0
#         if moment0 is not None:
#             moment0 = fits.open(moment0)
#             self._moment0 = moment0[0].data
#             self._moment0_err = moment0[1].data
#         else:
#             try:
#                 moment0 = fits.open(os.path.join(moments_path,
#                                                  root_name + "_moment0.fits"))
#                 self._moment0 = moment0[0].data
#                 self._moment0_err = moment0[1].data
#             except IOError as e:
#                 self._moment0 = None
#                 self._moment0_err = None
#                 print(e)
#                 print("Moment 0 fits file not found.")

#         if centroid is not None:
#             moment1 = fits.open(centroid)
#             self._moment1 = moment1[0].data
#             self._moment1_err = moment1[1].data
#         else:
#             try:
#                 moment1 = fits.open(os.path.join(moments_path,
#                                                  root_name + "_centroid.fits"))
#                 self._moment1 = moment1[0].data
#                 self._moment1_err = moment1[1].data
#             except IOError as e:
#                 self._moment1 = None
#                 self._moment1_err = None
#                 print(e)
#                 print("Centroid fits file not found.")

#         if linewidth is not None:
#             linewidth = fits.open(linewidth)

#             self._linewidth = linewidth[0].data
#             self._linewidth_err = linewidth[1].data
#         else:
#             try:
#                 linewidth = \
#                     fits.open(os.path.join(moments_path,
#                                            root_name + "_linewidth.fits"))

#                 self._linewidth = linewidth[0].data
#                 self._linewidth_err = linewidth[1].data
#             except IOError as e:
#                 self._linewidth = None
#                 self._linewidth_err = None
#                 print(e)
#                 print("Linewidth fits file not found.")

#         return self

#     def _get_int_intensity(self, axis=0):
#         '''
#         Get an integrated intensity image of the cube.
#         Parameters
#         ----------
#         axis : int, optional
#             Axis to perform operations along.
#         '''

#         shape = self.cube.shape
#         view = [slice(None)] * 3

#         if self.moment_how is 'cube':
#             channel_max = \
#                 np.nanmax(self.cube.filled_data[:].reshape(-1, shape[1] * shape[2]),
#                           axis=1).value
#         else:
#             channel_max = np.empty((shape[axis]))
#             for i in range(shape[axis]):
#                 view[axis] = i
#                 plane = self.cube[view]

#                 channel_max[i] = np.nanmax(plane).value
#             channel_max = u.Quantity(channel_max, unit=self.cube.unit)

#         good_channels = np.where(channel_max > self.clip * self.scale)[0]

#         if not np.any(good_channels):
#             raise ValueError("Cannot find any channels with signal.")

#         # Get the longest sequence
#         good_channels = longestSequence(good_channels)

#         self.channel_range = self.cube.spectral_axis[good_channels][[0, -1]]

#         slab = self.cube.spectral_slab(*self.channel_range)

#         return slab.moment0(axis=axis, how=self.moment_how)

#     def _get_int_intensity_err(self, axis=0, how='auto'):
#         '''
#         Parameters
#         ----------
#         axis : int, optional
#             Axis to perform operations along.
#         '''
#         slab = self.cube.spectral_slab(*self.channel_range)

#         return moment0_error(slab, self.scale, axis=axis, how=self.moment_how)

