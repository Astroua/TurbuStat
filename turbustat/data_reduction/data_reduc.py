# Licensed under an MIT open source license - see LICENSE


'''

Data Reduction Routines for PPV data cubes

'''

import numpy as np
from scipy import ndimage as nd
from operator import itemgetter
from itertools import groupby
from astropy.io import fits
import copy
from scipy.optimize import curve_fit
from astropy.convolution import convolve


class property_arrays(object):

    '''

    Create property arrays from a data cube

    Creates centroid (moment 1), integrated intensity,
    velocity dispersion (moment 2), total intensity (moment 0).

    '''

    def __init__(self, cube, clip_level=3, rms_noise=None,
                 kernel_size=None, save_name=None):
        super(property_arrays, self).__init__()
        self.cube = cube[0]  # cube.data
        self.header = cube[1]  # cube.header
        self.shape = (self.cube.shape[1], self.cube.shape[2])
        self.save_name = save_name

        self.clean_cube = np.ones(self.cube.shape)
        self.noise_array = None
        self.nan_mask = np.invert(np.isnan(self.cube), dtype=bool)
        self.weight_cube = np.ones(self.cube.shape)

        vel_channels = np.arange(1, self.cube.shape[0] + 1)
        self.weight_cube = np.tile(vel_channels[:, np.newaxis, np.newaxis],
                                   (1, self.shape[0], self.shape[1]))

        self.sigma = None

        self.property_dict = {"cube": [self.cube, self.header]}

        if rms_noise is not None:
            if isinstance(rms_noise, float):
                self.noise_type_flag = 1
                self.sigma = rms_noise
                self.noise_array = np.ones(self.shape) * self.sigma
                self.noise_mask = np.ones(self.shape)
                self.clean_cube[self.cube < (clip_level * self.sigma)] = 0.0
                self.clean_cube *= np.ma.masked_invalid(self.cube)

            else:
                self.noise_type_flag = 2
                self.clean_cube, self.noise_array, self.sigma = \
                    given_noise_cube(self.cube, rms_noise, clip_level)
                self.noise_mask = self.noise_array < (clip_level * self.sigma)
        else:
            if not kernel_size:
                raise ValueError(
                    "Kernel Size must be given for moment masking.")
            self.noise_type_flag = 0
            self.clean_cube, self.mask_cube, self.sigma = \
                moment_masking(self.cube, clip_level, kernel_size)
            # self.noise_mask = self.noise_array < (clip_level * self.sigma)
            self.nan_mask += self.mask_cube

    def moment0(self):

        moment0_array = np.sum(self.clean_cube * self.nan_mask, axis=0)
        moment0_array[np.where(moment0_array == 0.0)] = np.NaN
        # moment0_array *= self.noise_mask

        error_array = self.sigma * \
            np.sqrt(np.sum(self.nan_mask * (self.clean_cube > 0), axis=0))
        # error_array *= self.noise_mask

        self.property_dict["moment0"] = [moment0_array]
        self.property_dict["moment0_error"] = [error_array]

        return self

    def centroid(self):

        centroid_array = \
            np.sum(self.clean_cube * self.nan_mask * self.weight_cube, axis=0)\
            / self.property_dict["moment0"][0]
        # centroid_array *= self.noise_mask

        first_err_term = \
            self.sigma ** 2. * \
            np.sqrt(np.sum(self.weight_cube[np.nonzero(self.clean_cube * self.nan_mask)], axis=0)) \
            / self.property_dict["moment0"][0] ** 2.
        second_err_term = self.property_dict["moment0_error"][0] ** 2. / \
                          self.property_dict["moment0"][0] ** 2.
        error_array = np.sqrt(first_err_term + second_err_term)
        # error_array *= self.noise_mask

        self.property_dict["centroid"] = [centroid_array]
        self.property_dict["centroid_error"] = [error_array]

        return self

    def integrated_intensity(self):

        masked_clean = self.clean_cube * self.nan_mask
        int_intensity_array = np.ones(self.shape)
        error_array = np.ones(self.shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                z = np.where(masked_clean[:, i, j] > 0)
                continuous_sections = []
                for _, g in groupby(enumerate(z[0]), lambda (i, x): i - x):
                    continuous_sections.append(map(itemgetter(1), g))
                try:
                    integrating_section = max(continuous_sections, key=len)
                    int_intensity_array[i, j] = np.sum(
                        [masked_clean[k, i, j] for k in integrating_section])
                    error_array[i, j] = (
                        np.sqrt(len(integrating_section))) ** -1. * self.sigma

                except ValueError:
                    int_intensity_array[i, j] = np.NaN
                    error_array[i, j] = np.NaN

        self.property_dict["integrated_intensity"] = [int_intensity_array]
        self.property_dict["integrated_intensity_error"] = [error_array]
        return self

    def linewidth(self):

        masked_clean = self.clean_cube * self.nan_mask
        weight_clean = self.weight_cube * self.nan_mask

        linewidth_array = np.empty(self.shape)
        error_array = np.empty(self.shape)

        for i in range(self.shape[0]):
            for j in range(self.shape[1]):

                linewidth_array[i, j] = \
                    np.sqrt(np.sum((weight_clean[:, i, j] -
                                    self.property_dict["centroid"][0][i, j]) ** 2. *
                                   masked_clean[:, i, j])
                            / self.property_dict["moment0"][0][i, j])

                if np.isclose(linewidth_array[i, j], 0.0, rtol=1e-5):
                    error_array[i, j] = 0.0

                else:
                    first_err_term = \
                        (2 * np.sum((weight_clean[:, i, j] -
                        self.property_dict["centroid"][0][i, j]) *
                        masked_clean[:, i, j]) *
                        self.property_dict["centroid_error"][0][i, j] ** 2. +
                        self.sigma ** 2. *
                        np.sum((weight_clean[:, i, j] -
                                self.property_dict["centroid"][0][i, j]) ** 2.)) /\
                        np.sum((weight_clean[:, i, j] -
                               self.property_dict["centroid"][0][i, j]) ** 2. *
                               masked_clean[:, i, j]) ** 2.
                    second_err_term = \
                        self.sigma ** 2. * \
                        np.sum(self.nan_mask[:, i, j]) ** 2. / \
                        self.property_dict["moment0"][0][i, j] ** 2.

                    error_array[i, j] = np.sqrt(first_err_term +
                                                second_err_term)

        self.property_dict["linewidth"] = [linewidth_array]
        self.property_dict["linewidth_error"] = [error_array]

    def pixel_to_physical_units(self):

        # Lazy check to make sure we have units of km/s
        if np.abs(self.header["CDELT3"]) > 1:
            vel_pix_division = np.abs(self.header["CDELT3"]) / 1000.
            reference_velocity = self.header["CRVAL3"] / 1000.
        else:
            vel_pix_division = np.abs(self.header["CDELT3"])
            reference_velocity = self.header["CRVAL3"]

        # Centroid error needs to be recalculated when changing to physical
        # units
        physical_weights = \
            (np.sum(self.weight_cube, axis=0) * vel_pix_division) + \
            reference_velocity - (vel_pix_division * self.header["CRPIX3"])

        first_err_term = self.sigma ** 2. * \
            np.sqrt(np.sum(physical_weights *
                    (self.clean_cube > 0) * self.nan_mask, axis=0)) / \
            self.property_dict["moment0"][0] ** 2.
        second_err_term = self.property_dict["moment0_error"][0] ** 2. /\
            self.property_dict["moment0"][0] ** 2.
        cent_error_array = np.sqrt(first_err_term + second_err_term)
        # cent_error_array *= self.noise_mask

        self.property_dict["centroid"][0] = \
            (self.property_dict["centroid"][0] * vel_pix_division) + \
            reference_velocity - (vel_pix_division * self.header["CRPIX3"])

        self.property_dict["centroid_error"][0] = cent_error_array

        self.property_dict["integrated_intensity"][0] = \
            self.property_dict["integrated_intensity"][0] * vel_pix_division

        self.property_dict["integrated_intensity_error"][0] = \
            self.property_dict["integrated_intensity_error"][0] * vel_pix_division

        self.property_dict["linewidth"][0] = \
            self.property_dict["linewidth"][0] * vel_pix_division

        self.property_dict["linewidth_error"][0] = \
            self.property_dict["linewidth_error"][0] * vel_pix_division

        return self

    def make_headers(self):

        new_hdr = copy.deepcopy(self.header)
        del new_hdr["NAXIS3"], new_hdr["CRVAL3"], new_hdr[
            "CRPIX3"], new_hdr['CDELT3'], new_hdr['CTYPE3']
        new_hdr.update("NAXIS", 2)
        new_err_hdr = copy.deepcopy(new_hdr)

        if self.save_name is None:
            self.save_name = self.header["OBJECT"]

        moment0_specs = {
            'comment': "= Image of the Zeroth Moment",
            'BUNIT': 'K', 'name': 'moment0'}
        centroid_specs = {
            'comment': "= Image of the First Moment",
            'BUNIT': 'km/s', 'name': 'centroid'}
        linewidth_specs = {
            'comment': "= Image of the Second Moment",
            'BUNIT': 'km/s', 'name': 'linewidth'}
        int_int_specs = {'comment': "= Image of the Integrated Intensity",
                         'BUNIT': 'K km/s', 'name': 'integrated_intensity'}

        moment0_error_specs = {
            'comment': "= Image of the Zeroth Moment Error",
            'BUNIT': 'K', 'name': 'moment0'}
        centroid_error_specs = {
            'comment': "= Image of the First Moment Error",
            'BUNIT': 'km/s', 'name': 'centroid'}
        linewidth_error_specs = {
            'comment': "= Image of the Second Moment Error",
            'BUNIT': 'km/s', 'name': 'linewidth'}
        int_int_error_specs = {
            'comment': "= Image of the Integrated Intensity Error",
            'BUNIT': 'K km/s', 'name': 'integrated_intensity'}

        for prop in ['moment0', 'centroid', 'integrated_intensity',
                     'linewidth']:
            if prop == 'moment0':
                specs = moment0_specs
                specs_error = moment0_error_specs
            elif prop == 'centroid':
                specs = centroid_specs
                specs_error = centroid_error_specs
            elif prop == 'integrated_intensity':
                specs = int_int_specs
                specs_error = int_int_error_specs
            elif prop == 'linewidth':
                specs = linewidth_specs
                specs_error = linewidth_error_specs

            # Update header for array and the error array
            new_hdr.update("BUNIT", value=specs['BUNIT'], comment='')
            new_hdr.add_comment(specs["comment"])
            new_err_hdr.update("BUNIT", value=specs['BUNIT'], comment='')
            new_err_hdr.add_comment(specs_error["comment"])

            self.property_dict[prop].append(new_hdr)
            self.property_dict[prop+"_error"].append(new_err_hdr)

            # Reset the comments
            del new_hdr["COMMENT"]
            del new_err_hdr["COMMENT"]

        return self

    @property
    def dataset(self):
        return self.property_dict

    def return_all(self, save=True, physical_units=True, save_path=None):

        self.moment0()
        self.centroid()
        self.linewidth()
        self.integrated_intensity()

        if physical_units:
            self.pixel_to_physical_units()

        # Append headers in the dict.
        self.make_headers()

        if save:
            for prop in ['moment0', 'centroid', 'integrated_intensity',
                         'linewidth']:

                if save_path is not None:
                    filename = "".join(
                        [save_path, self.save_name, ".", prop, ".fits"])
                    filename_err = "".join(
                        [save_path, self.save_name, ".", prop,
                         "_error.fits"])
                else:
                    filename = "".join(
                        [self.save_name, ".", prop, ".fits"])
                    filename_err = "".join(
                        [self.save_name, ".", prop, "_error.fits"])

                fits.writeto(filename, self.property_dict[prop][0],
                             self.property_dict[prop][1])
                fits.writeto(filename_err, self.property_dict[prop+"_error"][0],
                             self.property_dict[prop+"_error"][1])

        return self


def given_noise_cube(data_cube, noise_cube, clip_level):
    if data_cube.shape != noise_cube.shape:
        raise ValueError("Error array has different dimensions.")

    assert clip_level is int

    noise_cube[np.where(noise_cube == 0)] = np.NaN

    clipped_cube = (data_cube / noise_cube) >= clip_level
    inv_cube = np.invert(clipped_cube, dtype=bool)

    noise_array = np.max(inv_cube * data_cube, axis=0)
    sigma = np.mean(noise_array)

    return clipped_cube * data_cube, noise_array, sigma


def __sigma__(data_cube, clip_level):

    flat_cube = np.ravel(data_cube[~np.isnan(data_cube)])

    hist, bins = np.histogram(flat_cube, bins=int(len(flat_cube) / 100.))
    centres = (bins[:-1] + bins[1:]) / 2

    def gaussian(x, *p):
        # Peak Height is p[0],Sigma is p[1],Mu is p[2]
        return p[0] * np.exp(-1 * np.power(x - p[2], 2) / (2 * np.power(p[1], 2)))

    p0 = (np.max(hist), 1.0, centres[np.argmax(hist)])

    opts, cov = curve_fit(
        gaussian, centres, hist, p0, maxfev=(100 * len(hist)) + 1)

    if opts[1] == p0[1]:
        print "Fitting Failed. Sigma is %s" % (opts[1])

    return opts[1]


def moment_masking(data_cube, clip_level, kernel_size):
    sigma_orig = __sigma__(data_cube, clip_level)

    if np.isnan(data_cube).any():
        print "Using astropy to convolve over nans"
        kernel = gauss_kern(kernel_size, ysize=kernel_size, zsize=kernel_size)
        smooth_cube = convolve(data_cube, kernel, normalize_kernel=True)
    else:
        smooth_cube = nd.gaussian_filter(data_cube, kernel_size, mode="mirror")

    sigma_smooth = __sigma__(smooth_cube, clip_level)

    mask_cube = smooth_cube > (clip_level * sigma_smooth)

    dilate_struct = nd.generate_binary_structure(3, 3)
    mask_cube = nd.binary_dilation(mask_cube, structure=dilate_struct)

    noise_cube = np.invert(mask_cube, dtype=bool) * data_cube

    # noise_array = np.max(noise_cube, axis=0)

    return (mask_cube * data_cube), mask_cube, sigma_orig


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

if __name__ == '__main__':
    pass
  # import sys
  # fib(int(sys.argv[1]))

  # from astropy.io.fits import getdata
  # cube, header = getdata("filename",header=True)
  # shape = cube.shape
  # cube[:,shape[0],:] = cube[:,0,:]
  # cube[:,:,shape[1]] = cube[:,:,0]
  # data = property_arrays((cube,header), rms_noise=0.001, save_name="filename")
  # data.return_all()
