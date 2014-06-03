
'''

Implementation of Modified Velocity Centroids (Lazarian & Esquivel, 03)

'''

import numpy as np
import scipy.ndimage as nd
from itertools import izip
from ..psds import pspec
import statsmodels.formula.api as sm
from pandas import Series, DataFrame

try:
    from scipy.fftpack import fft2, fftshift
except ImportError:
    from numpy.fft import fft2, fftshift


class MVC(object):

    """
    Implementation of Modified Velocity Centroids (Lazarian & Esquivel, 03)

    INPUTS
    ------

    centroid   - array
                 Normalized first moment array

    moment0  - array
                     Intensity array

    linewidth   - array
                  Normalized second moment array

    distances   - list
                  List of distances vector lengths
                  If None, MVC is computed using the distances in the distance
                  array.

    FUNCTIONS
    ---------

    correlation_function - computes the MVC function

    run_mvc - calls correlation_function for each distance

    compute_pspec - computes the power spectrum

    compute_radial_pspec - computes the radially-averaged power spectrum

    OUTPUTS
    -------

    correlation_array - array
                           Resultant 2D array from mvc

    correlation_spectrum - array
                           1D array of average value at each distance

    ps1D - array
           1D array of the radially-averaged power spectrum

    ps2D - array
           2D array of the power spectrum

    freq - spatial frequencies used to compute ps1D

    """

    def __init__(self, centroid, moment0, linewidth, header, distances=None):
        # super(MVC, self).__init__()
        self.centroid = centroid
        self.moment0 = moment0
        self.linewidth = linewidth

        # Get rid of nans.
        self.centroid[np.isnan(self.centroid)] = np.nanmin(self.centroid)
        self.moment0[np.isnan(self.moment0)] = np.nanmin(self.moment0)
        self.linewidth[np.isnan(self.linewidth)] = np.nanmin(self.linewidth)
        self.degperpix = np.abs(header["CDELT2"])

        assert self.centroid.shape == self.moment0.shape
        assert self.centroid.shape == self.linewidth.shape
        self.shape = self.centroid.shape

        self.center_pixel = []
        for i_shape in self.shape:
            if i_shape % 2. != 0:
                self.center_pixel.append(int((i_shape - 1) / 2.))
            else:
                self.center_pixel.append(int(i_shape / 2.))

        self.center_pixel = tuple(self.center_pixel)
        centering_array = np.ones(self.shape)
        centering_array[self.center_pixel] = 0

        self.distance_array = nd.distance_transform_edt(centering_array)

        if not distances:
            self.distances = np.unique(
                self.distance_array[np.nonzero(self.distance_array)])
        else:
            assert isinstance(distances, list)
            self.distances = distances

        self.correlation_array = np.ones(self.shape)
        self.correlation_spectrum = np.zeros((1, len(self.distances)))
        self.ps2D = None
        self.ps1D = None
        self.freq = None

    def correlation_function(self, points):
        '''

        Function which performs the correlation

        '''

        center_centroid = (
            self.centroid[self.center_pixel] * self.moment0[self.center_pixel])
        center_intensity = self.moment0[self.center_pixel]

        for i, j in izip(points[0], points[1]):
            first_term = (
                center_centroid - self.centroid[i, j] *
                self.moment0[i, j]) ** 2.
            second_term = self.linewidth[
                i, j] * (center_intensity - self.moment0[i, j]) ** 2.

            self.correlation_array[i, j] = first_term - second_term

        return self

    def compute_pspec(self):
        '''
        Compute the power spectrum of MVC
        '''

        mvc_fft = fft2(self.centroid.astype("f8")) - \
            self.linewidth * fft2(self.moment0.astype("f8"))
        mvc_fft = fftshift(mvc_fft)

        self.ps2D = np.abs(mvc_fft) ** 2.

        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False,
                             return_stddev=False, azbins=1, binsize=1.0,
                             view=False, **kwargs):
        '''

        Computes the radially averaged power spectrum
        Based on Adam Ginsburg's code

        '''

        self.freq, self.ps1D = pspec(self.ps2D, return_index=return_index,
                                     wavenumber=wavenumber,
                                     return_stddev=return_stddev,
                                     azbins=azbins, binsize=binsize,
                                     view=view, **kwargs)

        return self

    def run(self, distances=None, phys_units=False, verbose=False):
        '''
        Full computation of MVC
        '''

        if not distances:
            pass
        else:
            assert isinstance(distances, list)
            self.distances = distances

        for n, distance in enumerate(self.distances):
            points = np.where(self.distance_array == distance)
            self.correlation_function(points)
            self.correlation_spectrum[0, n] = np.sum(
                self.correlation_array[points]) / len(points[0])

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=True)

        if phys_units:
            self.freqs *= self.degperpix ** -1

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.imshow(
                np.log10(self.ps2D), origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.loglog(self.freq, self.ps1D, "bD-")
            p.show()

        return self


class MVC_distance(object):

    """

    Distance metric for MVC and wrapper for whole analysis

    INPUTS
    ------

    data1 - dictionary
            dictionary containing necessary property arrays

    data2 - dictionary
            dictionary containing necessary property arrays

    distances - list
                passed to MVC class


    FUNCTIONS
    ---------

    mvc_distance - computes the distance metric for 2 datasets

    OUTPUTS
    -------

    distance - float
               value of distance metric

    results  - statsmodels class
               results of the linear fit

    """

    def __init__(self, data1, data2, distances=None, fiducial_model=None):
        # super(mvc_distance, self).__init__()

        self.shape1 = data1["centroid"][0].shape
        self.shape2 = data2["centroid"][0].shape

        if fiducial_model is not None:
            self.mvc1 = fiducial_model
        else:
            self.mvc1 = MVC(data1["centroid"][0] * data1["centroid_error"][0] ** 2.,
                            data1["moment0"][0] * data1["moment0_error"][0] ** 2.,
                            data1["linewidth"][0] * data1["linewidth_error"][0] ** 2.,
                            data1["centroid"][1], distances=distances)
            self.mvc1.run()

        self.mvc2 = MVC(data2["centroid"][0] * data2["centroid_error"][0] ** 2.,
                        data2["moment0"][0] * data2["moment0_error"][0] ** 2.,
                        data2["linewidth"][0] * data2["linewidth_error"][0] ** 2.,
                        data2["centroid"][1], distances=distances)
        self.mvc2.run()

        self.results = None
        self.distance = None

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 MVC transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A statistical comparison is used on the powerlaw indexes.

        '''

        # Clipping from 8 pixels to half the box size
        # Noise effects dominate outside this region
        clip_mask1 = np.zeros((self.mvc1.freq.shape))
        for i, x in enumerate(self.mvc1.freq):
            if x > 8.0 and x < self.shape1[0] / 2.:
                clip_mask1[i] = 1
        clip_freq1 = self.mvc1.freq[np.where(clip_mask1 == 1)]
        clip_ps1D1 = self.mvc1.ps1D[np.where(clip_mask1 == 1)]

        clip_mask2 = np.zeros((self.mvc2.freq.shape))
        for i, x in enumerate(self.mvc2.freq):
            if x > 8.0 and x < self.shape2[0] / 2.:
                clip_mask2[i] = 1
        clip_freq2 = self.mvc2.freq[np.where(clip_mask2 == 1)]
        clip_ps1D2 = self.mvc2.ps1D[np.where(clip_mask2 == 1)]

        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(
            x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(
            formula="log_ps1D ~ dummy + scales + regressor", data=df)

        self.results = model.fit()

        self.distance = np.abs(self.results.tvalues["regressor"])

        if verbose:

            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_freq1), np.log10(clip_ps1D1), "bD",
                   np.log10(clip_freq2), np.log10(clip_ps1D2), "gD")
            p.plot(df["scales"][:len(clip_freq1)],
                   self.results.fittedvalues[:len(clip_freq1)], "b",
                   df["scales"][-len(clip_freq2):],
                   self.results.fittedvalues[-len(clip_freq2):], "g")
            p.grid(True)
            p.xlabel("log K")
            p.ylabel("MVC Power (K)")
            p.show()

        return self
