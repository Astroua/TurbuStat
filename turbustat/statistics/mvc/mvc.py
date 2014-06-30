
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

    Parameters
    ----------

    centroid : numpy.ndarray
        Normalized first moment array.

    moment0 : numpy.ndarray
        Moment 0 array.

    linewidth : numpy.ndarray
        Normalized second moment array

    header : FITS header
        Header of any of the arrays. Used only to get the
        spatial scale.

    """

    def __init__(self, centroid, moment0, linewidth, header):
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

        self.ps2D = None
        self.ps1D = None
        self.freq = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
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
        Computes the radially averaged power spectrum.
        This uses Adam Ginsburg's code (see https://github.com/keflavich/agpy).
        See the above url for parameter explanations.
        '''

        self.freq, self.ps1D = pspec(self.ps2D, return_index=return_index,
                                     wavenumber=wavenumber,
                                     return_stddev=return_stddev,
                                     azbins=azbins, binsize=binsize,
                                     view=view, **kwargs)

        return self

    def run(self, phys_units=False, verbose=False):
        '''
        Full computation of MVC.

        Parameters
        ----------
        phys_units : bool, optional
            Sets frequency scale to physical units.
        verbose: bool, optional
            Enables plotting.
        '''

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

    Parameters
    ----------

    data1 : dict
        dictionary containing necessary property arrays
    data2 : dict
        dictionary containing necessary property arrays
    fiducial_model : MVC
        Computed MVC object. use to avoid recomputing.
    """

    def __init__(self, data1, data2, fiducial_model=None):
        # super(mvc_distance, self).__init__()

        self.shape1 = data1["centroid"][0].shape
        self.shape2 = data2["centroid"][0].shape

        if fiducial_model is not None:
            self.mvc1 = fiducial_model
        else:
            self.mvc1 = MVC(data1["centroid"][0] * data1["centroid_error"][0] ** 2.,
                            data1["moment0"][0] * data1["moment0_error"][0] ** 2.,
                            data1["linewidth"][0] * data1["linewidth_error"][0] ** 2.,
                            data1["centroid"][1])
            self.mvc1.run()

        self.mvc2 = MVC(data2["centroid"][0] * data2["centroid_error"][0] ** 2.,
                        data2["moment0"][0] * data2["moment0_error"][0] ** 2.,
                        data2["linewidth"][0] * data2["linewidth_error"][0] ** 2.,
                        data2["centroid"][1])
        self.mvc2.run()

        self.results = None
        self.distance = None

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 MVC transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
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
