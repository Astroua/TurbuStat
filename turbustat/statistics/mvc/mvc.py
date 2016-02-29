# Licensed under an MIT open source license - see LICENSE


import numpy as np
import statsmodels.formula.api as sm
from pandas import Series, DataFrame
from numpy.fft import fft2, fftshift

from ..psds import pspec


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

        self._ps1D_stddev = None

    @property
    def ps2D(self):
        return self._ps2D

    @property
    def ps1D(self):
        return self._ps1D

    @property
    def ps1D_stddev(self):
        if not self._stddev_flag:
            Warning("scf_spectrum_stddev is only calculated when return_stddev"
                    " is enabled.")

        return self._ps1D_stddev

    @property
    def freqs(self):
        return self._freqs

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.

        The quantity calculated here is the same as Equation 3 in Lazarian &
        Esquivel (2003), but the inputted arrays are not in the same form as
        described. We can, however, adjust for the use of normalized Centroids
        and the linewidth.

        An unnormalized centroid can be constructed by multiplying the centroid
        array by the moment0. Velocity dispersion is the square of the linewidth
        subtracted by the square of the normalized centroid.
        '''

        term1 = fft2(self.centroid*self.moment0)

        term2 = np.power(self.linewidth, 2) + np.power(self.centroid, 2)

        mvc_fft = term1 - term2 * fft2(self.moment0)

        # Shift to the center
        mvc_fft = fftshift(mvc_fft)

        self._ps2D = np.abs(mvc_fft) ** 2.

        return self

    def compute_radial_pspec(self, return_stddev=True,
                             logspacing=True, **kwargs):
        '''
        Computes the radially averaged power spectrum.

        Parameters
        ----------
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        kwargs : passed to pspec
        '''

        if return_stddev:
            self._freqs, self._ps1D, self._ps1D_stddev = \
                pspec(self.ps2D, return_stddev=return_stddev,
                      logspacing=logspacing, **kwargs)
            self._stddev_flag = True
        else:
            self._freqs, self._ps1D = \
                pspec(self.ps2D, return_stddev=return_stddev,
                      logspacing=logspacing, **kwargs)
            self._stddev_flag = False

        return self

    def run(self, phys_units=False, verbose=False, logspacing=True,
            return_stddev=True):
        '''
        Full computation of MVC.

        Parameters
        ----------
        phys_units : bool, optional
            Sets frequency scale to physical units.
        verbose: bool, optional
            Enables plotting.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)

        if phys_units:
            self._freqs *= self.degperpix ** -1

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.imshow(
                np.log10(self.ps2D), origin="lower", interpolation="nearest")
            p.colorbar()
            ax = p.subplot(122)
            if self._stddev_flag:
                ax.errorbar(self.freqs, self.ps1D, yerr=self.ps1D_stddev,
                            fmt='D-', color='b', markersize=5, alpha=0.5)
                ax.set_xscale("log", nonposy='clip')
                ax.set_yscale("log", nonposy='clip')
            else:
                p.loglog(self.freqs, self.ps1D, "bD-", markersize=5,
                         alpha=0.5)

            if phys_units:
                ax.set_xlabel("Frequency (1/deg)")
            else:
                ax.set_xlabel("Frequency (pixels)")

            p.show()

        return self


class MVC_distance(object):

    """

    Distance metric for MVC.

    Parameters
    ----------
    data1 : dict
        A dictionary containing the centroid, moment 0, and linewidth arrays
        of a spectral cube. This output is created by Mask_and_Moments.to_dict.
        The minimum expected keys are 'centroid', 'moment0' and 'linewidth'. If
        weight_by_error is enabled, the dictionary should also contain the
        error arrays, where the keys are the three above with '_error'
        appended to the end.
    data2 : dict
        See data1.
    fiducial_model : MVC
        Computed MVC object. use to avoid recomputing.
    weight_by_error : bool, optional
        When enabled, the property arrays are weighted by the inverse
        squared of the error arrays.
    """

    def __init__(self, data1, data2, fiducial_model=None,
                 weight_by_error=False):
        # super(mvc_distance, self).__init__()

        self.shape1 = data1["centroid"][0].shape
        self.shape2 = data2["centroid"][0].shape

        # Create weighted or non-weighted versions
        if weight_by_error:
            centroid1 = data1["centroid"][0] * \
                data1["centroid_error"][0] ** -2.
            moment01 = data1["moment0"][0] * \
                data1["moment0_error"][0] ** -2.
            linewidth1 = data1["linewidth"][0] * \
                data1["linewidth_error"][0] ** -2.
            centroid2 = data2["centroid"][0] * \
                data2["centroid_error"][0] ** -2.
            moment02 = data2["moment0"][0] * \
                data2["moment0_error"][0] ** -2.
            linewidth2 = data2["linewidth"][0] * \
                data2["linewidth_error"][0] ** -2.
        else:
            centroid1 = data1["centroid"][0]
            moment01 = data1["moment0"][0]
            linewidth1 = data1["linewidth"][0]
            centroid2 = data2["centroid"][0]
            moment02 = data2["moment0"][0]
            linewidth2 = data2["linewidth"][0]

        if fiducial_model is not None:
            self.mvc1 = fiducial_model
        else:
            self.mvc1 = MVC(centroid1, moment01, linewidth1,
                            data1["centroid"][1])
            self.mvc1.run(phys_units=False)

        self.mvc2 = MVC(centroid2, moment02, linewidth2,
                        data2["centroid"][1])
        self.mvc2.run(phys_units=False)

        self.results = None
        self.distance = None

    def distance_metric(self, low_cut=None, high_cut=0.5, verbose=False):
        '''

        Implements the distance metric for 2 MVC transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        low_cut : int or float, optional
            Set the cut-off for low spatial frequencies. Visually, below ~2
            deviates from the power law (for the simulation set).
        high_cut : int or float, optional
            Set the cut-off for high spatial frequencies. Values beyond the
            size of the root grid are found to have no meaningful contribution
        verbose : bool, optional
            Enables plotting.
        '''

        if low_cut is None:
            # Default to a frequency of 1/2 the smallest axis in the images.
            low_cut = 2. / float(min(min(self.mvc1.ps2D.shape),
                                     min(self.mvc2.ps2D.shape)))

        keep_freqs1 = clip_func(self.mvc1.freqs, low_cut, high_cut)
        clip_freq1 = \
            self.mvc1.freqs[keep_freqs1]
        clip_ps1D1 = \
            self.mvc1.ps1D[keep_freqs1]
        clip_weights1 = \
            (0.434*self.mvc1.ps1D_stddev[keep_freqs1]/clip_ps1D1)**-2

        keep_freqs2 = clip_func(self.mvc2.freqs, low_cut, high_cut)
        clip_freq2 = \
            self.mvc2.freqs[keep_freqs2]
        clip_ps1D2 = \
            self.mvc2.ps1D[keep_freqs2]
        clip_weights2 = \
            (0.434*self.mvc2.ps1D_stddev[keep_freqs1]/clip_ps1D1)**-2

        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy

        weights = np.concatenate([clip_weights1, clip_weights2])

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(
            x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.wls(
            formula="log_ps1D ~ dummy + scales + regressor", data=df,
            weights=weights)

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


def clip_func(arr, low, high):
    return np.logical_and(arr > low, arr < high)
