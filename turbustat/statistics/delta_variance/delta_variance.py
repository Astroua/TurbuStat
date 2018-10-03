# Licensed under an MIT open source license - see LICENSE
from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from copy import copy
import statsmodels.api as sm
from astropy.extern.six import string_types
from warnings import warn
from astropy.utils.console import ProgressBar

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..stats_utils import common_scale, padwithzeros
from ..fitting_utils import check_fit_limits, residual_bootstrap
from .kernels import core_kernel, annulus_kernel
from ..stats_warnings import TurbuStatMetricWarning
from ..lm_seg import Lm_Seg
from ..convolve_wrapper import convolution_wrapper


class DeltaVariance(BaseStatisticMixIn):

    """

    The delta-variance technique as described in Ossenkopf et al. (2008).

    Parameters
    ----------

    img : %(dtypes)s
        The image calculate the delta-variance of.
    header : FITS header, optional
        Image header. Required when img is a `~numpy.ndarray`.
    weights : %(dtypes)s
        Weights to be used.
    diam_ratio : float, optional
        The ratio between the kernel sizes.
    lags : numpy.ndarray or list, optional
        The pixel scales to compute the delta-variance at.
    nlags : int, optional
        Number of lags to use.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.

    Example
    -------
    >>> from turbustat.statistics import DeltaVariance
    >>> from astropy.io import fits
    >>> moment0 = fits.open("2D.fits") # doctest: +SKIP
    >>> delvar = DeltaVariance(moment0) # doctest: +SKIP
    >>> delvar.run(verbose=True) # doctest: +SKIP
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, diam_ratio=1.5,
                 lags=None, nlags=25, distance=None):
        super(DeltaVariance, self).__init__()

        # Set the data and perform checks
        self.input_data_header(img, header)

        self.diam_ratio = diam_ratio

        if weights is None:
            # self.weights = np.ones(self.data.shape)
            self.weights = np.isfinite(self.data).astype(float)
        else:
            self.weights = input_data(weights, no_header=True)

        if distance is not None:
            self.distance = distance

        self.nanflag = False
        if np.isnan(self.data).any() or np.isnan(self.weights).any():
            self.nanflag = True

        if lags is None:
            min_size = 3.0
            self.lags = \
                np.logspace(np.log10(min_size),
                            np.log10(min(self.data.shape) / 2.), nlags) * u.pix
        else:
            # Check if the given lags are a Quantity
            # Default to pixel scales if it isn't
            if not hasattr(lags, "value"):
                self.lags = lags * u.pix
            else:
                self.lags = self._to_pixel(lags)

        self.convolved_arrays = []
        self.convolved_weights = []

    @property
    def lags(self):
        '''
        Lag values.
        '''
        return self._lags

    @lags.setter
    def lags(self, values):

        if not isinstance(values, u.Quantity):
            raise TypeError("lags must be given as an astropy.units.Quantity.")

        pix_lags = self._to_pixel(values)

        if np.any(pix_lags.value < 1):
            raise ValueError("At least one of the lags is smaller than one "
                             "pixel. Remove these lags from the array.")

        # Catch floating point issues in comparing to half the image shape
        half_comp = (np.floor(pix_lags.value) - min(self.data.shape) / 2.)

        if np.any(half_comp > 1e-10):
            raise ValueError("At least one of the lags is larger than half of"
                             " the image size. Remove these lags from the "
                             "array.")

        self._lags = values

    @property
    def weights(self):
        '''
        Array of weights.
        '''
        return self._weights

    @weights.setter
    def weights(self, arr):

        if arr.shape != self.data.shape:
            raise ValueError("Given weight array does not match the shape of "
                             "the given image.")

        self._weights = arr

    def do_convolutions(self, allow_huge=False, boundary='wrap',
                        min_weight_frac=0.01, nan_interpolate=True,
                        use_pyfftw=False, threads=1,
                        pyfftw_kwargs={},
                        show_progress=True):
        '''
        Perform the convolutions at all lags.

        Parameters
        ----------
        allow_huge : bool, optional
            Passed to `~astropy.convolve.convolve_fft`. Allows operations on
            images larger than 1 Gb.
        boundary : {"wrap", "fill"}, optional
            Use "wrap" for periodic boundaries, and "fill" for non-periodic.
        min_weight_frac : float, optional
            Set the fraction of the peak of the weight array to mask below.
            Default is 0.01. This will remove most edge artifacts, but is
            not guaranteed to! Increase this value if artifacts are
            encountered (this typically results in large spikes in the
            delta-variance curve).
        nan_interpolate : bool, optional
            Enable to interpolate over NaNs in the convolution. Default is
            True.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            See `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        '''

        if show_progress:
            bar = ProgressBar(len(self.lags))

        for i, lag in enumerate(self.lags.value):
            core = core_kernel(lag, self.data.shape[0], self.data.shape[1])
            annulus = annulus_kernel(lag, self.diam_ratio, self.data.shape[0],
                                     self.data.shape[1])

            if boundary == "wrap":
                # Don't pad for periodic boundaries
                pad_weights = self.weights
                pad_img = self.data * self.weights
            elif boundary == "fill":
                # Extend to avoid boundary effects from non-periodicity
                pad_weights = np.pad(self.weights, int(lag), padwithzeros)
                pad_img = np.pad(self.data, int(lag), padwithzeros) * \
                    pad_weights
            else:
                raise ValueError("boundary must be 'wrap' or 'fill'. "
                                 "Given {}".format(boundary))

            img_core = \
                convolution_wrapper(pad_img, core, boundary=boundary,
                                    fill_value=np.NaN,
                                    allow_huge=allow_huge,
                                    nan_interpolate=nan_interpolate,
                                    use_pyfftw=use_pyfftw,
                                    threads=threads,
                                    pyfftw_kwargs=pyfftw_kwargs)
            img_annulus = \
                convolution_wrapper(pad_img, annulus,
                                    boundary=boundary, fill_value=np.NaN,
                                    allow_huge=allow_huge,
                                    nan_interpolate=nan_interpolate,
                                    use_pyfftw=use_pyfftw,
                                    threads=threads,
                                    pyfftw_kwargs=pyfftw_kwargs)
            weights_core = \
                convolution_wrapper(pad_weights, core,
                                    boundary=boundary, fill_value=np.NaN,
                                    allow_huge=allow_huge,
                                    nan_interpolate=nan_interpolate,
                                    use_pyfftw=use_pyfftw,
                                    threads=threads,
                                    pyfftw_kwargs=pyfftw_kwargs)
            weights_annulus = \
                convolution_wrapper(pad_weights, annulus,
                                    boundary=boundary, fill_value=np.NaN,
                                    allow_huge=allow_huge,
                                    nan_interpolate=nan_interpolate,
                                    use_pyfftw=use_pyfftw,
                                    threads=threads,
                                    pyfftw_kwargs=pyfftw_kwargs)

            cutoff_val = min_weight_frac * self.weights.max()
            weights_core[np.where(weights_core <= cutoff_val)] = np.NaN
            weights_annulus[np.where(weights_annulus <= cutoff_val)] = np.NaN

            self.convolved_arrays.append((img_core / weights_core) -
                                         (img_annulus / weights_annulus))
            self.convolved_weights.append(weights_core * weights_annulus)

            if show_progress:
                bar.update(i + 1)

    def compute_deltavar(self):
        '''
        Computes the delta-variance values and errors.
        '''

        self._delta_var = np.empty((len(self.lags)))
        self._delta_var_error = np.empty((len(self.lags)))

        for i, (conv_arr,
                conv_weight,
                lag) in enumerate(zip(self.convolved_arrays,
                                      self.convolved_weights,
                                      self.lags.value)):

            val, err = _delvar(conv_arr, conv_weight, lag)

            if (val <= 0) or (err <= 0) or np.isnan(val) or np.isnan(err):
                self._delta_var[i] = np.NaN
                self._delta_var_error[i] = np.NaN
            else:
                self._delta_var[i] = val
                self._delta_var_error[i] = err

    @property
    def delta_var(self):
        '''
        Delta Variance values.
        '''
        return self._delta_var

    @property
    def delta_var_error(self):
        '''
        1-sigma errors on the Delta variance values.
        '''
        return self._delta_var_error

    def fit_plaw(self, xlow=None, xhigh=None, brk=None, verbose=False,
                 bootstrap=False, bootstrap_kwargs={},
                 **fit_kwargs):
        '''
        Fit a power-law to the Delta-variance spectrum.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        bootstrap : bool, optional
            Bootstrap using the model residuals to estimate the standard
            errors.
        bootstrap_kwargs : dict, optional
            Pass keyword arguments to `~turbustat.statistics.fitting_utils.residual_bootstrap`.
        verbose : bool, optional
            Show fit summary when enabled.
        '''

        x = np.log10(self.lags.value)
        y = np.log10(self.delta_var)

        if xlow is not None:
            xlow = self._to_pixel(xlow)

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.delta_var, dtype=bool)
            xlow = self.lags.min() * 0.99

        if xhigh is not None:
            xhigh = self._to_pixel(xhigh)

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.delta_var, dtype=bool)
            xhigh = self.lags.max() * 1.01

        self._fit_range = [xlow, xhigh]

        within_limits = np.logical_and(lower_limit, upper_limit)

        y = y[within_limits]
        x = x[within_limits]

        weights = self.delta_var_error[within_limits] ** -2

        min_fits_pts = 3

        if brk is not None:
            # Try fitting a segmented model

            pix_brk = self._to_pixel(brk)

            if pix_brk < xlow or pix_brk > xhigh:
                raise ValueError("brk must be within xlow and xhigh.")

            model = Lm_Seg(x, y, np.log10(pix_brk.value), weights=weights)

            fit_kwargs['verbose'] = verbose
            fit_kwargs['cov_type'] = 'HC3'

            model.fit_model(**fit_kwargs)

            self.fit = model.fit

            if model.params.size == 5:

                # Check to make sure this leaves enough to fit to.
                if sum(x < model.brk) < min_fits_pts:
                    warn("Not enough points to fit to." +
                         " Ignoring break.")

                    self._brk = None
                else:
                    good_pts = x.copy() < model.brk
                    x = x[good_pts]
                    y = y[good_pts]

                    self._brk = 10**model.brk / u.pix

                    self._slope = model.slopes

                    if bootstrap:
                        stderrs = residual_bootstrap(model.fit,
                                                     **bootstrap_kwargs)

                        self._slope_err = stderrs[1:-1]
                        self._brk_err = np.log(10) * self.brk.value * \
                            stderrs[-1] / u.pix

                    else:
                        self._slope_err = model.slope_errs
                        self._brk_err = np.log(10) * self.brk.value * \
                            model.brk_err / u.pix

                    self.fit = model.fit

            else:
                self._brk = None
                # Break fit failed, revert to normal model
                warn("Model with break failed, reverting to model\
                      without break.")
        else:
            self._brk = None

        # Revert to model without break if none is given, or if the segmented
        # model failed.
        if self.brk is None:

            x = sm.add_constant(x)

            # model = sm.OLS(y, x, missing='drop')
            model = sm.WLS(y, x, missing='drop', weights=weights)

            self.fit = model.fit(cov_type='HC3')

            self._slope = self.fit.params[1]

            if bootstrap:
                stderrs = residual_bootstrap(self.fit,
                                             **bootstrap_kwargs)
                self._slope_err = stderrs[1]

            else:
                self._slope_err = self.fit.bse[1]

        self._bootstrap_flag = bootstrap

        if verbose:
            print(self.fit.summary())

            if self._bootstrap_flag:
                print("Bootstrapping used to find stderrs! "
                      "Errors may not equal those shown above.")

        self._model = model

    @property
    def brk(self):
        '''
        Fitted break point.
        '''
        return self._brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point in the segmented linear model.
        '''
        return self._brk_err

    @property
    def slope(self):
        '''
        Fitted slope.
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        Standard error on the fitted slope.
        '''
        return self._slope_err

    @property
    def fit_range(self):
        '''
        Range of lags used in the fit.
        '''
        return self._fit_range

    def fitted_model(self, xvals):
        '''
        Computes the fitted power-law in log-log space using the
        given x values.

        Parameters
        ----------
        xvals : `~numpy.ndarray`
            Values of log(lags) to compute the model at (base 10 log).

        Returns
        -------
        model_values : `~numpy.ndarray`
            Values of the model at the given values.
        '''

        if isinstance(self._model, Lm_Seg):
            return self._model.model(xvals)
        else:
            return self.fit.params[0] + self.fit.params[1] * xvals

    def plot_fit(self, save_name=None, xunit=u.pix, color='r', fit_color=None,
                 show_residual=True):
        '''
        Plot the delta-variance curve and the fit.

        Parameters
        ----------
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        color : {str, RGB tuple}, optional
            Color to show the delta-variance curve in.
        fit_color : {str, RGB tuple}, optional
            Color of the fitted line. Defaults to `color` when no input is
            given.
        show_residual : bool, optional
            Plot the fit residuals.
        '''

        if fit_color is None:
            fit_color = color

        import matplotlib.pyplot as plt

        fig = plt.figure()

        if show_residual:
            ax = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
            ax_r = plt.subplot2grid((4, 1), (3, 0), colspan=1,
                                    rowspan=1,
                                    sharex=ax)
        else:
            ax = plt.subplot(111)

        ax.set_xscale("log")
        ax.set_yscale("log")

        lags = self._spatial_unit_conversion(self.lags, xunit).value

        # Check for NaNs
        fin_vals = np.logical_or(np.isfinite(self.delta_var),
                                 np.isfinite(self.delta_var_error))
        ax.errorbar(lags[fin_vals], self.delta_var[fin_vals],
                    yerr=self.delta_var_error[fin_vals],
                    fmt="D-", color=color)

        xvals = np.linspace(self._fit_range[0].value,
                            self._fit_range[1].value,
                            100) * self.lags.unit
        xvals_conv = self._spatial_unit_conversion(xvals, xunit).value

        ax.plot(xvals_conv, 10**self.fitted_model(np.log10(xvals.value)),
                '--', color=fit_color, linewidth=2, label='Fit')

        xlow = \
            self._spatial_unit_conversion(self._fit_range[0], xunit).value
        xhigh = \
            self._spatial_unit_conversion(self._fit_range[1], xunit).value

        ax.axvline(xlow, color=color, alpha=0.5, linestyle='-.')
        ax.axvline(xhigh, color=color, alpha=0.5, linestyle='-.')

        ax.legend(loc='best')
        ax.grid(True)

        if show_residual:
            resids = self.delta_var - 10**self.fitted_model(np.log10(lags))
            ax_r.errorbar(lags[fin_vals], resids[fin_vals],
                          yerr=self.delta_var_error[fin_vals],
                          fmt="D-", color=color)

            ax_r.set_ylabel("Residuals")

            ax_r.set_xlabel("Lag ({})".format(xunit))

            ax_r.axhline(0., color=fit_color, linestyle='--')

            ax_r.axvline(xlow, color=color, alpha=0.5, linestyle='-.')
            ax_r.axvline(xhigh, color=color, alpha=0.5, linestyle='-.')
            ax_r.grid()

            ax.get_xaxis().set_ticks([])

        else:
            ax.set_xlabel("Lag ({})".format(xunit))

        ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, verbose=False, xunit=u.pix,
            nan_interpolate=True, allow_huge=False, boundary='wrap',
            use_pyfftw=False, threads=1, pyfftw_kwargs={},
            xlow=None, xhigh=None,
            brk=None, fit_kwargs={},
            save_name=None):
        '''
        Compute the delta-variance.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        verbose : bool, optional
            Plot delta-variance transform.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        allow_huge : bool, optional
            See `~DeltaVariance.do_convolutions`.
        nan_interpolate : bool, optional
            Enable to interpolate over NaNs in the convolution. Default is
            True.
        boundary : {"wrap", "fill"}, optional
            Use "wrap" for periodic boundaries, and "cut" for non-periodic.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            See `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial break point guess. Enables fitting a segmented
            linear model.
        fit_kwargs : dict, optional
            Passed to `~turbustat.statistics.lm_seg.Lm_Seg.fit_model` when
            using a broken linear fit.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.do_convolutions(allow_huge=allow_huge, boundary=boundary,
                             nan_interpolate=nan_interpolate,
                             use_pyfftw=use_pyfftw,
                             threads=threads,
                             pyfftw_kwargs=pyfftw_kwargs,
                             show_progress=show_progress)
        self.compute_deltavar()
        self.fit_plaw(xlow=xlow, xhigh=xhigh, brk=brk, verbose=verbose,
                      **fit_kwargs)

        if verbose:
            self.plot_fit(save_name=save_name, xunit=xunit)

        return self


class DeltaVariance_Distance(object):

    """
    Compares 2 datasets using delta-variance. The distance between them is
    given by the Euclidean distance between the curves weighted by the
    bootstrapped errors.

    Parameters
    ----------

    dataset1 : %(dtypes)s
        Contains the data and header for one dataset.
    dataset2 : %(dtypes)s
        See above.
    weights1 : %(dtypes)s
        Weights for dataset1.
    weights2 : %(dtypes)s
        See above.
    diam_ratio : float, optional
        The ratio between the kernel sizes.
    lags : numpy.ndarray or list, optional
        The pixel scales to compute the delta-variance at.
    fiducial_model : DeltaVariance
        A computed DeltaVariance model. Used to avoid recomputing.
    ang_units : bool, optional
        Convert frequencies to angular units using the given header.
    xlow : float or np.ndarray, optional
        The lower lag fitting limit. An array with 2 elements can be passed to
        give separate lower limits for the datasets.
    xhigh : float or np.ndarray, optional
        The upper lag fitting limit. See `xlow` above.
    boundary : str, np.ndarray or list, optional
        Set how boundaries should be handled. If a string is not passed, a
        two element list/array with separate boundary conditions is expected.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, dataset1, dataset2, weights1=None, weights2=None,
                 diam_ratio=1.5, lags=None, fiducial_model=None,
                 xlow=None, xhigh=None, boundary='wrap'):
        super(DeltaVariance_Distance, self).__init__()

        dataset1 = copy(input_data(dataset1, no_header=False))
        dataset2 = copy(input_data(dataset2, no_header=False))

        # Enforce standardization on both datasets. Values for the
        # delta-variance then represents a relative fraction of structure on
        # different scales.
        # dataset1[0] = standardize(dataset1[0])
        # dataset2[0] = standardize(dataset2[0])

        # Create a default set of lags, in pixels
        if lags is None:
            min_size = 3.0
            nlags = 25
            shape1 = dataset1[0].shape
            shape2 = dataset2[0].shape
            if min(shape1) > min(shape2):
                lags = \
                    np.logspace(np.log10(min_size),
                                np.log10(min(shape2) / 2.),
                                nlags) * u.pix
            else:
                lags = \
                    np.logspace(np.log10(min_size),
                                np.log10(min(shape1) / 2.),
                                nlags) * u.pix

        # Check the limits are given in an understandable form.
        # The returned xlow and xhigh are arrays.
        xlow, xhigh = check_fit_limits(xlow, xhigh)

        # Allow separate boundary conditions to be passed
        if isinstance(boundary, string_types):
            boundary = [boundary] * 2
        else:
            if not len(boundary) == 2:
                raise ValueError("boundary must be a two-element list/array"
                                 " when a string is not passed.")

        # Now adjust the lags such they have a common scaling when the datasets
        # are not on a common grid.
        scale = common_scale(WCS(dataset1[1]), WCS(dataset2[1]))

        if scale == 1.0:
            lags1 = lags
            lags2 = lags
        elif scale > 1.0:
            lags1 = scale * lags
            lags2 = lags
        else:
            lags1 = lags
            lags2 = lags / float(scale)

        if fiducial_model is not None:
            self.delvar1 = fiducial_model
        else:
            self.delvar1 = DeltaVariance(dataset1,
                                         weights=weights1,
                                         diam_ratio=diam_ratio, lags=lags1)
            self.delvar1.run(xlow=xlow[0], xhigh=xhigh[0],
                             boundary=boundary[0])

        self.delvar2 = DeltaVariance(dataset2,
                                     weights=weights2,
                                     diam_ratio=diam_ratio, lags=lags2)
        self.delvar2.run(xlow=xlow[1], xhigh=xhigh[1], boundary=boundary[1])

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        xunit=u.pix, save_name=None):
        '''
        Applies the Euclidean distance to the delta-variance curves.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for dataset1
        label2 : str, optional
            Object or region name for dataset2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Check for NaNs and negatives
        nans1 = np.logical_or(np.isnan(self.delvar1.delta_var),
                              self.delvar1.delta_var <= 0.0)
        nans2 = np.logical_or(np.isnan(self.delvar2.delta_var),
                              self.delvar2.delta_var <= 0.0)

        all_nans = np.logical_or(nans1, nans2)

        # Cut the curves at the specified xlow and xhigh points
        fit_range1 = self.delvar1.fit_range
        fit_range2 = self.delvar1.fit_range

        # The curve metric only makes sense if the same range is used for both
        if fit_range1[0] == fit_range2[0] and fit_range1[1] == fit_range2[1]:

            # Lags are always in pixels. As are the limits
            cuts1 = np.logical_and(self.delvar1.lags >= fit_range1[0],
                                   self.delvar1.lags <= fit_range1[1])
            cuts2 = np.logical_and(self.delvar2.lags >= fit_range2[0],
                                   self.delvar2.lags <= fit_range2[1])

            valids1 = np.logical_and(cuts1, ~all_nans)
            valids2 = np.logical_and(cuts2, ~all_nans)

            deltavar1_sum = np.sum(self.delvar1.delta_var[valids1])
            deltavar1 = \
                np.log10(self.delvar1.delta_var[valids1] / deltavar1_sum)

            deltavar2_sum = np.sum(self.delvar2.delta_var[valids2])
            deltavar2 = \
                np.log10(self.delvar2.delta_var[valids2] / deltavar2_sum)

            # Distance between two normalized curves
            self.curve_distance = np.linalg.norm(deltavar1 - deltavar2)
        else:

            warn("The curve distance is only defined when the fit "
                 "range and lags for both datasets are equal. "
                 "Setting curve_distance to NaN.", TurbuStatMetricWarning)

            self.curve_distance = np.NaN

        # Distance between the fitted slopes (combined t-statistic)
        self.slope_distance = \
            np.abs(self.delvar1.slope - self.delvar2.slope) / \
            np.sqrt(self.delvar1.slope_err**2 + self.delvar2.slope_err**2)

        if verbose:
            import matplotlib.pyplot as p

            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")

            lags1 = self.delvar1._spatial_unit_conversion(self.delvar1.lags,
                                                          xunit).value
            lags2 = self.delvar2._spatial_unit_conversion(self.delvar2.lags,
                                                          xunit).value
            lags1 = lags1[~all_nans]
            lags2 = lags2[~all_nans]

            # Normalize the errors for when plotting. NOT log-scaled.
            deltavar1_err = \
                self.delvar1.delta_var_error[~all_nans] / deltavar1_sum
            deltavar2_err = \
                self.delvar2.delta_var_error[~all_nans] / deltavar2_sum

            p.errorbar(lags1, self.delvar1.delta_var[~all_nans] / deltavar1_sum,
                       yerr=deltavar1_err, fmt="bD-",
                       label=label1)
            p.errorbar(lags2, self.delvar2.delta_var[~all_nans] / deltavar2_sum,
                       yerr=deltavar2_err, fmt="go-",
                       label=label2)

            xvals1 = np.linspace(self.delvar1.fit_range[0].value,
                                 self.delvar1.fit_range[1].value,
                                 100) * self.delvar1.lags.unit
            xvals_conv1 = \
                self.delvar1._spatial_unit_conversion(xvals1, xunit).value

            xvals2 = np.linspace(self.delvar2.fit_range[0].value,
                                 self.delvar2.fit_range[1].value,
                                 100) * self.delvar2.lags.unit
            xvals_conv2 = \
                self.delvar2._spatial_unit_conversion(xvals2, xunit).value

            p.plot(xvals_conv1,
                   10**self.delvar1.fitted_model(np.log10(xvals1.value)) /
                   deltavar1_sum,
                   'b--', linewidth=8, alpha=0.75)
            p.plot(xvals_conv2,
                   10**self.delvar2.fitted_model(np.log10(xvals2.value)) /
                   deltavar2_sum,
                   'g--', linewidth=8, alpha=0.75)

            # Vertical lines to indicate fit region
            xlow1 = \
                self.delvar1._spatial_unit_conversion(fit_range1[0],
                                                      xunit).value
            xhigh1 = \
                self.delvar1._spatial_unit_conversion(fit_range1[1],
                                                      xunit).value
            fit_range2 = self.delvar2.fit_range
            xlow2 = \
                self.delvar2._spatial_unit_conversion(fit_range2[0],
                                                      xunit).value
            xhigh2 = \
                self.delvar2._spatial_unit_conversion(fit_range2[1],
                                                      xunit).value
            p.axvline(xlow1, color="b", alpha=0.5, linestyle='-.')
            p.axvline(xhigh1, color="b", alpha=0.5, linestyle='-.')

            p.axvline(xlow2, color="g", alpha=0.5, linestyle='-.')
            p.axvline(xhigh2, color="g", alpha=0.5, linestyle='-.')

            ax.legend(loc='best')
            ax.grid(True)
            ax.set_xlabel("Lag ({})".format(xunit))
            ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()

        return self


def _delvar(array, weight, lag):
    '''
    Computes the delta variance of the given array.
    '''
    arr_cent = array.copy() - np.nanmean(array, axis=None)

    val = np.nansum(arr_cent ** 2. * weight) /\
        np.nansum(weight)

    # The error needs to be normalized by the number of independent
    # pixels in the array.
    # Take width to be 1/2 FWHM. Note that lag is defined as 2*sigma.
    # So 2ln(2) sigma^2 = ln(2)/2 * lag^2
    kern_area = np.ceil(0.5 * np.pi * np.log(2) * lag**2).astype(int)
    nindep = np.sqrt(np.isfinite(arr_cent).sum() // kern_area)

    val_err = np.sqrt((np.nansum(arr_cent ** 4. * weight) /
                       np.nansum(weight)) - val**2) / nindep

    return val, val_err
