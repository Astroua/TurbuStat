# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import astropy.units as u
import statsmodels.api as sm
from warnings import warn
from astropy.utils.console import ProgressBar

try:
    from pyfftw.interfaces.numpy_fft import fftn, ifftn
    PYFFTW_FLAG = True
except ImportError:
    PYFFTW_FLAG = False

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types
from ..fitting_utils import check_fit_limits, residual_bootstrap
from ..lm_seg import Lm_Seg


class Wavelet(BaseStatisticMixIn):
    '''
    Compute the wavelet transform of a 2D array.

    Parameters
    ----------
    array : %(dtypes)s
        2D data.
    header : FITS header, optional
        Header for the array.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int, optional
        Number of scales to compute the transform at.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data, header=None, scales=None, num=50,
                 distance=None):

        self.input_data_header(data, header)

        # NOTE: can't use nan_interpolating from astropy
        # until the normalization for sum to zeros kernels is fixed!!!
        isnan = np.isnan(self.data)
        if isnan.any():
            self.data = self.data.copy()
            self.data[isnan] = 0.

        if distance is not None:
            self.distance = distance

        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            a_max = min(self.data.shape) / 2.
            # Log spaces scales up to half of the smallest size of the array
            scales = np.logspace(np.log10(a_min), np.log10(a_max), num) * u.pix

        self.scales = scales

    @property
    def scales(self):
        '''
        Wavelet scales.
        '''
        return self._scales

    @scales.setter
    def scales(self, values):

        if not isinstance(values, u.Quantity):
            raise TypeError("scales must be given as a "
                            "astropy.units.Quantity.")

        # Now make sure that we can convert into pixels before setting.
        try:
            pix_scal = self._to_pixel(values)
        except Exception as e:
            raise e

        # The radius should be larger than a pixel
        if np.any(pix_scal.value < 1):
            raise ValueError("One of the chosen lags is smaller than one "
                             "pixel."
                             " Ensure that all lag values are larger than one "
                             "pixel.")

        # Catch floating point issues in comparing to half the image shape
        half_comp = (np.floor(pix_scal.value) - min(self.data.shape) / 2.)

        if np.any(half_comp > 1e-10):
            raise ValueError("At least one of the lags is larger than half of"
                             " the image size. Remove these lags from the "
                             "array.")

        self._scales = values

    def compute_transform(self, show_progress=True, scale_normalization=True,
                          keep_convolved_arrays=False, convolve_kwargs={},
                          use_pyfftw=False, threads=1, pyfftw_kwargs={}):
        '''
        Compute the wavelet transform at each scale.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        scale_normalization: bool, optional
            Compute the transform with the correct scale-invariant
            normalization.
        keep_convolved_arrays: bool, optional
            Keep the image convolved at all wavelet scales. For large images,
            this can require a large amount memory. Default is False.
        convolve_kwargs : dict, optional
            Passed to `~astropy.convolution.convolve_fft`.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            See `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        '''

        if use_pyfftw:
            if PYFFTW_FLAG:
                use_fftn = fftn
                use_ifftn = ifftn
            else:
                warn("pyfftw not installed. Using numpy.fft functions.")
                use_fftn = np.fft.fftn
                use_ifftn = np.fft.ifftn
        else:
            use_fftn = np.fft.fftn
            use_ifftn = np.fft.ifftn

        n0, m0 = self.data.shape
        A = len(self.scales)

        if keep_convolved_arrays:
            self._Wf = np.zeros((A, n0, m0), dtype=np.float)
        else:
            self._Wf = None

        self._values = np.empty_like(self.scales.value)
        self._stddev = np.empty_like(self.scales.value)

        factor = 2
        if not scale_normalization:
            factor = 4
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

        pix_scales = self._to_pixel(self.scales).value

        if show_progress:
            bar = ProgressBar(len(pix_scales))

        for i, an in enumerate(pix_scales):
            psi = MexicanHat2DKernel(an)

            conv_arr = \
                convolve_fft(self.data, psi, normalize_kernel=False,
                             fftn=use_fftn, ifftn=use_ifftn,
                             nan_treatment='fill',
                             preserve_nan=True,
                             **convolve_kwargs).real * \
                an**factor

            if keep_convolved_arrays:
                self._Wf[i] = conv_arr

            self._values[i] = (conv_arr[conv_arr > 0]).mean()

            # The standard deviation should take into account the number of
            # kernel elements at that scale.
            kern_area = np.ceil(0.5 * np.pi * np.log(2) * an**2).astype(int)
            nindep = np.sqrt(np.isfinite(conv_arr).sum() // kern_area)

            self._stddev[i] = (conv_arr[conv_arr > 0]).std() / nindep

            if show_progress:
                bar.update(i + 1)

    @property
    def Wf(self):
        '''
        The wavelet transforms of the image. Each plane is the transform at
        different wavelet sizes.
        '''
        if self._Wf is None:
            warn("`keep_convolved_arrays` was disabled in "
                 "`compute_transform`.")

        return self._Wf

    @property
    def values(self):
        '''
        The 1-dimensional wavelet transform.
        '''
        return self._values

    @property
    def stddev(self):
        '''
        Standard deviation of the 1-dimensional wavelet transform.
        '''
        return self._stddev

    def fit_transform(self, xlow=None, xhigh=None, brk=None, min_fits_pts=3,
                      weighted_fit=False, bootstrap=False,
                      bootstrap_kwargs={}, **fit_kwargs):
        '''
        Perform a fit to the transform in log-log space.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower scale value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper scale value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        min_fits_pts : int, optional
            Minimum number of points required above or below the fitted break
            for it to be considered a valid fit. Only used when a segmented
            line is fit, i.e. when a value for `brk` is given.
        weighted_fit: bool, optional
            Use the `~Wavelet.stddev` to perform a weighted fit.
        bootstrap : bool, optional
            Bootstrap using the model residuals to estimate the standard
            errors.
        bootstrap_kwargs : dict, optional
            Pass keyword arguments to `~turbustat.statistics.fitting_utils.residual_bootstrap`.
        fit_kwargs : Passed to `turbustat.statistics.Lm_Seg.fit_model`
        '''

        pix_scales = self._to_pixel(self.scales)
        x = np.log10(pix_scales.value)
        y = np.log10(self.values)

        if weighted_fit:
            y_err = 0.434 * self.stddev / self.values
            y_err[y_err == 0.] = np.NaN

            weights = y_err**-2
        else:
            weights = None

        if xlow is not None:
            xlow = self._to_pixel(xlow)

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.values, dtype=bool)
            xlow = pix_scales.min() * 0.99

        if xhigh is not None:
            xhigh = self._to_pixel(xhigh)

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.values, dtype=bool)
            xhigh = pix_scales.max() * 1.01

        self._fit_range = [xlow, xhigh]

        within_limits = np.logical_and(lower_limit, upper_limit)

        y = y[within_limits]
        x = x[within_limits]

        if weighted_fit:
            weights = weights[within_limits]

        if brk is not None:
            # Try fitting a segmented model

            pix_brk = self._to_pixel(brk)

            if pix_brk < xlow or pix_brk > xhigh:
                raise ValueError("brk must be within xlow and xhigh.")

            model = Lm_Seg(x, y, np.log10(pix_brk.value), weights=weights)

            fit_kwargs['cov_type'] = 'HC3'

            model.fit_model(**fit_kwargs)

            self.fit = model.fit

            if model.params.size == 5:

                # Check to make sure this leaves enough to fit to.
                if sum(x < model.brk) < min_fits_pts:
                    warnings.warn("Not enough points to fit to." +
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
                warnings.warn("Model with break failed, reverting to model\
                               without break.")
        else:
            self._brk = None

        # Revert to model without break if none is given, or if the segmented
        # model failed.
        if self.brk is None:

            x = sm.add_constant(x)

            if weighted_fit:
                model = sm.WLS(y, x, missing='drop', weights=weights)
            else:
                model = sm.OLS(y, x, missing='drop')

            self.fit = model.fit(cov_type='HC3')

            self._slope = self.fit.params[1]

            if bootstrap:
                stderrs = residual_bootstrap(self.fit,
                                             **bootstrap_kwargs)
                self._slope_err = stderrs[1]

            else:
                self._slope_err = self.fit.bse[1]

        self._model = model

        self._bootstrap_flag = bootstrap

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
    def brk(self):
        '''
        Break point in the segmented linear model.
        '''
        return self._brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point in the segmented linear model.
        '''
        return self._brk_err

    @property
    def fit_range(self):
        '''
        Range of scales used in the fit.
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

    def plot_transform(self, save_name=None, xunit=u.pix,
                       color='r', symbol='o', fit_color='k',
                       label=None, show_residual=True):
        '''
        Plot the transform and the fit.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        xunit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        color : {str, RGB tuple}, optional
            Color to plot the wavelet curve.
        symbol : str, optional
            Symbol to use for the data.
        fit_color : {str, RGB tuple}, optional
            Color of the 1D fit.
        label : str, optional
            Label to later be used in a legend.
        show_residual : bool, optional
            Plot the fit residuals.
        '''

        import matplotlib.pyplot as plt

        if fit_color is None:
            fit_color = color

        # Check for already existing subplots
        fig = plt.gcf()
        axes = plt.gcf().get_axes()
        if len(axes) == 0:
            if show_residual:
                ax = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
                ax_r = plt.subplot2grid((4, 1), (3, 0), colspan=1,
                                        rowspan=1,
                                        sharex=ax)
            else:
                ax = plt.subplot(111)
        elif len(axes) == 1:
            ax = axes[0]
        else:
            ax = axes[0]
            ax_r = axes[1]

        ax.set_xscale("log")
        ax.set_yscale("log")

        pix_scales = self._to_pixel(self.scales)
        scales = self._spatial_unit_conversion(pix_scales, xunit).value

        # Check for NaNs
        fin_vals = np.logical_or(np.isfinite(self.values),
                                 np.isfinite(self.stddev))
        ax.errorbar(scales[fin_vals], self.values[fin_vals],
                    yerr=self.stddev[fin_vals],
                    fmt=symbol + "-", color=color,
                    label=label,
                    markersize=5, alpha=0.5, capsize=10,
                    elinewidth=3)

        # Plot the fit within the fitting range.
        low_lim = \
            self._spatial_unit_conversion(self._fit_range[0], xunit).value
        high_lim = \
            self._spatial_unit_conversion(self._fit_range[1], xunit).value

        ax.loglog(scales, 10**self.fitted_model(np.log10(pix_scales.value)),
                  '--', color=fit_color,
                  linewidth=3)

        ax.axvline(low_lim, color=color, alpha=0.5, linestyle='-')
        ax.axvline(high_lim, color=color, alpha=0.5, linestyle='-')

        ax.grid()

        ax.set_ylabel(r"$T_g$")

        if show_residual:
            resids = self.values - \
                10**self.fitted_model(np.log10(pix_scales.value))

            ax_r.errorbar(scales, resids, yerr=self.stddev[fin_vals],
                          fmt=symbol + "-", color=color, label=label,
                          markersize=5, alpha=0.5, capsize=10,
                          elinewidth=3)
            ax_r.axvline(low_lim, color=color, alpha=0.5, linestyle='-')
            ax_r.axvline(high_lim, color=color, alpha=0.5, linestyle='-')

            ax_r.axhline(0., color=fit_color, linestyle='--')

            ax_r.grid()

            ax_r.set_ylabel("Residuals")
            ax_r.set_xlabel("Scales ({})".format(xunit))

            plt.setp(ax.get_xticklabels(), visible=False)

        else:
            ax.set_xlabel("Scales ({})".format(xunit))

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, verbose=False, xunit=u.pix,
            convolve_kwargs={},
            use_pyfftw=False, threads=1,
            pyfftw_kwargs={}, scale_normalization=True,
            xlow=None, xhigh=None, brk=None, fit_kwargs={},
            save_name=None, **plot_kwargs):
        '''
        Compute the Wavelet transform.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        verbose : bool, optional
            Plot wavelet transform.
        xunit : u.Unit, optional
            Choose the unit to convert to when ang_units is enabled.
        convolve_kwargs : dict, optional
            Passed to `~astropy.convolution.convolve_fft`.
        scale_normalization: bool, optional
            Compute the transform with the correct scale-invariant
            normalization.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            See `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        scale_normalization: bool, optional
            Multiply the wavelet transform by the correct normalization
            factor.
        xlow : `~astropy.units.Quantity`, optional
            Lower scale value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper scale value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        fit_kwargs : dict, optional
            Passed to `~Wavelet.fit_transform`
        save_name : str,optional
            Save the figure when a file name is given.
        plot_kwargs : Passed to `~Wavelet.plot_transform`.
        '''
        self.compute_transform(scale_normalization=scale_normalization,
                               convolve_kwargs=convolve_kwargs,
                               use_pyfftw=use_pyfftw, threads=threads,
                               pyfftw_kwargs=pyfftw_kwargs,
                               show_progress=show_progress)
        self.fit_transform(xlow=xlow, xhigh=xhigh, brk=brk, **fit_kwargs)

        if verbose:
            print(self.fit.summary())

            if self._bootstrap_flag:
                print("Bootstrapping used to find stderrs! "
                      "Errors may not equal those shown above.")

            self.plot_transform(save_name=save_name, xunit=xunit,
                                **plot_kwargs)

        return self


class Wavelet_Distance(object):
    '''
    Compute the distance between the two cubes using the Wavelet transform.
    We fit a linear model to the two wavelet transforms. The distance is the
    t-statistic of the interaction term describing the difference in the
    slopes.

    Parameters
    ----------
    dataset1 : %(dtypes)s
        2D image.
    dataset2 : %(dtypes)s
        2D image.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int
        Number of scales to calculate the transform at.
    xlow : `astropy.units.Quantity`, optional
        The lower lag fitting limit. An array with 2 elements can be passed to
        give separate lower limits for the datasets.
    xhigh : `astropy.units.Quantity`, optional
        The upper lag fitting limit. See `xlow` above.
    fit_kwargs : dict, optional
        Passed to `~turbustat.statistics.Wavelet.run`.
    fit_kwargs2 : dict, optional
        Passed to `~turbustat.statistics.Wavelet.run` for `dataset2`. When
        `None` is given, `fit_kwargs` is used for `dataset2`.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, dataset1, dataset2,
                 scales=None, num=50, xlow=None, xhigh=None,
                 fit_kwargs={}, fit_kwargs2=None):
        super(Wavelet_Distance, self).__init__()

        xlow, xhigh = check_fit_limits(xlow, xhigh)

        # if fiducial_model is None:
        if isinstance(dataset1, Wavelet):
            self.wt1 = dataset1
            needs_run = False
            if not hasattr(self.wt1, '_slope'):
                warn("Wavelet class passed as `dataset1` does not have a "
                     "fitted slope. Computing Wavelet transform.")
                needs_run = True
        else:
            self.wt1 = Wavelet(dataset1, scales=scales)
            needs_run = True

        if needs_run:
            self.wt1.run(xlow=xlow[0], xhigh=xhigh[0], **fit_kwargs)

        if fit_kwargs2 is None:
            fit_kwargs2 = fit_kwargs

        if isinstance(dataset2, Wavelet):
            self.wt2 = dataset2
            needs_run = False
            if not hasattr(self.wt2, '_slope'):
                warn("Wavelet class passed as `dataset2` does not have a "
                     "fitted slope. Computing Wavelet transform.")
                needs_run = True
        else:
            self.wt2 = Wavelet(dataset2, scales=scales)
            needs_run = True

        if needs_run:
            self.wt2.run(xlow=xlow[1], xhigh=xhigh[1], **fit_kwargs2)

    def distance_metric(self, verbose=False, xunit=u.pix,
                        save_name=None, plot_kwargs1={},
                        plot_kwargs2={}):
        '''
        Implements the distance metric for 2 wavelet transforms.
        We fit the linear portion of the transform to represent the powerlaw

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        xunit : `~astropy.units.Unit`, optional
            Unit of the x-axis in the plot in pixel, angular, or
            physical units.
        save_name : str, optional
            Name of the save file. Enables saving the figure.
        plot_kwargs1 : dict, optional
            Pass kwargs to `~turbustat.statistics.Wavelet.plot_transform` for
            `dataset1`.
        plot_kwargs2 : dict, optional
            Pass kwargs to `~turbustat.statistics.Wavelet.plot_transform` for
            `dataset2`.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.wt1.slope - self.wt2.slope) /
                   np.sqrt(self.wt1.slope_err**2 +
                           self.wt2.slope_err**2))

        if verbose:

            print(self.wt1.fit.summary())
            print(self.wt2.fit.summary())

            import matplotlib.pyplot as plt

            defaults1 = {'color': 'b', 'symbol': 'D', 'label': '1'}
            defaults2 = {'color': 'g', 'symbol': 'o', 'label': '2'}

            for key in defaults1:
                if key not in plot_kwargs1:
                    plot_kwargs1[key] = defaults1[key]

            for key in defaults2:
                if key not in plot_kwargs2:
                    plot_kwargs2[key] = defaults2[key]

            if 'xunit' in plot_kwargs1:
                del plot_kwargs1['xunit']
            if 'xunit' in plot_kwargs2:
                del plot_kwargs2['xunit']

            self.wt1.plot_transform(xunit=xunit,
                                    **plot_kwargs1)
            self.wt2.plot_transform(xunit=xunit,
                                    **plot_kwargs2)
            axes = plt.gcf().get_axes()
            axes[0].legend(loc='best', frameon=True)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
