# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from numpy.fft import fftfreq
from astropy import units as u

from ..lm_seg import Lm_Seg
from ..rfft_to_fft import rfft_to_fft
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types
from ..fitting_utils import clip_func, residual_bootstrap


class VCS(BaseStatisticMixIn):

    '''
    The VCS technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header : FITS header, optional
        Corresponding FITS header.
    vel_units : bool, optional
        Convert frequencies to the spectral unit in the header.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None):
        super(VCS, self).__init__()

        self.input_data_header(cube, header)

        self._has_nan_flag = False

        isnan = np.isnan(self.data)
        if isnan.any():
            self.data = self.data.copy()
            self.data[isnan] = 0.
            self._has_nan_flag = True

        self.vel_channels = np.arange(1, self.data.shape[0], 1)

        self.freqs = \
            np.abs(fftfreq(self.data.shape[0])) / u.pix

    def compute_pspec(self, use_pyfftw=False, threads=1, **pyfftw_kwargs):
        '''
        Take the FFT of each spectrum in the velocity dimension and average.

        Parameters
        ----------
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        '''

        if self._has_nan_flag:
            # Is this the best way to be averaging the data?
            good_pixel_count = np.sum(self.data.max(axis=0) != 0)
        else:
            good_pixel_count = \
                float(self.data.shape[1] * self.data.shape[2])

        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')

        fft = rfft_to_fft(self.data, use_pyfftw=use_pyfftw,
                          keep_rfft=False,
                          threads=threads,
                          **pyfftw_kwargs)
        ps3D = np.power(fft, 2.)
        self._ps1D = np.nansum(ps3D, axis=(1, 2)) / good_pixel_count

    @property
    def ps1D(self):
        '''
        The 1D VCS spectrum.
        '''
        return self._ps1D

    def fit_pspec(self, breaks=None, log_break=True, low_cut=None,
                  high_cut=None, fit_verbose=False, bootstrap=False,
                  **bootstrap_kwargs):
        '''
        Fit the 1D Power spectrum using a segmented linear model. Note that
        the current implementation allows for only 1 break point in the
        model. If the break point is estimated via a spline, the breaks are
        tested, starting from the largest, until the model finds a good fit.

        Parameters
        ----------
        breaks : float or None, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error. If None, a spline is used to estimate the breaks.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        low_cut : `~astropy.units.Quantity`, optional
            Lowest frequency to consider in the fit.
        high_cut : `~astropy.units.Quantity`, optional
            Highest frequency to consider in the fit.
        fit_verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        bootstrap : bool, optional
            Bootstrap using the model residuals to estimate the standard
            errors.
        bootstrap_kwargs : dict, optional
            Pass keyword arguments to `~turbustat.statistics.fitting_utils.residual_bootstrap`.
        '''

        # Make the data to fit to
        if low_cut is None:
            # Default to the largest frequency, since this is just 1 pixel
            # But cut out fitting the total power point.
            self.low_cut = 0.98 / (float(self.data.shape[0]) * u.pix)
        else:
            self.low_cut = \
                self._spectral_freq_unit_conversion(low_cut, u.pix**-1)

        if high_cut is None:
            # Set to something larger than
            self.high_cut = (self.freqs.max().value * 1.01) / u.pix
        else:
            self.high_cut = \
                self._spectral_freq_unit_conversion(high_cut, u.pix**-1)

        # We keep both the real and imag frequencies, but we only need the real
        # component when fitting. We'll cut off the total power frequency too
        # in case it causes an extra break point (which seems to happen).
        shape = self.freqs.size
        rfreqs = self.freqs[1:shape // 2].value
        ps1D = self.ps1D[1:shape // 2]

        y = np.log10(ps1D[clip_func(rfreqs, self.low_cut.value,
                                    self.high_cut.value)])
        x = np.log10(rfreqs[clip_func(rfreqs, self.low_cut.value,
                                      self.high_cut.value)])

        if breaks is None:
            from scipy.interpolate import UnivariateSpline

            # Need to order the points
            spline = UnivariateSpline(x, y, k=1, s=1)

            # The first and last are just the min and max of x
            breaks = spline.get_knots()[1:-1]

            if fit_verbose:
                print("Breaks found from spline are: " + str(breaks))

            # Take the number according to max_breaks starting at the
            # largest x.
            breaks = breaks[::-1]

            # Ensure a break doesn't fall at the max or min.
            if breaks.size > 0:
                if breaks[0] == x.max():
                    breaks = breaks[1:]
            if breaks.size > 0:
                if breaks[-1] == x.min():
                    breaks = breaks[:-1]

            if x.size <= 3 or y.size <= 3:
                raise Warning("There are no points to fit to. Try lowering "
                              "'lg_scale_cut'.")

            # If no breaks, set to half-way before last point
            if breaks.size == 0:
                x_copy = np.sort(x.copy())
                breaks = np.array([0.5 * (x_copy[-1] + x_copy[-3])])

            # Now try these breaks until a good fit including the break is
            # found. If none are found, it accepts that there wasn't a good
            # break and continues on.
            i = 0
            while True:
                self.fit = \
                    Lm_Seg(x, y, breaks[i])
                self.fit.fit_model(verbose=fit_verbose, cov_type='HC3')

                if self.fit.params.size == 5:
                    # Success!
                    breaks = breaks[i]
                    break
                i += 1
                if i >= breaks.size:
                    warnings.warn("No good break point found. Returned fit\
                                   does not include a break!")
                    break

            # return self

        if not log_break:
            breaks = np.log10(breaks)

        # Fit the final model with whichever breaks were passed.
        self.fit = Lm_Seg(x, y, breaks)
        self.fit.fit_model(verbose=fit_verbose, cov_type='HC3')

        if bootstrap:
            stderrs = residual_bootstrap(self.fit.fit,
                                         **bootstrap_kwargs)

            self._slope_errs = stderrs[1:-1]
        else:
            self._slope_errs = self.fit.slope_errs

        self._slope = self.fit.slopes

        self._bootstrap_flag = bootstrap

    @property
    def slope(self):
        '''
        Power spectrum slope(s).
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        1-sigma error on the power spectrum slope(s).
        '''
        return self._slope_errs

    @property
    def brk(self):
        '''
        Fitted break point.
        '''
        return self.fit.brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point.
        '''
        return self.fit.brk_err

    def plot_fit(self, save_name=None, xunit=u.pix**-1, color='r',
                 symbol='o', fit_color='k', label=None, show_residual=True):
        '''
        Plot the VCS curve and the associated fit.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        xunit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        color : {str, RGB tuple}, optional
            Color to plot the VCS curve.
        symbol : str, optional
            Symbol to use for the data.
        fit_color : {str, RGB tuple}, optional
            Color of the 1D fit.
        label : str, optional
            Label to later be used in a legend.
        show_residual : bool, optional
            Plot the residuals for the 1D power-spectrum fit.

        '''
        import matplotlib.pyplot as plt

        if fit_color is None:
            fit_color = color

        fig = plt.gcf()
        axes = plt.gcf().get_axes()

        if len(axes) == 2:
            ax, ax_r = axes
        elif len(axes) == 1:
            ax = axes[0]
        else:
            if show_residual:
                ax = plt.subplot2grid((4, 1), (0, 0), rowspan=3)
                ax_r = plt.subplot2grid((4, 1), (3, 0), rowspan=1, sharex=ax)
            else:
                ax = plt.subplot(111)

        xlab = r"log $( k_v / $ (" + str(xunit**-1) + \
            ")$^{-1})$"

        shape = self.freqs.size
        rfreqs = self.freqs[1:shape // 2]
        ps1D = self.ps1D[1:shape // 2]

        good_interval = clip_func(rfreqs.value, self.low_cut.value,
                                  self.high_cut.value)

        freq = self._spectral_freq_unit_conversion(rfreqs, xunit)

        y_fit = \
            10**self.fit.model(np.log10(rfreqs.value[good_interval]))

        # Points in dark red
        ax.loglog(freq, ps1D, symbol, alpha=0.3, label=label,
                  color=color)
        ax.loglog(freq[good_interval], y_fit, color=fit_color,
                  linewidth=4)

        ax.set_ylabel(r"log P$_{1}$(k$_{v}$)")

        ax.axvline(self._spectral_freq_unit_conversion(self.low_cut,
                                                       xunit).value,
                   linestyle="--", color=color, alpha=0.5)
        ax.axvline(self._spectral_freq_unit_conversion(self.high_cut,
                                                       xunit).value,
                   linestyle="--", color=color, alpha=0.5)
        ax.grid(True)
        ax.legend(loc='best', frameon=True)

        if show_residual:
            resids = ps1D - 10**self.fit.model(np.log10(rfreqs.value))
            ax_r.semilogx(freq, resids, symbol, color=color)
            ax_r.axhline(0., color=fit_color)

            ax_r.grid()

            ax_r.set_ylabel("Residuals")

            ax_r.set_xlabel(xlab)
        else:
            ax.set_xlabel(xlab)

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, verbose=False, save_name=None, xunit=u.pix**-1,
            use_pyfftw=False, threads=1, pyfftw_kwargs={},
            **fit_kwargs):
        '''
        Run the entire computation.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis in the plot to.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`_
            for a list of accepted kwargs.
        fit_kwargs : Passed to `~VCS.fit_pspec`.
        '''

        # Remove threads if in dict
        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')
        self.compute_pspec(use_pyfftw=use_pyfftw, threads=threads,
                           **pyfftw_kwargs)

        self.fit_pspec(**fit_kwargs)

        if verbose:
            # Print the final fitted model when fit_verbose is not enabled.
            if not fit_kwargs.get("fit_verbose"):
                print(self.fit.fit.summary())
                if self._bootstrap_flag:
                    print("Bootstrapping used to find stderrs! "
                          "Errors may not equal those shown above.")

            self.plot_fit(save_name=save_name, xunit=xunit)

        return self


class VCS_Distance(object):

    '''
    Calculate the distance between two cubes using VCS. The 1D power spectrum
    is modeled by a broked linear model to account for the density and
    velocity dominated scales. The distance is the sum of  the t-statistics
    for each model.

    Parameters
    ----------
    cube1 : %(dtypes)s or `~VCS`
        Data cube. Or a `~VCS` class can be passed which may be pre-computed.
    cube2 : %(dtypes)s or `~VCS`
        See `data1`.
    slice_size : float, optional
        Slice to degrade the cube to.
    breaks : float, list or array, optional
        Specify where the break point is. If None, attempts to find using
        spline.
    fit_kwargs : dict, optional
        Passed to `~VCS.run`.
    fit_kwargs2 : dict or None, optional
        Passed to `~VCS.run` for `cube2`. When `None` is given, settings
        from `fit_kwargs` will be used for `cube2`.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, breaks=None,
                 fit_kwargs={}, fit_kwargs2=None):
        super(VCS_Distance, self).__init__()

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if isinstance(cube1, VCS):
            self.vcs1 = cube1
            needs_run = False
            if not hasattr(self.vcs1, '_slope'):
                warnings.warn("VCS class given as `cube1` does not have a "
                              "fitted slope. Computing VCS.")
                needs_run = True
        else:
            self.vcs1 = VCS(cube1)
            needs_run = True

        if needs_run:
            self.vcs1.run(breaks=breaks[0], **fit_kwargs)

        if fit_kwargs2 is None:
            fit_kwargs2 = fit_kwargs

        if isinstance(cube2, VCS):
            self.vcs2 = cube2
            needs_run = False
            if not hasattr(self.vcs2, '_slope'):
                warnings.warn("VCS class given as `cube2` does not have a "
                              "fitted slope. Computing VCS.")
                needs_run = True
        else:
            self.vcs2 = VCS(cube2)
            needs_run = True

        if needs_run:
            self.vcs2.run(breaks=breaks[1], **fit_kwargs2)

    def distance_metric(self, verbose=False, xunit=u.pix**-1,
                        save_name=None, plot_kwargs1={},
                        plot_kwargs2={}):
        '''

        Implements the distance metric for 2 VCS transforms.
        This distance is the t-statistic of the difference
        in the slopes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Now construct the t-statistics for each portion

        # There should always be the velocity distance
        self.large_scale_distance = \
            np.abs((self.vcs1.slope[0] - self.vcs2.slope[0]) /
                   np.sqrt(self.vcs1.slope_err[0]**2 +
                           self.vcs2.slope_err[0]**2))

        # A density distance is only found if a break was found
        if self.vcs1.slope.size == 1 or self.vcs2.slope.size == 1:
            self.small_scale_distance = np.NaN
            self.break_distance = np.NaN
        else:
            self.small_scale_distance = \
                np.abs((self.vcs1.slope[1] - self.vcs2.slope[1]) /
                       np.sqrt(self.vcs1.slope_err[1]**2 +
                               self.vcs2.slope_err[1]**2))

            self.break_distance = \
                np.abs((self.vcs1.brk - self.vcs2.brk) /
                       np.sqrt(self.vcs1.brk_err**2 +
                               self.vcs2.brk_err**2))

        # The overall distance is the sum from the two models
        self.distance = \
            np.nansum([self.large_scale_distance, self.small_scale_distance])

        if verbose:

            print(self.vcs1.fit.fit.summary())
            print(self.vcs2.fit.fit.summary())

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

            self.vcs1.plot_fit(xunit=xunit, **plot_kwargs1)
            self.vcs2.plot_fit(xunit=xunit, **plot_kwargs2)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
