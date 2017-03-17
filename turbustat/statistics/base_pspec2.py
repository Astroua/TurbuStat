# Licensed under an MIT open source license - see LICENSE


import numpy as np
import statsmodels.api as sm
import warnings
import astropy.units as u

from .lm_seg import Lm_Seg
from .psds import pspec
from .fitting_utils import clip_func


class StatisticBase_PSpec2D(object):
    """
    Common features shared by 2D power spectrum methods.
    """

    @property
    def ps2D(self):
        '''
        Two-dimensional power spectrum.
        '''
        return self._ps2D

    @property
    def ps1D(self):
        '''
        One-dimensional power spectrum.
        '''
        return self._ps1D

    @property
    def ps1D_stddev(self):
        '''
        1-sigma standard deviation of the 1D power spectrum.
        '''
        if not self._stddev_flag:
            Warning("ps1D_stddev is only calculated when return_stddev"
                    " is enabled.")

        return self._ps1D_stddev

    @property
    def freqs(self):
        '''
        Corresponding spatial frequencies of the 1D power spectrum.
        '''
        return self._freqs

    @property
    def wavenumbers(self):
        return self._freqs * min(self._ps2D.shape)

    def compute_radial_pspec(self, return_stddev=True,
                             logspacing=True, max_bin=None, **kwargs):
        '''
        Computes the radially averaged power spectrum.

        Parameters
        ----------
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        max_bin : float, optional
            Maximum spatial frequency to bin values at.
        kwargs : passed to `~turbustat.statistics.psds.pspec`.
        '''

        if return_stddev:
            self._freqs, self._ps1D, self._ps1D_stddev = \
                pspec(self.ps2D, return_stddev=return_stddev,
                      logspacing=logspacing, max_bin=max_bin, **kwargs)
            self._stddev_flag = True
        else:
            self._freqs, self._ps1D = \
                pspec(self.ps2D, return_stddev=return_stddev, max_bin=max_bin,
                      **kwargs)
            self._stddev_flag = False

        # Attach units to freqs
        self._freqs = self.freqs / u.pix

    def fit_pspec(self, brk=None, log_break=True, low_cut=None,
                  high_cut=None, min_fits_pts=10, verbose=False,
                  large_scale=1.):
        '''
        Fit the 1D Power spectrum using a segmented linear model. Note that
        the current implementation allows for only 1 break point in the
        model. If the break point is estimated via a spline, the breaks are
        tested, starting from the largest, until the model finds a good fit.

        Parameters
        ----------
        brk : float or None, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error. If None, a spline is used to estimate the breaks.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        low_cut : float, optional
            Lowest frequency to consider in the fit.
        high_cut : float, optional
            Highest frequency to consider in the fit.
        min_fits_pts : int, optional
            Sets the minimum number of points needed to fit. If not met, the
            break found is rejected.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        large_scale : float, optional
            Set fraction of array shape corresponding to the largest frequency
            to include while fitting (i.e., 1. uses all frequencies, 0.5 limits
            the maximum frequency to half of the smallest array shape).
        '''

        # Make the data to fit to
        if low_cut is None:
            # Default to the largest frequency, since this is just 1 pixel
            # in the 2D PSpec.
            self.low_cut = 1. / (large_scale * float(max(self.ps2D.shape)))
        else:
            self.low_cut = low_cut

        if high_cut is None:
            self.high_cut = self.freqs.max().value + 1
        else:
            self.high_cut = high_cut

        x = np.log10(self.freqs[clip_func(self.freqs.value, self.low_cut,
                                          self.high_cut)].value)
        y = np.log10(self.ps1D[clip_func(self.freqs.value, self.low_cut,
                                         self.high_cut)])

        if brk is not None:
            # Try the fit with a break in it.
            if not log_break:
                brk = np.log10(brk)

            brk_fit = \
                Lm_Seg(x, y, brk)
            brk_fit.fit_model(verbose=verbose)

            if brk_fit.params.size == 5:

                # Check to make sure this leaves enough to fit to.
                if sum(x < brk_fit.brk) < min_fits_pts:
                    warnings.warn("Not enough points to fit to." +
                                  " Ignoring break.")

                    self.high_cut = self.freqs.max().value
                else:
                    good_pts = x.copy() < brk_fit.brk
                    x = x[good_pts]
                    y = y[good_pts]

                    self.high_cut = 10**brk_fit.brk

            else:
                self.high_cut = self.freqs.max().value
                # Break fit failed, revert to normal model
                warnings.warn("Model with break failed, reverting to model\
                               without break.")

        x = sm.add_constant(x)

        model = sm.OLS(y, x, missing='drop')

        self.fit = model.fit()

        self._slope = self.fit.params[1]
        self._slope_err = self.fit.bse[1]

    @property
    def slope(self):
        '''
        Power spectrum slope.
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        1-sigma error on the power spectrum slope.
        '''
        return self._slope_err

    def plot_fit(self, show=True, show_2D=False, color='r', label=None,
                 symbol="D", ang_units=False, unit=u.deg, save_name=None,
                 use_wavenumber=False):
        '''
        Plot the fitted model.
        '''

        import matplotlib.pyplot as p

        if ang_units:
            if use_wavenumber:
                xlab = r"k/" + unit.to_string() + "$^{-1}$"
            else:
                xlab = r"Spatial Frequency/" + unit.to_string() + "$^{-1}$"
        else:
            if use_wavenumber:
                xlab = r"k/pixel$^{-1}$"
            else:
                xlab = r"Spatial Frequency/pixel$^{-1}$"

        # 2D Spectrum is shown alongside 1D. Otherwise only 1D is returned.
        if show_2D:
            p.subplot(122)
            p.imshow(np.log10(self.ps2D), interpolation="nearest",
                     origin="lower")
            p.colorbar()

            ax = p.subplot(121)
        else:
            ax = p.subplot(111)

        good_interval = clip_func(self.freqs.value, self.low_cut,
                                  self.high_cut)

        y_fit = self.fit.fittedvalues
        fit_index = np.logical_and(np.isfinite(self.ps1D), good_interval)

        # Set the x-values to use (freqs or k)
        if use_wavenumber:
            xvals = self.wavenumbers
        else:
            xvals = self.freqs

        if ang_units:
            xvals = 1. / (1. / xvals).to(unit,
                                         equivalencies=self.angular_equiv)

        xvals = xvals.value

        if self._stddev_flag:
            ax.errorbar(np.log10(xvals),
                        np.log10(self.ps1D),
                        yerr=0.434 * (self.ps1D_stddev / self.ps1D),
                        color=color,
                        fmt=symbol, markersize=5, alpha=0.5, capsize=10,
                        elinewidth=3)

            ax.plot(np.log10(xvals[fit_index]), y_fit, color + '-',
                    label=label, linewidth=2)
            ax.set_xlabel("log " + xlab)
            ax.set_ylabel(r"log P$_2(K)$")

        else:
            ax.loglog(self.xvals[fit_index], 10**y_fit, color + '-',
                      label=label, linewidth=2)

            ax.loglog(self.xvals, self.ps1D, color + symbol, alpha=0.5,
                      markersize=5)

            ax.set_xlabel(xlab)
            ax.set_ylabel(r"P$_2(K)$")

        # Show the fitting extents
        low_cut = self.low_cut if not use_wavenumber else \
            self.low_cut * min(self._ps2D.shape)
        high_cut = self.high_cut if not use_wavenumber else \
            self.high_cut * min(self._ps2D.shape)
        p.axvline(np.log10(low_cut), color=color, alpha=0.5, linestyle='--')
        p.axvline(np.log10(high_cut), color=color, alpha=0.5, linestyle='--')

        p.grid(True)

        if save_name is not None:
            p.savefig(save_name)

        if show:
            p.show()
