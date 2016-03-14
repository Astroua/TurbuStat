# Licensed under an MIT open source license - see LICENSE


import numpy as np
import statsmodels.api as sm
import warnings

from .lm_seg import Lm_Seg


class StatisticBase_PSpec2D(object):
    """
    Common features shared by 2D power spectrum methods.
    """

    @property
    def ps2D(self):
        return self._ps2D

    @property
    def ps1D(self):
        return self._ps1D

    @property
    def ps1D_stddev(self):
        if not self._stddev_flag:
            Warning("ps1D_stddev is only calculated when return_stddev"
                    " is enabled.")

        return self._ps1D_stddev

    @property
    def freqs(self):
        return self._freqs

    def fit_pspec(self, brk=None, log_break=False, low_cut=None,
                  min_fits_pts=10, verbose=False):
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
        lg_scale_cut : int, optional
            Cuts off largest scales, which deviate from the powerlaw.
        min_fits_pts : int, optional
            Sets the minimum number of points needed to fit. If not met, the
            break found is rejected.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

        # Make the data to fit to
        if low_cut is None:
            # Default to the largest frequency, since this is just 1 pixel
            # in the 2D PSpec.
            self.low_cut = 1/float(max(self.ps2D.shape))
        else:
            self.low_cut = low_cut
        x = np.log10(self.freqs[self.freqs > self.low_cut])
        y = np.log10(self.ps1D[self.freqs > self.low_cut])

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

                    self.high_cut = self.freqs.max()
                else:
                    x = x[x < brk_fit.brk]
                    y = y[x < brk_fit.brk]

                    self.high_cut = 10**brk_fit.brk

            else:
                self.high_cut = self.freqs.max()
                # Break fit failed, revert to normal model
                warnings.warn("Model with break failed, reverting to model\
                               without break.")
        else:
            self.high_cut = self.freqs.max()

        x = sm.add_constant(x)

        model = sm.OLS(y, x, missing='drop')

        self.fit = model.fit()

        self._slope = self.fit.params[1]

        cov_matrix = self.fit.cov_params()
        self._slope_err = np.sqrt(cov_matrix[1, 1])

        return self

    @property
    def slope(self):
        return self._slope

    @property
    def slope_err(self):
        return self._slope_err

    def plot_fit(self, show=True, show_2D=False, color='r', label=None):
        '''
        Plot the fitted model.
        '''

        import matplotlib.pyplot as p

        if self.phys_units_flag:
            xlab = r"log K"
        else:
            xlab = r"K (pixel$^{-1}$)"

        # 2D Spectrum is shown alongside 1D. Otherwise only 1D is returned.
        if show_2D:
            p.subplot(122)
            p.imshow(np.log10(self.ps2D), interpolation="nearest",
                     origin="lower")
            p.colorbar()

            ax = p.subplot(121)
        else:
            ax = p.subplot(111)

        good_interval = np.logical_and(self.freqs >= self.low_cut,
                                       self.freqs <= self.high_cut)

        y_fit = self.fit.fittedvalues
        fit_index = np.logical_and(np.isfinite(self.ps1D), good_interval)

        ax.loglog(self.freqs[fit_index], 10**y_fit, color+'-',
                  label=label, linewidth=2)
        ax.set_xlabel(xlab)
        ax.set_ylabel(r"P$_2(K)$")

        if self._stddev_flag:
            ax.errorbar(self.freqs[good_interval], self.ps1D[good_interval],
                        yerr=self.ps1D_stddev[good_interval], color=color,
                        fmt='D', markersize=5, alpha=0.5, capsize=10,
                        elinewidth=3)
            ax.set_xscale("log", nonposy='clip')
            ax.set_yscale("log", nonposy='clip')
        else:
            p.loglog(self.freqs[good_interval],
                     self.ps1D[good_interval], color+"D", alpha=0.5,
                     markersize=5)

        p.grid(True)

        if show:
            p.show()
