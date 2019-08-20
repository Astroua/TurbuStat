# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from six import string_types
import statsmodels.api as sm
from warnings import warn
from astropy.utils.console import ProgressBar
from itertools import product

from ..psds import pspec, make_radial_arrays
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data
from ..stats_utils import common_scale, fourier_shift, pixel_shift
from ..fitting_utils import clip_func, residual_bootstrap
from ..elliptical_powerlaw import (fit_elliptical_powerlaw,
                                   inverse_interval_transform,
                                   inverse_interval_transform_stderr)
from ..stats_warnings import TurbuStatMetricWarning


class SCF(BaseStatisticMixIn):
    '''
    Computes the Spectral Correlation Function of a data cube
    (Rosolowsky et al, 1999).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header :  FITS header, optional
        Header for the cube.
    size : int, optional
        The total size of the lags used in one dimension in pixels. The maximum
        lag size will be (size - 1) / 2 in each direction.
    roll_lags : `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        Pass a custom array of lag values. An odd number of lags, centered at
        0, must be given. If no units are given, it is
        assumed that the lags are in pixels. The lags should have
        symmetric positive and negative values (e.g., [-1, 0, 1]).
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.

    Examples
    --------
    >>> from spectral_cube import SpectralCube
    >>> from turbustat.statistics import SCF
    >>> cube = SpectralCube.read("Design4.13co.fits")  # doctest: +SKIP
    >>> scf = SCF(cube)  # doctest: +SKIP
    >>> scf.run(verbose=True)  # doctest: +SKIP
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, size=11, roll_lags=None,
                 distance=None):
        super(SCF, self).__init__()

        # Set data and header
        self.input_data_header(cube, header)

        if distance is not None:
            self.distance = distance

        if roll_lags is None:
            if size % 2 == 0:
                Warning("Size must be odd. Reducing size to next lowest odd"
                        " number.")
                size = size - 1
            self.roll_lags = (np.arange(size) - size // 2) * u.pix
        else:
            if roll_lags.size % 2 == 0:
                Warning("Size of roll_lags must be odd. Reducing size to next"
                        "lowest odd number.")
                roll_lags = roll_lags[: -1]

            if isinstance(roll_lags, u.Quantity):
                pass
            elif isinstance(roll_lags, np.ndarray):
                roll_lags = roll_lags * u.pix
            else:
                raise TypeError("roll_lags must be an astropy.units.Quantity"
                                " array or a numpy.ndarray.")

            self.roll_lags = roll_lags

            # Make sure that we can convert the lags
            self._to_pixel(self.roll_lags)

        self.size = self.roll_lags.size

        self._scf_surface = None
        self._scf_spectrum_stddev = None

        self._fit2D_flag = False

    @property
    def roll_lags(self):
        '''
        Pixel values that the cube is rolled by to compute the SCF correlation
        surface.
        '''
        return self._roll_lags

    @roll_lags.setter
    def roll_lags(self, value):

        # Needs to be a quantity with a unit
        if not hasattr(value, "unit"):
            raise ValueError("roll_lags must be an astropy.units.Quantity.")

        try:
            self._to_pixel(value)
        except u.UnitConversionError:
            raise u.UnitConversionError("Cannot convert given roll lags to "
                                        "pixel units. `roll_lags` must have"
                                        " pixel, angular, or physical (if a"
                                        " distance is given) units.")
        self._roll_lags = value

    @property
    def scf_surface(self):
        '''
        SCF correlation array
        '''
        return self._scf_surface

    @property
    def scf_spectrum(self):
        '''
        Azimuthally averaged 1D SCF spectrum
        '''
        return self._scf_spectrum

    @property
    def scf_spectrum_stddev(self):
        '''
        Standard deviation of the `~SCF.scf_spectrum`
        '''
        return self._scf_spectrum_stddev

    @property
    def lags(self):
        '''
        Values of the lags, in pixels, to compute SCF at
        '''
        return self._lags

    def compute_surface(self, boundary='continuous', show_progress=True):
        '''
        Computes the SCF up to the given lag value. This is an
        expensive operation and could take a long time to calculate.

        Parameters
        ----------
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        show_progress : bool, optional
            Show a progress bar when computing the surface. =
        '''

        if boundary not in ["continuous", "cut"]:
            raise ValueError("boundary must be 'continuous' or 'cut'.")

        self._scf_surface = np.zeros((self.size, self.size))

        # Convert the lags into pixel units.
        pix_lags = self._to_pixel(self.roll_lags).value

        dx = pix_lags.copy()
        dy = pix_lags.copy()

        if show_progress:
            bar = ProgressBar(len(dx) * len(dy))

        for n, (x_shift, y_shift) in enumerate(product(dx, dy)):

            i, j = np.unravel_index(n, (len(dx), len(dy)))

            if x_shift == 0 and y_shift == 0:
                self._scf_surface[j, i] = 1.

            if x_shift == 0:
                tmp = self.data
            else:
                if float(x_shift).is_integer():
                    shift_func = pixel_shift
                else:
                    shift_func = fourier_shift
                tmp = shift_func(self.data, x_shift, axis=1)

            if y_shift != 0:
                if float(y_shift).is_integer():
                    shift_func = pixel_shift
                else:
                    shift_func = fourier_shift
                tmp = shift_func(tmp, y_shift, axis=2)

            if boundary is "cut":
                # Always round up to the nearest integer.
                x_shift = np.ceil(x_shift).astype(int)
                y_shift = np.ceil(y_shift).astype(int)
                if x_shift < 0:
                    x_slice_data = slice(None, tmp.shape[1] + x_shift)
                    x_slice_tmp = slice(-x_shift, None)
                else:
                    x_slice_data = slice(x_shift, None)
                    x_slice_tmp = slice(None, tmp.shape[1] - x_shift)

                if y_shift < 0:
                    y_slice_data = slice(None, tmp.shape[2] + y_shift)
                    y_slice_tmp = slice(-y_shift, None)
                else:
                    y_slice_data = slice(y_shift, None)
                    y_slice_tmp = slice(None, tmp.shape[2] - y_shift)

                data_slice = (slice(None), x_slice_data, y_slice_data)
                tmp_slice = (slice(None), x_slice_tmp, y_slice_tmp)
            elif boundary is "continuous":
                data_slice = (slice(None),) * 3
                tmp_slice = (slice(None),) * 3

            values = \
                np.nansum(((self.data[data_slice] - tmp[tmp_slice]) ** 2),
                          axis=0) / \
                (np.nansum(self.data[data_slice] ** 2, axis=0) +
                 np.nansum(tmp[tmp_slice] ** 2, axis=0))

            scf_value = 1. - \
                np.sqrt(np.nansum(values) / np.sum(np.isfinite(values)))

            if scf_value > 1:
                raise ValueError("Cannot have a correlation above 1. Check "
                                 "your input data. Contact the TurbuStat "
                                 "authors if the problem persists.")

            self._scf_surface[j, i] = scf_value

            if show_progress:
                bar.update(n + 1)

    def compute_spectrum(self, **kwargs):
        '''
        Compute the 1D spectrum as a function of lag. Can optionally
        use log-spaced bins. kwargs are passed into the pspec function,
        which provides many options. The default settings are applicable in
        nearly all use cases.

        Parameters
        ----------
        kwargs : passed to `turbustat.statistics.psds.pspec`
        '''

        # If scf_surface hasn't been computed, do it
        if self.scf_surface is None:
            self.compute_surface()

        if kwargs.get("logspacing"):
            warn("Disabled log-spaced bins. This does not work well for the"
                 " SCF.", TurbuStatMetricWarning)
            kwargs.pop('logspacing')

        if kwargs.get("theta_0"):
            azim_constraint_flag = True
        else:
            azim_constraint_flag = False

        out = pspec(self.scf_surface, return_stddev=True,
                    logspacing=False, return_freqs=False, **kwargs)

        self._azim_constraint_flag = azim_constraint_flag

        if azim_constraint_flag:
            self._lags, self._scf_spectrum, self._scf_spectrum_stddev, \
                self._azim_mask = out
        else:
            self._lags, self._scf_spectrum, self._scf_spectrum_stddev = out

        roll_lag_diff = np.abs(self.roll_lags[1] - self.roll_lags[0])

        self._lags = self._lags * roll_lag_diff

    def fit_plaw(self, xlow=None, xhigh=None, verbose=False, bootstrap=False,
                 **bootstrap_kwargs):
        '''
        Fit a power-law to the SCF spectrum.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value limit to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value limit to consider in the fit.
        verbose : bool, optional
            Show fit summary when enabled.
        '''

        pix_lags = self._to_pixel(self.lags)

        x = np.log10(pix_lags.value)
        y = np.log10(self.scf_spectrum)

        if xlow is not None:
            if not isinstance(xlow, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")

            # Convert xlow into the same units as the lags
            xlow = self._to_pixel(xlow)

            self._xlow = xlow

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.scf_spectrum, dtype=bool)
            self._xlow = np.abs(self.lags).min()

        if xhigh is not None:
            if not isinstance(xhigh, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")
            # Convert xhigh into the same units as the lags
            xhigh = self._to_pixel(xhigh)

            self._xhigh = xhigh

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.scf_spectrum, dtype=bool)
            self._xhigh = np.abs(self.lags).max()

        within_limits = np.logical_and(lower_limit, upper_limit)

        if not within_limits.any():
            raise ValueError("Limits have removed all lag values. Make xlow"
                             " and xhigh less restrictive.")

        y = y[within_limits]
        x = x[within_limits]

        x = sm.add_constant(x)

        # If the std were computed, use them as weights
        # Converting to the log stds doesn't matter since the weights
        # remain proportional to 1/sigma^2, and an overal normalization is
        # applied in the fitting routine.
        weights = self.scf_spectrum_stddev[within_limits] ** -2

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

    @property
    def slope(self):
        '''
        SCF spectrum slope
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        1-sigma error on the SCF spectrum slope
        '''
        return self._slope_err

    @property
    def xlow(self):
        '''
        Lower limit for lags to consider in fits.
        '''
        return self._xlow

    @property
    def xhigh(self):
        '''
        Upper limit for lags to consider in fits.
        '''
        return self._xhigh

    def fitted_model(self, xvals):
        '''
        Computes the modelled power-law using the given x values.

        Parameters
        ----------
        xvals : `~astropy.Quantity`
            Values of lags to compute the model at.

        Returns
        -------
        model_values : `~numpy.ndarray`
            Values of the model at the given values. Equivalent to log values
            of the SCF spectrum.
        '''

        if not isinstance(xvals, u.Quantity):
            raise TypeError("xvals must be an astropy.units.Quantity.")

        # Convert into the lag units used for the fit
        xvals = self._to_pixel(xvals).value

        model_values = \
            self.fit.params[0] + self.fit.params[1] * np.log10(xvals)

        return 10**model_values

    def fit_2Dplaw(self, fit_method='LevMarq', p0=(), xlow=None,
                   xhigh=None, bootstrap=True, niters=100, use_azimmask=False):
        '''
        Model the 2D power-spectrum surface with an elliptical power-law model.

        Parameters
        ----------
        fit_method : str, optional
            The algorithm fitting to use. Only 'LevMarq' is currently
            available.
        p0 : tuple, optional
            Initial parameters for fitting. If no values are given, the initial
            parameters start from the 1D fit parameters.
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value limit to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value limit to consider in the fit.
        bootstrap : bool, optional
            Bootstrap using the model residuals to estimate the parameter
            standard errors. This tends to give more realistic intervals than
            the covariance matrix.
        niters : int, optional
            Number of bootstrap iterations.
        use_azimmask : bool, optional
            Use the azimuthal mask defined for the 1D spectrum, when azimuthal
            limit have been given.
        '''

        # Adjust the distance based on the separation of the lags
        pix_lag_diff = np.diff(self._to_pixel(self.lags))[0].value

        if xlow is not None:
            # Convert xlow into the same units as the lags
            xlow = self._to_pixel(xlow)
            self._xlow = xlow
        else:
            self._xlow = np.abs(self.lags).min()

        if xhigh is not None:
            # Convert xhigh into the same units as the lags
            xhigh = self._to_pixel(xhigh)
            self._xhigh = xhigh
        else:
            self._xhigh = np.abs(self.lags).max()

        xlow_pix = self._to_pixel(self.xlow).value
        xhigh_pix = self._to_pixel(self.xhigh).value

        yy, xx = make_radial_arrays(self.scf_surface.shape)

        # Needed to make sure the definition of theta is consistent with
        # azimuthal masking and the elliptical p-law
        yy = yy[::-1]
        xx = xx[::-1]

        dists = np.sqrt(yy**2 + xx**2) * pix_lag_diff

        mask = clip_func(dists, xlow_pix, xhigh_pix)

        if hasattr(self, "_azim_mask") and use_azimmask:
            mask = np.logical_and(mask, self._azim_mask)

        if not mask.any():
            raise ValueError("Limits have removed all lag values. Make xlow"
                             " and xhigh less restrictive.")

        if len(p0) == 0:
            if hasattr(self, 'slope'):
                slope_guess = self.slope
                amp_guess = self.fit.params[0]
            else:
                # Let's guess it's going to be ~ -0.2
                slope_guess = -0.2
                amp_guess = 1.0

            # Use an initial guess pi / 2 for theta
            theta = np.pi / 2.
            # For ellip = 0.5
            ellip_conv = 0
            p0 = (amp_guess, ellip_conv, theta, slope_guess)

        params, stderrs, fit_2Dmodel, fitter = \
            fit_elliptical_powerlaw(np.log10(self.scf_surface[mask]),
                                    xx[mask],
                                    yy[mask], p0,
                                    fit_method=fit_method,
                                    bootstrap=bootstrap,
                                    niters=niters)

        self.fit2D = fit_2Dmodel
        self._fitter = fitter

        self._slope2D = params[3]
        self._slope2D_err = stderrs[3]

        self._theta2D = params[2] % np.pi
        self._theta2D_err = stderrs[2]

        # Apply transforms to convert back to the [0, 1) ellipticity range
        self._ellip2D = inverse_interval_transform(params[1], 0, 1)
        self._ellip2D_err = \
            inverse_interval_transform_stderr(stderrs[1], params[1], 0, 1)

        self._fit2D_flag = True

    @property
    def slope2D(self):
        '''
        Fitted slope of the 2D power-law.
        '''
        return self._slope2D

    @property
    def slope2D_err(self):
        '''
        Slope standard error of the 2D power-law.
        '''
        return self._slope2D_err

    @property
    def theta2D(self):
        '''
        Fitted position angle of the 2D power-law.
        '''
        return self._theta2D

    @property
    def theta2D_err(self):
        '''
        Position angle standard error of the 2D power-law.
        '''
        return self._theta2D_err

    @property
    def ellip2D(self):
        '''
        Fitted ellipticity of the 2D power-law.
        '''
        return self._ellip2D

    @property
    def ellip2D_err(self):
        '''
        Ellipticity standard error of the 2D power-law.
        '''
        return self._ellip2D_err

    def plot_fit(self, save_name=None, show_radial=True,
                 show_residual=True,
                 show_surface=True, contour_color='k',
                 cmap='viridis', data_color='r', fit_color='k',
                 xunit=u.pix):
        '''
        Plot the SCF surface, radial profiles, and associated fits.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        show_radial : bool, optional
            Show the azimuthally-averaged 1D SCF spectrum and fit.
        show_surface : bool, optional
            Show the SCF surface and (if performed) fit.
        show_residual : bool, optional
            Plot the residuals for the 1D SCF fit.
        contour_color : {str, RGB tuple}, optional
            Color of the 2D fit contours.
        cmap : {str, matplotlib color map}, optional
            Colormap to use in the plots. Default is viridis.
        data_color : {str, RGB tuple}, optional
            Color of the azimuthally-averaged data.
        fit_color : {str, RGB tuple}, optional
            Color of the 1D fit.
        xunit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        fig = plt.gcf()
        axes = plt.gcf().get_axes()
        if len(axes) == 3:
            ax, ax2, ax_r = axes
        elif len(axes) == 2:
            if show_surface and not show_residual:
                ax, ax2 = axes
            else:
                ax2, ax_r = axes
        elif len(axes) == 1:
            if show_radial:
                ax = axes[0]
            else:
                ax2 = axes[0]
        else:
            if show_surface:
                if show_radial:
                    ax = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=4)
                    if show_residual:
                        ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=2,
                                               rowspan=3)
                        ax_r = plt.subplot2grid((4, 4), (3, 0), colspan=2,
                                                rowspan=1, sharex=ax2)
                    else:
                        ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=2,
                                               rowspan=4)
                else:
                    ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=4)
            else:
                if show_residual:
                    ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=4,
                                           rowspan=3)
                    ax_r = plt.subplot2grid((4, 4), (3, 0), colspan=4,
                                            rowspan=1, sharex=ax2)
                else:
                    ax2 = plt.subplot2grid((4, 4), (0, 0), colspan=4,
                                           rowspan=4)

        if show_surface:

            im1 = ax.imshow(self.scf_surface, origin="lower",
                            interpolation="nearest",
                            cmap=cmap)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", "5%", pad="3%")
            cb = plt.colorbar(im1, cax=cax)

            cb.set_label("SCF Value")

            yy, xx = make_radial_arrays(self.scf_surface.shape)

            pix_lag_diff = np.diff(self._to_pixel(self.lags))[0].value
            dists = np.sqrt(yy**2 + xx**2) * pix_lag_diff

            xlow_pix = self._to_pixel(self.xlow).value
            xhigh_pix = self._to_pixel(self.xhigh).value

            mask = clip_func(dists, xlow_pix, xhigh_pix)

            if not mask.all():
                ax.contour(mask, colors='b', linestyles='-.', levels=[0.5])

            if self._fit2D_flag:

                ax.contour(self.fit2D(xx, yy)[::-1], colors=contour_color,
                           linestyles='-')

            if self._azim_constraint_flag:
                if not np.all(self._azim_mask):
                    ax.contour(self._azim_mask, colors='b', linestyles='-.',
                               levels=[0.5])
                else:
                    warn("Azimuthal mask includes all data. No contours will "
                         "be drawn.")

        if show_radial:

            pix_lags = self._to_pixel(self.lags)
            lags = self._spatial_unit_conversion(pix_lags, xunit).value

            ax2.errorbar(lags, self.scf_spectrum,
                         yerr=self.scf_spectrum_stddev,
                         fmt='o', color=data_color,
                         markersize=5)
            ax2.set_xscale("log")  # , nonposy='clip')
            ax2.set_yscale("log")  # , nonposy='clip')

            ax2.set_xlim(lags.min() * 0.75, lags.max() * 1.25)
            ax2.set_ylim(np.nanmin(self.scf_spectrum) * 0.75,
                         np.nanmax(self.scf_spectrum) * 1.25)

            # Overlay the fit. Use points 5% lower than the min and max.
            xvals = np.linspace(lags.min() * 0.95,
                                lags.max() * 1.05, 50) * xunit
            ax2.loglog(xvals, self.fitted_model(xvals), '--', linewidth=2,
                       label='Fit', color=fit_color)
            # Show the fit limits
            xlow = self._spatial_unit_conversion(self._xlow, xunit).value
            xhigh = self._spatial_unit_conversion(self._xhigh, xunit).value
            ax2.axvline(xlow, color='b', alpha=0.5, linestyle='-.')
            ax2.axvline(xhigh, color='b', alpha=0.5, linestyle='-.')

            ax2.legend()

            ax2.set_ylabel("SCF Value")

            if show_residual:
                resids = self.scf_spectrum - self.fitted_model(pix_lags)

                ax_r.errorbar(lags, resids,
                              yerr=self.scf_spectrum_stddev,
                              fmt='o', color=data_color,
                              markersize=5)

                ax_r.axvline(xlow, color='b', alpha=0.5, linestyle='-.')
                ax_r.axvline(xhigh, color='b', alpha=0.5, linestyle='-.')

                ax_r.axhline(0., color=fit_color, linestyle='--')

                ax_r.set_ylabel("Residuals")

                ax_r.set_xlabel("Lag ({})".format(xunit))

                # ax2.get_xaxis().set_ticks([])

            else:
                ax2.set_xlabel("Lag ({})".format(xunit))

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, boundary='continuous',
            show_progress=True, xlow=None, xhigh=None,
            fit_kwargs={}, fit_2D=True,
            fit_2D_kwargs={}, radialavg_kwargs={},
            verbose=False, xunit=u.pix, save_name=None):
        '''
        Computes all SCF outputs.

        Parameters
        ----------
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        xlow : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        xhigh : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        fit_kwargs : dict, optional
            Keyword arguments for `SCF.fit_plaw`. Use the
            `xlow` and `xhigh` keywords to provide fit limits.
        fit_2D : bool, optional
            Fit an elliptical power-law model to the 2D spectrum.
        fit_2D_kwargs : dict, optional
            Keyword arguments for `SCF.fit_2Dplaw`. Use the
            `xlow` and `xhigh` keywords to provide fit limits.
        radialavg_kwargs : dict, optional
            Passed to `~SCF.compute_spectrum`.
        verbose : bool, optional
            Enables plotting.
        xunit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str, optional
            Save the figure when a file name is given.
        '''

        self.compute_surface(boundary=boundary, show_progress=show_progress)
        self.compute_spectrum(**radialavg_kwargs)
        self.fit_plaw(verbose=verbose, xlow=xlow, xhigh=xhigh, **fit_kwargs)

        if fit_2D:
            self.fit_2Dplaw(xlow=xlow, xhigh=xhigh,
                            **fit_2D_kwargs)

        if verbose:
            self.plot_fit(save_name=save_name, xunit=xunit)

        return self


class SCF_Distance(object):

    '''
    Calculates the distance between two data cubes based on their SCF surfaces.
    The distance is the L2 norm between the surfaces. We weight the surface by
    1/r^2 where r is the distance from the centre.

    .. note:: When a pre-computed `~SCF` class is given for `cube1` or `cube2`,
              it needs to have the same set of lags between the cubes, defined
              by as the angular scales based on the FITS header. If the lags
              are not equivalent, the SCF will be re-computed with new lags.

    Parameters
    ----------
    cube1 : %(dtypes)s or `~SCF`
        Data cube. Or a `~SCF` class can be passed which may be pre-computed.
    cube2 : %(dtypes)s or `~SCF`
        Data cube. Or a `~SCF` class can be passed which may be pre-computed.
    size : int, optional
        Maximum size roll, in pixels, over which SCF will be calculated. If
        the angular scale is different between the data cubes, the lags are
        scaled to have the same angular scales.
    boundary : {"continuous", "cut"}
        Treat the boundary as continuous (wrap-around) or cut values
        beyond the edge (i.e., for most observational data). A two element
        list can also be passed for treating the boundaries differently
        between the given cubes.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, size=11, boundary='continuous',
                 show_progress=True):

        if isinstance(cube1, SCF):
            self.scf1 = cube1
            _has_data1 = False
        else:
            dataset1 = input_data(cube1, no_header=False)
            _has_data1 = True

        if isinstance(cube2, SCF):
            self.scf2 = cube2
            _has_data2 = False
        else:
            dataset2 = input_data(cube2, no_header=False)
            _has_data2 = True

        # Create a default set of lags, in pixels
        if size % 2 == 0:
            Warning("Size must be odd. Reducing size to next lowest odd"
                    " number.")
            size = size - 1

        self.size = size
        roll_lags = (np.arange(size) - size // 2) * u.pix

        # Now adjust the lags such they have a common scaling when the datasets
        # are not on a common grid.
        wcs1 = WCS(dataset1[1]) if _has_data1 else self.scf1._wcs
        wcs2 = WCS(dataset2[1]) if _has_data2 else self.scf2._wcs
        scale = common_scale(wcs1, wcs2)

        if scale == 1.0:
            roll_lags1 = roll_lags
            roll_lags2 = roll_lags
        elif scale > 1.0:
            roll_lags1 = scale * roll_lags
            roll_lags2 = roll_lags
        else:
            roll_lags1 = roll_lags
            roll_lags2 = roll_lags / float(scale)

        if not isinstance(boundary, string_types):
            if len(boundary) != 2:
                raise ValueError("If boundary is not a string, it must be a "
                                 "list or array of 2 string elements.")
        else:
            boundary = [boundary, boundary]

        # if fiducial_model is not None:
        #     self.scf1 = fiducial_model
        if _has_data1:
            self.scf1 = SCF(cube1, roll_lags=roll_lags1)
            needs_run = True
        else:
            needs_run = False
            lag_check = (roll_lags1 == self.scf1.roll_lags).all()
            compute_check = hasattr(self.scf1, "_scf_spectrum")
            if not lag_check:
                warn("SCF given as cube1 needs to be recomputed as the lags"
                     " must match the common set of lags between the two data"
                     " sets. Recomputing SCF.")
                needs_run = True
                self.scf1.roll_lags = roll_lags1

            if not compute_check:
                warn("SCF given as cube1 does not have an SCF"
                     " spectrum computed. Recomputing SCF.")
                needs_run = True

        if needs_run:
            self.scf1.compute_surface(boundary=boundary[0],
                                      show_progress=show_progress)
            # This is for the plot, not the distance, so stick with default
            # params
            self.scf1.compute_spectrum()

        if _has_data2:
            self.scf2 = SCF(cube2, roll_lags=roll_lags2)
            needs_run = True
        else:
            needs_run = False
            lag_check = (roll_lags2 == self.scf2.roll_lags).all()
            compute_check = hasattr(self.scf2, "_scf_spectrum")
            if not lag_check:
                warn("SCF given as cube2 needs to be recomputed as the lags"
                     " must match the common set of lags between the two data"
                     " sets. Recomputing SCF.")
                needs_run = True
                self.scf2.roll_lags = roll_lags2

            if not compute_check:
                warn("SCF given as cube2 does not have an SCF"
                     " spectrum computed. Recomputing SCF.")
                needs_run = True

        if needs_run:
            self.scf2.compute_surface(boundary=boundary[1],
                                      show_progress=show_progress)
            # This is for the plot, not the distance, so stick with default
            # params
            self.scf2.compute_spectrum()

    def distance_metric(self, weighted=True, verbose=False,
                        plot_kwargs1={'color': 'b', 'marker': 'D',
                                      'label': '1'},
                        plot_kwargs2={'color': 'g', 'marker': 'o',
                                      'label': '2'},
                        xunit=u.pix, save_name=None):
        '''
        Compute the distance between the surfaces.

        Parameters
        ----------
        weighted : bool, optional
            Sets whether to apply the 1/r weighting to the distance.
        verbose : bool, optional
            Enables plotting.
        plot_kwargs1 : dict, optional
            Pass kwargs to `~matplotlib.pyplot.plot` for
            `cube1`.
        plot_kwargs2 : dict, optional
            Pass kwargs to `~matplotlib.pyplot.plot` for
            `cube2`.
        xunit : `~astropy.units.Unit`, optional
            Unit of the x-axis in the plot in pixel, angular, or
            physical units.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Since the angular scales are matched, we can assume that they will
        # have the same weights. So just use the shape of the lags to create
        # the weight surface.
        dx = np.arange(self.size) - self.size // 2
        dy = np.arange(self.size) - self.size // 2

        a, b = np.meshgrid(dx, dy)
        if weighted:
            dist_weight = 1 / np.sqrt(a ** 2 + b ** 2)
            # Centre pixel set to 1
            dist_weight[np.where((a == 0) & (b == 0))] = 1.
        else:
            dist_weight = np.ones((self.size, self.size))

        difference = (self.scf1.scf_surface - self.scf2.scf_surface) ** 2. * \
            dist_weight
        self.distance = np.sqrt(np.sum(difference) / np.sum(dist_weight))

        if verbose:
            import matplotlib.pyplot as plt

            defaults1 = {'color': 'b', 'marker': 'D', 'label': '1'}
            defaults2 = {'color': 'g', 'marker': 'o', 'label': '2'}

            for key in defaults1:
                if key not in plot_kwargs1:
                    plot_kwargs1[key] = defaults1[key]
            for key in defaults2:
                if key not in plot_kwargs2:
                    plot_kwargs2[key] = defaults2[key]

            fig = plt.figure()

            ax0 = fig.add_subplot(2, 2, 1)
            ax1 = fig.add_subplot(2, 2, 2, sharex=ax0, sharey=ax0)
            ax2 = fig.add_subplot(2, 2, 3, sharex=ax0, sharey=ax0)
            ax3 = fig.add_subplot(2, 2, 4)

            vmin = min(self.scf1.scf_surface.min(),
                       self.scf2.scf_surface.min())

            im0 = ax0.imshow(self.scf1.scf_surface, origin="lower",
                             interpolation="nearest", vmin=vmin)
            ax0.set_title(plot_kwargs1['label'])
            fig.colorbar(im0, ax=ax0)

            im1 = ax1.imshow(self.scf2.scf_surface, origin="lower",
                             interpolation="nearest", vmin=vmin)
            ax1.set_title(plot_kwargs2['label'])
            fig.colorbar(im1, ax=ax1)

            im2 = ax2.imshow(difference, origin="lower",
                             interpolation="nearest")
            ax2.set_title("")
            fig.colorbar(im2, ax=ax2)

            pix_lags1 = self.scf1._to_pixel(self.scf1.lags)
            lags1 = self.scf1._spatial_unit_conversion(pix_lags1, xunit).value

            pix_lags2 = self.scf2._to_pixel(self.scf2.lags)
            lags2 = self.scf2._spatial_unit_conversion(pix_lags2, xunit).value

            ax3.errorbar(lags1, self.scf1.scf_spectrum,
                         yerr=self.scf1.scf_spectrum_stddev,
                         fmt=plot_kwargs1['marker'],
                         color=plot_kwargs1['color'],
                         markersize=5,
                         label=plot_kwargs1['label'])
            ax3.errorbar(lags2, self.scf2.scf_spectrum,
                         yerr=self.scf2.scf_spectrum_stddev,
                         fmt=plot_kwargs2['marker'],
                         color=plot_kwargs2['color'],
                         markersize=5,
                         label=plot_kwargs2['label'])
            ax3.set_xscale("log")
            ax3.set_yscale("log")

            ax3.set_xlim(min(lags1.min(), lags2.min()) * 0.75,
                         max(lags1.max(), lags2.max()) * 1.25)
            ax3.set_ylim(min(self.scf1.scf_spectrum.min(),
                             self.scf2.scf_spectrum.min()) * 0.75,
                         max(self.scf1.scf_spectrum.max(),
                             self.scf2.scf_spectrum.max()) * 1.25)

            ax3.grid(True)

            ax3.set_xlabel("Lags ({})".format(xunit))

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
