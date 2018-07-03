# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.extern.six import string_types
import statsmodels.api as sm
from warnings import warn

from ..psds import pspec, make_radial_arrays
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data
from ..stats_utils import common_scale, fourier_shift, pixel_shift
from ..fitting_utils import clip_func
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

    Example
    -------
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
        if not self._stddev_flag:
            Warning("scf_spectrum_stddev is only calculated when return_stddev"
                    " is enabled.")
        return self._scf_spectrum_stddev

    @property
    def lags(self):
        '''
        Values of the lags, in pixels, to compute SCF at
        '''
        return self._lags

    def compute_surface(self, boundary='continuous'):
        '''
        Computes the SCF up to the given lag value.

        Parameters
        ----------
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        '''

        if boundary not in ["continuous", "cut"]:
            raise ValueError("boundary must be 'continuous' or 'cut'.")

        self._scf_surface = np.zeros((self.size, self.size))

        # Convert the lags into pixel units.
        pix_lags = self._to_pixel(self.roll_lags).value

        dx = pix_lags.copy()
        dy = pix_lags.copy()

        for i, x_shift in enumerate(dx):
            for j, y_shift in enumerate(dy):

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
                self._scf_surface[j, i] = scf_value

    def compute_spectrum(self, return_stddev=True,
                         **kwargs):
        '''
        Compute the 1D spectrum as a function of lag. Can optionally
        use log-spaced bins. kwargs are passed into the pspec function,
        which provides many options. The default settings are applicable in
        nearly all use cases.

        Parameters
        ----------
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins. Default is True.
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

        out = pspec(self.scf_surface, return_stddev=return_stddev,
                    logspacing=False, return_freqs=False, **kwargs)

        self._stddev_flag = return_stddev
        self._azim_constraint_flag = azim_constraint_flag

        if return_stddev and azim_constraint_flag:
            self._lags, self._scf_spectrum, self._scf_spectrum_stddev, \
                self._azim_mask = out
        elif return_stddev:
            self._lags, self._scf_spectrum, self._scf_spectrum_stddev = out
        elif azim_constraint_flag:
            self._lags, self._scf_spectrum, self._azim_mask = out
        else:
            self._lags, self._scf_spectrum = out

        roll_lag_diff = np.abs(self.roll_lags[1] - self.roll_lags[0])

        self._lags = self._lags * roll_lag_diff

    def fit_plaw(self, xlow=None, xhigh=None, verbose=False):
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
        if self._stddev_flag:

            # Converting to the log stds doesn't matter since the weights
            # remain proportional to 1/sigma^2, and an overal normalization is
            # applied in the fitting routine.
            weights = self.scf_spectrum_stddev[within_limits] ** -2

            model = sm.WLS(y, x, missing='drop', weights=weights)
        else:
            model = sm.OLS(y, x, missing='drop')

        self.fit = model.fit()

        if verbose:
            print(self.fit.summary())

        self._slope = self.fit.params[1]
        self._slope_err = self.fit.bse[1]

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
                 show_surface=True, contour_color='k',
                 cmap='viridis', data_color='k', fit_color=None,
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

        if show_surface:

            plt.subplot(1, 2, 1)

            plt.imshow(self.scf_surface, origin="lower",
                       interpolation="nearest",
                       cmap=cmap)
            cb = plt.colorbar()
            cb.set_label("SCF Value")

            yy, xx = make_radial_arrays(self.scf_surface.shape)

            pix_lag_diff = np.diff(self._to_pixel(self.lags))[0].value
            dists = np.sqrt(yy**2 + xx**2) * pix_lag_diff

            xlow_pix = self._to_pixel(self.xlow).value
            xhigh_pix = self._to_pixel(self.xhigh).value

            mask = clip_func(dists, xlow_pix, xhigh_pix)

            if not mask.all():
                plt.contour(mask, colors='b', linestyles='-.', levels=[0.5])

            if self._fit2D_flag:

                plt.contour(self.fit2D(xx, yy)[::-1], colors=contour_color,
                            linestyles='-')

            if self._azim_constraint_flag:
                if not np.all(self._azim_mask):
                    plt.contour(self._azim_mask, colors='b', linestyles='-.',
                                levels=[0.5])
                else:
                    warn("Azimuthal mask includes all data. No contours will "
                         "be drawn.")

            if show_radial:
                plt.subplot(2, 2, 2)
            else:
                plt.subplot(1, 2, 2)

            plt.hist(self.scf_surface.ravel(), bins='auto')
            plt.xlabel("SCF Value")

        if show_radial:
            if show_surface:
                ax = plt.subplot(2, 2, 4)
            else:
                ax = plt.subplot(1, 1, 1)

            pix_lags = self._to_pixel(self.lags)
            lags = self._spatial_unit_conversion(pix_lags, xunit).value

            if self._stddev_flag:
                ax.errorbar(lags, self.scf_spectrum,
                            yerr=self.scf_spectrum_stddev,
                            fmt='D', color=data_color,
                            markersize=5, label="Data")
                ax.set_xscale("log", nonposy='clip')
                ax.set_yscale("log", nonposy='clip')
            else:
                plt.loglog(self.lags, self.scf_spectrum, 'D',
                           markersize=5, label="Data",
                           color=data_color)

            ax.set_xlim(lags.min() * 0.75, lags.max() * 1.25)
            ax.set_ylim(np.nanmin(self.scf_spectrum) * 0.75,
                        np.nanmax(self.scf_spectrum) * 1.25)

            # Overlay the fit. Use points 5% lower than the min and max.
            xvals = np.linspace(lags.min() * 0.95,
                                lags.max() * 1.05, 50) * xunit
            plt.loglog(xvals, self.fitted_model(xvals), '--', linewidth=2,
                       label='Fit', color=fit_color)
            # Show the fit limits
            xlow = self._spatial_unit_conversion(self._xlow, xunit).value
            xhigh = self._spatial_unit_conversion(self._xhigh, xunit).value
            plt.axvline(xlow, color='b', alpha=0.5, linestyle='-.')
            plt.axvline(xhigh, color='b', alpha=0.5, linestyle='-.')

            plt.legend()

            ax.set_xlabel("Lag ({})".format(xunit))

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, return_stddev=True, boundary='continuous',
            xlow=None, xhigh=None, save_results=False, output_name=None,
            fit_2D=True, fit_2D_kwargs={}, radialavg_kwargs={},
            verbose=False, xunit=u.pix, save_name=None):
        '''
        Computes all SCF outputs.

        Parameters
        ----------
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        xlow : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        xhigh : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        save_results : bool, optional
            Pickle the results.
        output_name : str, optional
            Name of the outputted pickle file.
        fit_2D : bool, optional
            Fit an elliptical power-law model to the 2D spectrum.
        fit_2D_kwargs : dict, optional
            Keyword arguments for `SCF.fit_plaw`. Use the
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

        self.compute_surface(boundary=boundary)
        self.compute_spectrum(return_stddev=return_stddev,
                              **radialavg_kwargs)
        self.fit_plaw(verbose=verbose, xlow=xlow, xhigh=xhigh)

        if fit_2D:
            self.fit_2Dplaw(xlow=xlow, xhigh=xhigh,
                            **fit_2D_kwargs)

        if save_results:
            self.save_results(output_name=output_name)

        if verbose:
            self.plot_fit(save_name=save_name, xunit=xunit)

        return self


class SCF_Distance(object):

    '''
    Calculates the distance between two data cubes based on their SCF surfaces.
    The distance is the L2 norm between the surfaces. We weight the surface by
    1/r^2 where r is the distance from the centre.

    Parameters
    ----------
    cube1 : %(dtypes)s
        Data cube.
    cube2 : %(dtypes)s
        Data cube.
    size : `~astropy.units.Quantity`, optional
        Maximum size roll over which SCF will be calculated.
    boundary : {"continuous", "cut"}
        Treat the boundary as continuous (wrap-around) or cut values
        beyond the edge (i.e., for most observational data). A two element
        list can also be passed for treating the boundaries differently
        between the given cubes.
    fiducial_model : SCF
        Computed SCF object. Use to avoid recomputing.
    weighted : bool, optional
        Sets whether to apply the 1/r^2 weighting to the distance.
    phys_distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, size=21 * u.pix, boundary='continuous',
                 fiducial_model=None, weighted=True, phys_distance=None):
        super(SCF_Distance, self).__init__()
        self.weighted = weighted

        dataset1 = input_data(cube1, no_header=False)
        dataset2 = input_data(cube2, no_header=False)

        # Create a default set of lags, in pixels
        if size % 2 == 0:
            Warning("Size must be odd. Reducing size to next lowest odd"
                    " number.")
            size = size - 1

        self.size = size
        roll_lags = np.arange(size) - size // 2

        # Now adjust the lags such they have a common scaling when the datasets
        # are not on a common grid.
        scale = common_scale(WCS(dataset1[1]), WCS(dataset2[1]))

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

        if fiducial_model is not None:
            self.scf1 = fiducial_model
        else:
            self.scf1 = SCF(cube1, roll_lags=roll_lags1,
                            distance=phys_distance)
            self.scf1.run(return_stddev=True, boundary=boundary[0],
                          fit_2D=False)

        self.scf2 = SCF(cube2, roll_lags=roll_lags2,
                        distance=phys_distance)
        self.scf2.run(return_stddev=True, boundary=boundary[1],
                      fit_2D=False)

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        ang_units=False, unit=u.deg, save_name=None):
        '''
        Compute the distance between the surfaces.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Since the angular scales are matched, we can assume that they will
        # have the same weights. So just use the shape of the lags to create
        # the weight surface.
        dx = np.arange(self.size) - self.size // 2
        dy = np.arange(self.size) - self.size // 2

        a, b = np.meshgrid(dx, dy)
        if self.weighted:
            # Centre pixel set to 1
            a[np.where(a == 0)] = 1.
            b[np.where(b == 0)] = 1.
            dist_weight = 1 / np.sqrt(a ** 2 + b ** 2)
        else:
            dist_weight = np.ones((self.size, self.size))

        difference = (self.scf1.scf_surface - self.scf2.scf_surface) ** 2. * \
            dist_weight
        self.distance = np.sqrt(np.sum(difference) / np.sum(dist_weight))

        if verbose:
            import matplotlib.pyplot as p

            # print "Distance: %s" % (self.distance)

            p.subplot(2, 2, 1)
            p.imshow(
                self.scf1.scf_surface, origin="lower", interpolation="nearest")
            p.title(label1)
            p.colorbar()
            p.subplot(2, 2, 2)
            p.imshow(
                self.scf2.scf_surface, origin="lower", interpolation="nearest",
                label=label2)
            p.title(label2)
            p.colorbar()
            p.subplot(2, 2, 3)
            p.imshow(difference, origin="lower", interpolation="nearest")
            p.title("Weighted Difference")
            p.colorbar()
            ax = p.subplot(2, 2, 4)
            if ang_units:
                lags1 = \
                    self.scf1.lags.to(unit, self.scf1.angular_equiv).value
                lags2 = \
                    self.scf2.lags.to(unit, self.scf2.angular_equiv).value
            else:
                lags1 = self.scf1.lags.value
                lags2 = self.scf2.lags.value

            ax.errorbar(lags1, self.scf1.scf_spectrum,
                        yerr=self.scf1.scf_spectrum_stddev,
                        fmt='D', color='b', markersize=5, label=label1)
            ax.errorbar(lags2, self.scf2.scf_spectrum,
                        yerr=self.scf2.scf_spectrum_stddev,
                        fmt='o', color='g', markersize=5, label=label2)
            ax.set_xscale("log", nonposy='clip')
            ax.set_yscale("log", nonposy='clip')

            ax.set_xlim(min(lags1.min(), lags2.min()) * 0.75,
                        max(lags1.max(), lags2.max()) * 1.25)
            ax.set_ylim(min(self.scf1.scf_spectrum.min(),
                            self.scf2.scf_spectrum.min()) * 0.75,
                        max(self.scf1.scf_spectrum.max(),
                            self.scf2.scf_spectrum.max()) * 1.25)

            # Overlay the fit. Use points 5% lower than the min and max.
            xvals = np.linspace(np.log10(min(lags1.min(),
                                             lags2.min()) * 0.95),
                                np.log10(max(lags1.max(),
                                             lags2.max()) * 1.05), 50)
            p.plot(10**xvals, 10**self.scf1.fitted_model(xvals), 'b--',
                   linewidth=2)
            p.plot(10**xvals, 10**self.scf2.fitted_model(xvals), 'g--',
                   linewidth=2)
            ax.legend(loc='upper right')
            p.tight_layout()

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()
        return self
