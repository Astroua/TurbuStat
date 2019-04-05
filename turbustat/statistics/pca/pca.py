# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import astropy.units as u
from warnings import warn

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data, find_beam_width

# PCA utilities
from ..threeD_to_twoD import var_cov_cube
from .width_estimate import WidthEstimate1D, WidthEstimate2D

# Fitting utilities
from ..fitting_utils import bayes_linear, leastsq_linear


class PCA(BaseStatisticMixIn):

    '''
    Implementation of Principal Component Analysis (Heyer & Brunt, 2002)

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    n_eigs : int
        Deprecated. Input using `~PCA.compute_pca` or `~PCA.run`.
    distance : `~astropy.units.Quantity`, optional
        Distance to object in physical units. The output spatial widths will
        be converted to the units given here.

    Examples
    --------
    >>> from turbustat.statistics import PCA
    >>> from spectral_cube import SpectralCube
    >>> import astropy.units as u
    >>> cube = SpectralCube.read("adv.fits") # doctest: +SKIP
    >>> pca = PCA(cube, distance=250 * u.pc) # doctest: +SKIP
    >>> pca.run(verbose=True) # doctest: +SKIP

    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, n_eigs=None, distance=None):
        super(PCA, self).__init__()

        self.data, self.header = input_data(cube)

        _enforce_velocity_axis(self)

        # We need to check for completely empty channels. These cause
        # issues for the decomposition of the covariance matrix (eigenvectors
        # will have significant imaginary components).
        # Now doing this on a per-channel basis in var_cov_cube
        # self._data[np.isnan(self.data)] = np.finfo(self.data.dtype).eps

        self.spectral_shape = self.data.shape[0]

        if n_eigs is not None:
            raise DeprecationWarning("Input of n_eigs is deprecated. Use "
                                     "inputs in `compute_pca` or `run`.")

        if distance is not None:
            self.distance = distance

    @property
    def n_eigs(self):
        return self._n_eigs

    @n_eigs.setter
    def n_eigs(self, value):
        if value <= 0:
            raise ValueError("n_eigs must be > 0.")

        self._n_eigs = value

    def compute_pca(self, mean_sub=False, n_eigs='auto', min_eigval=None,
                    eigen_cut_method='value', show_progress=True):
        '''
        Create the covariance matrix and its eigenvalues.

        If `mean_sub` is disabled, the first eigenvalue is dominated by the
        mean of the data, not the variance.

        Parameters
        ----------
        mean_sub : bool, optional
            When enabled, subtracts the means of the channels before
            calculating the covariance. By default, this is disabled to
            match the Heyer & Brunt method.
        n_eigs : {int, 'auto'}, optional
            Number of eigenvalues to compute. The default setting is 'auto',
            which requires `min_eigval` to be set. Otherwise, the number of
            eigenvalues used can be set using an int. Setting to -1 will use
            all of the eigenvalues.
        min_eigval : float, optional
            The cut-off value to determine the number of important eigenvalues.
            When `eigen_cut_method` is `proportional`, min_eigval is the
            total proportion of variance described up to the Nth eigenvalue.
            When `eigen_cut_method` is `value`, min_eigval is the minimum
            variance described by that eigenvalue.
        eigen_cut_method : {'proportion', 'value'}, optional
            Set whether `min_eigval` is the proportion of variance determined
            up to the Nth eigenvalue (`proportion`) or the minimum value of
            variance (`value`).
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        '''

        # Define the decomposition-only flag, if not yet set. Will get
        # overridden later if the size-line width relation is fit.
        if not hasattr(self, "_decomp_only"):
            self._decomp_only = True
        else:
            self._decomp_only = True

        if n_eigs == 'auto' and min_eigval is None:
            raise ValueError("min_eigval must be given when using "
                             "n_eigs='auto'.")

        self.cov_matrix = var_cov_cube(self.data, mean_sub=mean_sub,
                                       progress_bar=show_progress)

        all_eigsvals, eigvecs = np.linalg.eigh(self.cov_matrix)
        all_eigsvals = np.real_if_close(all_eigsvals)
        eigvecs = eigvecs[:, np.argsort(all_eigsvals)[::-1]]
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        if n_eigs == 'auto':
            self.n_eigs = set_n_eigs(all_eigsvals, min_eigval,
                                     method=eigen_cut_method)
        elif n_eigs == -1:
            self.n_eigs = self.spectral_shape
        elif n_eigs < -1 or n_eigs > self.spectral_shape or n_eigs == 0:
            raise Warning("n_eigs must be less than the number of velocity"
                          " channels ({}) or -1 for"
                          " all".format(self.spectral_shape))
        else:
            self.n_eigs = n_eigs

        if mean_sub:
            self._total_variance = np.sum(all_eigsvals)
            self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
                self.total_variance
        else:
            self._total_variance = np.sum(all_eigsvals[1:])
            self._var_prop = np.sum(all_eigsvals[1:self.n_eigs]) / \
                self.total_variance

        self._eigvals = all_eigsvals
        self._eigvecs = eigvecs

        self._mean_sub = mean_sub

    @property
    def var_proportion(self):
        '''
        Proportion of variance described by the first `~PCA.n_eigs`
        eigenvalues.
        '''
        return self._var_prop

    @property
    def total_variance(self):
        '''
        Total variance of all eigenvalues.
        '''
        return self._total_variance

    @property
    def eigvals(self):
        '''
        All eigenvalues.
        '''
        return self._eigvals

    @property
    def eigvecs(self):
        '''
        All eigenvectors.
        '''
        return self._eigvecs

    def _valid_eigenvectors(self):
        '''
        Find the indices where the eigenvalues are above the machine precision
        limit of self.data's dtype. This stops us from running into
        eigenvectors with significant imaginary components (which we expect to
        get for empty channels).
        '''

        return np.where(self.eigvals >= np.finfo(self.data.dtype).eps)[0]

    def eigimages(self, n_eigs=None):
        '''
        Create eigenimages up to the n_eigs.

        Parameters
        ----------
        n_eigs : None or int
            The number of eigenimages to create. When n_eigs is negative, the
            last -n_eig eigenimages are created. If None is given, the number
            in `~PCA.n_eigs` will be returned.

        Returns
        -------
        eigimgs : `~numpy.ndarray`
            3D array, where the first dimension if the number of eigenvalues.
        '''

        if n_eigs is None:
            n_eigs = self.n_eigs

        if n_eigs > 0:
            iterat = range(n_eigs)
        elif n_eigs < 0:
            # We're looking for the noisy components whenever n_eigs < 0
            # Find where we have valid eigenvalues, and use the last
            # n_eigs of those.
            iterat = self._valid_eigenvectors()[n_eigs:]

        for ct, idx in enumerate(iterat):
            eigimg = np.zeros(self.data.shape[1:], dtype=float)
            for channel in range(self.data.shape[0]):
                if self._mean_sub:
                    mean_value = np.nanmean(self.data[channel])
                    eigimg += np.nan_to_num((self.data[channel] - mean_value) *
                                            np.real_if_close(
                                                self.eigvecs[channel, idx]))
                else:
                    eigimg += np.nan_to_num(self.data[channel] *
                                            np.real_if_close(
                                                self.eigvecs[channel, idx]))
            if ct == 0:
                eigimgs = eigimg
            else:
                eigimgs = np.dstack((eigimgs, eigimg))

        if eigimgs.ndim == 3:
            return eigimgs.swapaxes(0, 2)
        else:
            return eigimgs

    def autocorr_images(self, n_eigs=None):
        '''
        Create the autocorrelation of the eigenimages.

        Parameters
        ----------
        n_eigs : None or int
            The number of autocorrelation images to create. When n_eigs is
            negative, the last -n_eig autocorrelation images are created.
            If None is given, the number in `~PCA.n_eigs` will be returned.

        Returns
        -------
        acors : np.ndarray
            3D array, where the first dimension if the number of eigenvalues.
        '''

        if n_eigs is None:
            n_eigs = self.n_eigs

        # Calculate the eigenimages
        eigimgs = self.eigimages(n_eigs=n_eigs)

        for idx, image in enumerate(eigimgs):
            fftx = np.fft.fft2(image)
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft2((fftx - fftx.mean()) * (fftxs - fftxs.mean()))
            acor = np.fft.fftshift(acor)

            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        if acors.ndim == 3:
            return acors.swapaxes(0, 2)
        else:
            return acors

    def autocorr_spec(self, n_eigs=None):
        '''
        Create the autocorrelation spectra of the eigenvectors.

        Parameters
        ----------
        n_eigs : None or int
            The number of autocorrelation vectors to create. When n_eigs is
            negative, the last -n_eig autocorrelation vectors are created.
            If None is given, the number in `~PCA.n_eigs` will be returned.

        Returns
        -------
        acors : np.ndarray
            2D array, where the first dimension if the number of eigenvalues.
        '''
        if n_eigs is None:
            n_eigs = self.n_eigs

        for idx in range(n_eigs):
            fftx = np.fft.fft(self.eigvecs[:, idx])
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft((fftx - fftx.mean()) * (fftxs - fftxs.mean()))
            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        return acors.swapaxes(0, 1).squeeze()

    def noise_ACF(self, n_eigs=-10):
        '''
        Create the noise autocorrelation function based off of the eigenvalues
        beyond `PCA.n_eigs`. By default the final 10 eigenvectors **whose
        eigenvalues are above the machine precision limit of the data cube's
        dtype** are used.

        Parameters
        ----------
        n_eigs : int, optional
            The number of eigenvalues to use for estimating the noise ACF.
            The default is to use the last 10 eigenvectors.
        '''

        if n_eigs is None:
            n_eigs = self.n_eigs

        acors = self.autocorr_images(n_eigs=n_eigs)

        noise_ACF = np.nansum(acors, axis=0) / float(n_eigs)

        return noise_ACF

    def find_spatial_widths(self, method='contour',
                            brunt_beamcorrect=True, beam_fwhm=None,
                            distance=None,
                            diagnosticplots=False, **fit_kwargs):
        '''
        Derive the spatial widths using the autocorrelation of the
        eigenimages.

        Parameters
        ----------
        methods : {'contour', 'fit', 'interpolate', 'xinterpolate'}, optional
            Spatial fitting method to use. The default method is 'contour'
            (fits an ellipse to the 1/e contour about the peak; this is the
            method used by the Heyer & Brunt works).
            See `~turbustat.statistics.pca.WidthEstimate2D` for a description
            of all methods.
        brunt_beamcorrect : bool, optional
            Apply the beam correction described in Chris Brunt's
            `thesis <http://search.proquest.com/docview/304529913>`_. A beam
            will be searched for in the given header (looking for "BMAJ"). If
            this fails, the value must be given in `beam_fwhm` with angular
            units.
        beam_fwhm : None of `~astropy.units.Quantity`, optional
            When beam correction is enabled, the FWHM beam size can be given
            here.
        distance : `~astropy.units.Quantity`, optional
            Distance to object in physical units. The output spatial widths
            will be converted to the units given here.
        diagnosticplots : bool, optional
            Plot the first 9 autocorrelation images with the contour fits.
            *Only implemented for* `method='contour'`.
        fit_kwargs : dict, optional
            Used when method is 'contour'. Passed to
            `turbustat.statistics.stats_utils.EllipseModel.estimate_stderrs`.

        '''
        # Try reading beam width from the header is it is not given.
        if brunt_beamcorrect:
            if beam_fwhm is None:
                beam_fwhm = find_beam_width(self.header)

        acors = self.autocorr_images(n_eigs=self.n_eigs)
        noise_ACF = self.noise_ACF()

        self._spatial_width, self._spatial_width_error = \
            WidthEstimate2D(acors, noise_ACF=noise_ACF, method=method,
                            brunt_beamcorrect=brunt_beamcorrect,
                            beam_fwhm=beam_fwhm,
                            spatial_cdelt=self._ang_size,
                            diagnosticplots=diagnosticplots,
                            **fit_kwargs)

        self._spatial_width = self._spatial_width * u.pix
        self._spatial_width_error = self._spatial_width_error * u.pix

    def spatial_width(self, unit=u.pix):
        '''
        Spatial widths for the first `~PCA.n_eigs` components.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spatial unit to convert the widths to. Can be in pixels,
            an angular unit, or (if distance is given) a physical unit.
        '''
        return self._spatial_unit_conversion(self._spatial_width, unit)

    def spatial_width_error(self, unit=u.pix):
        '''
        The 1-sigma error bounds on the spatial widths for the first
        `~PCA.n_eigs` components.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spatial unit to convert the widths to. Can be in pixels,
            an angular unit, or (if distance is given) a physical unit.
        '''

        return self._spatial_unit_conversion(self._spatial_width_error, unit)

    def find_spectral_widths(self, method='walk-down'):
        '''
        Derive the spectral widths using the autocorrelation of the
        eigenvectors.

        Parameters
        ----------
        method : {"walk-down", "fit", "interpolate"}, optional
            Spectral fitting method to use. The default method is 'walk-down'
            (starting at the peak, continue until reaching 1/e of the peak;
            this is the method used by the Heyer & Brunt works). See
            `~turbustat.statistics.pca.WidthEstimate1D` for a description
            of all methods.
        '''

        acorr_spec = self.autocorr_spec(n_eigs=self.n_eigs)

        self._spectral_width, self._spectral_width_error = \
            WidthEstimate1D(acorr_spec, method=method)

        self._spectral_width = self._spectral_width * u.pix
        self._spectral_width_error = self._spectral_width_error * u.pix

    def spectral_width(self, unit=u.pix):
        '''
        Spectral widths for the first `~PCA.n_eigs` components.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spectral unit to convert the widths to. Can be in pixels,
            or a spectral unit equivalent to the unit specified in the
            `~PCA.header`.
        '''

        return self._to_spectral(self._spectral_width, unit)

    def spectral_width_error(self, unit=u.pix):
        '''
        The error bounds on the spectral widths for the first
        `~PCA.n_eigs` components.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spectral unit to convert the widths to. Can be in pixels,
            or a spectral unit equivalent to the unit specified in the
            `~PCA.header`.
        '''

        return self._to_spectral(self._spectral_width_error, unit)

    def fit_plaw(self, xlow=None, xhigh=None, fit_method='odr',
                 verbose=False, **kwargs):
        '''
        Fit the size-linewidth relation. This is done through Orthogonal
        Distance Regression (via scipy), or through MCMC (requires
        installing `emcee <http://dan.iel.fm/emcee/current/>`_).

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower spatial scale limit to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper spatial scale value limit to consider in the fit.
        fit_method : {'odr', 'bayes'}, optional
            Set the type of fitting to perform. Options are 'odr'
            (orthogonal distance regression) or 'bayes' (MCMC). Note that
            'bayes' requires the emcee package to be installed.
        verbose : bool, optional
            Prints out additional information about the fitting and plots the
            solution.
        **kwargs
            Passed to `~turbustat.statistics.fitting_utils.bayes_linear`
            when fit_method is bayes.
        '''

        # Set the decomposition flag to False if not defined yet
        if not hasattr(self, "_decomp_only"):
            self._decomp_only = False
        else:
            self._decomp_only = False

        if fit_method != 'odr' and fit_method != 'bayes':
            raise TypeError("fit_method must be 'odr' or 'bayes'.")

        # Hang on to this. We can't propagate the asymmetric errors when using
        # MCMC in sonic_length, and the laziest way is just to keep the samples
        # around, and estimate CIs directly.
        self._fit_method = fit_method

        x = np.log10(self.spatial_width().value)

        # Set limits on scales to consider for the fit
        if xlow is not None:
            if not isinstance(xlow, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")

            # Convert xlow into the same units as the lags
            xlow = self._to_pixel(xlow)

            self._xlow = xlow

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(x, dtype=bool)
            self._xlow = np.nanmin(self.spatial_width())

        if xhigh is not None:
            if not isinstance(xhigh, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")
            # Convert xhigh into the same units as the lags
            xhigh = self._to_pixel(xhigh)

            self._xhigh = xhigh

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(x, dtype=bool)
            self._xhigh = np.nanmax(self.spatial_width())

        within_limits = np.logical_and(lower_limit, upper_limit)

        if not within_limits.any():
            raise ValueError("Limits have removed all lag values. Make xlow"
                             " and xhigh less restrictive.")

        # Only keep the width estimations that worked
        are_finite = np.isfinite(self.spectral_width()) * \
            np.isfinite(self.spatial_width()) * \
            np.isfinite(self.spatial_width_error()) * \
            np.isfinite(self.spectral_width_error())

        fit_mask = np.logical_and(within_limits, are_finite)

        # Check to make sure there are enough points to fit to (minimum 2).
        num_pts = self.spectral_width().size - (~fit_mask).sum()
        if num_pts < 2:
            raise Warning("Less then 2 valid points. Cannot fit model.")
        elif num_pts < 5:
            warn("There are less than 5 points to fit to. The fit will not be"
                 " well constrained and results should be closely examined.")

        y = np.log10(self.spectral_width()[fit_mask].value)
        x = np.log10(self.spatial_width()[fit_mask].value)

        y_err = 0.434 * self.spectral_width_error()[fit_mask].value / \
            self.spectral_width()[fit_mask].value
        x_err = 0.434 * self.spatial_width_error()[fit_mask].value / \
            self.spatial_width()[fit_mask].value

        if fit_method == 'odr':
            params, errors = leastsq_linear(x, y, x_err, y_err,
                                            verbose=verbose)
            # Turn into +/- range values, as would be returned by the MCMC fit
            errors = np.vstack([params - errors, params + errors]).T
        else:
            params, errors, samps = bayes_linear(x, y, x_err, y_err,
                                                 verbose=verbose,
                                                 return_samples=True, **kwargs)
            self._samps = samps

        self._index = params[0]
        self._index_error_range = errors[0]

        # Take the intercept out of log scale
        self._intercept = 10 ** params[1] * self._spectral_width.unit
        self._intercept_error_range = 10 ** errors[1] * \
            self._spectral_width.unit

    @property
    def index(self):
        '''
        Power-law index.
        '''
        return self._index

    @property
    def index_error_range(self):
        '''
        One-sigma error bounds on the index.
        '''
        return self._index_error_range

    @property
    def gamma(self):
        '''
        Slope of the size-linewidth relation with correction from
        `Brunt & Heyer 2002 <https://ui.adsabs.harvard.edu/#abs/2002ApJ...566..276B/abstract>`_
        '''
        return float(brunt_index_correct(self.index))

    @property
    def gamma_error_range(self):
        '''
        One-sigma error bounds on gamma.
        '''
        return brunt_index_correct_range(*self.index_error_range)

    def intercept(self, unit=u.pix):
        '''
        Intercept from the fits, converted out of the log value.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spectral unit to convert the widths to. Can be in pixels,
            or a spectral unit equivalent to the unit specified in the
            `~PCA.header`.
        '''
        return self._to_spectral(self._intercept, unit)

    def intercept_error_range(self, unit=u.pix):
        '''
        One-sigma error bounds on the intercept.

        Parameters
        ----------
        unit : `~astropy.units.Unit`, optional
            The spectral unit to convert the widths to. Can be in pixels,
            or a spectral unit equivalent to the unit specified in the
            `~PCA.header`.
        '''
        return self._to_spectral(self._intercept_error_range, unit)

    def model(self, x):
        '''
        Model with the fit parameters from `~PCA.fit_plaw`
        '''
        return self.index * x + np.log10(self.intercept)

    def sonic_length(self, T_k=10 * u.K, mu=1.36, use_gamma=True,
                     unit=u.pix):
        '''
        Estimate of the sonic length based on a given temperature. Uses the
        intercept from the fit.

        Based on sonic.pro used in the Heyer & Brunt PCA implementation.

        Because error from the MCMC fit need not be symmetric, the MCMC
        samples are needed to provide the correct CIs for the sonic length.

        Parameters
        ----------
        T_k : `~astropy.units.Quantity`, optional
            Temperature given in units convertible to Kelvin.
        mu : float, optional
            Factor to multiply by m_H to account for He and metals.
        use_gamma : bool, optional
            Toggle whether to use `~PCA.gamma` or `~PCA.index`. See link given
            in `~PCA.gamma`.

        Returns
        -------
        lambd : `~astropy.units.Quantity`
            Value of the sonic length. If distance was provided, this will
            be in the units given in the distance. Otherwise, the result will
            be in pixel units.
        lambd_error_range : `~astropy.units.Quantity`
            The 1-sigma bounds on the sonic length. The units will match lambd.
        unit : `~astropy.units.Unit`, optional
            The spatial unit to convert the widths to. Can be in pixels,
            an angular unit, or (if distance is given) a physical unit.
        '''

        import astropy.constants as const

        try:
            T_k = T_k.to(u.K)
        except u.UnitConversionError:
            raise u.UnitConversionError("Cannot convert T_k to Kelvin.")

        # Sound speed in m/s
        c_s = np.sqrt(const.k_B.decompose() * T_k / (mu * const.m_p))
        # Convert to the same spectral unit
        c_s = self._to_spectral(c_s, u.pix)

        if use_gamma:
            index = self.gamma
            index_error_range = self.gamma_error_range
        else:
            index = self.index
            index_error_range = self.index_error_range

        lambd = np.power(c_s / self.intercept(), 1. / index).value

        if self._fit_method == 'odr':
            # Added in quadrature and simplified
            index_err = np.abs(index - index_error_range[0])
            intercept_err = 0.434 * np.abs(self.intercept() -
                                           self.intercept_error_range()[0]) / \
                self.intercept()

            term1 = np.log10(c_s / self.intercept()) * \
                (index_err / index)
            term2 = intercept_err / self.intercept()
            lambd_error = (lambd / index) * \
                np.sqrt(term1.value**2 + term2.value**2)
            lambd_error_range = np.array([lambd - lambd_error,
                                          lambd + lambd_error])
        else:
            # Don't propagate asymmetric errors in quadrature! Instead,
            # calculate CI directly from the samples.
            percentiles = [15, 85]
            slopes = self._samps[0]
            intercepts = self._samps[1]

            if use_gamma:
                slopes = np.array([brunt_index_correct(slope)
                                   for slope in slopes])

            all_lambds = np.power(c_s / 10 ** intercepts, 1. / index)

            lambd_error_range = np.percentile(all_lambds, percentiles)

        # Convert to specified units.
        lambd = self._spatial_unit_conversion(lambd * u.pix, unit)
        lambd_error_range = \
            self._spatial_unit_conversion(lambd_error_range * u.pix, unit)

        return lambd, lambd_error_range

    def plot_fit(self, save_name=None, show_cov_bar=True, show_sl_fit=True,
                 n_eigs=None, color='r', fit_color='k', symbol='o',
                 cov_cmap='viridis',
                 spatial_unit=u.pix, spectral_unit=u.pix, show_residual=True):
        '''
        Plot the covariance matrix, bar plot of eigenvalues, and the fitted
        size-line width relation.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        show_cov_bar : bool, optional
            Show the covariance matrix and eigenvalue variance bar plot.
        show_sl_fit : bool, optional
            Show the size-line width relation, if fit.
        n_eigs : int, optional
            Number of eigenvalues to show in the bar plot. Defaults to the
            automatically-set value (`PCA.n_eigs`).
        color : {str, RGB tuple}, optional
            Color to use in the plots. Defaults to red.
        fit_color : {str, RBG tuple}, optional
            Colour to show the fit line in. Defaults to `color` when `None` is
            given.
        symbol : str, optional
            Marker shape to plot the data.
        cov_cmap : {str, matplotlib colormap}, optional
            Colormap to show the covariance matrix in.
        show_residual : bool, optional
            Plot the fit residuals.
        '''

        if self._decomp_only and show_sl_fit:
            warn("Size-line width fit not performed. Disabling show_sl_fit.")
            show_sl_fit = False

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if show_cov_bar:
            if show_sl_fit:
                plt.subplot2grid((4, 4), (0, 0), rowspan=2, colspan=2)

            else:
                plt.subplot2grid((4, 4), (0, 0), rowspan=4, colspan=2)

            im1 = plt.imshow(self.cov_matrix, origin="lower",
                             interpolation="nearest", cmap=cov_cmap)
            divider = make_axes_locatable(plt.gca())
            cax = divider.append_axes("right", "5%", pad="3%")
            cb = plt.colorbar(im1, cax=cax)

            cb.set_label("Covariance")

            if show_sl_fit:
                plt.subplot2grid((4, 4), (2, 0), rowspan=2, colspan=2)
            else:
                plt.subplot2grid((4, 4), (0, 2), rowspan=4, colspan=2)

            if n_eigs is None:
                n_eigs = self.n_eigs

            plt.bar(np.arange(1, self.n_eigs + 1), self.eigvals[:n_eigs],
                    0.5, color=color)
            plt.xlim([0, n_eigs + 1])
            plt.xlabel('Eigenvalues')
            plt.ylabel('Variance')

        if show_sl_fit:
            if fit_color is None:
                fit_color = color

            if show_cov_bar:
                if show_residual:
                    plt.subplot2grid((4, 4), (0, 2), rowspan=3, colspan=2)
                else:
                    plt.subplot2grid((4, 4), (0, 2), rowspan=4, colspan=2)
            else:
                if show_residual:
                    plt.subplot2grid((4, 1), (0, 0), rowspan=3, colspan=1)
                else:
                    plt.subplot(111)

            spatial_width = self.spatial_width(unit=spatial_unit)
            spatial_width_error = self.spatial_width_error(unit=spatial_unit)

            spectral_width = self.spectral_width(unit=spectral_unit)
            spectral_width_error = \
                self.spectral_width_error(unit=spectral_unit)

            plt.errorbar(np.log10(spatial_width.value),
                         np.log10(spectral_width.value),
                         xerr=0.434 * spatial_width_error /
                         spatial_width,
                         yerr=0.434 * spectral_width_error /
                         spectral_width, fmt=symbol, color=color)

            # Show fitting extents
            xlow = self._spatial_unit_conversion(self._xlow,
                                                 spatial_unit).value
            xhigh = self._spatial_unit_conversion(self._xhigh,
                                                  spatial_unit).value
            plt.axvline(np.log10(xlow), color=fit_color,
                        alpha=0.5, linestyle='-.')
            plt.axvline(np.log10(xhigh), color=fit_color,
                        alpha=0.5, linestyle='-.')

            plt.ylabel("log Linewidth / "
                       "{}".format(spectral_width.unit.to_string()))
            plt.grid()

            xvals = np.linspace(np.log10(np.nanmin(spatial_width).value),
                                np.log10(np.nanmax(spatial_width).value),
                                spatial_width.size * 10)

            xvals_pix = \
                np.linspace(np.log10(np.nanmin(self.spatial_width(u.pix).value)),
                            np.log10(np.nanmax(self.spatial_width(u.pix).value)),
                            spatial_width.size * 10)

            intercept = self.intercept(unit=u.pix)

            spec_conv = self._to_spectral(1 * u.pix, spectral_unit).value

            plt.plot(xvals,
                     np.log10(10**(self.index * xvals_pix +
                                   np.log10(intercept.value)) * spec_conv),
                     '-', color=fit_color)

            # Some very large error bars makes it difficult to see the model
            # Limit the range shown in the plot.
            x_range = \
                np.ptp(np.log10(spatial_width.value)
                       [np.isfinite(spatial_width)])
            y_range = \
                np.ptp(np.log10(spectral_width.value)
                       [np.isfinite(spectral_width)])
            plt.xlim([np.log10(np.nanmin(spatial_width.value)) -
                      y_range / 4,
                      np.log10(np.nanmax(spatial_width.value)) +
                      y_range / 4])
            plt.ylim([np.log10(np.nanmin(spectral_width.value)) -
                      x_range / 4,
                      np.log10(np.nanmax(spectral_width.value)) +
                      x_range / 4])

            if show_residual:
                if show_cov_bar:
                    plt.subplot2grid((4, 4), (3, 2), rowspan=1, colspan=2)
                else:
                    plt.subplot2grid((4, 1), (3, 0), rowspan=1, colspan=1)

                x_log_pix = np.log10(self.spatial_width(u.pix).value)
                resids = np.log10(spectral_width.value) - \
                    np.log10(10**(self.index * x_log_pix +
                                  np.log10(intercept.value)) * spec_conv)

                plt.errorbar(np.log10(spatial_width.value), resids,
                             xerr=0.434 * spatial_width_error /
                             spatial_width,
                             yerr=0.434 * spectral_width_error /
                             spectral_width, fmt='o', color=color)
                plt.grid()

                plt.axvline(np.log10(xlow), color=fit_color,
                            alpha=0.5, linestyle='-.')
                plt.axvline(np.log10(xhigh), color=fit_color,
                            alpha=0.5, linestyle='-.')

                plt.axhline(0., color=fit_color)
                plt.ylabel("Residuals")

                plt.xlim([np.log10(np.nanmin(spatial_width.value)) -
                          y_range / 4,
                          np.log10(np.nanmax(spatial_width.value)) +
                          y_range / 4])

            plt.xlabel("log Spatial Length / "
                       "{}".format(spatial_width.unit.to_string()))

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, verbose=False, save_name=None,
            mean_sub=False, decomp_only=False, n_eigs='auto', min_eigval=None,
            eigen_cut_method='value', spatial_method='contour',
            spectral_method='walk-down',
            xlow=None, xhigh=None, fit_method='odr',
            beam_fwhm=None, brunt_beamcorrect=True,
            spatial_output_unit=u.pix, spectral_output_unit=u.pix):
        '''
        Run the decomposition and fitting in one step.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
            Enabled by default.
        verbose : bool, optional
            Enables plotting of the results.
        save_name : str,optional
            Save the figure when a file name is given.
        mean_sub : bool, optional
            See `~PCA.compute_pca`
        decomp_only : bool, optional
            Only run the PCA decomposition, not the entire procedure to derive
            the size-linewidth relation. This should be enabled when using
            PCA_Distance.
        n_eigs : {"auto", int}, optional
            See `~PCA.compute_pca`
        min_eigval : float, optional
            See `~PCA.compute_pca`
        eigen_cut_method : {'proportion', 'value'}, optional
            See `~PCA.compute_pca`
        spatial_method : str, optional
            See `~PCA.fit_spatial_widths`.
        spectral_method : str, optional
            See `~PCA.fit_spectral_widths`.
        xlow : `~astropy.units.Quantity`, optional
            See `~PCA.fit_plaw`.
        xhigh : `~astropy.units.Quantity`, optional
            See `~PCA.fit_plaw`.
        fit_method : str, optional
            See `~PCA.fit_plaw`.
        beam_fwhm : None of `~astropy.units.Quantity`, optional
            See `~PCA.fit_spatial_widths`.
        brunt_beamcorrect : bool, optional
            See `~PCA.fit_spatial_widths`.
        spatial_output_unit : `astropy.units.Unit`, optional
            Pixel, anglular, or physical unit to convert the spatial sizes to
            when plotting. Defaults to pixels. Physical unit conversion
            requires a distance to be given.
        spectral_output_unit : `astropy.units.Unit`, optional
            Pixel or spectral unit to convert spectral sizes to when plotting.
            Defaults to pixels. The spectral unit *MUST* match the spectral
            unit defined in the data cube.
        '''

        # Check if the beam can be loaded. Otherwise, turn off the beam
        # correction before computing the covariance matrix
        if beam_fwhm is None and brunt_beamcorrect and not decomp_only:
            try:
                beam_fwhm = find_beam_width(self.header)
            # Don't check for type. Otherwise I need to check if radio_beam
            # is installed.
            except Exception:
                raise ValueError("Cannot load beam size from the header. "
                                 "Please give the beam FWHM or set "
                                 "`brunt_beamcorrect=False`.")

        self.compute_pca(mean_sub=mean_sub, n_eigs=n_eigs,
                         min_eigval=min_eigval,
                         eigen_cut_method=eigen_cut_method,
                         show_progress=show_progress)

        self._decomp_only = decomp_only

        # Run rest of the analysis
        if not decomp_only:
            self.find_spatial_widths(method=spatial_method,
                                     beam_fwhm=beam_fwhm,
                                     brunt_beamcorrect=brunt_beamcorrect)
            self.find_spectral_widths(method=spectral_method)
            self.fit_plaw(xlow=xlow, xhigh=xhigh,
                          fit_method=fit_method)

        if verbose:

            print('Proportion of Variance kept: %s' % (self.var_proportion))

            if not decomp_only:
                print("Index: {0:.2f} ({1:.2f}, {2:.2f})"
                      .format(self.index, *self.index_error_range))
                print("Gamma: {0:.2f} ({1:.2f}, {2:.2f})"
                      .format(self.gamma, *self.gamma_error_range))

                # Compute sonic length assuming 10 K
                T_k = 10. * u.K
                sl, sl_range = self.sonic_length(T_k=T_k)
                print("Sonic length: {0:.3e} ({4:.3e}, {5:.3e}) {1} at {2} {3}"
                      .format(sl.value, sl.unit.to_string(), T_k.value,
                              T_k.unit.to_string(), *sl_range.value))

            self.plot_fit(save_name=save_name, show_sl_fit=not decomp_only,
                          spatial_unit=spatial_output_unit,
                          spectral_unit=spectral_output_unit)

        return self


class PCA_Distance(object):

    '''
    Compare two data cubes based on the eigenvalues of the PCA decomposition.
    The distance is the Euclidean distance between the eigenvalues.

    Parameters
    ----------
    cube1 : %(dtypes)s or `~PCA`
        Data cube. Or a `~PCA` class can be given which may be pre-computed.
    cube2 : %(dtypes)s or `~PCA`
        Data cube. Or a `~PCA` class can be given which may be pre-computed.
    n_eigs : int
        Number of eigenvalues to compute.
    fiducial_model : PCA
        Computed PCA object. Use to avoid recomputing.
    mean_sub : bool, optional
        Subtracts the mean before computing the covariance matrix. Not
        subtracting the mean is done in the original Heyer & Brunt works.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None,
                 mean_sub=True):
        super(PCA_Distance, self).__init__()

        if n_eigs == 'auto':
            raise ValueError("'auto' n_eigs mode is disabled for distance "
                             "computation. The metric requires having the same"
                             " number of eigenvalues to compare.")

        # if fiducial_model is not None:
        #     self.pca1 = fiducial_model
        if isinstance(cube1, PCA):
            self.pca1 = cube1
            needs_run = False
            # Set the number of eigvals. This is fine b/c we keep
            # all of them, even if they're not used.

            if not hasattr(self.pca1, 'eigvals'):
                needs_run = True
                warn("PCA given as cube1 does not have eigenvalues"
                     " defined. Re-running PCA decomposition.")
            else:
                self.pca1.n_eigs = n_eigs
                if n_eigs >= self.pca1.eigvals.size:
                    raise ValueError("n_eigs exceeds the total number of "
                                     "spectral channel for the class given "
                                     "as `cube1`. Choose a smaller `n_eigs`.")

        else:
            self.pca1 = PCA(cube1)
            needs_run = True

        if needs_run:
            self.pca1.run(mean_sub=mean_sub, n_eigs=n_eigs, decomp_only=True)

        if isinstance(cube2, PCA):
            self.pca2 = cube2
            needs_run = False
            # Set the number of eigvals. This is fine b/c we keep
            # all of them, even if they're not used.

            if not hasattr(self.pca2, 'eigvals'):
                needs_run = True
                warn("PCA given as cube2 does not have eigenvalues"
                     " defined. Re-running PCA decomposition.")
            else:
                self.pca2.n_eigs = n_eigs
                if n_eigs >= self.pca2.eigvals.size:
                    raise ValueError("n_eigs exceeds the total number of "
                                     "spectral channel for the class given "
                                     "as `cube2`. Choose a smaller `n_eigs`.")

        else:
            self.pca2 = PCA(cube2)
            needs_run = True

        if needs_run:
            self.pca2.run(mean_sub=mean_sub, n_eigs=n_eigs, decomp_only=True)

        self._mean_sub = mean_sub
        self._n_eigs = n_eigs

    def distance_metric(self, verbose=False, save_name=None,
                        plot_kwargs1={},
                        plot_kwargs2={},
                        cmap='viridis'):
        '''
        Computes the distance between the cubes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str, optional
            Save the figure when a file name is given.
        plot_kwargs1 : dict, optional
            Set the color, symbol, and label for dataset1
            (e.g., plot_kwargs1={'color': 'b', 'symbol': 'D', 'label': '1'}).
        plot_kwargs2 : dict, optional
            Set the color, symbol, and label for dataset2.
        cmap : str, optional
            The colormap to use when plotting the covariance matrices.
        '''

        # The eigenvalues need to be normalized before being compared. If
        # mean_sub is False, the first eigenvalue is not used.
        if self._mean_sub:
            slicer = slice(0, self._n_eigs)
        else:
            slicer = slice(1, self._n_eigs)

        eigvals1 = self.pca1.eigvals[slicer] / \
            np.sum(self.pca1.eigvals[slicer])
        eigvals2 = self.pca2.eigvals[slicer] / \
            np.sum(self.pca2.eigvals[slicer])

        self.distance = np.linalg.norm(eigvals1 - eigvals2)

        if verbose:
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

            print("Proportions of total variance: 1 - %0.3f, 2 - %0.3f" %
                  (self.pca1.var_proportion, self.pca2.var_proportion))

            plt.subplot(2, 2, 1)
            plt.imshow(self.pca1.cov_matrix,
                       cmap=cmap,
                       origin="lower", interpolation="nearest",
                       vmin=np.median(self.pca1.cov_matrix))
            plt.colorbar()
            plt.title(plot_kwargs1['label'])

            plt.subplot(2, 2, 3)
            plt.bar(np.arange(1, len(eigvals1) + 1), eigvals1, 0.5,
                    color=plot_kwargs1['color'])
            plt.xlim([0, self.pca1.n_eigs + 1])
            plt.xlabel('Eigenvalues')
            plt.ylabel("Proportion of Variance")

            plt.subplot(2, 2, 2)
            plt.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca2.cov_matrix))
            plt.colorbar()
            plt.title(plot_kwargs2['label'])
            plt.subplot(2, 2, 4)
            plt.bar(np.arange(1, len(eigvals2) + 1), eigvals2, 0.5,
                    color=plot_kwargs2['color'])
            plt.xlim([0, self.pca2.n_eigs + 1])
            plt.xlabel('Eigenvalues')

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self


def brunt_index_correct(alpha):
    '''
    Apply empirical corrections from Heyer & Brunt

    Using the empirical correction from Brunt & Heyer 2002a, where

    .. math::
        \alpha = (0.33 \pm 0.04)\beta -(0.05 \pm 0.08)

    :math:`\beta` is the index of the integrated velocity spectrum. This is
    Equation 4.1. The relation to the velocity structure index is:

    .. math::
        \beta = 2\gamma + 1

    Then the conversion to :math:`\gamma` is:

    .. math::
        \gamma = (1.5 \pm 0.18) \alpha - (0.19 \pm 0.20)

    These values are based off a broken linear fit in Section 3.3.1 from
    Chris Brunt's thesis. These are based off calibrating against uniform
    density field's with different indices.
    '''

    # term1 = 1.52 * alpha
    # term1_err = term1 * np.sqrt((0.18 / 1.52)**2 + (alpha_err / alpha)**2)

    return 1.52 * alpha - 0.19


def brunt_index_correct_range(alpha_low, alpha_up):
    '''
    Upper and low error ranges. See `brunt_index_correct`.
    '''

    return (1.52 - 0.18) * alpha_low - (0.19 + 0.2), \
        (1.52 + 0.18) * alpha_up - (0.19 - 0.2)


def set_n_eigs(eigenvalues, min_eigval, method='value'):
    '''
    Based on a minimum eigenvalue, find the number of components to consider.
    The cut-off may be the proportion of variance (method='proportion') or a
    minimum value for the variance (method='value')

    Parameters
    ----------
    eigenvalues : `~numpy.ndarray`
        Array of eigenvalues.
    min_eigval : float
        Value to determine the cut-off for important eigenvalues.
    method : {"value", "proportion"}, optional
        If `value`, `min_eigval` is the smallest eigenvalue to consider
        important. If `proportion`, `min_eigval` is the proportion of
        variance at which to cut at (i.e., 0.99 for 99%).

    Returns
    -------
    above.size : int
        The number of eigenvalues satisfying the given criteria.
    '''

    if method == "value":
        above = np.where(eigenvalues >= min_eigval)[0]

        return above.size

    elif method == "proportion":

        cumulative = np.cumsum(eigenvalues / eigenvalues.sum())

        above = np.where(cumulative <= min_eigval)[0]

        if above.size == 0:
            # The first one has a greater proportion than the limit.
            # Set to 1
            return 1
        else:
            return above.size

    else:
        raise ValueError("method must be 'value' or 'proportion'.")


def _enforce_velocity_axis(pca_obj):
    '''
    Enforce spectral_size be in velocity units.
    '''

    if not pca_obj._spectral_size.unit.is_equivalent(u.m / u.s):
        raise Warning("PCA requires the spectral axis to be in velocity units."
                      " If using a spectral cube, perform this conversion with"
                      " 'cube_vel = cube.with_spectral_unit(u.m / u.s, "
                      "rest_value=113 * u.GHz)', changing to the appropriate"
                      " rest frequency and desired velocity unit.")
