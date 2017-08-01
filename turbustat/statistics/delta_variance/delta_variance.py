# Licensed under an MIT open source license - see LICENSE

import numpy as np
from astropy.convolution import convolve_fft
from astropy import units as u
from astropy.wcs import WCS
from copy import copy
import statsmodels.api as sm

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..stats_utils import common_scale, padwithzeros
from ..fitting_utils import check_fit_limits


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
    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits") # doctest: +SKIP
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
            self.weights = np.ones(self.data.shape)
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

    def do_convolutions(self, allow_huge=False, boundary='wrap'):
        '''
        Perform the convolutions at all lags.

        Parameters
        ----------
        allow_huge : bool, optional
            Passed to `~astropy.convolve.convolve_fft`. Allows operations on
            images larger than 1 Gb.
        boundary : {"wrap", "fill"}, optional
            Use "wrap" for periodic boundaries, and "fill" for non-periodic.
        '''
        for i, lag in enumerate(self.lags.value):
            core = core_kernel(lag, self.data.shape[0], self.data.shape[1])
            annulus = annulus_kernel(
                lag, self.diam_ratio, self.data.shape[0], self.data.shape[1])

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

            img_core = convolve_fft(
                pad_img, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge,
                boundary=boundary)
            img_annulus = convolve_fft(
                pad_img, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge,
                boundary=boundary)
            weights_core = convolve_fft(
                pad_weights, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge,
                boundary=boundary)
            weights_annulus = convolve_fft(
                pad_weights, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge,
                boundary=boundary)

            weights_core[np.where(weights_core == 0)] = np.NaN
            weights_annulus[np.where(weights_annulus == 0)] = np.NaN

            self.convolved_arrays.append(
                (img_core / weights_core) - (img_annulus / weights_annulus))
            self.convolved_weights.append(weights_core * weights_annulus)

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

    def fit_plaw(self, xlow=None, xhigh=None, verbose=False):
        '''
        Fit a power-law to the SCF spectrum.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value to consider in the fit.
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

        x = sm.add_constant(x)

        # If the std were computed, use them as weights
        weighted_fit = True
        if weighted_fit:

            # Converting to the log stds doesn't matter since the weights
            # remain proportional to 1/sigma^2, and an overal normalization is
            # applied in the fitting routine.
            weights = self.delta_var_error[within_limits] ** -2

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

        model_values = self.fit.params[0] + self.fit.params[1] * xvals

        return model_values

    def run(self, verbose=False, xunit=u.pix, allow_huge=False,
            boundary='wrap', xlow=None, xhigh=None, save_name=None):
        '''
        Compute the delta-variance.

        Parameters
        ----------
        verbose : bool, optional
            Plot delta-variance transform.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        allow_huge : bool, optional
            See `~DeltaVariance.do_convolutions`.
        boundary : {"wrap", "fill"}, optional
            Use "wrap" for periodic boundaries, and "cut" for non-periodic.
        xlow : float, optional
            Lower lag value to consider in the fit.
        xhigh : float, optional
            Upper lag value to consider in the fit.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.do_convolutions(allow_huge=allow_huge, boundary=boundary)
        self.compute_deltavar()
        self.fit_plaw(xlow=xlow, xhigh=xhigh, verbose=verbose)

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")

            lags = self._spatial_unit_conversion(self.lags, xunit).value

            # Check for NaNs
            fin_vals = np.logical_or(np.isfinite(self.delta_var),
                                     np.isfinite(self.delta_var_error))
            p.errorbar(lags, self.delta_var[fin_vals],
                       yerr=self.delta_var_error[fin_vals],
                       fmt="bD-", label="Data")

            xvals = np.linspace(self._fit_range[0].value,
                                self._fit_range[1].value,
                                100) * self.lags.unit
            xvals_conv = self._spatial_unit_conversion(xvals, xunit).value

            p.plot(xvals_conv, 10**self.fitted_model(np.log10(xvals.value)),
                   'r--', linewidth=2, label='Fit')

            xlow = \
                self._spatial_unit_conversion(self._fit_range[0], xunit).value
            xhigh = \
                self._spatial_unit_conversion(self._fit_range[1], xunit).value
            p.axvline(xlow, color="r", alpha=0.5, linestyle='-.')
            p.axvline(xhigh, color="r", alpha=0.5, linestyle='-.')

            p.legend(loc='best')
            ax.grid(True)

            ax.set_xlabel("Lag ({})".format(xunit))
            ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()

        return self


def core_kernel(lag, x_size, y_size):
    '''
    Core Kernel for convolution.

    Parameters
    ----------

    lag : int
        Size of the lag. Set the kernel size.
    x_size : int
        Grid size to use in the x direction
    y_size_size : int
        Grid size to use in the y_size direction

    Returns
    -------

    kernel : numpy.ndarray
        Normalized kernel.
    '''

    x, y = np.meshgrid(np.arange(-x_size / 2, x_size / 2 + 1, 1),
                       np.arange(-y_size / 2, y_size / 2 + 1, 1))
    kernel = ((4 / np.pi * lag**2)) * \
        np.exp(-(x ** 2. + y ** 2.) / (lag / 2.) ** 2.)

    return kernel / np.sum(kernel)


def annulus_kernel(lag, diam_ratio, x_size, y_size):
    '''

    Annulus Kernel for convolution.

    Parameters
    ----------
    lag : int
        Size of the lag. Set the kernel size.
    diam_ratio : float
                 Ratio between kernel diameters.
    x_size : int
        Grid size to use in the x direction
    y_size_size : int
        Grid size to use in the y_size direction

    Returns
    -------

    kernel : numpy.ndarray
        Normalized kernel.
    '''

    x, y = np.meshgrid(np.arange(-x_size / 2, x_size / 2 + 1, 1),
                       np.arange(-y_size / 2, y_size / 2 + 1, 1))

    inner = np.exp(-(x ** 2. + y ** 2.) / (lag / 2.) ** 2.)
    outer = np.exp(-(x ** 2. + y ** 2.) / (diam_ratio * lag / 2.) ** 2.)

    kernel = 4 / (np.pi * lag**2 * (diam_ratio ** 2. - 1)) * (outer - inner)

    return kernel / np.sum(kernel)


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
        if isinstance(boundary, basestring):
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
                        ang_units=False, unit=u.deg, save_name=None):
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

        deltavar1_sum = np.sum(self.delvar1.delta_var[~all_nans])
        deltavar1 = np.log10(self.delvar1.delta_var[~all_nans] / deltavar1_sum)

        deltavar2_sum = np.sum(self.delvar2.delta_var[~all_nans])
        deltavar2 = np.log10(self.delvar2.delta_var[~all_nans] / deltavar2_sum)

        # Distance between two normalized curves
        self.curve_distance = np.linalg.norm(deltavar1 - deltavar2)

        # Distance between the fitted slopes (combined t-statistic)
        self.slope_distance = \
            np.abs(self.delvar1.slope - self.delvar2.slope) / \
            np.sqrt(self.delvar1.slope_err**2 + self.delvar2.slope_err**2)

        if verbose:
            import matplotlib.pyplot as p

            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")
            if ang_units:
                ang_equiv1 = self.delvar1.angular_equiv
                ang_equiv2 = self.delvar2.angular_equiv
                lags1 = \
                    self.delvar1.lags.to(unit, equivalencies=ang_equiv1).value
                lags2 = \
                    self.delvar2.lags.to(unit, equivalencies=ang_equiv2).value
            else:
                lags1 = self.delvar1.lags.value
                lags2 = self.delvar2.lags.value

            lags1 = lags1[~all_nans]
            lags2 = lags2[~all_nans]

            # Normalize the errors for when plotting. NOT log-scaled.
            deltavar1_err = \
                self.delvar1.delta_var_error[~all_nans] / deltavar1_sum
            deltavar2_err = \
                self.delvar2.delta_var_error[~all_nans] / deltavar2_sum

            p.errorbar(lags1, 10**deltavar1,
                       yerr=deltavar1_err, fmt="bD-",
                       label=label1)
            p.errorbar(lags2, 10**deltavar2,
                       yerr=deltavar2_err, fmt="go-",
                       label=label2)

            lims1 = self.delvar1.fit_range
            xvals1 = \
                np.linspace(self.delvar1.lags.min().value if lims1[0] is None
                            else lims1[0],
                            self.delvar1.lags.max().value if lims1[1] is None
                            else lims1[1],
                            100)
            lims2 = self.delvar2.fit_range
            xvals2 = \
                np.linspace(self.delvar2.lags.min().value if lims2[0] is None
                            else lims2[0],
                            self.delvar2.lags.max().value if lims2[1] is None
                            else lims2[1],
                            100)
            if ang_units:
                xvals1_conv = xvals1.to(unit, equivalencies=ang_equiv1).value
                xvals2_conv = xvals2.to(unit, equivalencies=ang_equiv2).value
            else:
                xvals1_conv = xvals1
                xvals2_conv = xvals2

            p.plot(xvals1_conv,
                   10**self.delvar1.fitted_model(np.log10(xvals1)) / deltavar1_sum,
                   'b--', linewidth=8, alpha=0.75)
            p.plot(xvals2_conv,
                   10**self.delvar2.fitted_model(np.log10(xvals2)) / deltavar2_sum,
                   'g--', linewidth=8, alpha=0.75)

            # Vertical lines to indicate fit region
            p.axvline(self.delvar1.lags.min().value if lims1[0] is None
                      else lims1[0], color='b', alpha=0.5, linestyle='-')
            p.axvline(self.delvar1.lags.max().value if lims1[1] is None
                      else lims1[1], color='b', alpha=0.5, linestyle='-')

            p.axvline(self.delvar2.lags.min().value if lims2[0] is None
                      else lims2[0], color='g', alpha=0.5, linestyle='-')
            p.axvline(self.delvar2.lags.max().value if lims2[1] is None
                      else lims2[1], color='g', alpha=0.5, linestyle='-')

            ax.legend(loc='best')
            ax.grid(True)
            if ang_units:
                ax.set_xlabel("Lag ({})".format(unit))
            else:
                ax.set_xlabel("Lag (pixels)")
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
    nindep = np.sqrt(np.isfinite(arr_cent).sum() / kern_area)

    val_err = np.sqrt((np.nansum(arr_cent ** 4. * weight) /
                       np.nansum(weight)) - val**2) / nindep

    return val, val_err
