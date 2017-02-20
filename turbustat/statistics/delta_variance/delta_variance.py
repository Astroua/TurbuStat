# Licensed under an MIT open source license - see LICENSE

import numpy as np
from astropy.convolution import convolve_fft
from astropy import units as u
from astropy.wcs import WCS
from copy import copy
import statsmodels.api as sm

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..stats_utils import common_scale, standardize


class DeltaVariance(BaseStatisticMixIn):

    """

    The delta-variance technique as described in Ossenkopf et al. (2008).

    Parameters
    ----------

    img : %(dtypes)s
        The image calculate the delta-variance of.
    header : FITS header, optional
        Image header.
    weights : %(dtypes)s
        Weights to be used.
    diam_ratio : float, optional
        The ratio between the kernel sizes.
    lags : numpy.ndarray or list, optional
        The pixel scales to compute the delta-variance at.
    nlags : int, optional
        Number of lags to use.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, diam_ratio=1.5,
                 lags=None, nlags=25):
        super(DeltaVariance, self).__init__()

        # Set the data and perform checks
        self.input_data_header(img, header)

        self.diam_ratio = diam_ratio

        if weights is None:
            self.weights = np.ones(self.data.shape)
        else:
            self.weights = input_data(weights, no_header=True)

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
                self.lags = self.to_pixel(lags)

        self.convolved_arrays = []
        self.convolved_weights = []
        self.delta_var = np.empty((len(self.lags)))
        self.delta_var_error = np.empty((len(self.lags)))

    def do_convolutions(self, allow_huge=False):
        for i, lag in enumerate(self.lags.value):
            core = core_kernel(lag, self.data.shape[0], self.data.shape[1])
            annulus = annulus_kernel(
                lag, self.diam_ratio, self.data.shape[0], self.data.shape[1])

            # Extend to avoid boundary effects from non-periodicity
            pad_weights = np.pad(self.weights, int(lag), padwithzeros)
            pad_img = np.pad(self.data, int(lag), padwithzeros) * pad_weights

            img_core = convolve_fft(
                pad_img, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge)
            img_annulus = convolve_fft(
                pad_img, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge)
            weights_core = convolve_fft(
                pad_weights, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge)
            weights_annulus = convolve_fft(
                pad_weights, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True, allow_huge=allow_huge)

            weights_core[np.where(weights_core == 0)] = np.NaN
            weights_annulus[np.where(weights_annulus == 0)] = np.NaN

            self.convolved_arrays.append(
                (img_core / weights_core) - (img_annulus / weights_annulus))
            self.convolved_weights.append(weights_core * weights_annulus)

    def compute_deltavar(self):

        for i, (conv_arr,
                conv_weight,
                lag) in enumerate(zip(self.convolved_arrays,
                                      self.convolved_weights,
                                      self.lags.value)):

            val, err = _delvar(conv_arr, conv_weight, lag)

            self.delta_var[i] = val
            self.delta_var_error[i] = err

    def fit_plaw(self, xlow=None, xhigh=None, verbose=False):
        '''
        Fit a power-law to the SCF spectrum.

        Parameters
        ----------
        xlow : float, optional
            Lower lag value to consider in the fit.
        xhigh : float, optional
            Upper lag value to consider in the fit.
        verbose : bool, optional
            Show fit summary when enabled.
        '''

        x = np.log10(self.lags.value)
        y = np.log10(self.delta_var)

        if xlow is not None:
            lower_limit = x >= np.log10(xlow)
        else:
            lower_limit = \
                np.ones_like(self.delta_var, dtype=bool)

        if xhigh is not None:
            upper_limit = x <= np.log10(xhigh)
        else:
            upper_limit = \
                np.ones_like(self.delta_var, dtype=bool)

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
        return self._slope

    @property
    def slope_err(self):
        return self._slope_err

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
            Values of the model at the given values. Equivalent to log values
            of the SCF spectrum.
        '''

        model_values = self.fit.params[0] + self.fit.params[1] * xvals

        return model_values

    def run(self, verbose=False, ang_units=False, unit=u.deg,
            allow_huge=False, xlow=None, xhigh=None):
        '''
        Compute the delta-variance.

        Parameters
        ----------
        verbose : bool, optional
            Plot delta-variance transform.
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        self.do_convolutions(allow_huge=allow_huge)
        self.compute_deltavar()
        self.fit_plaw(xlow=xlow, xhigh=xhigh, verbose=verbose)

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")
            if ang_units:
                lags = \
                    self.lags.to(unit, equivalencies=self.angular_equiv).value
            else:
                lags = self.lags.value
            p.errorbar(lags, self.delta_var, yerr=self.delta_var_error,
                       fmt="bD-", label="Data")
            xvals = \
                np.linspace(self.lags.min().value if xlow is None else xlow,
                            self.lags.max().value if xhigh is None else xhigh,
                            100)
            p.plot(xvals, 10**self.fitted_model(np.log10(xvals)), 'r--',
                   linewidth=2, label='Fit')
            p.legend(loc='best')
            ax.grid(True)

            if ang_units:
                ax.set_xlabel("Lag ({})".format(unit))
            else:
                ax.set_xlabel("Lag (pixels)")
            ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")
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


def padwithzeros(vector, pad_width, iaxis, kwargs):
    '''
    Pad array with zeros.
    '''
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


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
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, dataset1, dataset2, weights1=None, weights2=None,
                 diam_ratio=1.5, lags=None, fiducial_model=None):
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
            self.delvar1.run()

        self.delvar2 = DeltaVariance(dataset2,
                                     weights=weights2,
                                     diam_ratio=diam_ratio, lags=lags2)
        self.delvar2.run()

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        ang_units=False, unit=u.deg):
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
        '''

        # Check for NaNs and negatives
        nans1 = np.logical_or(np.isnan(self.delvar1.delta_var),
                              self.delvar1.delta_var <= 0.0)
        nans2 = np.logical_or(np.isnan(self.delvar2.delta_var),
                              self.delvar2.delta_var <= 0.0)

        all_nans = np.logical_or(nans1, nans2)

        deltavar1 = np.log10(self.delvar1.delta_var)[~all_nans]
        deltavar2 = np.log10(self.delvar2.delta_var)[~all_nans]

        self.distance = np.linalg.norm(deltavar1 - deltavar2)

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")
            if ang_units:
                lags1 = \
                    self.delvar1.lags.to(unit, equivalencies=self.delvar1.angular_equiv).value
                lags2 = \
                    self.delvar2.lags.to(unit, equivalencies=self.delvar2.angular_equiv).value
            else:
                lags1 = self.delvar1.lags.value
                lags2 = self.delvar2.lags.value
            p.errorbar(lags1, self.delvar1.delta_var,
                       yerr=self.delvar1.delta_var_error, fmt="bD-",
                       label=label1)
            p.errorbar(lags2, self.delvar2.delta_var,
                       yerr=self.delvar2.delta_var_error, fmt="go-",
                       label=label2)
            ax.legend(loc='best')
            ax.grid(True)
            if ang_units:
                ax.set_xlabel("Lag ({})".format(unit))
            else:
                ax.set_xlabel("Lag (pixels)")
            ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")
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
