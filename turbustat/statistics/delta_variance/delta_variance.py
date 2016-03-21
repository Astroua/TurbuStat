# Licensed under an MIT open source license - see LICENSE

import numpy as np
from scipy.stats import nanmean
from astropy.convolution import convolve_fft
from astropy import units as u

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class DeltaVariance(BaseStatisticMixIn):

    """

    The delta-variance technique as described in Ossenkopf et al. (2008).

    Parameters
    ----------

    img : %(dtypes)s
        The image calculate the delta-variance of.
    header : FITS header
        Image header.
    weights : %(dtypes)s
        Weights to be used.
    diam_ratio : float, optional
        The ratio between the kernel sizes.
    lags : numpy.ndarray or list, optional
        The pixel scales to compute the delta-variance at.
    nlags : int, optional
        Number of lags to use.
    ang_units : bool, optional
        Convert frequencies to angular units using the given header.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, diam_ratio=1.5,
                 lags=None, nlags=25, ang_units=False):
        super(DeltaVariance, self).__init__()

        # Set the data and perform checks
        self.input_data_header(img, header)

        self.diam_ratio = diam_ratio
        self.ang_units = ang_units

        if weights is None:
            self.weights = np.ones(img.shape)
        else:
            self.weights = input_data(weights, no_header=True)

        self.nanflag = False
        if np.isnan(self.data).any() or np.isnan(self.weights).any():
            self.nanflag = True

        if lags is None:
            try:
                self.ang_size = self.header["CDELT2"] * u.deg
                min_size = (0.1 * u.arcmin) / self.ang_size.to(u.arcmin)
                # Can't be smaller than one pixel, set to 3 to
                # avoid noisy pixels. Can't be bigger than half the image
                #(deals with issue from sim cubes).
                if min_size < 3.0 or min_size > min(self.data.shape) / 2.:
                    min_size = 3.0
            except ValueError:
                print("No CDELT2 in header. Using pixel scales.")
                self.ang_size = 1.0 * u.astrophys.pixel
                min_size = 3.0
            self.lags = np.logspace(np.log10(min_size),
                                    np.log10(min(self.data.shape) / 2.), nlags)
        else:
            self.lags = lags

        if self.ang_units:
            self.lags *= self.ang_size

        self.convolved_arrays = []
        self.convolved_weights = []
        self.delta_var = np.empty((len(self.lags, )))
        self.delta_var_error = np.empty((len(self.lags, )))

    def do_convolutions(self):
        for i, lag in enumerate(self.lags):
            core = core_kernel(lag, self.data.shape[0], self.data.shape[1])
            annulus = annulus_kernel(
                lag, self.diam_ratio, self.data.shape[0], self.data.shape[1])

            # Extend to avoid boundary effects from non-periodicity
            pad_weights = np.pad(self.weights, int(lag), padwithzeros)
            pad_img = np.pad(self.data, int(lag), padwithzeros) * pad_weights

            img_core = convolve_fft(
                pad_img, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True)
            img_annulus = convolve_fft(
                pad_img, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True)
            weights_core = convolve_fft(
                pad_weights, core, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True)
            weights_annulus = convolve_fft(
                pad_weights, annulus, normalize_kernel=True,
                interpolate_nan=self.nanflag,
                ignore_edge_zeros=True)

            weights_core[np.where(weights_core == 0)] = np.NaN
            weights_annulus[np.where(weights_annulus == 0)] = np.NaN

            self.convolved_arrays.append(
                (img_core / weights_core) - (img_annulus / weights_annulus))
            self.convolved_weights.append(weights_core * weights_annulus)

        return self

    def compute_deltavar(self):

        for i, (conv_arr,
                conv_weight,
                lag) in enumerate(zip(self.convolved_arrays,
                                      self.convolved_weights,
                                      self.lags)):

            val, err = _delvar(conv_arr, conv_weight, lag)

            self.delta_var[i] = val
            self.delta_var_error[i] = err

        return self

    def run(self, verbose=False):

        self.do_convolutions()
        self.compute_deltavar()

        if verbose:
            import matplotlib.pyplot as p
            ax = p.subplot(111)
            ax.set_xscale("log", nonposx="clip")
            ax.set_yscale("log", nonposx="clip")
            p.errorbar(self.lags, self.delta_var, yerr=self.delta_var_error,
                       fmt="bD-")
            ax.grid(True)

            if self.ang_units:
                ax.set_xlabel("Lag (deg)")
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
                 diam_ratio=1.5, lags=None, fiducial_model=None,
                 ang_units=False):
        super(DeltaVariance_Distance, self).__init__()

        self.ang_units = ang_units

        if fiducial_model is not None:
            self.delvar1 = fiducial_model
        else:
            self.delvar1 = DeltaVariance(dataset1[0], dataset1[1],
                                         weights=weights1,
                                         diam_ratio=diam_ratio, lags=lags,
                                         ang_units=ang_units)
            self.delvar1.run()

        self.delvar2 = DeltaVariance(dataset2[0], dataset2[1],
                                     weights=weights2,
                                     diam_ratio=diam_ratio, lags=lags,
                                     ang_units=ang_units)
        self.delvar2.run()

    def distance_metric(self, verbose=False, label1=None, label2=None):
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
            p.errorbar(self.delvar1.lags, self.delvar1.delta_var,
                       yerr=self.delvar1.delta_var_error, fmt="bD-",
                       label=label1)
            p.errorbar(self.delvar2.lags, self.delvar2.delta_var,
                       yerr=self.delvar2.delta_var_error, fmt="go-",
                       label=label2)
            ax.legend(loc='best')
            ax.grid(True)
            if self.ang_units:
                ax.set_xlabel("Lag (deg)")
            else:
                ax.set_xlabel("Lag (pixels)")
            ax.set_ylabel(r"$\sigma^{2}_{\Delta}$")
            p.show()

        return self


def _delvar(array, weight, lag):
    '''
    Computes the delta variance of the given array.
    '''
    arr_cent = array.copy() - nanmean(array, axis=None)

    val = np.nansum(arr_cent ** 2. * weight) /\
        np.nansum(weight)

    # The error needs to be normalized by the number of independent
    # pixels in the array.
    # Take width to be 1/2 FWHM. Note that lag is defined as 2*sigma.
    # So 2ln(2) sigma^2 = ln(2)/2 * lag^2
    kern_area = np.ceil(0.5*np.pi*np.log(2)*lag**2).astype(int)
    nindep = np.sqrt(np.isfinite(arr_cent).sum() / kern_area)

    val_err = np.sqrt((np.nansum(arr_cent ** 4. * weight) /
                       np.nansum(weight)) - val**2) / nindep

    return val, val_err
