
'''

Implementation of the Delta-Variance Method from Stutzki et al. 1998.

'''

import numpy as np
from scipy.stats import nanmean
from astropy.convolution import convolve_fft
from astropy import units as u

class DeltaVariance(object):

    """

    docstring for DeltaVariance

    """

    def __init__(self, img, header, weights, diam_ratio=1.5, lags=None):
        super(DeltaVariance, self).__init__()

        self.img = img
        self.header = header
        self.weights = weights
        self.diam_ratio = diam_ratio

        self.nanflag = False
        if np.isnan(self.img).any():
            self.nanflag = True

        if lags is None:
            try:
                self.ang_size = self.header["CDELT2"] * u.deg
                min_size = (0.1 * u.arcmin) / self.ang_size.to(u.arcmin)
                # Can't be smaller than one pixel, set to 3 to avoid noisy pixels
                if min_size < 3.0:
                    min_size = 3.0
            except KeyError:
                print "No CDELT2 in header. Using pixel scales."
                self.ang_size = 1.0 * u.astrophys.pixel
                min_size = 3.0
            self.lags = np.logspace(np.log10(min_size), np.log10(min(self.img.shape) / 2.), 25)
        else:
            self.lags = lags

        self.convolved_arrays = []
        self.convolved_weights = []
        self.delta_var = np.empty((1, len(self.lags)))
        self.delta_var_error = np.empty((1, len(self.lags)))

    def do_convolutions(self):
        for i, lag in enumerate(self.lags):
            core = core_kernel(lag, self.img.shape[0], self.img.shape[1])
            annulus = annulus_kernel(
                lag, self.diam_ratio, self.img.shape[0], self.img.shape[1])

            # Extend to avoid boundary effects from non-periodicity
            pad_weights = np.pad(self.weights, int(lag), padwithzeros)
            pad_img = np.pad(self.img, int(lag), padwithzeros) * pad_weights

            interpolate_nan = False
            if self.nanflag:
                interpolate_nan = True

            img_core = convolve_fft(
                pad_img, core, normalize_kernel=True,
                interpolate_nan=interpolate_nan,
                ignore_edge_zeros=True)
            img_annulus = convolve_fft(
                pad_img, annulus, normalize_kernel=True,
                interpolate_nan=interpolate_nan,
                ignore_edge_zeros=True)
            weights_core = convolve_fft(
                pad_weights, core, normalize_kernel=True,
                interpolate_nan=interpolate_nan,
                ignore_edge_zeros=True)
            weights_annulus = convolve_fft(
                pad_weights, annulus, normalize_kernel=True,
                interpolate_nan=interpolate_nan,
                ignore_edge_zeros=True)

            weights_core[np.where(weights_core == 0)] = np.NaN
            weights_annulus[np.where(weights_annulus == 0)] = np.NaN

            self.convolved_arrays.append(
                (img_core / weights_core) - (img_annulus / weights_annulus))
            self.convolved_weights.append(weights_core * weights_annulus)

        return self

    def compute_deltavar(self):

        for i, (convolved_array, convolved_weight) in enumerate(zip(self.convolved_arrays, self.convolved_weights)):
            avg_value = nanmean(convolved_array, axis=None)

            delta_var_val = np.nansum(
                (convolved_array - avg_value) ** 2. * convolved_weight) / \
                np.nansum(convolved_weight)

            error = np.nansum(
                ((convolved_array - avg_value) ** 2. - delta_var_val) ** 2.) /\
                np.nansum(convolved_weight)

            self.delta_var[0, i] = delta_var_val
            self.delta_var_error[0, i] = error

        return self

    def run(self, verbose=False, ang_units=True):

        self.do_convolutions()
        self.compute_deltavar()

        if ang_units:
            self.lags *= self.ang_size

        if verbose:
            import matplotlib.pyplot as p
            # p.errorbar(self.lags, self.delta_var, yerr=self.delta_var_error)
            p.loglog(self.lags, self.delta_var[0, :], "bD-")
            p.grid(True)
            p.xlabel("Lag (arcmin)")
            p.ylabel(r"$\sigma^{2}_{\Delta}$")
            p.show()

        return self


def core_kernel(lag, x_size, y_size):
    '''
    Core Kernel for convolution.

    INPUTS
    ------

    lag : float


    OUTPUTS
    -------

    kernel : numpy.ndarray
    '''

    x, y = np.meshgrid(np.arange(-x_size / 2, x_size / 2 + 1, 1),
                       np.arange(-y_size / 2, y_size / 2 + 1, 1))
    kernel = ((4 / np.pi * lag)) * \
        np.exp(-(x ** 2. + y ** 2.) / (lag / 2.) ** 2.)

    return kernel / np.sum(kernel)


def annulus_kernel(lag, diam_ratio, x_size, y_size):
    '''

    Annulus Kernel for convolution.

    INPUTS
    ------

    lag - float
          size of kernel, same as width

    diam_ratio - float
                 ratio between kernel diameters

    '''

    x, y = np.meshgrid(np.arange(-x_size / 2, x_size / 2 + 1, 1),
                       np.arange(-y_size / 2, y_size / 2 + 1, 1))

    inner = np.exp(-(x ** 2. + y ** 2.) / (lag / 2.) ** 2.)
    outer = np.exp(-(x ** 2. + y ** 2.) / (diam_ratio * lag / 2.) ** 2.)

    kernel = 4 / (np.pi * lag * (diam_ratio ** 2. - 1)) * (outer - inner)

    return kernel / np.sum(kernel)


def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


class DeltaVariance_Distance(object):

    """


    """

    def __init__(self, dataset1, weights1, dataset2, weights2, diam_ratio=1.5,
                 lags=None, fiducial_model=None):
        super(DeltaVariance_Distance, self).__init__()

        if fiducial_model is not None:
            self.delvar1 = fiducial_model
        else:
            self.delvar1 = DeltaVariance(dataset1[0], dataset1[1], weights1, diam_ratio=diam_ratio,
                                         lags=lags)
            self.delvar1.run()

        self.delvar2 = DeltaVariance(dataset2[0], dataset2[1], weights2, diam_ratio=diam_ratio,
                                     lags=lags)
        self.delvar2.run()

        self.distance = None

    def distance_metric(self, verbose=False):

        self.distance = np.linalg.norm(np.log10(self.delvar1.delta_var[0, :]) -
                                       np.log10(self.delvar2.delta_var[0, :]))

        if verbose:
            import matplotlib.pyplot as p

            # print "Distance: %s" % (self.distance)

            p.loglog(self.delvar1.lags, self.delvar1.delta_var[0, :], "bD-",
                     label="Delta Var 1")
            p.loglog(self.delvar2.lags, self.delvar2.delta_var[0, :], "rD-",
                     label="Delta Var 2")
            p.legend()
            p.grid(True)
            p.xlabel("Lag (arcmin)")
            p.ylabel(r"$\sigma^{2}_{\Delta}$")

            p.show()

        return self
