# Licensed under an MIT open source license - see LICENSE


import numpy as np
from scipy.stats import chisquare
from scipy.optimize import curve_fit

from ..stats_utils import standardize
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class Tsallis(BaseStatisticMixIn):
    """
    The Tsallis Distribution (see Tofflemire et al., 2011)

    Parameters
    ----------
    img : %(dtypes)s
        2D image.
    lags : numpy.ndarray or list
        Lags to calculate at.
    num_bins : int, optional
        Number of bins to use in the histograms.
    periodic : bool, optional
        Sets whether the boundaries are periodic.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, lags=None, num_bins=500, periodic=False):
        '''
        Parameters
        ----------

        periodic : bool, optional
                   Use for simulations with periodic boundaries.
        '''

        self.need_header_flag = False
        self.header = None

        self.data = input_data(img, no_header=True)
        self.num_bins = num_bins
        self.periodic = periodic

        if lags is None:
            self.lags = [1, 2, 4, 8, 16, 32, 64]
        else:
            self.lags = lags

        self.tsallis_arrays = np.empty(
            (len(self.lags), self.data.shape[0], self.data.shape[1]))
        self.tsallis_distrib = np.empty((len(self.lags), 2, num_bins))
        self.tsallis_fits = np.empty((len(self.lags), 7))

    def make_tsallis(self):
        '''
        Calculate the Tsallis distribution at each lag.
        We standardize each distribution such that it has a mean of zero and
        variance of one.
        '''

        for i, lag in enumerate(self.lags):
            if self.periodic:
                pad_img = self.data
            else:
                pad_img = np.pad(self.data, lag, padwithzeros)
            rolls = np.roll(pad_img, lag, axis=0) +\
                np.roll(pad_img, (-1) * lag, axis=0) +\
                np.roll(pad_img, lag, axis=1) +\
                np.roll(pad_img, (-1) * lag, axis=1)

            #  Remove the padding
            if self.periodic:
                clip_resulting = (rolls / 4.) - pad_img
            else:
                clip_resulting = (rolls[lag:-lag, lag:-lag] / 4.) -\
                    pad_img[lag:-lag, lag:-lag]
            # Normalize the data
            data = standardize(clip_resulting)

            # Ignore nans for the histogram
            hist, bin_edges = np.histogram(data[~np.isnan(data)],
                                           bins=self.num_bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            normlog_hist = np.log10(hist / np.sum(hist, dtype="float"))

            # Keep results
            self.tsallis_arrays[i, :] = data
            self.tsallis_distrib[i, 0, :] = bin_centres
            self.tsallis_distrib[i, 1, :] = normlog_hist

    def fit_tsallis(self, sigma_clip=2):
        '''
        Fit the Tsallis distributions.

        Parameters
        ----------
        sigma_clip : float
            Sets the sigma value to clip data at.
        '''

        for i, dist in enumerate(self.tsallis_distrib):
            clipped = clip_to_sigma(dist[0], dist[1], sigma=sigma_clip)
            params, pcov = curve_fit(tsallis_function, clipped[0], clipped[1],
                                     p0=(-np.max(clipped[1]), 1., 2.),
                                     maxfev=100 * len(dist[0]))
            fitted_vals = tsallis_function(clipped[0], *params)
            self.tsallis_fits[i, :3] = params
            self.tsallis_fits[i, 3:6] = np.diag(pcov)
            self.tsallis_fits[i, 6] = chisquare(
                np.exp(fitted_vals), f_exp=np.exp(clipped[1]), ddof=3)[0]

    def run(self, verbose=False, sigma_clip=2):
        '''
        Run all steps.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        sigma_clip : float
            Sets the sigma value to clip data at.
            Passed to :func:`fit_tsallis`.
        '''

        self.make_tsallis()
        self.fit_tsallis(sigma_clip=sigma_clip)

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.lags)

            i = 1
            for dist, arr, params in zip(self.tsallis_distrib,
                                         self.tsallis_arrays,
                                         self.tsallis_fits):

                p.subplot(num, 2, i)
                # This doesn't plot the last image
                p.imshow(arr, origin="lower", interpolation="nearest")
                p.colorbar()
                p.subplot(num, 2, i + 1)
                p.plot(dist[0], tsallis_function(dist[0], *params), "r")
                p.plot(dist[0], dist[1], 'rD', label="".join(
                    ["Tsallis Distribution with Lag ", str(self.lags[i / 2])]))
                p.legend(loc="best")

                i += 2
            p.show()

        return self


class Tsallis_Distance(object):

    '''
    Distance Metric for the Tsallis Distribution.

    Parameters
    ----------
    array1 : %(dtypes)s
        2D datas.
    array2 : %(dtypes)s
        2D datas.
    lags : numpy.ndarray or list
        Lags to calculate at.
    num_bins : int, optional
        Number of bins to use in the histograms.
    fiducial_model : Tsallis
        Computed Tsallis object. use to avoid recomputing.
    periodic : bool, optional
        Sets whether the boundaries are periodic.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, array1, array2, lags=None, num_bins=500,
                 fiducial_model=None, periodic=False):
        super(Tsallis_Distance, self).__init__()

        if fiducial_model is not None:
            self.tsallis1 = fiducial_model
        else:
            self.tsallis1 = Tsallis(
                array1, lags=lags, num_bins=num_bins,
                periodic=periodic).run(verbose=False)

        self.tsallis2 = Tsallis(
            array2, lags=lags, num_bins=num_bins,
            periodic=periodic).run(verbose=False)

        self.distance = None

    def distance_metric(self, verbose=False):
        '''

        We do not consider the parameter a in the distance metric. Since we
        are fitting to a PDF, a is related to the number of data points and
        is therefore not a true measure of the differences between the data
        sets. The distance is computed by summing the squared difference of
        the parameters, normalized by the sums of the squares, for each lag.
        The total distance the sum between the two parameters.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.

        '''

        w1 = self.tsallis1.tsallis_fits[:, 1]
        w2 = self.tsallis2.tsallis_fits[:, 1]

        q1 = self.tsallis1.tsallis_fits[:, 2]
        q2 = self.tsallis2.tsallis_fits[:, 2]

        # diff_a = (a1-a2)**2.
        diff_w = (w1 - w2) ** 2. / (w1 ** 2. + w2 ** 2.)
        diff_q = (q1 - q2) ** 2. / (q1 ** 2. + q2 ** 2.)

        self.distance = np.sum(diff_w + diff_q)

        if verbose:
            import matplotlib.pyplot as p
            lags = self.tsallis1.lags
            p.plot(lags, diff_w, "rD", label="Difference of w")
            p.plot(lags, diff_q, "go", label="Difference of q")
            p.legend()
            p.xscale('log', basex=2)
            p.ylabel("Normalized Difference")
            p.xlabel("Lags (pixels)")
            p.grid(True)
            p.show()

        return self


def tsallis_function(x, *p):
    '''
    Tsallis distribution function as given in Tofflemire
    Implemented in log form

    Parameters
    ----------
    x : numpy.ndarray or list
        x-data
    params : list
        Contains the three parameter values.
    '''
    loga, wsquare, q = p
    return (-1 / (q - 1)) * (np.log10(1 + (q - 1) *
                                      (x ** 2. / wsquare)) + loga)


def clip_to_sigma(x, y, sigma=2):
    '''
    Clip to values between -2 and 2 sigma.

    Parameters
    ----------
    x : numpy.ndarray
        x-data
    y : numpy.ndarray
        y-data
    '''
    clip_mask = np.zeros(x.shape)
    for i, val in enumerate(x):
        if val < sigma or val > -sigma:
            clip_mask[i] = 1
    clip_x = x[np.where(clip_mask == 1)]
    clip_y = y[np.where(clip_mask == 1)]

    return clip_x[np.isfinite(clip_y)], clip_y[np.isfinite(clip_y)]


def padwithzeros(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector
