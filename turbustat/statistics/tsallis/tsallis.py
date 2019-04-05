# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from scipy.stats import chisquare
from scipy.optimize import curve_fit
import astropy.units as u
from astropy.table import Table

from ..stats_utils import standardize, padwithzeros
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types


class Tsallis(BaseStatisticMixIn):
    """
    The Tsallis Distribution (see Tofflemire et al., 2011)

    Parameters
    ----------
    img : %(dtypes)s
        2D image.
    header : FITS header, optional
        The image header. Needed for the pixel scale.
    lags : `~astropy.units.Quantity`, optional
        Give the spatial lag values to compute the distribution at. The
        default lag sizes are powers of 2 up to half the image size (so for a
        128 by 128 image, the lags will be [1, 2, 4, 8, 16, 32, 64]).
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, lags=None, distance=None):

        self.input_data_header(img, header)

        if distance is not None:
            self.distance = distance

        if lags is None:
            # Find the next smallest power of 2 from the smallest axis
            max_power = \
                np.floor(np.log2(min(self.data.shape) / 2.)).astype(int)
            self.lags = [2**i for i in range(max_power + 1)] * u.pix
        else:
            self.lags = lags

    @property
    def lags(self):
        '''
        Lag values to calculate the statistics at.
        '''
        return self._lags

    @lags.setter
    def lags(self, values):

        if not isinstance(values, u.Quantity):
            raise TypeError("lags must be given as a astropy.units.Quantity.")

        # Now make sure that we can convert into pixels before setting.
        try:
            pix_rad = self._to_pixel(values)
        except Exception as e:
            raise e

        # The radius should be larger than a pixel
        if np.any(pix_rad.value < 1):
            raise ValueError("One of the chosen lags is smaller than one "
                             "pixel."
                             " Ensure that all lag values are larger than one "
                             "pixel.")

        half_comp = (np.floor(pix_rad.value) - min(self.data.shape) / 2.)

        if np.any(half_comp > 1e-10):
            raise ValueError("At least one of the lags is larger than half of"
                             " the image size. Remove these lags from the "
                             "array.")

        self._lags = values

    def make_tsallis(self, periodic=True, num_bins=None):
        '''
        Calculate the Tsallis distribution at each lag.
        We standardize each distribution such that it has a mean of zero and
        variance of one before fitting.

        If the lag values are fractions of a pixel when converted to pixel
        units, the lag is rounded down to the next smallest integer value.

        Parameters
        ----------
        periodic : bool, optional
                   Use for simulations with periodic boundaries.
        num_bins : int, optional
            Number of bins to use in the histograms. Defaults to the
            square-root of the number of finite points in the image.

        '''

        if num_bins is None:
            num_bins = \
                np.ceil(np.sqrt(np.isfinite(self.data).sum())).astype(int)

        self._lag_arrays = np.empty((len(self.lags),
                                     self.data.shape[0],
                                     self.data.shape[1]))
        self._lag_distribs = np.empty((len(self.lags), 2, num_bins))

        # Convert the lags into pixels
        pix_lags = np.floor(self._to_pixel(self.lags).value).astype(int)

        for i, lag in enumerate(pix_lags):
            if periodic:
                pad_img = self.data
            else:
                pad_img = np.pad(self.data, lag, padwithzeros)
            rolls = np.roll(pad_img, lag, axis=0) +\
                np.roll(pad_img, -lag, axis=0) +\
                np.roll(pad_img, lag, axis=1) +\
                np.roll(pad_img, -lag, axis=1)

            #  Remove the padding
            if periodic:
                clip_resulting = (rolls / 4.) - pad_img
            else:
                clip_resulting = (rolls[lag:-lag, lag:-lag] / 4.) -\
                    pad_img[lag:-lag, lag:-lag]
            # Normalize the data
            data = standardize(clip_resulting)

            # Ignore nans for the histogram
            hist, bin_edges = np.histogram(data[~np.isnan(data)],
                                           bins=num_bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
            normlog_hist = np.log10(hist / np.sum(hist, dtype="float"))

            # Keep results
            self._lag_arrays[i, :] = data
            self._lag_distribs[i, 0, :] = bin_centres
            self._lag_distribs[i, 1, :] = normlog_hist

    @property
    def lag_arrays(self):
        '''
        Arrays of the image computed at different lags.
        '''
        return self._lag_arrays

    @property
    def lag_distribs(self):
        '''
        Histogram bins and values compute from `~Tsallis.lag_arrays`. The
        histogram values are in log10.
        '''
        return self._lag_distribs

    def fit_tsallis(self, sigma_clip=5):
        '''
        Fit the Tsallis distributions.

        Parameters
        ----------
        sigma_clip : float
            Sets the sigma value to clip data at. If `None`,
            no clipping is performed on the data. Defaults to 5.
        '''

        if not hasattr(self, 'lag_distribs'):
            raise Exception("Calculate the distributions first with "
                            "Tsallis.make_tsallis.")

        self._sigma_clip = sigma_clip

        self._tsallis_params = np.empty((len(self.lags), 3))
        self._tsallis_stderrs = np.empty((len(self.lags), 3))
        self._tsallis_chisq = np.empty((len(self.lags), 1))

        for i, dist in enumerate(self.lag_distribs):
            if sigma_clip is None:
                # Keep all finite data
                finite_mask = np.logical_and(np.isfinite(dist[0]),
                                             np.isfinite(dist[1]))
                clipped = dist[0][finite_mask], dist[1][finite_mask]
            else:
                clipped = clip_to_sigma(dist[0], dist[1], sigma=sigma_clip)

            params, pcov = curve_fit(tsallis_function, clipped[0], clipped[1],
                                     p0=(-np.max(clipped[1]), 1., 2.),
                                     maxfev=100 * len(dist[0]))

            fitted_vals = tsallis_function(clipped[0], *params)
            self._tsallis_params[i] = params
            self._tsallis_stderrs[i] = np.sqrt(np.diag(pcov))
            self._tsallis_chisq[i] = chisquare(np.exp(fitted_vals),
                                               f_exp=np.exp(clipped[1]),
                                               ddof=3)[0]

    @property
    def tsallis_params(self):
        '''
        Parameters of the Tsallis distribution fit at each lag value.
        '''
        return self._tsallis_params

    @property
    def tsallis_stderrs(self):
        '''
        Standard errors of the Tsallis distribution fit at each lag value.
        '''
        return self._tsallis_stderrs

    @property
    def tsallis_chisq(self):
        '''
        Reduced chi-squared values for the fit at each lag value.
        '''
        return self._tsallis_chisq

    @property
    def tsallis_table(self):
        '''
        Return the fit parameters, standard error, and chi-squared values as
        an `~astropy.table.Table`.
        '''

        data = [self.lags] + [col for col in self.tsallis_params.T] + \
            [col for col in self.tsallis_stderrs.T] + [self.tsallis_chisq]

        names = ['lags', 'logA', 'w2', 'q', 'logA_stderr', 'w2_stderr',
                 'q_stderr', 'redchisq']

        return Table(data, names=names)

    def plot_parameters(self, save_name=None, **kwargs):
        '''
        Plot the fit parameters as a function of lag.

        Parameters
        ----------
        save_name : str,optional
            Save name for the figure. Enables saving the plot.
        kwargs : passed to `~matplotlib.pyplot.errorbar`.
        '''

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, sharex=True)

        ax1 = axes[0]
        ax1.errorbar(self.lags.value, self.tsallis_table['logA'],
                     yerr=self.tsallis_table['logA_stderr'],
                     **kwargs)
        ax1.set_ylabel(r"log A")
        ax1.grid()

        ax2 = axes[1]
        ax2.errorbar(self.lags.value, self.tsallis_table['w2'],
                     yerr=self.tsallis_table['w2_stderr'],
                     **kwargs)
        ax2.set_ylabel(r"$w^2$")
        ax2.grid()

        ax3 = axes[2]
        ax3.errorbar(self.lags.value, self.tsallis_table['q'],
                     yerr=self.tsallis_table['q_stderr'],
                     **kwargs)
        ax3.set_ylabel(r"q")
        ax3.grid()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def plot_fit(self, save_name=None, color='r',
                 fit_color='k'):
        '''
        Plot the distributions and fits to the Tsallis function.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.

        '''
        import matplotlib.pyplot as plt

        if fit_color is None:
            fit_color = color

        fig, axes = plt.subplots(len(self.lags), 1, sharex=True)

        for vals in zip(self.lags, self.lag_distribs,
                        self.lag_arrays, self.tsallis_params,
                        axes):

            lag, dist, arr, params, ax = vals

            ax.plot(dist[0], dist[1], 'D', color=color,
                    label="Lag {}".format(lag), alpha=0.5)
            ax.plot(dist[0], tsallis_function(dist[0], *params),
                    color=fit_color)

            # Indicate which data was used for the fits.
            # Only if sigma-clipping is applied.
            if self._sigma_clip is not None:
                ax.axvline(self._sigma_clip, color='r', linestyle='--',
                           alpha=0.7)
                ax.axvline(-self._sigma_clip, color='r', linestyle='--',
                           alpha=0.7)

            ax.legend(frameon=True, loc='best')

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, verbose=False, num_bins=None, periodic=True, sigma_clip=5,
            save_name=None):
        '''
        Run all steps.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        num_bins : int, optional
            Sets the number of bins to use in the lag histograms. Passed
            to `~Tsallis.make_tsallis`.
        periodic : bool, optional
            Treat periodic boundaries. Passed
            to `~Tsallis.make_tsallis`. Enabled by default.
        sigma_clip : float
            Sets the sigma value to clip data at.
            Passed to :func:`fit_tsallis`.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.make_tsallis(num_bins=num_bins, periodic=periodic)
        self.fit_tsallis(sigma_clip=sigma_clip)

        if verbose:
            # print the table of parameters
            print(self.tsallis_table)

            self.plot_fit(save_name=save_name)

        return self


# class Tsallis_Distance(object):

#     '''
#     Distance Metric for the Tsallis Distribution.

#     Parameters
#     ----------
#     array1 : %(dtypes)s
#         2D datas.
#     array2 : %(dtypes)s
#         2D datas.
#     lags : `~astropy.units.Quantity`
#         Lags to calculate at.
#     fiducial_model : Tsallis
#         Computed Tsallis object. use to avoid recomputing.
#     tsallis1_kwargs : dict, optional
#         Pass kwargs to `~Tsallis.run` for array1.
#     tsallis2_kwargs : dict, optional
#         Pass kwargs to `~Tsallis.run` for array2.
#     '''

#     __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

#     def __init__(self, array1, array2, lags=None, tsallis1_kwargs={},
#                  tsallis2_kwargs={}, fiducial_model=None,):
#         super(Tsallis_Distance, self).__init__()

#         if fiducial_model is not None:
#             self.tsallis1 = fiducial_model
#         else:
#             self.tsallis1 = \
#                 Tsallis(array1, lags=lags).run(verbose=False,
#                                                **tsallis1_kwargs)

#         self.tsallis2 = \
#             Tsallis(array2, lags=lags).run(verbose=False,
#                                            **tsallis2_kwargs)

#         self.distance = None

#     def distance_metric(self, verbose=False, save_name=None):
#         '''

#         We do not consider the parameter a in the distance metric. Since we
#         are fitting to a PDF, a is related to the number of data points and
#         is therefore not a true measure of the differences between the data
#         sets. The distance is computed by summing the squared difference of
#         the parameters, normalized by the sums of the squares, for each lag.
#         The total distance the sum between the two parameters.

#         Parameters
#         ----------
#         verbose : bool, optional
#             Enables plotting.
#         save_name : str,optional
#             Save the figure when a file name is given.
#         '''

#         w1 = self.tsallis1.tsallis_params[:, 1]
#         w2 = self.tsallis2.tsallis_params[:, 1]

#         q1 = self.tsallis1.tsallis_params[:, 2]
#         q2 = self.tsallis2.tsallis_params[:, 2]

#         # diff_a = (a1-a2)**2.
#         diff_w = (w1 - w2) ** 2. / (w1 ** 2. + w2 ** 2.)
#         diff_q = (q1 - q2) ** 2. / (q1 ** 2. + q2 ** 2.)

#         self.distance = np.sum(diff_w + diff_q)

#         if verbose:
#             import matplotlib.pyplot as p
#             lags = self.tsallis1.lags
#             p.plot(lags, diff_w, "rD", label="Difference of w")
#             p.plot(lags, diff_q, "go", label="Difference of q")
#             p.legend()
#             p.xscale('log', basex=2)
#             p.ylabel("Normalized Difference")
#             p.xlabel("Lags (pixels)")
#             p.grid(True)

#             if save_name is not None:
#                 p.savefig(save_name)
#                 p.close()
#             else:
#                 p.show()
#         return self


def tsallis_function(x, *p):
    '''
    Tsallis distribution function as given in Tofflemire
    Implemented in log form. The expected parameters are
    log A, w^2, and q.

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
    Clip to values between +/- sigma.

    Parameters
    ----------
    x : numpy.ndarray
        x-data
    y : numpy.ndarray
        y-data
    '''

    clip_mask = np.logical_and(y < sigma, y > -sigma)
    # And ensure all data is finite for fitting
    finite_mask = np.logical_and(np.isfinite(y), np.isfinite(x))

    all_mask = np.logical_and(clip_mask, finite_mask)

    return x[all_mask], y[all_mask]
