# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

'''

Dendrogram statistics as described in Burkhart et al. (2013)
Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

Requires the astrodendro package (http://github.com/astrodendro/dendro-core)

'''

import numpy as np
from warnings import warn
import statsmodels.api as sm
from astropy.utils.console import ProgressBar
import warnings

try:
    from astrodendro import Dendrogram, periodic_neighbours
    astrodendro_flag = True
except ImportError:
    Warning("Need to install astrodendro to use dendrogram statistics.")
    astrodendro_flag = False

from ..stats_utils import hellinger, common_histogram_bins, standardize
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, twod_types
from .mecdf import mecdf


class Dendrogram_Stats(BaseStatisticMixIn):

    """
    Dendrogram statistics as described in Burkhart et al. (2013)
    Two statistics are contained:
    * number of leaves & branches vs. :math:`\delta` parameter
    * statistical moments of the intensity histogram

    Parameters
    ----------

    data : %(dtypes)s
        Data to create the dendrogram from.
    min_deltas : {`~numpy.ndarray`, 'auto', None}, optional
        Minimum deltas of leaves in the dendrogram. Multiple values must
        be given in increasing order to correctly prune the dendrogram.
        The default estimates delta levels from percentiles in the data.
    dendro_params : dict
        Further parameters for the dendrogram algorithm
        (see www.dendrograms.org for more info).
    num_deltas : int, optional
        Number of min_delta values to use when `min_delta='auto'`.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types +
                                      threed_types)}

    def __init__(self, data, header=None, min_deltas='auto',
                 dendro_params=None, num_deltas=10):
        super(Dendrogram_Stats, self).__init__()

        if not astrodendro_flag:
            raise ImportError("astrodendro must be installed to use "
                              "Dendrogram_Stats.")

        self.input_data_header(data, header)

        if dendro_params is None:
            self.dendro_params = {"min_npix": 10,
                                  "min_value": 0.001,
                                  "min_delta": 0.1}
        else:
            self.dendro_params = dendro_params

        if min_deltas == 'auto':
            self.autoset_min_deltas(num=num_deltas)
        else:
            self.min_deltas = min_deltas

    @property
    def min_deltas(self):
        '''
        Array of min_delta values to compute the dendrogram.
        '''
        return self._min_deltas

    @min_deltas.setter
    def min_deltas(self, value):
        # In the case where only one min_delta is given
        if "min_delta" in self.dendro_params and value is None:
            self._min_deltas = np.array([self.dendro_params["min_delta"]])
        else:
            # Multiple values given. Ensure they are in increasing order
            if not (np.diff(value) > 0).all():
                raise ValueError("Multiple values of min_delta must be given "
                                 "in increasing order.")

            if not isinstance(value, np.ndarray):
                self._min_deltas = np.array([value])
            else:
                self._min_deltas = value

    def autoset_min_deltas(self, num=10):
        '''
        Create an array delta values that the dendrogram will be pruned to.
        Creates equally-spaced delta values between the minimum value set in
        `~Dendrogram_Stats.dendro_params` and the maximum in the data. The last
        delta (which would only occur at the peak in the data) is removed.

        Parameters
        ----------
        num : int, optional
            Number of delta values to create.
        '''

        min_val = self.dendro_params.get('min_value', -np.inf)

        min_delta = self.dendro_params.get('min_delta', 1e-5)

        # Calculate the ptp above the min_val
        ptp = np.nanmax(self.data[self.data > min_val]) - min_val

        self.min_deltas = np.linspace(min_delta, ptp, num + 1)[:-1]

    def compute_dendro(self, show_progress=False, save_dendro=False,
                       dendro_name=None, dendro_obj=None,
                       periodic_bounds=False):
        '''
        Compute the dendrogram and prune to the minimum deltas.
        ** min_deltas must be in ascending order! **

        Parameters
        ----------
        show_progress : optional, bool
            Enables the progress bar in astrodendro.
        save_dendro : optional, bool
            Saves the dendrogram in HDF5 format. **Requires pyHDF5**
        dendro_name : str, optional
            Save name when save_dendro is enabled. ".hdf5" appended
            automatically.
        dendro_obj : Dendrogram, optional
            Input a pre-computed dendrogram object. It is assumed that
            the dendrogram has already been computed!
        periodic_bounds : bool, optional
            Enable when the data is periodic in the spatial dimensions.
        '''

        self._numfeatures = np.empty(self.min_deltas.shape, dtype=int)
        self._values = []

        if dendro_obj is None:
            if periodic_bounds:
                # Find the spatial dimensions
                num_axes = self.data.ndim
                spat_axes = []
                for i, axis_type in enumerate(self._wcs.get_axis_types()):
                    if axis_type["coordinate_type"] == u"celestial":
                        spat_axes.append(num_axes - i - 1)
                neighbours = periodic_neighbours(spat_axes)
            else:
                neighbours = None

            d = Dendrogram.compute(self.data, verbose=show_progress,
                                   min_delta=self.min_deltas[0],
                                   min_value=self.dendro_params["min_value"],
                                   min_npix=self.dendro_params["min_npix"],
                                   neighbours=neighbours)
        else:
            d = dendro_obj
        self._numfeatures[0] = len(d)
        self._values.append(np.array([struct.vmax for struct in
                                      d.all_structures]))

        if len(self.min_deltas) > 1:

            # Another progress bar for pruning steps
            if show_progress:
                print("Pruning steps.")
                bar = ProgressBar(len(self.min_deltas[1:]))

            for i, delta in enumerate(self.min_deltas[1:]):
                d.prune(min_delta=delta)
                self._numfeatures[i + 1] = len(d)
                self._values.append(np.array([struct.vmax for struct in
                                              d.all_structures]))

                if show_progress:
                    bar.update(i + 1)

    @property
    def numfeatures(self):
        '''
        Number of branches and leaves at each value of min_delta
        '''
        return self._numfeatures

    @property
    def values(self):
        '''
        Array of peak intensity values of leaves and branches at all values of
        min_delta.
        '''
        return self._values

    def make_hists(self, min_number=10, **kwargs):
        '''
        Creates histograms based on values from the tree.
        *Note:* These histograms are remade when calculating the distance to
        ensure the proper form for the Hellinger distance.

        Parameters
        ----------
        min_number : int, optional
            Minimum number of structures needed to create a histogram.
        '''

        hists = []

        for value in self.values:

            if len(value) < min_number:
                hists.append([np.zeros((0, ))] * 2)
                continue

            if 'bins' not in kwargs:
                bins = int(np.sqrt(len(value)))
            else:
                bins = kwargs['bins']
                kwargs.pop('bins')

            hist, bins = np.histogram(value, bins=bins, **kwargs)
            bin_cents = (bins[:-1] + bins[1:]) / 2
            hists.append([bin_cents, hist])

        self._hists = hists

    @property
    def hists(self):
        '''
        Histogram values and bins computed from the peak intensity in all
        structures. One set of values and bins are returned for each value
        of `~Dendro_Statistics.min_deltas`
        '''
        return self._hists

    def fit_numfeat(self, size=5, verbose=False):
        '''
        Fit a line to the power-law tail. The break is approximated using
        a moving window, computing the standard deviation. A spike occurs at
        the break point.

        Parameters
        ----------
        size : int. optional
            Size of std. window. Passed to std_window.
        verbose : bool, optional
            Shows the model summary.
        '''

        if len(self.numfeatures) == 1:
            raise ValueError("Multiple min_delta values must be provided to "
                             "perform fitting. Only one value was given.")

        nums = self.numfeatures[self.numfeatures > 1]
        deltas = self.min_deltas[self.numfeatures > 1]

        # Find the position of the break
        break_pos = std_window(nums, size=size)
        self.break_pos = deltas[break_pos]

        # Still enough point to fit to?
        if len(deltas[break_pos:]) < 2:
            raise ValueError("Too few points to fit. Try running with more "
                             "min_deltas or lowering the std. window size.")

        # Remove points where there is only 1 feature or less.
        self._fitvals = [np.log10(deltas[break_pos:]),
                         np.log10(nums[break_pos:])]

        x = sm.add_constant(self.fitvals[0])

        self._model = sm.OLS(self.fitvals[1], x).fit(cov_type='HC3')

        if verbose:
            print(self.model.summary())

        errors = self.model.bse

        self._tail_slope = self.model.params[-1]
        self._tail_slope_err = errors[-1]

    @property
    def model(self):
        '''
        Power-law tail fit model.
        '''
        return self._model

    @property
    def fitvals(self):
        '''
        Log values of delta and number of structures used for the power-law
        tail fit.
        '''
        return self._fitvals

    @property
    def tail_slope(self):
        '''
        Slope of power-law tail.
        '''
        return self._tail_slope

    @property
    def tail_slope_err(self):
        '''
        1-sigma error on slope of power-law tail.
        '''
        return self._tail_slope_err

    @staticmethod
    def load_dendrogram(hdf5_file, min_deltas=None):
        '''
        Load in a previously saved dendrogram. **Requires pyHDF5**

        Parameters
        ----------
        hdf5_file : str
            Name of saved file.
        min_deltas : numpy.ndarray or list
            Minimum deltas of leaves in the dendrogram.
        '''

        dendro = Dendrogram.load_from(hdf5_file)

        self = Dendrogram_Stats(dendro.data, min_deltas=min_deltas,
                                dendro_params=dendro.params)

        return self

    def plot_fit(self, save_name=None, show_hists=True, color='r',
                 fit_color='k', symbol='o'):
        '''

        Parameters
        ----------
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            The unit to show the x-axis in.
        show_hists : bool, optional
            Plot the histograms of intensity. Requires
            `~Dendrogram_Stats.make_hists` to be run first.
        color : {str, RGB tuple}, optional
            Color to show the delta-variance curve in.
        fit_color : {str, RGB tuple}, optional
            Color of the fitted line. Defaults to `color` when no input is
            given.
        '''

        import matplotlib.pyplot as plt

        if not show_hists:
            ax1 = plt.subplot(111)
        else:
            ax1 = plt.subplot(121)

        if fit_color is None:
            fit_color = color

        ax1.plot(self.fitvals[0], self.fitvals[1], symbol, color=color)
        ax1.plot(self.fitvals[0], self.model.fittedvalues, color=fit_color)
        plt.xlabel(r"log $\delta$")
        plt.ylabel(r"log Number of Features")

        if show_hists:
            ax2 = plt.subplot(122)

            if not hasattr(self, "_hists"):
                raise ValueError("Histograms were not computed with "
                                 "Dendrogram_Stats.make_hists. Cannot plot.")

            for bins, vals in self.hists:
                if bins.size < 1:
                    continue
                bin_width = np.abs(bins[1] - bins[0])
                ax2.bar(bins, vals, align="center",
                        width=bin_width, alpha=0.25,
                        color=color)
                plt.xlabel("Data Value")

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, periodic_bounds=False, verbose=False, save_name=None,
            show_progress=True, dendro_obj=None, save_results=False,
            output_name=None, fit_kwargs={}, make_hists=True, hist_kwargs={}):
        '''

        Compute dendrograms. Necessary to maintain the package format.

        Parameters
        ----------
        periodic_bounds : bool or list, optional
            Enable when the data is periodic in the spatial dimensions. Passing
            a two-element list can be used to individually set how the
            boundaries are treated for the datasets.
        verbose : optional, bool
            Enable plotting of results.
        save_name : str,optional
            Save the figure when a file name is given.
        show_progress : optional, bool
            Enables progress bars while making the dendrogram.
        dendro_obj : Dendrogram, optional
            Pass a pre-computed dendrogram object. **MUST have min_delta set
            at or below the smallest value in`~Dendro_Statistics.min_deltas`.**
        save_results : bool, optional
            Save the statistic results as a pickle file. See
            `~Dendro_Statistics.save_results`.
        output_name : str, optional
            Filename used when `save_results` is enabled. Must be given when
            saving.
        fit_kwargs : dict, optional
            Passed to `~Dendro_Statistics.fit_numfeat`.
        make_hists : bool, optional
            Enable computing histograms.
        hist_kwargs : dict, optional
            Passed to `~Dendro_Statistics.make_hists`.
        '''
        self.compute_dendro(show_progress=show_progress, dendro_obj=dendro_obj,
                            periodic_bounds=periodic_bounds)
        self.fit_numfeat(verbose=verbose, **fit_kwargs)

        if make_hists:
            self.make_hists(**hist_kwargs)

        if verbose:
            self.plot_fit(save_name=save_name, show_hists=make_hists)

        if save_results:
            self.save_results(output_name=output_name)


class Dendrogram_Distance(object):

    """
    Calculate the distance between 2 cubes using dendrograms. The number of
    features vs. minimum delta is fit to a linear model, with an interaction
    term to gauge the difference. The distance is the t-statistic of that
    parameter. The Hellinger distance is computed for the histograms at each
    minimum delta value. The distance is the average of the Hellinger
    distances.

    .. note:: When passing a computed `~DeltaVariance` class for `dataset1`
              or `dataset2`, it may be necessary to recompute the
              dendrogram if `~Dendrogram_Stats.min_deltas` does not equal
              `min_deltas` generated here (or passed as kwarg).

    Parameters
    ----------
    dataset1 : %(dtypes)s or `~Dendrogram_Stats`
        Data cube or 2D image. Or pass a
        `~Dendrogram_Stats` class that may be pre-computed.
        where the dendrogram statistics are saved.
    dataset2 : %(dtypes)s or `~Dendrogram_Stats`
        See `dataset1` above.
    min_deltas : numpy.ndarray or list
        Minimum deltas (branch heights) of leaves in the dendrogram. The set
        of dendrograms must be computed with the same minimum branch heights.
    nbins : str or float, optional
        Number of bins for the histograms. 'best' sets
        that number using the square root of the average
        number of features between the histograms to be
        compared.
    min_features : int, optional
        The minimum number of features (branches and leaves) for the histogram
        be used in the histogram distance.
    dendro_params : dict or list of dicts, optional
        Further parameters for the dendrogram algorithm
        (see the `astrodendro documentation <dendrograms.readthedocs.io>`_
        for more info). If a list of dictionaries is
        given, the first list entry should be the dictionary for `dataset1`,
        and the second for `dataset2`.
    dendro_kwargs : dict, optional
        Passed to `~turbustat.statistics.Dendrogram_Stats.run`.
    dendro2_kwargs : None, dict, optional
        Passed to `~turbustat.statistics.Dendrogram_Stats.run` for `dataset2`.
        When `None` is given, parameters given in `dendro_kwargs` will be used
        for both datasets.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types +
                                      threed_types)}

    def __init__(self, dataset1, dataset2, min_deltas=None, nbins="best",
                 min_features=100, dendro_params=None,
                 dendro_kwargs={}, dendro2_kwargs=None):

        if not astrodendro_flag:
            raise ImportError("astrodendro must be installed to use "
                              "Dendrogram_Stats.")

        self.nbins = nbins

        if min_deltas is None:
            # min_deltas = np.append(np.logspace(-1.5, -0.7, 8),
            #                        np.logspace(-0.6, -0.35, 10))
            warnings.warn("Using default min_deltas ranging from 10^-2.5 to"
                          "10^0.5. Check whether this range is appropriate"
                          " for your data.")
            min_deltas = np.logspace(-2.5, 0.5, 100)

        if dendro_params is not None:
            if isinstance(dendro_params, list):
                dendro_params1 = dendro_params[0]
                dendro_params2 = dendro_params[1]
            elif isinstance(dendro_params, dict):
                dendro_params1 = dendro_params
                dendro_params2 = dendro_params
            else:
                raise TypeError("dendro_params is a {}. It must be a dictionary"
                                ", or a list containing a dictionary entries."
                                .format(type(dendro_params)))
        else:
            dendro_params1 = None
            dendro_params2 = None

        if dendro2_kwargs is None:
            dendro2_kwargs = dendro_kwargs

        # if fiducial_model is not None:
        #     self.dendro1 = fiducial_model
        # elif isinstance(dataset1, str):
        #     self.dendro1 = Dendrogram_Stats.load_results(dataset1)
        if isinstance(dataset1, Dendrogram_Stats):
            self.dendro1 = dataset1
            # Check if we need to re-run the stat
            has_slope = hasattr(self.dendro1, "_tail_slope")
            match_deltas = (self.dendro1.min_deltas == min_deltas).all()
            if not has_slope or not match_deltas:
                warn("Dendrogram_Stats needs to be re-run for dataset1 "
                     "to compute the slope or have the same set of "
                     "`min_deltas`.")
                dendro_kwargs.pop('make_hists', None)
                dendro_kwargs.pop('verbose', None)
                self.dendro1.run(verbose=False, make_hists=False,
                                 **dendro_kwargs)
        else:
            self.dendro1 = Dendrogram_Stats(dataset1, min_deltas=min_deltas,
                                            dendro_params=dendro_params1)
            dendro_kwargs.pop('make_hists', None)
            dendro_kwargs.pop('verbose', None)
            self.dendro1.run(verbose=False, make_hists=False,
                             **dendro_kwargs)

        # if isinstance(dataset2, str):
        #     self.dendro2 = Dendrogram_Stats.load_results(dataset2)
        if isinstance(dataset2, Dendrogram_Stats):
            self.dendro2 = dataset2
            # Check if we need to re-run the stat
            has_slope = hasattr(self.dendro2, "_tail_slope")
            match_deltas = (self.dendro2.min_deltas == min_deltas).all()
            if not has_slope or not match_deltas:
                warn("Dendrogram_Stats needs to be re-run for dataset2 "
                     "to compute the slope or have the same set of "
                     "`min_deltas`.")
                dendro_kwargs.pop('make_hists', None)
                dendro_kwargs.pop('verbose', None)
                self.dendro2.run(verbose=False, make_hists=False,
                                 **dendro2_kwargs)
        else:
            self.dendro2 = \
                Dendrogram_Stats(dataset2, min_deltas=min_deltas,
                                 dendro_params=dendro_params2)
            dendro_kwargs.pop('make_hists', None)
            dendro_kwargs.pop('verbose', None)
            self.dendro2.run(verbose=False, make_hists=False,
                             **dendro2_kwargs)

        # Set the minimum number of components to create a histogram
        cutoff1 = np.argwhere(self.dendro1.numfeatures > min_features)
        cutoff2 = np.argwhere(self.dendro2.numfeatures > min_features)
        if cutoff1.any():
            cutoff1 = cutoff1[-1]
        else:
            raise ValueError("The dendrogram from dataset1 does not contain the"
                             " necessary number of features, %s. Lower"
                             " min_features or alter min_deltas."
                             % (min_features))
        if cutoff2.any():
            cutoff2 = cutoff2[-1]
        else:
            raise ValueError("The dendrogram from dataset2 does not contain the"
                             " necessary number of features, %s. Lower"
                             " min_features or alter min_deltas."
                             % (min_features))

        self.cutoff = np.min([cutoff1, cutoff2])

    @property
    def num_distance(self):
        '''
        Distance between slopes from the for to the
        log Number of features vs. branch height.
        '''
        return self._num_distance

    def numfeature_stat(self, verbose=False,
                        save_name=None, plot_kwargs1={},
                        plot_kwargs2={}):
        '''
        Calculate the distance based on the number of features statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str, optional
            Saves the plot when a filename is given.
        plot_kwargs1 : dict, optional
            Set the color, symbol, and label for dataset1
            (e.g., plot_kwargs1={'color': 'b', 'symbol': 'D', 'label': '1'}).
        plot_kwargs2 : dict, optional
            Set the color, symbol, and label for dataset2.
        '''

        self._num_distance = \
            np.abs(self.dendro1.tail_slope - self.dendro2.tail_slope) / \
            np.sqrt(self.dendro1.tail_slope_err**2 +
                    self.dendro2.tail_slope_err**2)

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

            plt.figure()

            # Dendrogram 1
            plt.plot(self.dendro1.fitvals[0], self.dendro1.fitvals[1],
                     plot_kwargs1['symbol'], label=plot_kwargs1['label'],
                     color=plot_kwargs1['color'])
            plt.plot(self.dendro1.fitvals[0], self.dendro1.model.fittedvalues,
                     plot_kwargs1['color'])

            # Dendrogram 2
            plt.plot(self.dendro2.fitvals[0], self.dendro2.fitvals[1],
                     plot_kwargs2['symbol'], label=plot_kwargs2['label'],
                     color=plot_kwargs2['color'])
            plt.plot(self.dendro2.fitvals[0], self.dendro2.model.fittedvalues,
                     plot_kwargs2['color'])

            plt.grid(True)
            plt.xlabel(r"log $\delta$")
            plt.ylabel("log Number of Features")
            plt.legend(loc='best')

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self

    @property
    def histogram_distance(self):
        return self._histogram_distance

    def histogram_stat(self, verbose=False,
                       save_name=None,
                       plot_kwargs1={},
                       plot_kwargs2={}):
        '''
        Computes the distance using histograms.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str, optional
            Saves the plot when a filename is given.
        plot_kwargs1 : dict, optional
            Set the color, symbol, and label for dataset1
            (e.g., plot_kwargs1={'color': 'b', 'symbol': 'D', 'label': '1'}).
        plot_kwargs2 : dict, optional
            Set the color, symbol, and label for dataset2.
        '''

        if self.nbins == "best":
            self.nbins = [np.floor(np.sqrt((n1 + n2) / 2.)) for n1, n2 in
                          zip(self.dendro1.numfeatures[:self.cutoff],
                              self.dendro2.numfeatures[:self.cutoff])]
        else:
            self.nbins = [self.nbins] * \
                len(self.dendro1.numfeatures[:self.cutoff])

        self.nbins = np.array(self.nbins, dtype=int)

        self.histograms1 = \
            np.empty((len(self.dendro1.numfeatures[:self.cutoff]),
                      np.max(self.nbins)))
        self.histograms2 = \
            np.empty((len(self.dendro2.numfeatures[:self.cutoff]),
                      np.max(self.nbins)))

        self.bins = []

        for n, (data1, data2, nbin) in enumerate(
                zip(self.dendro1.values[:self.cutoff],
                    self.dendro2.values[:self.cutoff], self.nbins)):

            stand_data1 = standardize(data1)
            stand_data2 = standardize(data2)

            bins = common_histogram_bins(stand_data1, stand_data2,
                                         nbins=nbin + 1)

            self.bins.append(bins)

            hist1 = np.histogram(stand_data1, bins=bins,
                                 density=True)[0]
            self.histograms1[n, :] = \
                np.append(hist1, (np.max(self.nbins) -
                                  bins.size + 1) * [np.NaN])

            hist2 = np.histogram(stand_data2, bins=bins,
                                 density=True)[0]
            self.histograms2[n, :] = \
                np.append(hist2, (np.max(self.nbins) -
                                  bins.size + 1) * [np.NaN])

            # Normalize
            self.histograms1[n, :] /= np.nansum(self.histograms1[n, :])
            self.histograms2[n, :] /= np.nansum(self.histograms2[n, :])

        self.mecdf1 = mecdf(self.histograms1)
        self.mecdf2 = mecdf(self.histograms2)

        self._histogram_distance = hellinger_stat(self.histograms1,
                                                  self.histograms2)

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

            plt.figure()

            ax1 = plt.subplot(2, 2, 1)
            ax1.set_title(plot_kwargs1['label'])
            ax1.set_ylabel("ECDF")
            for n in range(len(self.dendro1.min_deltas[:self.cutoff])):
                ax1.plot((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                         self.mecdf1[n, :][:self.nbins[n]],
                         plot_kwargs1['symbol'],
                         color=plot_kwargs1['color'])
            ax1.axes.xaxis.set_ticklabels([])
            ax2 = plt.subplot(2, 2, 2)
            ax2.set_title(plot_kwargs2['label'])
            ax2.axes.xaxis.set_ticklabels([])
            ax2.axes.yaxis.set_ticklabels([])
            for n in range(len(self.dendro2.min_deltas[:self.cutoff])):
                ax2.plot((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                         self.mecdf2[n, :][:self.nbins[n]],
                         plot_kwargs2['symbol'],
                         color=plot_kwargs2['color'])
            ax3 = plt.subplot(2, 2, 3)
            ax3.set_ylabel("PDF")
            for n in range(len(self.dendro1.min_deltas[:self.cutoff])):
                bin_width = self.bins[n][1] - self.bins[n][0]
                ax3.bar((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                        self.histograms1[n, :][:self.nbins[n]],
                        align="center", width=bin_width, alpha=0.25,
                        color=plot_kwargs1['color'])

            ax3.set_xlabel("z-score")
            ax4 = plt.subplot(2, 2, 4)
            for n in range(len(self.dendro2.min_deltas[:self.cutoff])):
                bin_width = self.bins[n][1] - self.bins[n][0]
                ax4.bar((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                        self.histograms2[n, :][:self.nbins[n]],
                        align="center", width=bin_width, alpha=0.25,
                        color=plot_kwargs2['color'])

            ax4.set_xlabel("z-score")
            ax4.axes.yaxis.set_ticklabels([])

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self

    def distance_metric(self, verbose=False, save_name=None,
                        plot_kwargs1={}, plot_kwargs2={}):
        '''
        Calculate both distance metrics.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str, optional
            Save plots by passing a file name. `hist_distance` and
            `num_distance` will be appended to the file name to distinguish
            the plots made with the two metrics.
        plot_kwargs1 : dict, optional
            Set the color, symbol, and label for dataset1
            (e.g., plot_kwargs1={'color': 'b', 'symbol': 'D', 'label': '1'}).
        plot_kwargs2 : dict, optional
            Set the color, symbol, and label for dataset2.
        '''

        if save_name is not None:
            import os
            # Distinguish name for the two plots
            base_name, extens = os.path.splitext(save_name)

            save_name_hist = "{0}.hist_distance{1}".format(base_name, extens)
            save_name_num = "{0}.num_distance{1}".format(base_name, extens)
        else:
            save_name_hist = None
            save_name_num = None

        self.histogram_stat(verbose=verbose, plot_kwargs1=plot_kwargs1,
                            plot_kwargs2=plot_kwargs2,
                            save_name=save_name_hist)
        self.numfeature_stat(verbose=verbose, plot_kwargs1=plot_kwargs1,
                             plot_kwargs2=plot_kwargs2,
                             save_name=save_name_num)

        return self


def DendroDistance(*args, **kwargs):
    '''
    Old name for the Dendrogram_Distance class.
    '''

    warn("Use the new 'Dendrogram_Distance' class. 'DendroDistance' is deprecated and will"
         " be removed in a future release.", Warning)

    return Dendrogram_Distance(*args, **kwargs)


def hellinger_stat(x, y):
    '''
    Compute the Hellinger statistic of multiple samples.
    '''

    assert x.shape == y.shape

    if len(x.shape) == 1:
        return hellinger(x, y)
    else:
        dists = np.empty((x.shape[0], 1))
        for n in range(x.shape[0]):
            dists[n, 0] = hellinger(x[n, :], y[n, :])
        return np.mean(dists)


def std_window(y, size=5, return_results=False):
    '''
    Uses a moving standard deviation window to find where the powerlaw break
    is.

    Parameters
    ----------
    y : np.ndarray
        Data.
    size : int, optional
        Odd integer which sets the window size.
    return_results : bool, optional
        If enabled, returns the results of the window. Otherwise, only the
        position of the break is returned.
    '''

    half_size = (size - 1) // 2

    shape = max(y.shape)

    stds = np.empty((shape - size + 1))

    for i in range(half_size, shape - half_size):
        stds[i - half_size] = np.std(y[i - half_size: i + half_size])

    # Now find the max
    break_pos = np.argmax(stds) + half_size

    if return_results:
        return break_pos, stds

    return break_pos
