# Licensed under an MIT open source license - see LICENSE


'''

Dendrogram statistics as described in Burkhart et al. (2013)
Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

Requires the astrodendro package (http://github.com/astrodendro/dendro-core)

'''

import numpy as np
from copy import deepcopy
import cPickle as pickle
import statsmodels.api as sm
from mecdf import mecdf
from astrodendro import Dendrogram

from ..stats_utils import hellinger, common_histogram_bins, standardize


class Dendrogram_Stats(object):

    """
    Dendrogram statistics as described in Burkhart et al. (2013)
    Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

    Parameters
    ----------

    cube : numpy.ndarray
        Data cube.
    min_deltas : numpy.ndarray or list
        Minimum deltas of leaves in the dendrogram.
    dendro_params : dict
        Further parameters for the dendrogram algorithm
        (see www.dendrograms.org for more info).

    """

    def __init__(self, cube, min_deltas=None, dendro_params=None):
        super(Dendrogram_Stats, self).__init__()
        self.cube = cube
        self.min_deltas = min_deltas

        if dendro_params is None:
            self.dendro_params = {"min_npix": 10,
                                  "min_value": 0.001}
        else:
            # poss_keys = dir(pruning)
            # for key in dendro_params.keys():
            #     if key not in poss_keys:
            #         raise KeyError(key + " is not a valid pruning parameter.")
            self.dendro_params = dendro_params

        self.numfeatures = np.empty(self.min_deltas.shape)
        self.values = []
        self.histograms = []

    def compute_dendro(self, verbose=False, save_dendro=False,
                       dendro_name=None, dendro_obj=None):
        '''
        Compute the dendrogram and prune to the minimum deltas.
        ** min_deltas must be in ascending order! **

        Parameters
        ----------
        verbose : optional, bool
            Enables the progress bar in astrodendro.
        save_dendro : optional, bool
            Saves the dendrogram in HDF5 format. **Requires pyHDF5**
        dendro_name : str, optional
            Save name when save_dendro is enabled. ".hdf5" appended
            automatically.
        dendro_obj : Dendrogram, optional
            Input a pre-computed dendrogram object. It is assumed that
            the dendrogram has already been computed!

        '''

        if dendro_obj is None:
            d = \
                Dendrogram.compute(self.cube, verbose=verbose,
                                   min_delta=self.min_deltas[0],
                                   min_value=self.dendro_params["min_value"],
                                   min_npix=self.dendro_params["min_npix"])
        else:
            d = dendro_obj
        self.numfeatures[0] = len(d)
        self.values.append(
            np.asarray([struct.vmax for struct in d.all_structures]))

        for i, delta in enumerate(self.min_deltas[1:]):
            if verbose:
                print "On %s of %s" % (i + 1, len(self.min_deltas[1:]))
            d.prune(min_delta=delta)
            self.numfeatures[i + 1] = len(d)
            self.values.append([struct.vmax for struct in d.all_structures])

        return self

    def make_hist(self):
        '''
        Creates histograms based on values from the tree.
        *Note:* These histograms are remade whenc calculating the distance to
        ensure the proper form for the Hellinger distance.

        Returns
        -------
        hists : list
            Each list entry contains the histogram values and bins for a
            value of delta.
        '''

        hists = []

        for value in self.values:
            hist, bins = np.histogram(value, bins=int(np.sqrt(len(value))))
            hists.append([hist, bins])

        return hists

    def fit_numfeat(self, size=5, verbose=False):
        '''
        Fit a line to the power-law tail. The break is approximated using
        a moving window, computing the standard deviation. A spike occurs at
        the break point.

        Parameters
        ----------
        size : int. optional
            Size of window. Passed to std_window.
        verbose : bool, optional
            Shows the model summary.
        '''

        nums = self.numfeatures[self.numfeatures > 1]
        deltas = self.min_deltas[self.numfeatures > 1]

        # Find the position of the break
        break_pos = std_window(nums, size=size)
        self.break_pos = deltas[break_pos]

        # Remove points where there is only 1 feature or less.
        self.x = np.log10(deltas[break_pos:])
        self.y = np.log10(nums[break_pos:])

        x = sm.add_constant(self.x)

        self.model = sm.OLS(self.y, x).fit()

        if verbose:
            print self.model.summary()

        cov_matrix = self.model.cov_params()
        errors = \
            np.asarray([np.sqrt(cov_matrix[i, i])
                        for i in range(cov_matrix.shape[0])])

        self.tail_slope = self.model.params[-1]
        self.tail_slope_err = errors[-1]

        return self

    def save_results(self, output_name=None, keep_data=False):
        '''
        Save the results of the dendrogram statistics to avoid re-computing.
        The pickled file will not include the data cube by default.
        '''

        if output_name is None:
            output_name = "dendrogram_stats_output.pkl"

        if output_name[-4:] != ".pkl":
            output_name += ".pkl"

        self_copy = deepcopy(self)

        # Don't keep the whole cube unless keep_data enabled.
        if not keep_data:
            self_copy.cube = None

        with open(output_name, 'wb') as output:
                pickle.dump(self_copy, output, -1)

    @staticmethod
    def load_results(pickle_file):
        '''
        Load in a saved pickle file.

        Parameters
        ----------
        pickle_file : str
            Name of filename to load in.
        '''

        with open(pickle_file, 'rb') as input:
                self = pickle.load(input)

        return self

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

        self.compute_dendro(dendro_obj=dendro)

        return self

    def run(self, verbose=False, dendro_verbose=False,
            save_results=False, output_name=None):
        '''

        Compute dendrograms. Necessary to maintain the package format.

        Parameters
        ----------
        verbose : optional, bool

        dendro_verbose : optional, bool
            Prints out updates while making the dendrogram.
        '''
        self.compute_dendro(verbose=dendro_verbose)
        self.fit_numfeat(verbose=verbose)

        if verbose:
            import matplotlib.pyplot as p

            p.plot(self.x, self.y, 'bD')
            p.plot(self.x, self.model.fittedvalues, 'g')
            p.show()

        if save_results:
            self.save_results(output_name=output_name)


class DendroDistance(object):

    """
    Calculate the distance between 2 cubes using dendrograms. The number of
    features vs. minimum delta is fit to a linear model, with an interaction
    term o gauge the difference. The distance is the t-statistic of that
    parameter. The Hellinger distance is computed for the histograms at each
    minimum delta value. The distance is the average of the Hellinger
    distances.

    Parameters
    ----------
    cube1 : numpy.ndarray or str
        Data cube. If a str, it should be the filename of a pickle file saved
        using Dendrogram_Stats.
    cube2 : numpy.ndarray or str
        Data cube. If a str, it should be the filename of a pickle file saved
        using Dendrogram_Stats.
    min_deltas : numpy.ndarray or list
        Minimum deltas of leaves in the dendrogram.
    nbins : str or float, optional
        Number of bins for the histograms. 'best' sets
        that number using the square root of the average
        number of features between the histograms to be
        compared.
    min_features : int, optional
        The minimum number of features necessary to compare
        the histograms.
    fiducial_model : Dendrogram_Stats
        Computed dendrogram and statistic values. Use to avoid
        re-computing.
    dendro_params : dict or list of dicts, optional
        Further parameters for the dendrogram algorithm
        (see www.dendrograms.org for more info). If a list of dictionaries is
        given, the first list entry should be the dictionary for cube1, and the
        second for cube2.

    """

    def __init__(self, cube1, cube2, min_deltas=None, nbins="best",
                 min_features=100, fiducial_model=None, dendro_params=None):
        super(DendroDistance, self).__init__()

        self.nbins = nbins

        if min_deltas is None:
            # min_deltas = np.append(np.logspace(-1.5, -0.7, 8),
            #                        np.logspace(-0.6, -0.35, 10))
            min_deltas = np.logspace(-2.5, 0.5, 100)

        if dendro_params is not None:
            if isinstance(dendro_params, list):
                dendro_params1 = dendro_params[0]
                dendro_params2 = dendro_params[1]
            elif isinstance(dendro_params, dict):
                dendro_params1 = dendro_params
                dendro_params2 = dendro_params
            else:
                raise TypeError("dendro_params is a "+str(type(dendro_params)) +
                                "It must be a dictionary, or a list containing" +
                                " a dictionary entries.")
        else:
            dendro_params1 = None
            dendro_params2 = None

        if fiducial_model is not None:
            self.dendro1 = fiducial_model
        elif isinstance(cube1, str):
            self.dendro1 = Dendrogram_Stats.load_results(cube1)
        else:
            self.dendro1 = Dendrogram_Stats(
                cube1, min_deltas=min_deltas, dendro_params=dendro_params1)
            self.dendro1.run(verbose=False)

        if isinstance(cube2, str):
            self.dendro2 = Dendrogram_Stats.load_results(cube2)
        else:
            self.dendro2 = \
                Dendrogram_Stats(cube2, min_deltas=min_deltas,
                                 dendro_params=dendro_params2)
            self.dendro2.run(verbose=False)

        # Set the minimum number of components to create a histogram
        cutoff1 = np.argwhere(self.dendro1.numfeatures > min_features)
        cutoff2 = np.argwhere(self.dendro2.numfeatures > min_features)
        if cutoff1.any():
            cutoff1 = cutoff1[-1]
        else:
            raise ValueError("The dendrogram from cube1 does not contain the \
                              necessary number of features, %s. Lower \
                              min_features or alter min_deltas."
                             % (min_features))
        if cutoff2.any():
            cutoff2 = cutoff2[-1]
        else:
            raise ValueError("The dendrogram from cube2 does not contain the \
                              necessary number of features, %s. Lower \
                              min_features or alter min_deltas."
                             % (min_features))

        self.cutoff = np.min([cutoff1, cutoff2])

        self.bins = []
        self.mecdf1 = None
        self.mecdf2 = None

        self.num_results = None
        self.num_distance = None
        self.histogram_distance = None

    def numfeature_stat(self, verbose=False):
        '''
        Calculate the distance based on the number of features statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        self.num_distance = \
            np.abs(self.dendro1.tail_slope - self.dendro2.tail_slope) / \
            np.sqrt(self.dendro1.tail_slope_err**2 +
                    self.dendro2.tail_slope_err**2)

        if verbose:

            import matplotlib.pyplot as p

            # Dendrogram 1
            p.plot(self.dendro1.x, self.dendro1.y, 'gD', label='Dendro 1')
            p.plot(self.dendro1.x, self.dendro1.model.fittedvalues, 'g')

            # Dendrogram 2
            p.plot(self.dendro2.x, self.dendro2.y, 'bD', label='Dendro 2')
            p.plot(self.dendro2.x, self.dendro2.model.fittedvalues, 'b')

            p.grid(True)
            p.xlabel(r"log $\delta$")
            p.ylabel("log Number of Features")
            p.legend()
            p.show()

        return self

    def histogram_stat(self, verbose=False):
        '''
        Computes the distance using histograms.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        if self.nbins == "best":
            self.nbins = [np.floor(np.mean([n1, n2])) for n1, n2 in
                          zip(self.dendro1.numfeatures[:self.cutoff],
                              self.dendro2.numfeatures[:self.cutoff])]
        else:
            self.nbins = [self.nbins] * \
                len(self.dendro1.numfeatures[:self.cutoff])

        self.histograms1 = \
            np.empty((len(self.dendro1.numfeatures[:self.cutoff]),
                     np.max(self.nbins)))
        self.histograms2 = \
            np.empty((len(self.dendro2.numfeatures[:self.cutoff]),
                     np.max(self.nbins)))

        for n, (data1, data2, nbin) in enumerate(
                zip(self.dendro1.values[:self.cutoff],
                    self.dendro2.values[:self.cutoff], self.nbins)):

            stand_data1 = standardize(data1)
            stand_data2 = standardize(data2)

            bins = common_histogram_bins(stand_data1, stand_data2,
                                         nbins=nbin)

            self.bins.append(bins)

            hist1 = np.histogram(stand_data1, bins=bins,
                                 density=True)[0]
            self.histograms1[n, :] = \
                np.append(hist1, (np.max(self.nbins) - bins.size) * [np.NaN])

            hist2 = np.histogram(stand_data2, bins=bins,
                                 density=True)[0]
            self.histograms2[n, :] = \
                np.append(hist2, (np.max(self.nbins) - bins.size) * [np.NaN])

            # Normalize
            self.histograms1[n, :] /= np.nansum(self.histograms1[n, :])
            self.histograms2[n, :] /= np.nansum(self.histograms2[n, :])

        self.mecdf1 = mecdf(self.histograms1)
        self.mecdf2 = mecdf(self.histograms2)

        self.histogram_distance = hellinger_stat(
            self.histograms1, self.histograms2)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(2, 2, 1)
            p.title("ECDF 1")
            p.xlabel("Intensities")
            for n in range(len(self.dendro1.min_deltas[:self.cutoff])):
                p.plot((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                       self.mecdf1[n, :][:self.nbins[n]])
            p.subplot(2, 2, 2)
            p.title("ECDF 2")
            p.xlabel("Intensities")
            for n in range(len(self.dendro2.min_deltas[:self.cutoff])):
                p.plot((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                       self.mecdf2[n, :][:self.nbins[n]])
            p.subplot(2, 2, 3)
            p.title("PDF 1")
            for n in range(len(self.dendro1.min_deltas[:self.cutoff])):
                bin_width = self.bins[n][1] - self.bins[n][0]
                p.bar((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                      self.histograms1[n, :][:self.nbins[n]],
                      align="center", width=bin_width, alpha=0.25)
            p.subplot(2, 2, 4)
            p.title("PDF 2")
            for n in range(len(self.dendro2.min_deltas[:self.cutoff])):
                bin_width = self.bins[n][1] - self.bins[n][0]
                p.bar((self.bins[n][:-1] + self.bins[n][1:]) / 2,
                      self.histograms2[n, :][:self.nbins[n]],
                      align="center", width=bin_width, alpha=0.25)
            p.show()

        return self

    def distance_metric(self, verbose=False):
        '''
        '''

        self.histogram_stat(verbose=verbose)
        self.numfeature_stat(verbose=verbose)

        return self


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

    half_size = (size - 1)/2

    shape = max(y.shape)

    stds = np.empty((shape - size + 1))

    for i in range(half_size, shape - half_size):
        stds[i - half_size] = np.std(y[i - half_size: i + half_size])

    # Now find the max
    break_pos = np.argmax(stds) + half_size

    if return_results:
        return break_pos, stds

    return break_pos
