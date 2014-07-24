# Licensed under an MIT open source license - see LICENSE


'''

Dendrogram statistics as described in Burkhart et al. (2013)
Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

Requires the astrodendro package (http://github.com/astrodendro/dendro-core)

'''

import numpy as np
from pandas import Series, DataFrame
import statsmodels.formula.api as sm
from scipy.interpolate import UnivariateSpline
from mecdf import mecdf
from astrodendro import pruning, Dendrogram


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

    def compute_dendro(self, verbose=False):
        '''
        Compute the dendrogram and prune to the minimum deltas.
        ** min_deltas must be in ascending order! **

        Parameters
        ----------
        verbose : optional, bool

        '''
        d = Dendrogram.compute(self.cube, verbose=verbose,
                               min_delta=self.min_deltas[0],
                               min_value=self.dendro_params["min_value"],
                               min_npix=self.dendro_params["min_npix"])
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

    def run(self, verbose=False):
        '''

        Compute dendrograms. Necessary to maintain the package format.

        Parameters
        ----------
        verbose : optional, bool

        '''
        self.compute_dendro(verbose=verbose)
        if verbose:
            # import matplotlib.pyplot as p
            pass  # Write up some quick plots.


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
    cube1 : numpy.ndarray
        Data cube.
    cube2 : numpy.ndarray
        Data cube.
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
    dendro_params : dict
        Further parameters for the dendrogram algorithm
        (see www.dendrograms.org for more info).

    """

    def __init__(self, cube1, cube2, min_deltas=None, nbins="best",
                 min_features=100, fiducial_model=None, dendro_params=None):
        super(DendroDistance, self).__init__()

        self.nbins = nbins

        if min_deltas is None:
            # min_deltas = np.append(np.logspace(-1.5, -0.7, 8),
            #                        np.logspace(-0.6, -0.35, 10))
            min_deltas = np.logspace(-2.5, 0.5, 100)

        if fiducial_model is not None:
            self.dendro1 = fiducial_model
        else:
            self.dendro1 = Dendrogram_Stats(
                cube1, min_deltas=min_deltas, dendro_params=dendro_params)
            self.dendro1.run(verbose=False)

        self.dendro2 = Dendrogram_Stats(
            cube2, min_deltas=min_deltas, dendro_params=dendro_params)
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

    def numfeature_stat(self, use_spline=False, verbose=False):
        '''
        Calculate the distance based on the number of features statistic.

        Parameters
        ----------
        use_spline : bool, optional
            Pick out power-law tail using a spline. Otherwise, the last 20% of
            the data points are used.
        verbose : bool, optional
            Enables plotting.
        '''

        # Remove points where log(numdata)=0
        deltas1 = np.log10(
            self.dendro1.min_deltas[self.dendro1.numfeatures > 1])
        numfeatures1 = np.log10(
            self.dendro1.numfeatures[self.dendro1.numfeatures > 1])

        deltas2 = np.log10(
            self.dendro2.min_deltas[self.dendro2.numfeatures > 1])
        numfeatures2 = np.log10(
            self.dendro2.numfeatures[self.dendro2.numfeatures > 1])

        if use_spline:
            # Approximate knot location using linear splines with minimal smoothing
            break1 = break_spline(deltas1, numfeatures1, k=1, s=1)
            break2 = break_spline(deltas2, numfeatures2, k=1, s=1)

            # Keep values after the break
            numfeatures1 = numfeatures1[np.where(deltas1 > break1)]
            numfeatures2 = numfeatures2[np.where(deltas2 > break2)]
            deltas1 = deltas1[np.where(deltas1 > break1)]
            deltas2 = deltas2[np.where(deltas2 > break2)]
        else:
            # Try keeping 20% of the original number of delta values (same for
            # both dendrograms).
            keep_num = int(round(0.2 * len(self.dendro1.min_deltas)))

            # If there aren't enough points, don't do any clipping.
            if len(numfeatures1) >= keep_num:
                numfeatures1 = numfeatures1[-keep_num:]
                deltas1 = deltas1[-keep_num:]
            else:
                pass

            if len(numfeatures2) >= keep_num:
                numfeatures2 = numfeatures2[-keep_num:]
                deltas2 = deltas2[-keep_num:]
            else:
                pass

        # Set up columns for regression
        dummy = [0] * len(deltas1) + [1] * len(deltas2)
        x = np.concatenate((deltas1, deltas2))
        regressor = x.T * dummy

        numdata = np.concatenate((numfeatures1, numfeatures2))

        d = {"dummy": Series(dummy), "scales": Series(x),
             "log_numdata": Series(numdata), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(
            formula="log_numdata ~ dummy + scales + regressor", data=df)

        self.num_results = model.fit()

        self.num_distance = np.abs(self.num_results.tvalues["regressor"])
        if verbose:

            print self.num_results.summary()

            import matplotlib.pyplot as p
            p.plot(deltas1, numfeatures1, "bD",
                   deltas2, numfeatures2, "gD")
            p.plot(df["scales"][:len(numfeatures1)],
                   self.num_results.fittedvalues[:len(numfeatures1)], "b",
                   df["scales"][-len(numfeatures2):],
                   self.num_results.fittedvalues[-len(numfeatures2):], "g")
            p.grid(True)
            p.xlabel(r"log $\delta$")
            p.ylabel("log Number of Features")
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
            self.nbins = [int(round(np.sqrt((n1 + n2) / 2.))) for n1, n2 in
                          zip(self.dendro1.numfeatures[:self.cutoff],
                              self.dendro2.numfeatures[:self.cutoff])]
        else:
            self.nbins = [self.nbins] * \
                len(self.dendro1.numfeatures[:self.cutoff])

        self.histograms1 = np.empty(
            (len(self.dendro1.numfeatures[:self.cutoff]), np.max(self.nbins)))
        self.histograms2 = np.empty(
            (len(self.dendro2.numfeatures[:self.cutoff]), np.max(self.nbins)))

        for n, (data1, data2, nbin) in enumerate(
                zip(self.dendro1.values[:self.cutoff],
                    self.dendro2.values[:self.cutoff], self.nbins)):

            stand_data1 = standardize(data1)
            stand_data2 = standardize(data2)

            # Create bins for both from the relative minimum and maximum.
            bins = np.linspace(np.min(np.append(stand_data1, stand_data2)),
                               np.max(np.append(stand_data1, stand_data2)),
                               nbin + 1)
            self.bins.append(bins)

            hist1 = np.histogram(
                stand_data1, bins=bins, density=True)[0]
            self.histograms1[n, :] = \
                np.append(hist1, (np.max(self.nbins) - nbin) * [np.NaN])

            hist2 = np.histogram(
                stand_data2, bins=bins, density=True)[0]
            self.histograms2[n, :] = \
                np.append(hist2, (np.max(self.nbins) - nbin) * [np.NaN])

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

    hellinger = lambda i, j: (
        1 / np.sqrt(2)) * np.sqrt(np.nansum((np.sqrt(i) - np.sqrt(j)) ** 2.))

    if len(x.shape) == 1:
        return hellinger(x, y)
    else:
        dists = np.empty((x.shape[0], 1))
        for n in range(x.shape[0]):
            dists[n, 0] = hellinger(x[n, :], y[n, :])
        return np.mean(dists)


def break_spline(x, y, **kwargs):
    '''
    Calculate the break in 2 linear trends using a spline.
    '''

    s = UnivariateSpline(x, y, **kwargs)
    knots = s.get_knots()

    if len(knots) > 3:
        print "Many knots"
        knot_expec = -0.8
        posn = np.argmin(np.abs(knots - knot_expec))
        return knots[posn]  # Return the knot closest to the expected value

    elif len(knots) == 2:
        print "No knots"
        return knots[0]  # Set the threshold to the beginning

    else:
        return knots[1]


def standardize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)
