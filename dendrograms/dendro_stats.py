
'''

Dendrogram statistics as described in Burkhart et al. (2013)
Two statistics are contained:
    * number of leaves + branches vs. $\delta$ parameter
    * statistical moments of the intensity histogram

Requires the astrodendro package (http://github.com/astrodendro/dendro-core)

'''

from astrodendro import Dendrogram
import numpy as np
from matplotlib import pyplot as p
import matplotlib.cm as cm
import os
from pandas import HDFStore, Series, DataFrame
import statsmodels.formula.api as sm
from scipy.stats import nanmean, nanstd

class DendroDistance(object):
    """docstring for DendroDistance"""
    def __init__(self, file1, file2, timestep):
        super(DendroDistance, self).__init__()


        self.file1 = HDFStore(file1)
        self.file2 = HDFStore(file2)

        if timestep != "/": ## Saving in HDF5 adds this to the beginning
            timestep = "/"+timestep

        assert timestep in self.file1.keys()
        assert timestep in self.file2.keys()


        self.file1 = self.file1[timestep]
        self.file2 = self.file2[timestep]

        assert (self.file1["Deltas"] == self.file2["Deltas"]).all()
        self.deltas = self.file1["Deltas"]

        self.moments1 = {"mean": [], "variance": [], "skewness": [], "kurtosis": []}
        self.moments2 = {"mean": [], "variance": [], "skewness": [], "kurtosis": []}
        self.numdata1 = None
        self.numdata2 = None

        self.num_results = None
        self.num_distance = None
        self.hist_distance = None

    def numfeature_stat(self, verbose=False):
        '''
        '''

        numdata1 = np.asarray([np.log10(self.deltas), np.log10(self.file1["Num Features"])])
        numdata2 = np.asarray([np.log10(self.deltas), np.log10(self.file2["Num Features"])])

        self.numdata1 = np.log10(self.file1["Num Features"].ix[np.where(np.log10(self.deltas)>-0.9)])
        self.numdata2 = np.log10(self.file2["Num Features"].ix[np.where(np.log10(self.deltas)>-0.9)])
        clip_delta = self.deltas.ix[np.where(np.log10(self.deltas)>-0.9)]
        ## Set up columns for regression
        dummy = [0] * len(clip_delta) + [1] * len(clip_delta)
        x = np.concatenate((np.log10(clip_delta), np.log10(clip_delta)))
        regressor = x.T * dummy
        constant = np.array([[1] * (len(clip_delta) + len(clip_delta))])

        numdata = np.concatenate((np.log10(self.numdata1), np.log10(self.numdata2)))

        d = {"dummy": Series(dummy), "scales": Series(x), "log_numdata": Series(numdata), \
             "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(formula = "log_numdata ~ dummy + scales + regressor", data = df)

        self.num_results = model.fit()

        self.num_distance = np.abs(self.num_results.tvalues["regressor"])

        if verbose:

            print self.num_results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_delta), np.log10(self.numdata1), "bD", \
                    np.log10(clip_delta), np.log10(self.numdata2), "gD")
            p.plot(df["scales"][:len(self.numdata1)], self.num_results.fittedvalues[:len(self.numdata1)], "b", \
                   df["scales"][-len(self.numdata2):], self.num_results.fittedvalues[-len(self.numdata2):], "g")
            p.grid(True)
            p.xlabel(r"log $\delta$")
            p.ylabel("log Number of Features")
            p.show()

    def histogram_stat(self, verbose=False):
        '''
        '''

        for hist1, hist2 in zip(self.file1["Histograms"], self.file2["Histograms"]):

            mean1, var1, skew1, kurt1 = compute_moments(hist1)
            mean2, var2, skew2, kurt2 = compute_moments(hist2)

            self.moments1["mean"].append(mean1)
            self.moments1["variance"].append(var1)
            self.moments1["skewness"].append(skew1)
            self.moments1["kurtosis"].append(kurt1)

            self.moments2["mean"].append(mean2)
            self.moments2["variance"].append(var2)
            self.moments2["skewness"].append(skew2)
            self.moments2["kurtosis"].append(kurt2)

        if verbose:
            import matplotlib.pyplot as p

            for i, key in enumerate(self.moments1.keys()):
                p.subplot(2,2,i+1)
                p.title(key)
                p.xlabel("Delta")
                p.ylabel(key)
                p.plot(self.deltas, self.moments1[key], "bD-")
                p.plot(self.deltas, self.moments2[key], "gD-")
            p.show()

        return self

    def distance_metric(self, verbose=False):
        '''
        '''

        self.numfeature_stat(verbose=verbose)
        self.histogram_stat(verbose=verbose)

        return self

def compute_moments(pdf):

    mean = nanmean(pdf, axis=None)
    variance = nanstd(pdf, axis=None)**2.
    skewness = np.nansum(((pdf - mean)/np.sqrt(variance))**3.)/np.sum(~np.isnan(pdf))
    kurtosis = np.nansum(((pdf - mean)/np.sqrt(variance))**4.)/np.sum(~np.isnan(pdf)) - 3

    return mean, variance, skewness, kurtosis


