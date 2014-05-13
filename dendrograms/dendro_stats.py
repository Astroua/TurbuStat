
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
from mecdf import mecdf

class DendroDistance(object):
    """docstring for DendroDistance"""
    def __init__(self, file1, file2, timestep, nbins="best", min_features=1000):
        super(DendroDistance, self).__init__()


        self.file1 = HDFStore(file1)
        self.file2 = HDFStore(file2)
        self.nbins = nbins

        if timestep[0] != "/": ## Saving in HDF5 adds this to the beginning
            timestep = "/"+timestep

        assert timestep in self.file1.keys()
        assert timestep in self.file2.keys()


        self.file1 = self.file1[timestep]
        self.file2 = self.file2[timestep]

        assert (self.file1["Deltas"] == self.file2["Deltas"]).all()
        self.deltas = self.file1["Deltas"]

        ## Set the minimum number of components to create a histogram
        self.cutoff = min(np.argwhere((self.file1["Num Features"]<min_features)==1)[0], \
                         np.argwhere((self.file2["Num Features"]<min_features)==1)[0])[0]

        self.histograms1 = None
        self.histograms2 = None
        self.bins = []
        self.mecdf1 = None
        self.mecdf2 = None

        self.numdata1 = None
        self.numdata2 = None

        self.num_results = None
        self.num_distance = None
        self.histogram_distance = None

    def numfeature_stat(self, verbose=False):
        '''
        '''

        self.numdata1 = np.log10(self.file1["Num Features"].ix[np.where(np.log10(self.deltas)>-0.6)])
        self.numdata2 = np.log10(self.file2["Num Features"].ix[np.where(np.log10(self.deltas)>-0.6)])
        clip_delta = self.deltas.ix[np.where(np.log10(self.deltas)>-0.6)]

        if (self.numdata1==0).any() or (self.numdata2==0).any():
            filter11 = self.numdata1[self.numdata1>0]
            filter21 = self.numdata2[self.numdata1>0]
            filter12 = filter11[self.numdata2>0]
            filter22 = filter21[self.numdata2>0]

            clip_delta = clip_delta[self.numdata1>0]
            clip_delta = clip_delta[self.numdata2>0]

            if filter12.empty or filter22.empty: ## If we can't compare at all, just exit.
                self.num_distance = 0.0
                return self
            self.numdata1 = filter12
            self.numdata2 = filter22

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

        return self

    def histogram_stat(self, verbose=False):
        '''
        '''

        if self.nbins == "best":
            self.nbins = [int(round(np.sqrt((n1+n2)/2.))) for n1, n2 in zip(self.file1["Num Features"].ix[:self.cutoff], \
                            self.file2["Num Features"].ix[:self.cutoff])]
        else:
            self.nbins = [self.nbins] * len(self.deltas.ix[:self.cutoff])

        self.histograms1 = np.empty((len(self.deltas.ix[:self.cutoff]), np.max(self.nbins)))
        self.histograms2 = np.empty((len(self.deltas.ix[:self.cutoff]), np.max(self.nbins)))

        for n, (data1,data2,nbin) in enumerate(zip(self.file1["Histograms"].ix[:self.cutoff], \
                                      self.file2["Histograms"].ix[:self.cutoff], self.nbins)):
            hist1, bins = np.histogram(data1, bins=nbin, density=True, range=[0,1])[:2]
            self.histograms1[n,:] = np.append(hist1, (np.max(self.nbins) - nbin) * [np.NaN])
            self.bins.append(bins)
            hist2 = np.histogram(data2, bins=nbin, density=True, range=[0,1])[0]
            self.histograms2[n,:] = np.append(hist2, (np.max(self.nbins) - nbin) * [np.NaN])

            ## Normalize
            self.histograms1[n,:] /= np.nansum(self.histograms1[n,:])
            self.histograms2[n,:] /= np.nansum(self.histograms2[n,:])

        self.mecdf1 = mecdf(self.histograms1)
        self.mecdf2 = mecdf(self.histograms2)

        self.histogram_distance = hellinger_stat(self.histograms1, self.histograms2)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(2,2,1)
            p.title("ECDF 1")
            p.xlabel("Intensities")
            for n in range(len(self.deltas[:self.cutoff])):
                p.plot((self.bins[n][:-1] + self.bins[n][1:])/2, self.mecdf1[n,:][:self.nbins[n]])
            p.subplot(2,2,2)
            p.title("ECDF 2")
            p.xlabel("Intensities")
            for n in range(len(self.deltas[:self.cutoff])):
                p.plot((self.bins[n][:-1] + self.bins[n][1:])/2, self.mecdf2[n,:][:self.nbins[n]])
            p.subplot(2,2,3)
            p.title("PDF 1")
            for n in range(len(self.deltas[:self.cutoff])):
                bin_width = self.bins[n][1]-self.bins[n][0]
                p.bar((self.bins[n][:-1] + self.bins[n][1:])/2, self.histograms1[n,:][:self.nbins[n]], \
                    align="center", width=bin_width, alpha=0.25)
            p.subplot(2,2,4)
            p.title("PDF 2")
            for n in range(len(self.deltas[:self.cutoff])):
                bin_width = self.bins[n][1]-self.bins[n][0]
                p.bar((self.bins[n][:-1] + self.bins[n][1:])/2, self.histograms2[n,:][:self.nbins[n]], \
                    align="center", width=bin_width, alpha=0.25)
            p.show()

        return self

    def distance_metric(self, verbose=False):
        '''
        '''

        self.numfeature_stat(verbose=verbose)
        self.histogram_stat(verbose=verbose)

        return self


def hellinger_stat(x, y):

    assert x.shape == y.shape

    hellinger = lambda i,j : (1/np.sqrt(2)) * np.sqrt(np.nansum((np.sqrt(i) - np.sqrt(j))**2.))

    if len(x.shape)==1:
        return hellinger(x, y)
    else:
        dists = np.empty((x.shape[0], 1))
        for n in range(x.shape[0]):
            dists[n,0] = hellinger(x[n,:], y[n,:])
        return np.mean(dists)