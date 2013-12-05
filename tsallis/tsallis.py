
'''

Implementation of the Tsallis Distribution from Tofflemire et al. (2011)

'''

import numpy as np
from scipy.stats import nanmean, nanstd

class Tsallis(object):
    """

    The Tsallis Distribution

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, img, lags, num_bins=500):

        self.img = img
        self.lags = lags
        self.num_bins = num_bins

        self.tsallis_arrays = np.empty((len(lags), img.shape[0], img.shape[1]))
        self.tsallis_distrib = np.empty((len(lags), 2, num_bins))


    def make_tsallis(self):
        for i, lag in enumerate(self.lags):
            pad_img = self.img#np.pad(self.img, 2*lag, padwithzeros)
            rolls = np.roll(pad_img, lag, axis=0) + np.roll(pad_img, (-1)*lag, axis=0) + \
                     np.roll(pad_img, lag, axis=1) + np.roll(pad_img, (-1)*lag, axis=1)

            self.tsallis_arrays[i,:] = normalize((rolls/4.)- self.img)

            hist, bin_edges = np.histogram(self.tsallis_arrays[i,:].ravel(), bins=self.num_bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            normlog_hist = np.log(hist/np.sum(hist, dtype="float"))

            self.tsallis_distrib[i,0,:] = bin_centres
            self.tsallis_distrib[i,1,:] = normlog_hist

        return self

    def run(self, verbose=False):

        self.make_tsallis()

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.lags)

            i = 1
            for dist,arr in zip(self.tsallis_distrib,self.tsallis_arrays):

                p.subplot(num, 2, i)
                p.imshow(arr, origin="lower",interpolation="nearest") ## This doesn't plot the last image
                p.colorbar()

                p.subplot(num, 2, i+1)
                p.plot(dist[0],dist[1],'rD',label="".join(["Tsallis Distribution with Lag ",str(self.lags[i/2])]))
                p.legend(loc="upper left", prop={"size":10})

                i += 2
            p.show()

        return self

class Tsallis_Distance(object):
    """

    Distance Metric for the Tsallis Distribution.

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------


    """
    def __init__(self):
        super(Tsallis_Distance, self).__init__()
        raise NotImplementedError("")


def normalize(data):

    av_val = nanmean(data, axis=None)
    st_dev = nanstd(data, axis=None)

    return (data - av_val)/st_dev

def padwithzeros(vector,pad_width,iaxis,kwargs):
  vector[:pad_width[0]] = 0
  vector[-pad_width[1]:] = 0
  return vector