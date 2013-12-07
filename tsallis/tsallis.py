
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
        self.tsallis_fit_params = np.empty((len(lags),3))


    def make_tsallis(self):
        for i, lag in enumerate(self.lags):
            pad_img = self.img#np.pad(self.img, 2*lag, padwithzeros)
            rolls = np.roll(pad_img, lag, axis=0) + np.roll(pad_img, (-1)*lag, axis=0) + \
                     np.roll(pad_img, lag, axis=1) + np.roll(pad_img, (-1)*lag, axis=1)

            self.tsallis_arrays[i,:] = normalize((rolls/4.)- self.img)

            hist, bin_edges = np.histogram(self.tsallis_arrays[i,:].ravel(), bins=self.num_bins)
            bin_centres = (bin_edges[:-1] + bin_edges[1:])/2
            normlog_hist = np.log10(hist/np.sum(hist, dtype="float"))

            self.tsallis_distrib[i,0,:] = bin_centres
            self.tsallis_distrib[i,1,:] = normlog_hist

        return self

    def fit_tsallis(self):
        from scipy.optimize import curve_fit
        for i, dist in enumerate(self.tsallis_distrib):
            clipped = clip_to_good(dist[0], dist[1])
            params, pcov = curve_fit(tsallis_function, clipped[0], clipped[1], p0=(-np.max(clipped[1]),1.,2.),maxfev=100*len(dist[0]))
            self.tsallis_fit_params[i,:] = params

    def run(self, verbose=False):

        self.make_tsallis()
        self.fit_tsallis()

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.lags)

            i = 1
            for dist,arr,params in zip(self.tsallis_distrib, self.tsallis_arrays, self.tsallis_fit_params):

                p.subplot(num, 2, i)
                p.imshow(arr, origin="lower",interpolation="nearest") ## This doesn't plot the last image
                p.colorbar()
                p.subplot(num, 2, i+1)
                p.plot(dist[0], tsallis_function(dist[0], params[0], params[1], params[2]), "r")
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

def tsallis_function(x, *p):
    '''

    Tsallis distribution function as given in Tofflemire
    Implemented in log form

    INPUTS
    ------

    x - array or list
        x-data

    params - array or list
             contains the three parameter values
    '''
    loga, w, q = p
    return (-1/(q-1))*(np.log10(1+(q-1)*(x**2./w**2.)) + loga)

def clip_to_good(x, y):
    '''

    Clip to values between -2 and 2

    '''
    clip_mask = np.zeros(x.shape)
    for i,val in enumerate(x):
        if val<2.0 or val>-2.0:
            clip_mask[i] = 1
    clip_x = x[np.where(clip_mask==1)]
    clip_y = y[np.where(clip_mask==1)]

    return clip_x[np.isfinite(clip_y)], clip_y[np.isfinite(clip_y)]

def normalize(data):

    av_val = nanmean(data, axis=None)
    st_dev = nanstd(data, axis=None)

    return (data - av_val)/st_dev

def padwithzeros(vector,pad_width,iaxis,kwargs):
  vector[:pad_width[0]] = 0
  vector[-pad_width[1]:] = 0
  return vector

