
'''

Implementation of the Tsallis Distribution from Tofflemire et al. (2011)

'''

import numpy as np
from scipy.stats import nanmean, nanstd, chisquare

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
    def __init__(self, img, lags=None, num_bins=500):

        self.img = img
        self.num_bins = num_bins

        if lags is None:
            self.lags = [1,2,4,8,16,32,64]
        else:
            self.lags = lags

        self.tsallis_arrays = np.empty((len(self.lags), img.shape[0], img.shape[1]))
        self.tsallis_distrib = np.empty((len(self.lags), 2, num_bins))
        self.tsallis_fits = np.empty((len(self.lags),7))


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
            fitted_vals = tsallis_function(clipped[0],*params)
            self.tsallis_fits[i,:3] = params
            self.tsallis_fits[i,3:6] = np.diag(pcov)
            self.tsallis_fits[i,6] = chisquare(np.exp(fitted_vals),f_exp=np.exp(clipped[1]),ddof=3)[0]#np.nansum((clipped[1] - fitted_vals)**2.)/np.nansum(fitted_vals)

    def run(self, verbose=False):

        self.make_tsallis()
        self.fit_tsallis()

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.lags)

            i = 1
            for dist,arr,params in zip(self.tsallis_distrib, self.tsallis_arrays, self.tsallis_fits):

                p.subplot(num, 2, i)
                p.imshow(arr, origin="lower",interpolation="nearest") ## This doesn't plot the last image
                p.colorbar()
                p.subplot(num, 2, i+1)
                p.plot(dist[0], tsallis_function(dist[0], *params), "r")
                p.plot(dist[0],dist[1],'rD',label="".join(["Tsallis Distribution with Lag ",str(self.lags[i/2])]))
                p.legend(loc="upper left", prop={"size":10})

                i += 2
            p.show()

        return self

class Tsallis_Distance(object):
    """

    Distance Metric for the Tsallis Distribution.

    The distance between two data cubes is defined as the sum of the difference of
    the Tsallis fit parameters for a set of lags weighted by the $\Chi^2$ of the fit

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------


    """
    def __init__(self, array1, array2, lags=None, num_bins=500):
        super(Tsallis_Distance, self).__init__()
        self.array1 = array1
        self.array2 = array2

        self.tsallis1 = Tsallis(array1, lags=lags, num_bins=num_bins).run(verbose=False)
        self.tsallis2 = Tsallis(array2, lags=lags, num_bins=num_bins).run(verbose=False)

        self.distance = None

    def distance_metric(self, verbose=False):

        a1 = np.asarray([loga/(err**2.) for loga,err in zip(self.tsallis1.tsallis_fits[:,0],self.tsallis1.tsallis_fits[:,3])])
        a2 = np.asarray([loga/(err**2.) for loga,err in zip(self.tsallis2.tsallis_fits[:,0],self.tsallis2.tsallis_fits[:,3])])

        w1 = np.asarray([np.sqrt(w/err**2.) for w,err in zip(self.tsallis1.tsallis_fits[:,1],self.tsallis1.tsallis_fits[:,4])])
        w2 = np.asarray([np.sqrt(w/err**2.) for w,err in zip(self.tsallis2.tsallis_fits[:,1],self.tsallis2.tsallis_fits[:,4])])

        q1 = np.asarray([q/err**2.for q,err in zip(self.tsallis1.tsallis_fits[:,2],self.tsallis1.tsallis_fits[:,5])])
        q2 = np.asarray([q/err**2. for q,err in zip(self.tsallis2.tsallis_fits[:,2],self.tsallis2.tsallis_fits[:,5])])

        diff_a = (a1-a2)**2.
        diff_w = (w1-w2)**2.
        diff_q = (q1-q2)**2.

        print w1,w2
        print a1,a2
        print q1,q2

        print diff_a, diff_w, diff_q

        self.distance = np.sum(diff_a + diff_w + diff_q)

        if verbose:
            import matplotlib.pyplot as p
            lags = self.tsallis1.lags
            p.plot(lags, diff_a, "bD", label="Difference of a")
            p.plot(lags, diff_w, "rD", label="Difference of w")
            p.plot(lags, diff_q, "gD", label="Difference of q")
            p.legend()
            p.show()

        return self


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
    loga, wsquare, q = p
    return (-1/(q-1))*(np.log10(1+(q-1)*(x**2./wsquare)) + loga)

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

