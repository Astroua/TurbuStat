'''

Higher Order Statistical Moments following the method by Burkhart et al. (2010)

'''

import numpy as np
from scipy.stats import nanmean, nanstd

class StatMoments(object):
    """

    Statistical Moments of a given image are returned.

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, img, radius, periodic=True, bin_num=1000):
        super(StatMoments, self).__init__()

        self.img = img
        self.radius = radius
        self.periodic_flag = periodic
        self.bin_num = bin_num

        self.mean = None
        self.variance = None
        self.skewness = None
        self.kurtosis = None

        self.mean_array = np.empty(img.shape)
        self.variance_array = np.empty(img.shape)
        self.skewness_array = np.empty(img.shape)
        self.kurtosis_array = np.empty(img.shape)

        self.mean_hist = None
        self.variance_hist = None
        self.skewness_hist = None
        self.kurtosis_hist = None

    def array_moments(self):
        '''
        Moments of the entire image
        '''
        self.mean, self.variance, self.skewness, self.kurtosis  = compute_moments(self.img)

        return self

    def compute_spatial_distrib(self):

        if self.periodic_flag:
            pad_img = np.pad(self.img, self.radius, mode="wrap")
        else:
            pad_img = np.pad(self.img, self.radius, padwithnans)
        circle_mask = circular_region(self.radius)

        for i in range(self.radius,pad_img.shape[0]-self.radius):
            for j in range(self.radius,pad_img.shape[1]-self.radius):
                img_slice = pad_img[i-self.radius:i+self.radius+1,j-self.radius:j+self.radius+1]
                img_slice = img_slice*circle_mask

                moments = compute_moments(img_slice)

                self.mean_array[i-self.radius,j-self.radius] = moments[0]
                self.variance_array[i-self.radius,j-self.radius] = moments[1]
                self.skewness_array[i-self.radius,j-self.radius] = moments[2]
                self.kurtosis_array[i-self.radius,j-self.radius] = moments[3]

        return self

    def make_spatial_histograms(self):
        # Mean
        mean_hist, edges = np.histogram(self.mean_array, self.bin_num)
        bin_centres = (edges[:-1] + edges[1:])/2
        self.mean_hist = [bin_centres, mean_hist]
        # Variance
        variance_hist, edges = np.histogram(self.variance_array, self.bin_num)
        bin_centres = (edges[:-1] + edges[1:])/2
        self.variance_hist = [bin_centres, variance_hist]
        # Skewness
        skewness_hist, edges = np.histogram(self.skewness_array, self.bin_num)
        bin_centres = (edges[:-1] + edges[1:])/2
        self.skewness_hist = [bin_centres, skewness_hist]
        # Kurtosis
        kurtosis_hist, edges = np.histogram(self.kurtosis_array, self.bin_num)
        bin_centres = (edges[:-1] + edges[1:])/2
        self.kurtosis_hist = [bin_centres, kurtosis_hist]

        return self


    def run(self, verbose=True):

        self.array_moments()
        self.compute_spatial_distrib()
        self.make_spatial_histograms()

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(221)
            p.imshow(self.mean_array, cmap="binary", origin="lower", interpolation="nearest")
            p.title("Mean")
            p.colorbar()
            p.contour(self.img)
            p.subplot(222)
            p.imshow(self.variance_array, cmap="binary", origin="lower", interpolation="nearest")
            p.title("Variance")
            p.colorbar()
            p.contour(self.img)
            p.subplot(223)
            p.imshow(self.skewness_array, cmap="binary", origin="lower", interpolation="nearest")
            p.title("Skewness")
            p.colorbar()
            p.contour(self.img)
            p.subplot(224)
            p.imshow(self.kurtosis_array, cmap="binary", origin="lower", interpolation="nearest")
            p.title("Kurtosis")
            p.colorbar()
            p.contour(self.img)
            p.show()
        return self

def circular_region(radius):

    xx, yy = np.mgrid[-radius:radius+1,-radius:radius+1]

    circle = xx**2. + yy**2.
    circle = circle < radius**2.

    circle = circle.astype(float)
    circle[np.where(arr==0.)] = np.NaN

    return circle

def compute_moments(img):

    mean = nanmean(img, axis=None)
    variance = nanstd(img, axis=None)**2.
    skewness = np.nansum(((img - mean)/np.sqrt(variance))**3.)/np.sum(~np.isnan(img))
    kurtosis = np.nansum(((img - mean)/np.sqrt(variance))**4.)/np.sum(~np.isnan(img)) - 3

    return mean, variance, skewness, kurtosis

def padwithnans(vector,pad_width,iaxis,kwargs):
  vector[:pad_width[0]] = np.NaN
  vector[-pad_width[1]:] = np.NaN
  return vector


class VCA_Distance(object):
    """docstring for VCA_Distance"""
    def __init__(self, arg):
        super(VCA_Distance, self).__init__()
        self.arg = arg
        raise NotImplementedError("")

class VCS_Distance(object):
    """docstring for VCS_Distance"""
    def __init__(self, arg):
        super(VCS_Distance, self).__init__()
        self.arg = arg
        raise NotImplementedError("")