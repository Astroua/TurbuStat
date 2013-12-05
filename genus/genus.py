
'''

Genus Statistics Implementation (Chepurnov et al., 2008)

'''

import numpy as np
import scipy.ndimage as nd
import matplotlib.pyplot as p
from scipy.stats import scoreatpercentile, nanmean, nanstd
from skimage.morphology import remove_small_objects

from astrodendro import Dendrogram

class Genus(object):
    """

    Genus Statistics

    INPUTS
    ------

    img - array
          integrated intensity image

    density_threshold - int
                        percentile to set density threshold at

    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, img, lowdens_thresh=0, highdens_thresh=100, numpts=100, smoothing_radii=None, save_dendrograms=False, save_name=None):
        super(Genus, self).__init__()

        self.img = img

        self.save_name = save_name
        if not self.save_name:
            self.save_name = "untitled"


        self.nanflag = False
        if np.isnan(self.img).any():
            self.nanflag = True

        self.lowdens_thresh = scoreatpercentile(img[~np.isnan(img)], \
                                                   lowdens_thresh)
        self.highdens_thresh = scoreatpercentile(img[~np.isnan(img)], \
                                                   highdens_thresh)

        self.thresholds = np.linspace(self.lowdens_thresh, self.highdens_thresh, numpts)

        if smoothing_radii is not None:
            assert isinstance(smoothing_radii, list)
            self.smoothing_radii = smoothing_radii
        else:
            self.smoothing_radii = np.linspace(1.0, 0.1*min(img.shape),10)

        self.genus_stats = np.empty([numpts,len(self.smoothing_radii)])
        # self.genus_roots = []
        # self.variance = np.empty((1,len(self.smoothing_radii)))
        self.fft_images = []
        self.smoothed_images = []

        self.dendrograms = []
        self.save_dendrograms = save_dendrograms


    def make_smooth_arrays(self):
        from astropy.convolution import Gaussian2DKernel, convolve_fft

        for i, width in enumerate(self.smoothing_radii):
            kernel = Gaussian2DKernel(width=width, x_size=self.img.shape[0], y_size=self.img.shape[1])
            if self.nanflag:
                self.smoothed_images.append(convolve_fft(self.img, kernel, normalize_kernel=True, interpolate_nan=True))
            else:
                self.smoothed_images.append(convolve_fft(self.img, kernel))
        return self

    def clean_fft(self):
        from scipy.fftpack import fft2
        from numpy.fft import fftfreq

        for j, image in enumerate(self.smoothed_images):
            self.fft_images.append(fft2(image))

        return self

    def initialize_dendrograms(self, verbose=False, save_dendrograms=False):

        if self.save_dendrograms != save_dendrograms:
            self.save_dendrograms = save_dendrograms

        for j in range(len(self.smoothing_radii)):
            try:
                d = self.dendrograms[j]
                self.save_dendrograms = False ## Override save flag if it already is
            except IndexError:
                d = Dendrogram.compute(self.smoothed_images[j], verbose=verbose)
                self.dendrograms.append(d)

            if self.save_dendrograms:
                filename = "".join([self.save_name,"_smooth_radius_",str(self.smoothing_radii[j]),".hdf5"])
                d.save_to(filename)

        return self


    def make_genus_curve(self):

        self.genus_stats = compute_genus(self.smoothed_images, self.thresholds)

        # for j in range(len(self.smoothing_radii)):
        #     i = 0
        #     depth = 0
        #     while True:
        #         if depth == 0:
        #             structure = self.dendrograms[j].trunk[0]
        #         else:
        #             structure = next_branch
        #         print structure
        #         if structure.height < self.thresholds[i]:
        #             self.genus_stats[i,j] = len(structure.children)

        #         depth = structure.level + 1
        #         i += 1
        #         next_branch = [f for f in structure.children if not f.is_leaf]
        #         if not next_branch or i >= len(self.thresholds):
        #             break

        return self

    # def fit_genus_curve(self):

    #     for curve in self.genus_stats:
    #         self.genus_roots.append(polyroots(self.thresholds, self.genus_stats))

    #     return self

    # def genus_variance(self):
    #     from scipy.fftpack import ifft2
    #     thresholds = np.linspace(-np.pi, np.pi, 100)

    #     for k, ft_im in enumerate(self.fft_images):
    #         rand_nu0 = np.empty((1,10))
    #         for j in range(10):
    #             rand_array = np.empty(self.img.shape)
    #             rand_genus_stats = np.empty((1,len(thresholds)))
    #             phase_array = ft_im.imag
    #             phases = np.unique(phase_array)
    #             for val in phases:
    #                 rand_array[np.where(phase_array==val)] = np.random.uniform(-np.pi, np.pi)
    #             rand_ifft = ifft2(rand_array).real

    #             rand_genus_stats = compute_genus(rand_ifft, thresholds)
    #             p.subplot(211)
    #             p.imshow(rand_ifft)
    #             p.subplot(212)
    #             p.plot(thresholds, rand_genus_stats[0,:])
    #             p.show()

    #             nu0 = polyroots(thresholds, rand_genus_stats[0,:])
    #             if len(nu0)!=0:
    #                 rand_nu0[:,j] = nu0
    #             else: # If there is no root...
    #                 rand_nu0[:,j] = np.NaN

    #         self.variance[:,k] = np.var(rand_nu0)
    #     return self

    def run(self, verbose=True):

        self.make_smooth_arrays()
        self.clean_fft()
        # self.initialize_dendrograms(save_dendrograms=test.save_dendrogram)
        self.make_genus_curve()
        # self.fit_genus_curve()

        # print self.genus_roots

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.smoothing_radii)
            for i in range(1, num+1):
                p.subplot(num/2,2,i)
                p.plot(self.thresholds, self.genus_stats[i-1], "bD")
            p.show()


        return self


def polyroots(x, y, ranges=None, deg=5, **kwargs):
    '''

    Fit polynomial and find its roots in a given range.

    INPUTS
    ------

    x - array
        x data

    y - array
        y data

    ranges - list
             list of upper and low bounds within x. If None, uses whole x-range.
             low bound is first, upper bound second (ie. [2,10])

    kwargs - passed to np.polyfit

    OUTPUTS
    -------

    range_roots - list
                  roots in the given range

    '''

    if ranges is not None:
        assert len(ranges)==2
        if ranges[0] < min(x):
            print "Lower Bound below list minimum, using list minimum"
            ranges[0] = min(x)
        if ranges[1] > max(x):
            print "Upper Bound above list maximum, using list maximum"
            ranges[1] = max(x)
    else:
        ranges = [min(x), max(x)]

    p = np.polyfit(x, y, deg=deg, **kwargs)
    roots = np.roots(p)
    range_roots = [root for root in roots if root<ranges[1] and root > ranges[0]]

    return range_roots


def compute_genus(images, thresholds):
    '''

    Computes the Genus Statistic

    INPUTS
    ------

    image - list of array OR a single array
            images to compute genus from

    thresholds - list/array
                 thresholds to calculate statistic at

    OUTPUTS
    -------

    genus_stats - array
                  statistic calculated

    '''

    if isinstance(images, list):
        pass
    else:
        images = [images]

    genus_stats = np.empty((len(images),len(thresholds)))
    for j, image in enumerate(images):
        for i, thresh in enumerate(thresholds):
            high_density = remove_small_objects(image>thresh, min_size=4, connectivity=1)
            low_density = remove_small_objects(image<thresh, min_size=4, connectivity=1)
            high_density_labels, high_density_num = nd.label(high_density, np.ones((3,3))) # eight-connectivity
            low_density_labels, low_density_num = nd.label(low_density, np.ones((3,3))) # eight-connectivity

            genus_stats[j,i] = high_density_num - low_density_num

        # genus_stats[j,:] = clip_genus(genus_stats[j,:]) # Until I know this is needed, ignore.

    return genus_stats


def clip_genus(genus_curve, length_threshold=5):
    '''

    Clip out uninteresting regions in the genus curve (large regions with value of 0)

    INPUTS
    ------

    genus_curve - array
                  computed genus curve

    thresholds - list/array
                 thresholds to calculate statistic at

    length_threshold - int
                       minimum length to warrant clipping

    '''

    from operator import itemgetter
    from itertools import groupby


    zeros = np.where(genus_curve==0)
    continuous_sections = []
    for _, g in groupby(enumerate(zeros[0]), lambda (i,x): i-x):
        continuous_sections.append(map(itemgetter(1), g))

    try:
        max_cont_section = max(continuous_sections, key=len)
    except ValueError:
        max_cont_section = []

    if len(max_cont_section)>=length_threshold:
        genus_curve[max_cont_section] = np.NaN


    return genus_curve

def check_saved_dendrograms():
    raise NotImplementedError("")


class GenusDistance(object):
    """

    Distance Metric for the Genus Statistic.

    """
    def __init__(self, img1, img2, smoothing_radii=None, verbose=False):
        super(GenusDistance, self).__init__()
        self.genus1 = Genus(img1, smoothing_radii=smoothing_radii).run(verbose=verbose)
        self.genus2 = Genus(img2, smoothing_radii=smoothing_radii).run(verbose=verbose)

        self.distance = None


    def distance_metric(self, verbose=False):
        '''

        Data is centered and normalized (via normalize).
        The distance is the difference between cubic splines.

        '''
        from scipy.interpolate import UnivariateSpline

        norm1 = normalize(self.genus1.genus_stats[0,:])
        norm2 = normalize(self.genus2.genus_stats[0,:])

        interp1 = UnivariateSpline(self.genus1.thresholds, norm1, s=1, k=3) #small smoothing
        interp2 = UnivariateSpline(self.genus2.thresholds, norm2, s=1, k=3)

        self.distance = np.nansum(np.abs(interp1(self.genus1.thresholds) - interp2(self.genus2.thresholds)))

        if verbose:
            import matplotlib.pyplot as p

            p.plot(self.genus1.thresholds, norm1, "bD", label="".join(["Genus Curve 1:", self.genus1.save_name]))
            p.plot(self.genus2.thresholds, norm2, "rD", label="".join(["Genus Curve 2:", self.genus2.save_name]))
            p.plot(self.genus1.thresholds, interp1(self.genus1.thresholds), "b", label="".join(["Genus Fit 1:", self.genus1.save_name]))
            p.plot(self.genus2.thresholds, interp2(self.genus2.thresholds), "r", label="".join(["Genus Fit 2:", self.genus2.save_name]))
            p.grid(True)
            p.legend(loc="upper right")
            p.show()

        return self


def normalize(data):

    av_val = nanmean(data, axis=None)
    st_dev = nanstd(data, axis=None)

    return (data - av_val)/st_dev