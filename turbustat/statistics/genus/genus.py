# Licensed under an MIT open source license - see LICENSE

import numpy as np
import scipy.ndimage as nd
from scipy.stats import scoreatpercentile
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Gaussian2DKernel, convolve_fft
from operator import itemgetter
from itertools import groupby
from astropy.wcs import WCS

try:
    from scipy.fftpack import fft2
except ImportError:
    from numpy.fft import fft2

from ..stats_utils import standardize, common_scale
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class Genus(BaseStatisticMixIn):

    """

    Genus Statistics based off of Chepurnov et al. (2008).

    Parameters
    ----------

    img : %(dtypes)s
        2D image.
    lowdens_percent : float, optional
        Lower percentile of the data to use.
    highdens_percent : float, optional
        Upper percentile of the data to use.
    numpts : int, optional
        Number of thresholds to calculate statistic at.
    smoothing_radii : list, optional
        Kernel radii to smooth data to.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, lowdens_percent=0, highdens_percent=100,
                 numpts=100, smoothing_radii=None):
        super(Genus, self).__init__()

        # A header isn't needed. Disable the check flag
        self.need_header_flag = False
        self.header = None
        self.data = input_data(img, no_header=True)

        self.nanflag = False
        if np.isnan(self.data).any():
            self.nanflag = True

        self.lowdens_percent = \
            scoreatpercentile(self.data[~np.isnan(self.data)],
                              lowdens_percent)
        self.highdens_percent = \
            scoreatpercentile(self.data[~np.isnan(self.data)],
                              highdens_percent)

        self.thresholds = np.linspace(
            self.lowdens_percent, self.highdens_percent, numpts)

        if smoothing_radii is not None:
            assert isinstance(smoothing_radii, list)
            self.smoothing_radii = smoothing_radii
        else:
            self.smoothing_radii = np.linspace(1.0, 0.1 * min(self.data.shape), 5)

        self.genus_stats = np.empty([numpts, len(self.smoothing_radii)])
        self.fft_images = []
        self.smoothed_images = []

    def make_smooth_arrays(self):
        '''
        Smooth data using a Gaussian kernel.
        '''

        for i, width in enumerate(self.smoothing_radii):
            kernel = Gaussian2DKernel(
                width, x_size=self.data.shape[0], y_size=self.data.shape[1])
            if self.nanflag:
                self.smoothed_images.append(
                    convolve_fft(self.data, kernel,
                                 normalize_kernel=True,
                                 interpolate_nan=True))
            else:
                self.smoothed_images.append(convolve_fft(self.data, kernel))

    # def clean_fft(self):

    #     for j, image in enumerate(self.smoothed_images):
    #         self.fft_images.append(fft2(image))

    #     return self

    def make_genus_curve(self):
        '''
        Create the genus curve.
        '''

        self.genus_stats = compute_genus(self.smoothed_images, self.thresholds)

    def run(self, verbose=False):
        '''
        Run the whole statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        self.make_smooth_arrays()
        # self.clean_fft()
        self.make_genus_curve()

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.smoothing_radii)
            for i in range(1, num + 1):
                p.subplot(num / 2, 2, i)
                p.title(
                    "".join(["Smooth Size: ",
                            str(self.smoothing_radii[i - 1])]))
                p.plot(self.thresholds, self.genus_stats[i - 1], "bD")
                p.xlabel("Intensity")
                p.grid(True)
            p.show()

        return self


def compute_genus(images, thresholds):
    '''

    Computes the Genus Statistic.

    Parameters
    ----------

    image : list of numpy.ndarray OR a single numpy.ndarray
        Images(s) to compute the Genus of.

    thresholds : list or numpy.ndarray
        Thresholds to calculate the statistic at.

    Returns
    -------

    genus_stats : array
    The calculated statistic.

    '''

    if not isinstance(images, list):
        images = [images]

    genus_stats = np.empty((len(images), len(thresholds)))
    for j, image in enumerate(images):
        for i, thresh in enumerate(thresholds):
            high_density = remove_small_objects(
                image > thresh, min_size=4, connectivity=1)
            low_density = remove_small_objects(
                image < thresh, min_size=4, connectivity=1)
            high_density_labels, high_density_num = nd.label(
                high_density, np.ones((3, 3)))  # eight-connectivity
            low_density_labels, low_density_num = nd.label(
                low_density, np.ones((3, 3)))  # eight-connectivity

            genus_stats[j, i] = high_density_num - low_density_num

        # genus_stats[j,:] = clip_genus(genus_stats[j,:])

    return genus_stats


def clip_genus(genus_curve, length_threshold=5):
    '''

    Clip out uninteresting regions in the genus curve
    (large regions with value of 0).

    Parameters
    ----------

    genus_curve : array
        Computed genus curve.

    length_threshold : int, optional
        Minimum length to warrant clipping.

    Returns
    -------

    genus_curve : numpy.ndarray
        Clipped Genus Curve.

    '''

    zeros = np.where(genus_curve == 0)
    continuous_sections = []
    for _, g in groupby(enumerate(zeros[0]), lambda (i, x): i - x):
        continuous_sections.append(map(itemgetter(1), g))

    try:
        max_cont_section = max(continuous_sections, key=len)
    except ValueError:
        max_cont_section = []

    if len(max_cont_section) >= length_threshold:
        genus_curve[max_cont_section] = np.NaN

    return genus_curve


class GenusDistance(object):

    """

    Distance Metric for the Genus Statistic.

    Parameters
    ----------

    img1 : %(dtypes)s
        2D image.
    img2 : %(dtypes)s
        2D image.
    smoothing_radii : list, optional
        Kernel radii to smooth data to.
    fiducial_model : Genus
        Computed Genus object. Use to avoid recomputing.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img1, img2, smoothing_radii=None, fiducial_model=None):
        super(GenusDistance, self).__init__()

        # Standardize the intensity values in the images

        img1, hdr1 = input_data(img1)
        img2, hdr2 = input_data(img2)

        img1 = standardize(img1)
        img2 = standardize(img2)

        if fiducial_model is not None:
            self.genus1 = fiducial_model
        else:
            self.genus1 = \
                Genus(img1, smoothing_radii=smoothing_radii,
                      lowdens_percent=20).run()

        self.genus2 = \
            Genus(img2, smoothing_radii=smoothing_radii,
                  lowdens_percent=20).run()

        # When normalizing the genus curves for the distance metric, find
        # the scaling between the angular size of the grids.
        self.scale = common_scale(WCS(hdr1), WCS(hdr2))

    def distance_metric(self, verbose=False, label1=None, label2=None):
        '''

        Data is centered and normalized (via normalize).
        The distance is the difference between cubic splines of the curves.

        All values are normalized by the area of the image they were
        calculated from.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for img1
        label2 : str, optional
            Object or region name for img2
        '''

        # 2 times the average number between the two
        num_pts = \
            int((len(self.genus1.thresholds) +
                 len(self.genus2.thresholds)) / 2)

        # Get the min and the max of the thresholds
        min_pt = max(np.min(self.genus1.thresholds),
                     np.min(self.genus2.thresholds))

        max_pt = min(np.max(self.genus1.thresholds),
                     np.max(self.genus2.thresholds))

        points = np.linspace(min_pt, max_pt, 2 * num_pts)

        # Divide each by the area of the data. genus1 is additionally
        # adjusted by the scale factor of the angular size between the
        # datasets.
        genus1 = self.genus1.genus_stats[0, :] / \
            float(self.genus1.data.size / self.scale)
        genus2 = self.genus2.genus_stats[0, :] / float(self.genus2.data.size)

        interp1 = \
            InterpolatedUnivariateSpline(self.genus1.thresholds,
                                         genus1, k=3)
        interp2 = \
            InterpolatedUnivariateSpline(self.genus2.thresholds,
                                         genus2, k=3)

        self.distance = np.linalg.norm(interp1(points) -
                                       interp2(points))

        if verbose:
            import matplotlib.pyplot as p

            p.plot(self.genus1.thresholds,
                   genus1, "bD",
                   label=label1)
            p.plot(self.genus2.thresholds,
                   genus2, "go",
                   label=label2)
            p.plot(points, interp1(points), "b")
            p.plot(points, interp2(points), "g")
            p.xlabel("z-score")
            p.ylabel("Genus Score")
            p.grid(True)
            p.legend(loc="best")
            p.show()

        return self


def remove_small_objects(arr, min_size, connectivity=8):
    '''
    Remove objects less than the given size.
    Function is based on skimage.morphology.remove_small_objects

    Parameters
    ----------
    arr : numpy.ndarray
        Binary array containing the mask.
    min_size : int
        Smallest allowed size.
    connectivity : int, optional
        Connectivity of the neighborhood.
    '''

    struct = nd.generate_binary_structure(arr.ndim, connectivity)

    labels, num = nd.label(arr, struct)

    sizes = nd.sum(arr, labels, range(1, num + 1))

    for i, size in enumerate(sizes):
        if size >= min_size:
            continue

        posns = np.where(labels == i + 1)

        arr[posns] = 0

    return arr
