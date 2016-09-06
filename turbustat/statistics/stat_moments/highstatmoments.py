# Licensed under an MIT open source license - see LICENSE


import numpy as np
from astropy.wcs import WCS

from ..stats_utils import (hellinger, kl_divergence, common_histogram_bins,
                           common_scale)
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class StatMoments(BaseStatisticMixIn):
    """
    Statistical Moments of a given image are returned.
    See Burkhart et al. (2010) for methods used.

    Parameters
    ----------
    img : %(dtypes)s
        2D Image.
    radius : int, optional
        Radius of circle to use when computing moments.
    periodic : bool, optional
        If the data is periodic (ie. from asimulation), wrap the data.
    bins : array or int, optional
        Number of bins to use in the histogram.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, radius=5, periodic=True, nbins=None):
        super(StatMoments, self).__init__()

        self.need_header_flag = False
        self.header = None

        self.data = input_data(img, no_header=True)

        self.radius = radius
        self.periodic_flag = periodic

        if nbins is None:
            self.nbins = np.sqrt(self.data.size)
        else:
            self.nbins = nbins

        self.mean = None
        self.variance = None
        self.skewness = None
        self.kurtosis = None

        self.mean_array = np.empty(self.data.shape)
        self.variance_array = np.empty(self.data.shape)
        self.skewness_array = np.empty(self.data.shape)
        self.kurtosis_array = np.empty(self.data.shape)

        self.mean_hist = None
        self.variance_hist = None
        self.skewness_hist = None
        self.kurtosis_hist = None

    def array_moments(self):
        '''
        Moments over the entire image.
        '''
        self.mean, self.variance, self.skewness, self.kurtosis =\
            compute_moments(self.data)

    def compute_spatial_distrib(self):
        '''
        Compute the moments over circular region with the specified radius.
        '''

        if self.periodic_flag:
            pad_img = np.pad(self.data, self.radius, mode="wrap")
        else:
            pad_img = np.pad(self.data, self.radius, padwithnans)
        circle_mask = circular_region(self.radius)

        for i in range(self.radius, pad_img.shape[0] - self.radius):
            for j in range(self.radius, pad_img.shape[1] - self.radius):
                img_slice = pad_img[
                    i - self.radius:i + self.radius + 1,
                    j - self.radius:j + self.radius + 1]

                if np.isnan(img_slice).all():
                    self.mean_array[i - self.radius, j - self.radius] = np.NaN
                    self.variance_array[
                        i - self.radius, j - self.radius] = np.NaN
                    self.skewness_array[
                        i - self.radius, j - self.radius] = np.NaN
                    self.kurtosis_array[
                        i - self.radius, j - self.radius] = np.NaN

                else:
                    img_slice = img_slice * circle_mask

                    moments = compute_moments(img_slice)

                    self.mean_array[
                        i - self.radius, j - self.radius] = moments[0]
                    self.variance_array[
                        i - self.radius, j - self.radius] = moments[1]
                    self.skewness_array[
                        i - self.radius, j - self.radius] = moments[2]
                    self.kurtosis_array[
                        i - self.radius, j - self.radius] = moments[3]

    @property
    def mean_extrema(self):
        return np.nanmin(self.mean_array), np.nanmax(self.mean_array)

    @property
    def variance_extrema(self):
        return np.nanmin(self.variance_array), np.nanmax(self.variance_array)

    @property
    def skewness_extrema(self):
        return np.nanmin(self.skewness_array), np.nanmax(self.skewness_array)

    @property
    def kurtosis_extrema(self):
        return np.nanmin(self.kurtosis_array), np.nanmax(self.kurtosis_array)

    def make_spatial_histograms(self, mean_bins=None, variance_bins=None,
                                skewness_bins=None, kurtosis_bins=None):
        '''
        Create histograms of the moments.

        Parameters
        ----------
        mean_bins : array, optional
            Bins to use for the histogram of the mean array
        variance_bins : array, optional
            Bins to use for the histogram of the variance array
        skewness_bins : array, optional
            Bins to use for the histogram of the skewness array
        kurtosis_bins : array, optional
            Bins to use for the histogram of the kurtosis array
        '''
        # Mean
        if mean_bins is None:
            mean_bins = self.nbins
        mean_hist, edges = \
            np.histogram(self.mean_array[~np.isnan(self.mean_array)],
                         mean_bins, density=True)
        mean_bin_centres = (edges[:-1] + edges[1:]) / 2
        self.mean_hist = [mean_bin_centres, mean_hist]

        # Variance
        if variance_bins is None:
            variance_bins = self.nbins
        variance_hist, edges = \
            np.histogram(self.variance_array[~np.isnan(self.variance_array)],
                         variance_bins, density=True)
        var_bin_centres = (edges[:-1] + edges[1:]) / 2
        self.variance_hist = [var_bin_centres, variance_hist]

        # Skewness
        if skewness_bins is None:
            skewness_bins = self.nbins
        skewness_hist, edges = \
            np.histogram(self.skewness_array[~np.isnan(self.skewness_array)],
                         skewness_bins, density=True)
        skew_bin_centres = (edges[:-1] + edges[1:]) / 2
        self.skewness_hist = [skew_bin_centres, skewness_hist]
        # Kurtosis
        if kurtosis_bins is None:
            kurtosis_bins = self.nbins
        kurtosis_hist, edges = \
            np.histogram(self.kurtosis_array[~np.isnan(self.kurtosis_array)],
                         kurtosis_bins, density=True)
        kurt_bin_centres = (edges[:-1] + edges[1:]) / 2
        self.kurtosis_hist = [kurt_bin_centres, kurtosis_hist]

    def run(self, verbose=False, kwargs={}):
        '''
        Compute the entire method.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        kwargs : dict, optional
            Passed to `make_spatial_histograms`. Bins to use in the histograms
            can be given here.
        '''

        self.array_moments()
        self.compute_spatial_distrib()
        self.make_spatial_histograms(**kwargs)

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(221)
            p.imshow(self.mean_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Mean")
            p.colorbar()
            p.contour(self.data)
            p.subplot(222)
            p.imshow(self.variance_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Variance")
            p.colorbar()
            p.contour(self.data)
            p.subplot(223)
            p.imshow(self.skewness_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Skewness")
            p.colorbar()
            p.contour(self.data)
            p.subplot(224)
            p.imshow(self.kurtosis_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Kurtosis")
            p.colorbar()
            p.contour(self.data)
            p.show()
        return self


class StatMoments_Distance(object):

    '''
    Compute the distance between two images based on their moments.
    The distance is calculated for the skewness and kurtosis. The distance
    values for each for computed using the Hellinger Distance (default),
    or the Kullback-Leidler Divergence.

    Unlike the other distance classes in TurbuStat, the computation of the
    histograms needed for the distance metric has been split into its own
    method. However, the change is fairly transparent, since it is called
    within distance_metric.

    Parameters
    ----------
    image1 : %(dtypes)s
        2D Image.
    image2 : %(dtypes)s
        2D Image.
    radius : int, optional
        Radius of circle to use when computing moments. This is the pixel size
        applied to the coarsest grid (if the datasets are not on a common
        grid). The radius for the finer grid is adjusted so the angular scales
        match.
    nbins : int, optional
        Number of bins to use when constructing histograms.
    periodic1 : bool, optional
        If image1 is periodic in the spatial boundaries, set to True.
    periodic2 : bool, optional
        If image2 is periodic in the spatial boundaries, set to True.
    fiducial_model : StatMoments
        Computed StatMoments object. use to avoid recomputing.

    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, image1, image2, radius=5, nbins=None,
                 periodic1=False, periodic2=False, fiducial_model=None):
        super(StatMoments_Distance, self).__init__()

        image1 = input_data(image1, no_header=False)
        image2 = input_data(image2, no_header=False)

        # Compute the scale so the radius is common between the two datasets
        scale = common_scale(WCS(image1[1]), WCS(image2[1]))

        if scale == 1.0:
            radius1 = radius
            radius2 = radius
        elif scale > 1.0:
            radius1 = int(np.round(scale * radius))
            radius2 = radius
        else:
            radius1 = radius
            radius2 = int(np.round(radius / float(scale)))

        if nbins is None:
            self.nbins = _auto_nbins(image1[0].size, image2[0].size)
        else:
            self.nbins = nbins

        if fiducial_model is not None:
            self.moments1 = fiducial_model
        else:
            self.moments1 = StatMoments(image1, radius1, nbins=self.nbins,
                                        periodic=periodic1)
            self.moments1.compute_spatial_distrib()

        self.moments2 = StatMoments(image2, radius2, nbins=self.nbins,
                                    periodic=periodic1)
        self.moments2.compute_spatial_distrib()

    def create_common_histograms(self, nbins=None):
        '''
        Calculate the histograms using a common set of bins. Only
        histograms of the kurtosis and skewness are calculated, since only
        they are used in the distance metric.

        Parameters
        ----------
        nbins : int, optional
            Bins to use in the histogram calculation.
        '''

        skew_bins = \
            common_histogram_bins(self.moments1.skewness_array.flatten(),
                                  self.moments2.skewness_array.flatten(),
                                  nbins=nbins)

        kurt_bins = \
            common_histogram_bins(self.moments1.kurtosis_array.flatten(),
                                  self.moments2.kurtosis_array.flatten(),
                                  nbins=nbins)

        self.moments1.make_spatial_histograms(skewness_bins=skew_bins,
                                              kurtosis_bins=kurt_bins)

        self.moments2.make_spatial_histograms(skewness_bins=skew_bins,
                                              kurtosis_bins=kurt_bins)

    def distance_metric(self, metric='Hellinger', verbose=False, nbins=None,
                        label1=None, label2=None):
        '''
        Compute the distance.

        Parameters
        ----------
        metric : 'Hellinger' (default) or "KL Divergence", optional
            Set the metric to use compare the histograms.
        verbose : bool, optional
            Enables plotting.
        nbins : int, optional
            Bins to use in the histogram calculation.
        label1 : str, optional
            Object or region name for image1
        label2 : str, optional
            Object or region name for image2
        '''

        self.create_common_histograms(nbins=nbins)

        if metric == "Hellinger":
            kurt_bw = np.diff(self.moments1.kurtosis_hist[0])[0]
            self.kurtosis_distance = hellinger(self.moments1.kurtosis_hist[1],
                                               self.moments2.kurtosis_hist[1],
                                               bin_width=kurt_bw)

            skew_bw = np.diff(self.moments1.skewness_hist[0])[0]
            self.skewness_distance = hellinger(self.moments1.skewness_hist[1],
                                               self.moments2.skewness_hist[1],
                                               bin_width=skew_bw)
        elif metric == "KL Divergence":
            self.kurtosis_distance = np.abs(
                kl_divergence(self.moments1.kurtosis_hist[1],
                              self.moments2.kurtosis_hist[1]))
            self.skewness_distance = np.abs(
                kl_divergence(self.moments1.skewness_hist[1],
                              self.moments2.skewness_hist[1]))
        else:
            raise ValueError("metric must be 'Hellinger' or 'KL Divergence'. "
                             "Was given as "+str(metric))

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.plot(self.moments1.kurtosis_hist[0],
                   self.moments1.kurtosis_hist[1], 'b', label=label1)
            p.plot(self.moments2.kurtosis_hist[0],
                   self.moments2.kurtosis_hist[1], 'g', label=label2)
            p.fill(self.moments1.kurtosis_hist[0],
                   self.moments1.kurtosis_hist[1], 'b',
                   self.moments2.kurtosis_hist[0],
                   self.moments2.kurtosis_hist[1], 'g', alpha=0.5)
            p.xlabel("Kurtosis")
            p.ylabel("PDF")
            p.legend(loc='upper right')
            p.subplot(122)
            p.plot(self.moments1.skewness_hist[0],
                   self.moments1.skewness_hist[1], 'b',
                   self.moments2.skewness_hist[0],
                   self.moments2.skewness_hist[1], 'g')
            p.fill(self.moments1.skewness_hist[0],
                   self.moments1.skewness_hist[1], 'b',
                   self.moments2.skewness_hist[0],
                   self.moments2.skewness_hist[1], 'g', alpha=0.5)
            p.xlabel("Skewness")
            p.show()

        return self


def circular_region(radius):
    '''
    Create a circular region with nans outside the radius.
    '''

    xx, yy = np.mgrid[-radius:radius + 1, -radius:radius + 1]

    circle = xx ** 2. + yy ** 2.
    circle = circle < radius ** 2.

    circle = circle.astype(float)
    circle[np.where(circle == 0.)] = np.NaN

    return circle


def compute_moments(img):
    '''
    Compute the moments of the given image.

    Parameters
    ----------
    img : numpy.ndarray
        2D image.

    Returns
    -------
    mean : float
        The 1st moment.
    variance : float
        The 2nd moment.
    skewness : float
        The 3rd moment.
    kurtosis : float
        The 4th moment.

    '''

    mean = np.nanmean(img, axis=None)
    variance = np.nanvar(img, axis=None)
    skewness = np.nansum(
        ((img - mean) / np.sqrt(variance)) ** 3.) / np.sum(~np.isnan(img))
    kurtosis = np.nansum(
        ((img - mean) / np.sqrt(variance)) ** 4.) / np.sum(~np.isnan(img)) - 3

    return mean, variance, skewness, kurtosis


def padwithnans(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = np.NaN
    vector[-pad_width[1]:] = np.NaN
    return vector


def _auto_nbins(size1, size2):
    return int((size1 + size2) / 2.)
