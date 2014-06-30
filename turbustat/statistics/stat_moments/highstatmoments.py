
import numpy as np
from scipy.stats import nanmean, nanstd


class StatMoments(object):

    """

    Statistical Moments of a given image are returned.
    See Burkhart et al. (2010) for methods used.

    Parameters
    ----------
    img : numpy.ndarray
        2D Image.
    radius : int
        Radius of circle to use when computing moments.
    periodic : bool, optional
        If the data is periodic (ie. from asimulation), wrap the data.
    bin_num : int, optional
        Number of bins to use in the histogram.

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
        Moments over the entire image.
        '''
        self.mean, self.variance, self.skewness, self.kurtosis =\
            compute_moments(self.img)

        return self

    def compute_spatial_distrib(self):
        '''
        Compute the moments over circular region with the specified radius.
        '''

        if self.periodic_flag:
            pad_img = np.pad(self.img, self.radius, mode="wrap")
        else:
            pad_img = np.pad(self.img, self.radius, padwithnans)
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

        return self

    def make_spatial_histograms(self):
        '''
        Create histograms of the moments.
        '''
        # Mean
        mean_hist, edges = np.histogram(
            self.mean_array[~np.isnan(self.mean_array)], self.bin_num,
            density=True)
        bin_centres = (edges[:-1] + edges[1:]) / 2
        self.mean_hist = [bin_centres, mean_hist]
        # Variance
        variance_hist, edges = np.histogram(
            self.variance_array[~np.isnan(self.variance_array)], self.bin_num,
            density=True)
        bin_centres = (edges[:-1] + edges[1:]) / 2
        self.variance_hist = [bin_centres, variance_hist]
        # Skewness
        skewness_hist, edges = np.histogram(
            self.skewness_array[~np.isnan(self.skewness_array)], self.bin_num,
            density=True)
        bin_centres = (edges[:-1] + edges[1:]) / 2
        self.skewness_hist = [bin_centres, skewness_hist]
        # Kurtosis
        kurtosis_hist, edges = np.histogram(
            self.kurtosis_array[~np.isnan(self.kurtosis_array)], self.bin_num,
            density=True)
        bin_centres = (edges[:-1] + edges[1:]) / 2
        self.kurtosis_hist = [bin_centres, kurtosis_hist]

        return self

    def run(self, verbose=False):
        '''
        Compute the entire method.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.

        '''

        self.array_moments()
        self.compute_spatial_distrib()
        self.make_spatial_histograms()

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(221)
            p.imshow(self.mean_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Mean")
            p.colorbar()
            p.contour(self.img)
            p.subplot(222)
            p.imshow(self.variance_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Variance")
            p.colorbar()
            p.contour(self.img)
            p.subplot(223)
            p.imshow(self.skewness_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Skewness")
            p.colorbar()
            p.contour(self.img)
            p.subplot(224)
            p.imshow(self.kurtosis_array, cmap="binary",
                     origin="lower", interpolation="nearest")
            p.title("Kurtosis")
            p.colorbar()
            p.contour(self.img)
            p.show()
        return self


class StatMomentsDistance(object):

    '''
    Compute the distance between two images based on their moments.
    The distance is calculated for the skewness and kurtosis. The distance
    values for each for computed using the Kullback-Leidler Divergence.

    Parameters
    ----------
    image1 : numpy.ndarray
        2D Image.
    image2 : numpy.ndarray
        2D Image.
    radius : int, optional
        Radius of circle to use when computing moments.
    fiducial_model : StatMoments
        Computed StatMoments object. use to avoid recomputing.

    '''

    ######### ADD ADDITIONAL ARGS !!!!!
    def __init__(self, image1, image2, radius=5, fiducial_model=None):
        super(StatMomentsDistance, self).__init__()

        if fiducial_model is not None:
            self.moments1 = fiducial_model
        else:
            self.moments1 = StatMoments(image1, radius).run()

        self.moments2 = StatMoments(image2, radius).run()

        self.kurtosis_distance = None
        self.skewness_distance = None

    def distance_metric(self, verbose=False):
        '''
        Compute the distance.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.

        '''
        self.kurtosis_distance = np.abs(
            kl_divergence(self.moments1.kurtosis_hist[1],
                          self.moments2.kurtosis_hist[1]))
        self.skewness_distance = np.abs(
            kl_divergence(self.moments1.skewness_hist[1],
                          self.moments2.skewness_hist[1]))

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.plot(self.moments1.kurtosis_hist[0],
                   self.moments1.kurtosis_hist[1], 'b',
                   self.moments2.kurtosis_hist[0],
                   self.moments2.kurtosis_hist[1], 'g')
            p.fill(self.moments1.kurtosis_hist[0],
                   self.moments1.kurtosis_hist[1], 'b',
                   self.moments2.kurtosis_hist[0],
                   self.moments2.kurtosis_hist[1], 'g', alpha=0.5)
            p.xlabel("Kurtosis")
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

    mean = nanmean(img, axis=None)
    variance = nanstd(img, axis=None) ** 2.
    skewness = np.nansum(
        ((img - mean) / np.sqrt(variance)) ** 3.) / np.sum(~np.isnan(img))
    kurtosis = np.nansum(
        ((img - mean) / np.sqrt(variance)) ** 4.) / np.sum(~np.isnan(img)) - 3

    return mean, variance, skewness, kurtosis


def padwithnans(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = np.NaN
    vector[-pad_width[1]:] = np.NaN
    return vector


def kl_divergence(P, Q):
    '''
    Kullback Leidler Divergence

    Parameters
    ----------

    P,Q : numpy.ndarray
        Two Discrete Probability distributions

    Returns
    -------

    kl_divergence : float
    '''
    P = P[~np.isnan(P)]
    Q = Q[~np.isnan(Q)]
    P = P[np.isfinite(P)]
    Q = Q[np.isfinite(Q)]
    return np.nansum(np.where(Q != 0, P * np.log(P / Q), 0))
