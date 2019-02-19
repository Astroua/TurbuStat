# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.wcs import WCS
import astropy.units as u
from astropy.utils.console import ProgressBar
from itertools import product
import warnings

from ..stats_utils import (hellinger, kl_divergence, common_histogram_bins,
                           common_scale, padwithnans)
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class StatMoments(BaseStatisticMixIn):
    """
    Statistical Moments of an image. See Burkhart et al. (2010) for the methods
    used. By specifying the radius of circular mask, the mean, variance,
    skewness, and kurtosis are calculated within the circular mask for every
    pixel in the image. The distributions of these moments can be compared
    between data sets.

    Parameters
    ----------
    img : %(dtypes)s
        2D image.
    header : FITS header, optional
        The image header. Needed for the pixel scale.
    weights : %(dtypes)s
        2D array of weights. Uniform weights are used if none are given.
    radius : `~astropy.units.Quantity`, optional
        Radius of circle to use when computing moments. When angular or
        physical units are given, they will be rounded *down* to the nearest
        pixel size.
    nbins : array or int, optional
        Number of bins to use in the histogram.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, radius=5 * u.pix,
                 nbins=None, distance=None):
        super(StatMoments, self).__init__()

        self.input_data_header(img, header)

        if weights is None:
            self.weights = np.ones_like(self.data)
        else:
            self.weights = input_data(weights, no_header=True)

        if distance is not None:
            self.distance = distance

        self.radius = radius

        if nbins is None:
            self.nbins = np.sqrt(self.data.size)
        else:
            self.nbins = nbins

        self.nbins = int(self.nbins)

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):

        if not isinstance(value, u.Quantity):
            raise TypeError("radius must be an astropy.units.Quantity.")

        # Must be convertible to pixel scales!
        try:
            pix_rad = self._to_pixel(value)
        except Exception as e:
            raise e

        # The radius should be larger than a pixel
        if pix_rad.value < 2:
            raise ValueError("The chosen radius is smaller than two pixels. "
                             "Increase the size of the radius.")

        # Finally, limit the radius to a maximum of half the image size.
        if pix_rad.value > min(self.data.shape) / 2.:
            raise ValueError("The chosen radius is larger than half the image "
                             "size. Reduce the size of the radius.")

        self._radius = value

    def array_moments(self):
        '''
        Moments over the entire image.
        '''
        self.mean, self.variance, self.skewness, self.kurtosis = \
            compute_moments(self.data, self.weights)

    def compute_spatial_distrib(self, radius=None, periodic=True,
                                min_frac=0.8, show_progress=True):
        '''
        Compute the moments over circular region with the specified radius.

        Parameters
        ----------
        radius : `~astropy.units.Quantity`, optional
            Override the radius size of the region.
        periodic : bool, optional
            Specify whether the boundaries can be wrapped. Default is True.
        min_frac : float, optional
            A number between 0 and 1 that sets the minimum fraction of data in
            each region that are finite. A value of 1.0 requires that no NaNs
            be in the region.
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        '''

        # Require the fraction to be > 0 and <=1
        if min_frac <= 0.0 or min_frac > 1.:
            raise ValueError("min_frac must be larger than 0 and less than"
                             "or equal to 1.")

        self._mean_array = np.empty(self.data.shape)
        self._variance_array = np.empty(self.data.shape)
        self._skewness_array = np.empty(self.data.shape)
        self._kurtosis_array = np.empty(self.data.shape)

        # Use the new radius when another given
        if radius is not None:
            self.radius = radius

        # Convert to pixels. We need this to be an integer so round down to
        # the nearest integer values
        pix_rad = np.ceil(self._to_pixel(self.radius).value).astype(int)

        if periodic:
            pad_img = np.pad(self.data, pix_rad, mode="wrap")
            pad_weights = np.pad(self.weights, pix_rad, mode="wrap")
        else:
            pad_img = np.pad(self.data, pix_rad, padwithnans)
            pad_weights = np.pad(self.weights, pix_rad, padwithnans)

        circle_mask = circular_region(pix_rad)

        if show_progress:
            bar = ProgressBar((pad_img.shape[0] - 2 * pix_rad) *
                              (pad_img.shape[1] - 2 * pix_rad))

        # Loop through every point within the non-padded shape.
        prod = product(range(pix_rad, pad_img.shape[0] - pix_rad),
                       range(pix_rad, pad_img.shape[1] - pix_rad))

        for n, (i, j) in enumerate(prod):
                img_slice = pad_img[i - pix_rad:i + pix_rad + 1,
                                    j - pix_rad:j + pix_rad + 1]
                wgt_slice = pad_weights[i - pix_rad:i + pix_rad + 1,
                                        j - pix_rad:j + pix_rad + 1]

                valid_img_frac = \
                    np.isfinite(img_slice).sum() / float(img_slice.size)
                valid_wgt_frac = \
                    np.isfinite(wgt_slice).sum() / float(wgt_slice.size)

                if valid_img_frac < min_frac or valid_wgt_frac < min_frac:
                    self.mean_array[i - pix_rad, j - pix_rad] = np.NaN
                    self.variance_array[i - pix_rad, j - pix_rad] = np.NaN
                    self.skewness_array[i - pix_rad, j - pix_rad] = np.NaN
                    self.kurtosis_array[i - pix_rad, j - pix_rad] = np.NaN

                else:
                    img_slice = img_slice * circle_mask
                    wgt_slice = wgt_slice * circle_mask

                    moments = compute_moments(img_slice, wgt_slice)

                    self.mean_array[i - pix_rad, j - pix_rad] = moments[0]
                    self.variance_array[i - pix_rad, j - pix_rad] = moments[1]
                    self.skewness_array[i - pix_rad, j - pix_rad] = moments[2]
                    self.kurtosis_array[i - pix_rad, j - pix_rad] = moments[3]

                if show_progress:
                    bar.update(n + 1)

    @property
    def mean_array(self):
        '''
        The array of local means.
        '''
        return self._mean_array

    @property
    def variance_array(self):
        '''
        The array of local variances.
        '''
        return self._variance_array

    @property
    def skewness_array(self):
        '''
        The array of local skewnesss.
        '''
        return self._skewness_array

    @property
    def kurtosis_array(self):
        '''
        The array of local kurtosiss.
        '''
        return self._kurtosis_array

    @property
    def mean_extrema(self):
        '''
        The extrema of the mean array.
        '''
        return np.nanmin(self.mean_array), np.nanmax(self.mean_array)

    @property
    def variance_extrema(self):
        '''
        The extrema of the variance array.
        '''
        return np.nanmin(self.variance_array), np.nanmax(self.variance_array)

    @property
    def skewness_extrema(self):
        '''
        The extrema of the skewness array.
        '''
        return np.nanmin(self.skewness_array), np.nanmax(self.skewness_array)

    @property
    def kurtosis_extrema(self):
        '''
        The extrema of the kurtosis array.
        '''
        return np.nanmin(self.kurtosis_array), np.nanmax(self.kurtosis_array)

    def make_spatial_histograms(self, mean_bins=None, variance_bins=None,
                                skewness_bins=None, kurtosis_bins=None):
        '''
        Create histograms of the moments. If an optional set of bins is not
        given, :math:`\sqrt{N}` equally-size bins will be created, where
        :math:`N` is the number of elements in the array. The histogram
        values are normalized so that the sum of the values in the bins,
        multiplied by the bin width is 1.

        Parameters
        ----------
        mean_bins : array, optional
            Bins to use for the histogram of the mean array.
        variance_bins : array, optional
            Bins to use for the histogram of the variance array.
        skewness_bins : array, optional
            Bins to use for the histogram of the skewness array.
        kurtosis_bins : array, optional
            Bins to use for the histogram of the kurtosis array.
        '''
        # Mean
        if mean_bins is None:
            mean_bins = self.nbins
        mean_hist, edges = \
            np.histogram(self.mean_array[~np.isnan(self.mean_array)],
                         mean_bins, density=True)
        mean_bin_centres = (edges[:-1] + edges[1:]) / 2
        self._mean_hist = [mean_bin_centres, mean_hist]

        # Variance
        if variance_bins is None:
            variance_bins = self.nbins
        variance_hist, edges = \
            np.histogram(self.variance_array[~np.isnan(self.variance_array)],
                         variance_bins, density=True)
        var_bin_centres = (edges[:-1] + edges[1:]) / 2
        self._variance_hist = [var_bin_centres, variance_hist]

        # Skewness
        if skewness_bins is None:
            skewness_bins = self.nbins
        skewness_hist, edges = \
            np.histogram(self.skewness_array[~np.isnan(self.skewness_array)],
                         skewness_bins, density=True)
        skew_bin_centres = (edges[:-1] + edges[1:]) / 2
        self._skewness_hist = [skew_bin_centres, skewness_hist]
        # Kurtosis
        if kurtosis_bins is None:
            kurtosis_bins = self.nbins
        kurtosis_hist, edges = \
            np.histogram(self.kurtosis_array[~np.isnan(self.kurtosis_array)],
                         kurtosis_bins, density=True)
        kurt_bin_centres = (edges[:-1] + edges[1:]) / 2
        self._kurtosis_hist = [kurt_bin_centres, kurtosis_hist]

    @property
    def mean_hist(self):
        '''
        The histogram bins and values for the mean array. The first element is
        the array of bins, and the second contains the values.
        '''
        return self._mean_hist

    @property
    def variance_hist(self):
        '''
        The histogram bins and values for the variance array. The first element
        is the array of bins, and the second contains the values.
        '''
        return self._variance_hist

    @property
    def skewness_hist(self):
        '''
        The histogram bins and values for the skewness array. The first element
        is the array of bins, and the second contains the values.
        '''
        return self._skewness_hist

    @property
    def kurtosis_hist(self):
        '''
        The histogram bins and values for the kurtosis array. The first element
        is the array of bins, and the second contains the values.
        '''
        return self._kurtosis_hist

    def plot_histograms(self, new_figure=True, save_name=None,
                        hist_color='r', face_color='k'):
        '''
        Plot the histograms of each moment.

        Parameters
        ----------
        new_figure : bool, optional
            Creates a new matplotlib figure.
        save_name : str, optional
            The filename to save the plot as. This enables saving of the plot.
        '''

        if not hasattr(self, "mean_hist"):
            raise Exception("The histograms have not been computed. Run"
                            " StatMoments.make_spatial_histograms first.")

        import matplotlib.pyplot as plt

        if new_figure:
            plt.figure()

        alpha = 0.5

        plt.subplot(221)
        plt.plot(self.mean_hist[0],
                 self.mean_hist[1], color=hist_color)
        plt.fill_between(self.mean_hist[0], 0,
                         self.mean_hist[1], facecolor=face_color,
                         alpha=alpha)
        plt.xlabel("Mean")
        plt.ylabel("PDF")

        plt.subplot(222)
        plt.plot(self.variance_hist[0],
                 self.variance_hist[1], color=hist_color)
        plt.fill_between(self.variance_hist[0], 0,
                         self.variance_hist[1], facecolor=face_color,
                         alpha=alpha)
        plt.xlabel("Variance")

        plt.subplot(223)
        plt.plot(self.skewness_hist[0],
                 self.skewness_hist[1], color=hist_color)
        plt.fill_between(self.skewness_hist[0], 0,
                         self.skewness_hist[1], facecolor=face_color,
                         alpha=alpha)
        plt.xlabel("Skewness")
        plt.ylabel("PDF")

        plt.subplot(224)
        plt.plot(self.kurtosis_hist[0],
                 self.kurtosis_hist[1], color=hist_color)
        plt.fill_between(self.kurtosis_hist[0], 0,
                         self.kurtosis_hist[1], facecolor=face_color,
                         alpha=alpha)
        plt.xlabel("Kurtosis")

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def plot_maps(self, save_name=None, cmap='binary',
                  contour_cmap='viridis'):
        '''
        Plot the maps of locally-estimated moments.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        cmap : {str, matplotlib colormap}, optional
            Colormap for the images.
        contour_cmap : {str, matplotlib colormap}, optional
            Colormap for the contours.
        '''
        import matplotlib.pyplot as plt

        plt.subplot(221)
        plt.imshow(self.mean_array, cmap=cmap,
                   origin="lower", interpolation="nearest")
        plt.title("Mean")
        plt.colorbar()
        plt.contour(self.data, cmap=contour_cmap)
        plt.subplot(222)
        plt.imshow(self.variance_array, cmap=cmap,
                   origin="lower", interpolation="nearest")
        plt.title("Variance")
        plt.colorbar()
        plt.contour(self.data, cmap=contour_cmap)
        plt.subplot(223)
        plt.imshow(self.skewness_array, cmap=cmap,
                   origin="lower", interpolation="nearest")
        plt.title("Skewness")
        plt.colorbar()
        plt.contour(self.data, cmap=contour_cmap)
        plt.subplot(224)
        plt.imshow(self.kurtosis_array, cmap=cmap,
                   origin="lower", interpolation="nearest")
        plt.title("Kurtosis")
        plt.colorbar()
        plt.contour(self.data, cmap=contour_cmap)

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, verbose=False, save_name=None,
            radius=None, periodic=True, min_frac=0.8, **hist_kwargs):
        '''
        Compute the entire method.

        Parameters
        ----------
        show_progress : bool, optional
            Show a progress bar during the creation of the covariance matrix.
        verbose : bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given.
        radius : `~astropy.units.Quantity`, optional
            Override the radius size of the region.
        periodic : bool, optional
            Specify whether the boundaries can be wrapped. Default is True.
        min_frac : float, optional
            A number between 0 and 1 that sets the minimum fraction of data in
            each region that are finite. A value of 1.0 requires that no NaNs
            be in the region.
        hist_kwargs : Passed to `~StatMoments.make_spatial_histograms`.
        '''

        self.array_moments()
        self.compute_spatial_distrib(periodic=periodic, radius=radius,
                                     show_progress=show_progress)
        self.make_spatial_histograms(**hist_kwargs)

        if verbose:
            self.plot_maps(save_name=save_name)

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

    .. note:: When passing `~StatMoments` classes as `image1` or `image2`,
              if the radius does not match the given radius, or the common
              angular radius between the two datasets,
              `~StatMoments.compute_spatial_distrib` will be re-run with a
              new radius.

    Parameters
    ----------
    image1 : %(dtypes)s or `~StatMoments`
        2D image. Or a `~StatMoments` class can be passed which may be
        pre-computed.
    image2 : %(dtypes)s or `~StatMoments`
        See `image1`.
    radius : `~astropy.units.Quantity`, optional
        Radius of circle to use when computing moments. When given in pixel
        units, the radius will be adjusted such that a common *angular* scale
        is used between the two images, defined by whichever image has the
        coarser spatial grid. *This assumes the pixels can be treated as square
        in the celestial frame!* If an angular unit is given, there
        is no adjustment made.
    weights1 : %(dtypes)s
        2D array of weights. Uniform weights are used if none are given.
    weights2 : %(dtypes)s
        2D array of weights. Uniform weights are used if none are given.
    nbins : int, optional
        Number of bins to use when constructing histograms.
    periodic1 : bool, optional
        If image1 is periodic in the spatial boundaries, set to True.
    periodic2 : bool, optional
        If image2 is periodic in the spatial boundaries, set to True.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, image1, image2, radius=5 * u.pix, min_frac=0.8,
                 weights1=None, weights2=None,
                 nbins=None, periodic1=True, periodic2=True):
        super(StatMoments_Distance, self).__init__()

        if isinstance(image1, StatMoments):
            self.moments1 = image1
            _has_data1 = False
        else:
            image1 = input_data(image1, no_header=False)
            _has_data1 = True
        if isinstance(image2, StatMoments):
            self.moments2 = image2
            _has_data2 = False
        else:
            image2 = input_data(image2, no_header=False)
            _has_data2 = True

        # Compute the scale so the radius is common between the two datasets
        if radius.unit.is_equivalent(u.pix):

            wcs1 = WCS(image1[1]) if _has_data1 else self.moments1._wcs
            wcs2 = WCS(image2[1]) if _has_data2 else self.moments2._wcs
            scale = common_scale(wcs1, wcs2)

            if scale == 1.0:
                radius1 = radius
                radius2 = radius
            elif scale > 1.0:
                radius1 = scale * radius
                radius2 = radius
            else:
                radius1 = radius
                radius2 = radius / float(scale)
        else:
            radius1 = radius
            radius2 = radius

        if nbins is None:
            size1 = image1[0].size if _has_data1 else self.moments1.data.size
            size2 = image2[0].size if _has_data2 else self.moments2.data.size
            self.nbins = _auto_nbins(size1, size2)
        else:
            self.nbins = nbins

        if _has_data1:
            self.moments1 = StatMoments(image1, radius=radius1,
                                        nbins=self.nbins,
                                        weights=weights1)
            needs_run = True
        else:
            needs_run = False
            if not hasattr(self.moments1, '_kurtosis_array'):
                warnings.warn("StatMoments class given as `image1` does not"
                              " have"
                              " skewness/kurtosis maps. Computing spatial "
                              "distributions for `moments1`.")
                needs_run = True

            pix_rad = self.moments1._to_pixel(self.moments1.radius).value
            pix_rad1 = self.moments1._to_pixel(radius1).value

            if np.abs(pix_rad - pix_rad1) >= 1.:
                warnings.warn("Spatial radius differs "
                              "between the given radius"
                              " or common radius found by "
                              "StatMoments_Distance. Recomputing `moments1`.")
                self.moments1.radius = radius1
                self.moments1.nbins = nbins
                needs_run = True

        if needs_run:
            self.moments1.compute_spatial_distrib(periodic=periodic1,
                                                  min_frac=min_frac)

        if _has_data2:
            self.moments2 = StatMoments(image2, radius=radius2,
                                        nbins=self.nbins,
                                        weights=weights2)
            needs_run = True
        else:
            needs_run = False
            if not hasattr(self.moments2, '_kurtosis_array'):
                warnings.warn("StatMoments class given as `image2` does not"
                              " have"
                              " skewness/kurtosis maps. Computing spatial "
                              "distributions for `moments2`.")
                needs_run = True

            pix_rad = self.moments2._to_pixel(self.moments2.radius).value
            pix_rad2 = self.moments2._to_pixel(radius2).value
            if np.abs(pix_rad - pix_rad2) >= 1.:
                warnings.warn("Spatial radius differs between the given radius"
                              " or common radius found by "
                              "StatMoments_Distance. Recomputing `moments2`.")
                self.moments2.radius = radius2
                needs_run = True

        if needs_run:
            self.moments2.compute_spatial_distrib(periodic=periodic2,
                                                  min_frac=min_frac)

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
                                  nbins=self.nbins if nbins is None else nbins)

        kurt_bins = \
            common_histogram_bins(self.moments1.kurtosis_array.flatten(),
                                  self.moments2.kurtosis_array.flatten(),
                                  nbins=self.nbins if nbins is None else nbins)

        self.moments1.make_spatial_histograms(skewness_bins=skew_bins,
                                              kurtosis_bins=kurt_bins)

        self.moments2.make_spatial_histograms(skewness_bins=skew_bins,
                                              kurtosis_bins=kurt_bins)

    def distance_metric(self, verbose=False, nbins=None,
                        plot_kwargs1={'color': 'b', 'label': '1'},
                        plot_kwargs2={'color': 'g', 'label': '2'},
                        save_name=None):
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
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.create_common_histograms(nbins=nbins)

        kurt_bw = np.diff(self.moments1.kurtosis_hist[0])[0]
        self.kurtosis_distance = hellinger(self.moments1.kurtosis_hist[1],
                                           self.moments2.kurtosis_hist[1],
                                           bin_width=kurt_bw)

        skew_bw = np.diff(self.moments1.skewness_hist[0])[0]
        self.skewness_distance = hellinger(self.moments1.skewness_hist[1],
                                           self.moments2.skewness_hist[1],
                                           bin_width=skew_bw)

        if verbose:
            import matplotlib.pyplot as plt

            defaults1 = {'color': 'b', 'label': '1'}
            defaults2 = {'color': 'g', 'label': '2'}

            for key in defaults1:
                if key not in plot_kwargs1:
                    plot_kwargs1[key] = defaults1[key]
            for key in defaults2:
                if key not in plot_kwargs2:
                    plot_kwargs2[key] = defaults2[key]

            plt.subplot(121)
            plt.plot(self.moments1.kurtosis_hist[0],
                     self.moments1.kurtosis_hist[1],
                     plot_kwargs1['color'],
                     label=plot_kwargs1['label'])
            plt.plot(self.moments2.kurtosis_hist[0],
                     self.moments2.kurtosis_hist[1],
                     plot_kwargs2['color'],
                     label=plot_kwargs2['label'])
            plt.fill(self.moments1.kurtosis_hist[0],
                     self.moments1.kurtosis_hist[1],
                     plot_kwargs1['color'],
                     self.moments2.kurtosis_hist[0],
                     self.moments2.kurtosis_hist[1],
                     plot_kwargs2['color'],
                     alpha=0.5)
            plt.xlabel("Kurtosis")
            plt.ylabel("PDF")
            plt.legend(loc='upper right')

            plt.subplot(122)
            plt.plot(self.moments1.skewness_hist[0],
                     self.moments1.skewness_hist[1],
                     plot_kwargs1['color'],
                     label=plot_kwargs1['label'])
            plt.plot(self.moments2.skewness_hist[0],
                     self.moments2.skewness_hist[1],
                     plot_kwargs2['color'],
                     label=plot_kwargs2['label'])
            plt.fill(self.moments1.skewness_hist[0],
                     self.moments1.skewness_hist[1],
                     plot_kwargs1['color'],
                     self.moments2.skewness_hist[0],
                     self.moments2.skewness_hist[1],
                     plot_kwargs2['color'],
                     alpha=0.5)
            plt.xlabel("Skewness")

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()
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


def compute_moments(img, weights):
    '''
    Compute the moments of the given image.

    Parameters
    ----------
    img : numpy.ndarray
        2D image.
    weights : numpy.ndarray
        2D weight image.

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

    mean = np.nansum(img * weights) / np.nansum(weights)
    variance = np.nansum(weights * (img - mean) ** 2.) / np.nansum(weights)

    std = np.sqrt(variance)

    skewness = np.nansum(weights * ((img - mean) / std) ** 3.) / \
        np.nansum(weights)
    kurtosis = np.nansum(weights * ((img - mean) / std) ** 4.) / \
        np.nansum(weights) - 3

    return mean, variance, skewness, kurtosis


def _auto_nbins(size1, size2):
    '''
    Set bins to the sqrt of the smaller size.
    '''
    return int(np.sqrt(min(size1, size2)))
