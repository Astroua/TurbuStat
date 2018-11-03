# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import scipy.ndimage as nd
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Gaussian2DKernel, convolve_fft
from astropy.wcs import WCS
import astropy.units as u
from warnings import warn

from ..stats_utils import standardize, common_scale
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data, find_beam_properties


class Genus(BaseStatisticMixIn):

    """

    Genus Statistics based off of Chepurnov et al. (2008).

    Parameters
    ----------

    img : %(dtypes)s
        2D image.
    min_value : `~astropy.units.Quantity` or float, optional
        Minimum value in the data to consider. If None, the minimum is used.
        When `img` has an attached brightness unit, `min_value` must have the
        same units.
    max_value : `~astropy.units.Quantity` or float, optional
        Maximum value in the data to consider. If None, the maximum is used.
        When `img` has an attached brightness unit, `min_value` must have the
        same units.
    lowdens_percent : float, optional
        Lower percentile of the data to use. Defaults to the minimum value.
        Overrides `min_value` when the value of this percentile is greater
        than `min_value`.
    highdens_percent : float, optional
        Upper percentile of the data to use. Defaults to the maximum value.
        Overrides `max_value` when the value of this percentile is lower than
        `max_value`.
    numpts : int, optional
        Number of thresholds to calculate statistic at.
    smoothing_radii : np.ndarray or `astropy.units.Quantity`, optional
        Kernel radii to smooth data to. If units are not attached, the radii
        are assumed to be in pixels. If no radii are given, 5 smoothing radii
        will be used ranging from 1 pixel to one-tenth the smallest dimension
        size.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.

    Example
    -------
    >>> from turbustat.statistics import Genus
    >>> from astropy.io import fits
    >>> import astropy.units as u
    >>> import numpy as np
    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits")[0]  # doctest: +SKIP
    >>> genus = Genus(moment0, lowdens_percent=15, highdens_percent=85)  # doctest: +SKIP
    >>> genus.run()  # doctest: +SKIP

    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, min_value=None, max_value=None, lowdens_percent=0,
                 highdens_percent=100, numpts=100, smoothing_radii=None,
                 distance=None):
        super(Genus, self).__init__()

        if isinstance(img, np.ndarray):
            self.need_header_flag = False
            self.data = input_data(img, no_header=True)
            self.header = None
        else:
            self.need_header_flag = True
            self.data, self.header = input_data(img, no_header=False)

        self.nanflag = False
        if np.isnan(self.data).any():
            self.nanflag = True

        if distance is not None:
            self.distance = distance

        if min_value is None:
            min_value = np.nanmin(self.data)
        else:
            if hasattr(self.data, 'unit'):
                if not hasattr(min_value, 'unit'):
                    raise TypeError("data has units of {}. 'min_value' must "
                                    "have equivalent units."
                                    .format(self.data.unit))
                if not min_value.unit.is_equivalent(self.data.unit):
                    raise u.UnitsError("min_value does not have an equivalent "
                                       "units to the img unit.")

                min_value = min_value.to(self.data.unit)

        if max_value is None:
            max_value = np.nanmax(self.data)
        else:
            if hasattr(self.data, 'unit'):
                if not hasattr(max_value, 'unit'):
                    raise TypeError("data has units of {}. 'max_value' must "
                                    "have equivalent units."
                                    .format(self.data.unit))
                if not max_value.unit.is_equivalent(self.data.unit):
                    raise u.UnitsError("max_value does not have an equivalent "
                                       "units to the img unit.")

                max_value = max_value.to(self.data.unit)

        min_percent = \
            np.percentile(self.data[~np.isnan(self.data)],
                          lowdens_percent)
        max_percent = \
            np.percentile(self.data[~np.isnan(self.data)],
                          highdens_percent)

        if min_value is None or min_percent > min_value:
            min_value = min_percent

        if max_value is None or max_percent > max_value:
            max_value = max_percent

        self._thresholds = np.linspace(min_value, max_value, numpts)

        if smoothing_radii is None:
            self.smoothing_radii = \
                np.linspace(1.0, 0.1 * min(self.data.shape), 5)
        else:
            if isinstance(smoothing_radii, u.Quantity):
                print(smoothing_radii)
                self.smoothing_radii = self._to_pixel(smoothing_radii).value
            else:
                self.smoothing_radii = smoothing_radii

    @property
    def thresholds(self):
        '''
        Values of the data to compute the Genus statistics at.
        '''
        return self._thresholds

    @property
    def smoothing_radii(self):
        '''
        Pixel radii used to smooth the data.
        '''
        return self._smoothing_radii

    @smoothing_radii.setter
    def smoothing_radii(self, values):

        if np.any(values < 1.0):
            raise ValueError("All smoothing radii must be larger than one"
                             " pixel.")

        if np.any(values > 0.5 * min(self.data.shape)):
            raise ValueError("All smoothing radii must be smaller than half of"
                             " the image shape.")

        self._smoothing_radii = values

    @property
    def smoothed_images(self):
        '''
        List of smoothed versions of the image, using the radii in
        `~Genus.smoothing_radii`.
        '''
        if not hasattr(self, '_smoothed_images'):
            raise ValueError("Set `keep_smoothed_images=True` in "
                             "Genus.make_genus_curve")

        return self._smoothed_images

    def make_genus_curve(self, use_beam=False, min_size=4,
                         connectivity=1, keep_smoothed_images=False,
                         match_kernel=False,
                         **convolution_kwargs):
        '''
        Smooth the data with a Gaussian kernel to create the genus curve from
        at the specified thresholds.

        Parameters
        ----------
        use_beam : bool, optional
            When enabled, will use the given `beam_fwhm` or try to load it from
            the header. When disabled, the minimum size is set by `min_size`.
        min_size : int or `~astropy.units.Quantity`, optional
            Directly specify the minimum
            area a region must have to be counted. Integer values with no units
            are assumed to be in pixels.
        connectivity : {1, 2}, optional
            Connectivity used when removing regions below min_size.
        keep_smoothed_images : bool, optional
            Keep the convolved images in the `~Genus.smoothed_images` list.
            Default is `False`.
        match_kernel : bool, optional
            Match kernel shape to the data shape when convolving. Default is
            `False`. Enable to reproduce behaviour of `~Genus` prior to
            version 1.0 of TurbuStat.
        convolution_kwargs: Passed to `~astropy.convolve.convolve_fft`.

        '''

        if keep_smoothed_images:
            self._smoothed_images = []

        if use_beam:
            major, minor = find_beam_properties(self.header)[:2]
            major = self._to_pixel(major)
            minor = self._to_pixel(minor)
            pix_area = np.pi * major * minor
            min_size = int(np.floor(pix_area.value))
        else:
            if isinstance(min_size, u.Quantity):
                # Convert to pixel area
                min_size = self._to_pixel_area(min_size)

                min_size = int(np.floor(min_size.value))
            else:
                min_size = int(min_size)

        self._genus_stats = np.empty((len(self.smoothing_radii),
                                      len(self.thresholds)))

        for j, width in enumerate(self.smoothing_radii):

            if match_kernel:
                kernel = Gaussian2DKernel(width, x_size=self.data.shape[0],
                                          y_size=self.data.shape[1])
            else:
                kernel = Gaussian2DKernel(width)

            smooth_img = convolve_fft(self.data, kernel, **convolution_kwargs)

            if keep_smoothed_images:
                self._keep_smoothed_images.append(smooth_img)

            for i, thresh in enumerate(self.thresholds):
                high_density = remove_small_objects(smooth_img > thresh,
                                                    min_size=min_size,
                                                    connectivity=connectivity)
                low_density = remove_small_objects(smooth_img < thresh,
                                                   min_size=min_size,
                                                   connectivity=connectivity)
                # eight-connectivity to count the regions
                high_density_labels, high_density_num = \
                    nd.label(high_density, np.ones((3, 3)))
                low_density_labels, low_density_num = \
                    nd.label(low_density, np.ones((3, 3)))

                self._genus_stats[j, i] = high_density_num - low_density_num

    @property
    def genus_stats(self):
        '''
        Array of genus statistic values for all smoothed images (0th axis) and
        all threshold values (1st axis).
        '''
        return self._genus_stats

    def plot_fit(self, save_name=None, color='b'):
        '''
        Plot the Genus curves.

        Parameters
        ----------
        save_name : str,optional
            Save the figure when a file name is given.
        color : {str, RGB tuple}, optional
            Color to show the Genus curves in.
        '''

        import matplotlib.pyplot as plt

        num = len(self.smoothing_radii)
        num_cols = num / 2 if num % 2 == 0 else (num / 2) + 1

        for i in range(1, num + 1):
            if num == 1:
                ax = plt.subplot(111)
            else:
                ax = plt.subplot(num_cols, 2, i)
            # plt.title("Smooth Size: {0}".format(self.smoothing_radii[i - 1]))
            ax.text(0.3, 0.1,
                    "Smooth Size: {0:.2f}".format(self.smoothing_radii[i - 1]),
                    transform=ax.transAxes, fontsize=12)
            plt.plot(self.thresholds, self.genus_stats[i - 1], "D-",
                     color=color)
            plt.xlabel("Intensity")
            plt.grid(True)

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, verbose=False, save_name=None,
            color='b', **kwargs):
        '''
        Run the whole statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given. Must have `verbose`
            enabled for plotting.
        kwargs : See `~Genus.make_genus_curve`.
        '''

        self.make_genus_curve(**kwargs)

        if verbose:
            self.plot_fit(save_name=save_name, color=color)

        return self


class Genus_Distance(object):

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

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        save_name=None):
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
        save_name : str,optional
            Save the figure when a file name is given.
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

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()

        return self


def GenusDistance(*args, **kwargs):
    '''
    Old name for the Genus_Distance class.
    '''

    warn("Use the new 'Genus_Distance' class. 'GenusDistance' is deprecated and will"
         " be removed in a future release.", Warning)

    return Genus_Distance(*args, **kwargs)


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
