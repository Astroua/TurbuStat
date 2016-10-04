# Licensed under an MIT open source license - see LICENSE

import numpy as np
import scipy.ndimage as nd
from scipy.stats import scoreatpercentile
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.convolution import Gaussian2DKernel, convolve_fft
from operator import itemgetter
from itertools import groupby
from astropy.wcs import WCS
import astropy.units as u

try:
    from scipy.fftpack import fft2
except ImportError:
    from numpy.fft import fft2

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
    lowdens_percent : float, optional
        Lower percentile of the data to use.
    highdens_percent : float, optional
        Upper percentile of the data to use.
    numpts : int, optional
        Number of thresholds to calculate statistic at.
    smoothing_radii : list, optional
        Kernel radii to smooth data to.

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

    def __init__(self, img, lowdens_percent=0, highdens_percent=100,
                 numpts=100, smoothing_radii=None):
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

        self.lowdens_percent = \
            scoreatpercentile(self.data[~np.isnan(self.data)],
                              lowdens_percent)
        self.highdens_percent = \
            scoreatpercentile(self.data[~np.isnan(self.data)],
                              highdens_percent)

        self._thresholds = np.linspace(
            self.lowdens_percent, self.highdens_percent, numpts)

        if smoothing_radii is not None:
            try:
                self._smoothing_radii = np.asarray(smoothing_radii)
            except Exception:
                raise TypeError("smoothing_radii must be convertible to a "
                                "numpy array.")
        else:
            self._smoothing_radii = \
                np.linspace(1.0, 0.1 * min(self.data.shape), 5)

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

    def make_smooth_arrays(self, **kwargs):
        '''
        Smooth data using a Gaussian kernel. NaN interpolation during
        convolution is automatically used when the data contains any NaNs.

        Parameters
        ----------
        kwargs: Passed to `~astropy.convolve.convolve_fft`.
        '''

        self._smoothed_images = []

        for i, width in enumerate(self.smoothing_radii):
            kernel = Gaussian2DKernel(width, x_size=self.data.shape[0],
                                      y_size=self.data.shape[1])
            if self.nanflag:
                self._smoothed_images.append(
                    convolve_fft(self.data, kernel,
                                 normalize_kernel=True,
                                 interpolate_nan=True,
                                 **kwargs))
            else:
                self._smoothed_images.append(
                    convolve_fft(self.data, kernel, **kwargs))

    @property
    def smoothed_images(self):
        '''
        List of smoothed versions of the image, using the radii in
        `~Genus.smoothing_radii`.
        '''
        return self._smoothed_images

    # def clean_fft(self):
    #     self.fft_images = []

    #     for j, image in enumerate(self.smoothed_images):
    #         self.fft_images.append(fft2(image))

    #     return self

    def make_genus_curve(self, use_beam=False, beam_area=None, min_size=4,
                         connectivity=1):
        '''
        Create the genus curve from the smoothed_images at the specified\
        thresholds.

        Parameters
        ----------
        use_beam : bool, optional
            When enabled, will use the given `beam_fwhm` or try to load it from
            the header. When disabled, the minimum size is set by `min_size`.
        beam_area : `~astropy.units.Quantity`, optional
            The angular area of the beam size. Requires a header to be given.
        min_size : int, optional
            Directly specify the number of pixels to be used as the minimum
            area a region must have to be counted.
        connectivity : {1, 2}, optional
            Connectivity used when removing regions below min_size.
        '''

        if use_beam:
            if self.header is None:
                raise TypeError("A header must be provided with the data to "
                                "use the beam area.")

            if beam_area is None:
                major, minor = find_beam_properties(self.header)[:2]
                major = major.to(u.pixel, equivalencies=self.angular_equiv)
                minor = minor.to(u.pixel, equivalencies=self.angular_equiv)
                pix_area = np.pi * major * minor
            else:
                if not beam_area.unit.is_equivalent(u.sr):
                    raise u.UnitsError("beam_area must be in angular units "
                                       "equivalent to solid angle. The given "
                                       "units are {}".format(beam_area.unit))

                # Can't use angular_equiv to do deg**2 to u.pix**2, so do sqrt
                pix_area = \
                    (np.sqrt(beam_area)
                     .to(u.pix, equivalencies=self.angular_equiv)) ** 2

            min_size = int(np.floor(pix_area.value))

        self._genus_stats = np.empty((len(self.smoothed_images),
                                      len(self.thresholds)))

        for j, image in enumerate(self.smoothed_images):
            for i, thresh in enumerate(self.thresholds):
                high_density = remove_small_objects(image > thresh,
                                                    min_size=min_size,
                                                    connectivity=connectivity)
                low_density = remove_small_objects(image < thresh,
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

    def run(self, verbose=False, use_beam=False, beam_area=None, min_size=4,
            **kwargs):
        '''
        Run the whole statistic.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        use_beam : bool, optional
            See `~Genus.make_genus_curve`.
        beam_area : `~astropy.units.Quantity`, optional
            See `~Genus.make_genus_curve`.
        min_size : int, optional
            See `~Genus.make_genus_curve`.
        kwargs : See `~Genus.make_smooth_arrays`.
        '''

        self.make_smooth_arrays(**kwargs)
        # self.clean_fft()
        self.make_genus_curve(use_beam=use_beam, beam_area=beam_area,
                              min_size=min_size)

        if verbose:
            import matplotlib.pyplot as p
            num = len(self.smoothing_radii)
            num_cols = num / 2 if num % 2 == 0 else (num / 2) + 1
            for i in range(1, num + 1):
                if num == 1:
                    p.subplot(111)
                else:
                    p.subplot(num_cols, 2, i)
                p.title(
                    "".join(["Smooth Size: ",
                            str(self.smoothing_radii[i - 1])]))
                p.plot(self.thresholds, self.genus_stats[i - 1], "bD")
                p.xlabel("Intensity")
                p.grid(True)
            p.tight_layout()
            p.show()

        return self


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
