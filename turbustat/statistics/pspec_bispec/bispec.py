# Licensed under an MIT open source license - see LICENSE
# from __future__ import print_function, absolute_import, division

import numpy as np
import numpy.random as ra
import astropy.units as u
from scipy.stats import binned_statistic
from warnings import warn
from astropy.utils.console import ProgressBar
from astropy.utils import NumpyRNGContext
from itertools import product

try:
    from pyfftw.interfaces.numpy_fft import fft2
    PYFFTW_FLAG = True
except ImportError:
    PYFFTW_FLAG = False

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..psds import make_radial_arrays


class Bispectrum(BaseStatisticMixIn):

    """
    Computes the bispectrum (three-point correlation function) of the given
    image (Burkhart et al., 2010).
    The bispectrum and the bicoherence are returned. The bicoherence is
    a normalized version (real and to unity) of the bispectrum.

    Parameters
    ----------
    img : %(dtypes)s
        2D image.


    Examples
    --------
    >>> from turbustat.statistics import Bispectrum
    >>> from astropy.io import fits
    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits") # doctest: +SKIP
    >>> bispec = Bispectrum(moment0) # doctest: +SKIP
    >>> bispec.run(verbose=True, nsamples=1000) # doctest: +SKIP

    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img):

        self.need_header_flag = False
        self.header = None

        self.data = input_data(img, no_header=True)
        self.shape = self.data.shape

        # Set nans to min
        isnan = np.isnan(self.data)
        if isnan.any():
            self.data = self.data.copy()
            self.data[isnan] = 0.

    def compute_bispectrum(self, show_progress=True, use_pyfftw=False,
                           threads=1, nsamples=100, seed=1000,
                           mean_subtract=False, **pyfftw_kwargs):
        '''
        Do the computation.

        Parameters
        ----------
        show_progress : optional, bool
            Show progress bar while sampling the bispectrum.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        nsamples : int, optional
            Sets the number of samples to take at each vector
            magnitude.
        seed : int, optional
            Sets the seed for the distribution draws.
        mean_subtract : bool, optional
            Subtract the mean from the data before computing. This removes the
            "zero frequency" (i.e., constant) portion of the power, resulting
            in a loss of phase coherence along the k_1=k_2 line.
        pyfft_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html#interfaces-additional-args>`_
            for a list of accepted kwargs.
        '''

        if mean_subtract:
            norm_data = self.data - self.data.mean()
        else:
            norm_data = self.data

        if use_pyfftw:
            if PYFFTW_FLAG:
                if pyfftw_kwargs.get('threads') is not None:
                    pyfftw_kwargs.pop('threads')

                fftarr = fft2(norm_data,
                              threads=threads,
                              **pyfftw_kwargs)
            else:
                warn("pyfftw not installed. Reverting to using numpy.")
                use_pyfftw = False

        if not use_pyfftw:
            fftarr = np.fft.fft2(norm_data)

        conjfft = np.conj(fftarr)

        bispec_shape = (int(self.shape[0] / 2.), int(self.shape[1] / 2.))

        self._bispectrum = np.zeros(bispec_shape, dtype=np.complex)
        self._bicoherence = np.zeros(bispec_shape, dtype=np.float)
        self._tracker = np.zeros(self.shape, dtype=np.int16)

        biconorm = np.ones_like(self.bispectrum, dtype=float)

        if show_progress:
            bar = ProgressBar(np.prod(fftarr.shape) / 4.)

        prod = product(range(int(fftarr.shape[0] / 2.)),
                       range(int(fftarr.shape[1] / 2.)))

        with NumpyRNGContext(seed):
            for n, (k1mag, k2mag) in enumerate(prod):
                phi1 = ra.uniform(0, 2 * np.pi, nsamples)
                phi2 = ra.uniform(0, 2 * np.pi, nsamples)

                k1x = np.asarray([int(k1mag * np.cos(angle))
                                  for angle in phi1])
                k2x = np.asarray([int(k2mag * np.cos(angle))
                                  for angle in phi2])
                k1y = np.asarray([int(k1mag * np.sin(angle))
                                  for angle in phi1])
                k2y = np.asarray([int(k2mag * np.sin(angle))
                                  for angle in phi2])

                k3x = np.asarray([int(k1mag * np.cos(ang1) +
                                      k2mag * np.cos(ang2))
                                  for ang1, ang2 in zip(phi1, phi2)])
                k3y = np.asarray([int(k1mag * np.sin(ang1) +
                                      k2mag * np.sin(ang2))
                                  for ang1, ang2 in zip(phi1, phi2)])

                samps = fftarr[k1x, k1y] * fftarr[k2x, k2y] * conjfft[k3x, k3y]

                self._bispectrum[k1mag, k2mag] = np.sum(samps)

                biconorm[k1mag, k2mag] = np.sum(np.abs(samps))

                # Track where we're sampling from in fourier space
                self._tracker[k1x, k1y] += 1
                self._tracker[k2x, k2y] += 1
                self._tracker[k3x, k3y] += 1

                if show_progress:
                    bar.update(n + 1)

        self._bicoherence = (np.abs(self.bispectrum) / biconorm)
        self._bispectrum_amp = np.log10(np.abs(self.bispectrum))

    @property
    def bispectrum(self):
        '''
        Bispectrum array.
        '''
        return self._bispectrum

    @property
    def bispectrum_logamp(self):
        '''
        log amplitudes of the bispectrum.
        '''
        return self._bispectrum_amp

    @property
    def bicoherence(self):
        '''
        Bicoherence array.
        '''
        return self._bicoherence

    @property
    def tracker(self):
        '''
        Array showing the number of samples in each k_1 k_2 bin.
        '''
        return self._tracker

    def azimuthal_slice(self, radii, delta_radii, bin_width=5. * u.deg,
                        value='bispectrum', return_masks=False):
        '''
        Create an azimuthal slice of the bispectrum or bicoherence
        surfaces.

        Parameters
        ----------
        radii : float or np.ndarray
            Radii in the bispectrum plane to extract slices at. Multiple
            slices are returned if radii is an array.
        delta_radii : float or np.ndarray
            The width around the radii in the bispectrum plane. If multiple
            radii are given, `delta_radii` must match the length of `radii`.
        bin_width : `~astropy.units.Quantity`, optional
            The angular size of the bins used to create the slice.
        value : str, optional
            Which surface to create a profile from. Can be "bispectrum"
            (default), "bispectrum_logamp", or "bicoherence".
        return_masks : bool, optional
            Return the radial masks used to create the slices.

        Returns
        -------
        azimuthal_slices : dict
            Dictionary with the azimuthal slices. Each radius given in radii
            is a key in the dictionary. Each slice is a numpy array containing
            the bin centers, the averaged values, and the standard deviations
            in the azimuthal bins.
        '''

        if value == "bispectrum":
            value_arr = self.bispectrum
        elif value == "bispectrum_logamp":
            value_arr = self.bispectrum_logamp
        elif value == "bicoherence":
            value_arr = self.bicoherence
        else:
            raise TypeError("value must be 'bispectrum'"
                            ", 'bispectrum_logamp', or 'bicoherence'")

        if isinstance(radii, np.ndarray) or isinstance(radii, list):
            if not isinstance(delta_radii, np.ndarray):
                delta_radii = np.array([delta_radii] * len(radii))
            if len(radii) != len(delta_radii):
                raise ValueError("Length of radii and delta_radii must match.")
        else:
            radii = np.array([radii])

        if not isinstance(delta_radii, np.ndarray):
            delta_radii = np.array([delta_radii])

        if not hasattr(bin_width, "unit"):
            raise TypeError("bin_width must have an attached angular unit.")
        elif not bin_width.unit.is_equivalent(u.rad):
            raise TypeError("bin_width must have an attached angular unit.")
        else:
            bin_width = bin_width.to(u.rad).value

        kky, kkx = make_radial_arrays(self.bispectrum.shape, y_center=0,
                                      x_center=0)

        dist = np.sqrt(kky**2 + kkx**2)
        theta = np.arctan2(kky, kkx)

        nbins = np.floor(np.pi / bin_width).astype(int)
        bins = np.linspace(0, np.pi, nbins)

        azimuthal_slices = {}
        if return_masks:
            masks = []

        for rad, del_rad in zip(radii, delta_radii):

            # Create the mask of the radii to extract the profile at.
            mask = np.logical_and(dist >= rad - del_rad / 2.,
                                  dist <= rad + del_rad / 2.)

            vals, bin_edge, cts = binned_statistic(theta[mask].ravel(),
                                                   value_arr[mask].ravel(),
                                                   bins=bins,
                                                   statistic=np.nanmean)

            stds, bin_edge, cts = binned_statistic(theta[mask].ravel(),
                                                   value_arr[mask].ravel(),
                                                   bins=bins,
                                                   statistic=np.nanstd)

            bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.

            azimuthal_slices[rad] = np.array([bin_cents, vals, stds])
            if return_masks:
                masks.append(mask)

        if return_masks:
            return azimuthal_slices, masks

        return azimuthal_slices

    def radial_slice(self, thetas, delta_thetas, bin_width=1.,
                     value='bispectrum', return_masks=False):
        '''
        Create a radial slice of the bispectrum (or bicoherence) plane.

        Parameters
        ----------
        thetas : `~astropy.units.Quantity`
            Azimuthal angles in the bispectrum plane to extract slices at.
            Multiple slices are returned if `thetas` is an array.
        delta_thetas : `~astropy.units.Quantity`
            The width around the angle in the bispectrum plane. If multiple
            angles are given, `delta_thetas` must match the length of `thetas`.
        bin_width : float, optional
            The radial size of the bins used to create the slice.
        value : str, optional
            Which surface to create a profile from. Can be "bispectrum"
            (default), "bispectrum_logamp", or "bicoherence".
        return_masks : bool, optional
            Return the radial masks used to create the slices.

        Returns
        -------
        radial_slices : dict
            Dictionary with the radial slices. Each angle given in thetas
            is a key in the dictionary. Each slice is a numpy array containing
            the bin centers, the averaged values, and the standard deviations
            in the radial bins.
        '''

        if value == "bispectrum":
            value_arr = self.bispectrum
        elif value == "bispectrum_logamp":
            value_arr = self.bispectrum_logamp
        elif value == "bicoherence":
            value_arr = self.bicoherence
        else:
            raise TypeError("value must be 'bispectrum'"
                            ", 'bispectrum_logamp', or 'bicoherence'")

        orig_thetas = thetas.copy().value

        # Check units
        if not hasattr(thetas, "unit") or not hasattr(delta_thetas, "unit"):
            raise TypeError("thetas must have an attached angular unit.")
        elif (not thetas.unit.is_equivalent(u.rad) or
              not delta_thetas.unit.is_equivalent(u.rad)):
            raise TypeError("thetas must have an attached angular unit.")
        else:
            thetas = thetas.to(u.rad).value
            delta_thetas = delta_thetas.to(u.rad).value

        # Make sure the lengths match if multiple values are given.
        if isinstance(thetas, np.ndarray):
            if not isinstance(delta_thetas, np.ndarray):
                delta_thetas = np.array([delta_thetas] * len(thetas))
            if len(thetas) != len(delta_thetas):
                raise ValueError("Length of thetas and delta_thetas must "
                                 "match.")
        else:
            thetas = np.array([thetas])
            orig_thetas = np.array([orig_thetas])

        if not isinstance(delta_thetas, np.ndarray):
            delta_thetas = np.array([delta_thetas])

        kky, kkx = make_radial_arrays(self.bispectrum.shape, y_center=0,
                                      x_center=0)

        radial_slices = dict.fromkeys(orig_thetas)
        if return_masks:
            masks = []

        dist = np.sqrt(kky**2 + kkx**2)
        theta_arr = np.arctan2(kky, kkx)

        nbins = np.floor(dist.max() / bin_width).astype(int)
        bins = np.linspace(0, dist.max(), nbins)

        for theta, del_theta, theta0 in zip(thetas, delta_thetas, orig_thetas):

            # Create the mask of the radii to extract the profile at.
            mask = np.logical_and(theta_arr >= theta - del_theta / 2.,
                                  theta_arr <= theta + del_theta / 2.)

            vals, bin_edge, cts = binned_statistic(dist[mask].ravel(),
                                                   value_arr[mask].ravel(),
                                                   bins=bins,
                                                   statistic=np.nanmean)

            stds, bin_edge, cts = binned_statistic(dist[mask].ravel(),
                                                   value_arr[mask].ravel(),
                                                   bins=bins,
                                                   statistic=np.nanstd)

            bin_cents = (bin_edge[1:] + bin_edge[:-1]) / 2.

            radial_slices[theta0] = np.array([bin_cents, vals, stds])
            if return_masks:
                masks.append(mask)

        if return_masks:
            return radial_slices, masks

        return radial_slices

    def plot_surface(self, save_name=None, show_bicoh=True,
                     cmap='viridis', contour_color='k'):
        '''
        Plot the bispectrum amplitude and (optionally) the bicoherence.

        Parameters
        ----------
        save_name : str, optional
            Save name for the figure. Enables saving the plot.
        show_bicoh : bool, optional
            Plot the bicoherence surface. Enabled by default.
        cmap : {str, matplotlib color map}, optional
            Colormap to use in the plots. Default is viridis.
        contour_color : {str, RGB tuple}, optional
            Color of the amplitude contours.
        '''

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if show_bicoh:
            ax1 = plt.subplot(1, 2, 1)
        else:
            ax1 = plt.subplot(1, 1, 1)
        im1 = ax1.imshow(self.bispectrum_logamp, origin="lower",
                         interpolation="nearest", cmap=cmap)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar1 = plt.colorbar(im1, cax=cax)
        cbar1.set_label(r"log$_{10}$ Bispectrum Amplitude")
        ax1.contour(self.bispectrum_logamp,
                    colors=contour_color)
        ax1.set_xlabel(r"$k_1$")
        ax1.set_ylabel(r"$k_2$")

        if show_bicoh:
            ax2 = plt.subplot(1, 2, 2)
            im2 = ax2.imshow(self.bicoherence, origin="lower",
                             interpolation="nearest",
                             cmap=cmap)
            divider = make_axes_locatable(ax2)
            cax2 = divider.append_axes("right", size="5%", pad=0.05)
            cbar2 = plt.colorbar(im2, cax=cax2)
            cbar2.set_label("Bicoherence")
            ax2.set_xlabel(r"$k_1$")
            ax2.set_ylabel(r"$k_2$")

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, show_progress=True, use_pyfftw=False, threads=1,
            nsamples=100, seed=1000,
            mean_subtract=False, verbose=False,
            save_name=None, **pyfftw_kwargs):
        '''
        Compute the bispectrum. Necessary to maintain package standards.

        Parameters
        ----------
        show_progress : optional, bool
            Show progress bar while sampling the bispectrum.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        nsamples : int, optional
            See `~BiSpectrum.compute_bispectrum`.
        seed : int, optional
            See `~BiSpectrum.compute_bispectrum`.
        mean_subtract : bool, optional
            See `~BiSpectrum.compute_bispectrum`.
        verbose : bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given.
        pyfftw_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html#interfaces-additional-args>`_
            for a list of accepted kwargs.
        '''

        self.compute_bispectrum(show_progress=show_progress,
                                use_pyfftw=use_pyfftw,
                                threads=threads,
                                nsamples=nsamples,
                                mean_subtract=mean_subtract,
                                seed=seed, **pyfftw_kwargs)

        if verbose:
            self.plot_surface(save_name=save_name)

        return self


def BiSpectrum(*args, **kwargs):
    '''
    Old name for the Bispectrum class.
    '''

    warn("Use the new 'Bispectrum' class. 'BiSpectrum' is deprecated and will"
         " be removed in a future release.", Warning)

    return Bispectrum(*args, **kwargs)


class Bispectrum_Distance(object):

    '''
    Calculate the distance between two images based on their bicoherence.
    The distance is the L2 norm between the bicoherence surfaces.

    Parameters
    ----------
    data1 : %(dtypes)s or `~Bispectrum`
        Contains the data and header of the image. Or a `~Bispectrum` class may
        be given which can be pre-computed.
    data2 : %(dtypes)s or `~Bispectrum`
        Contains the data and header of the second image. Or a `~Bispectrum`
        class may be given which can be pre-computed.
    stat_kwargs : dict, optional
        Passed to `~Bispectrum.run`.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data1, data2, stat_kwargs={}):

        # if fiducial_model is not None:
        #     self.bispec1 = fiducial_model

        if isinstance(data1, Bispectrum):
            self.bispec1 = data1
            needs_run = False
            if not hasattr(self.bispec1, '_bicoherence'):
                needs_run = True
                warn("Bispectrum given as data1 does not have a"
                     " bicoherence surface. Re-running Bispectrum.")
        else:
            self.bispec1 = BiSpectrum(data1)
            needs_run = True

        if needs_run:
            self.bispec1.run(**stat_kwargs)

        if isinstance(data2, Bispectrum):
            self.bispec2 = data2
            needs_run = False
            if not hasattr(self.bispec2, '_bicoherence'):
                needs_run = True
                warn("Bispectrum given as data2 does not have a"
                     " bicoherence surface. Re-running Bispectrum.")
        else:
            self.bispec2 = BiSpectrum(data2)
            needs_run = True

        if needs_run:
            self.bispec2.run(**stat_kwargs)

    @property
    def surface_distance(self):
        '''
        L2 distance between the bicoherence surfaces.
        '''
        return self._surface_distance

    @property
    def mean_distance(self):
        '''
        Absolute difference between the mean bicoherence.
        '''
        return self._mean_distance

    def distance_metric(self, verbose=False, label1=None,
                        label2=None, save_name=None,
                        cmap='viridis'):
        '''
        verbose : bool, optional
            Enable plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        save_name : str,optional
            Save the figure when a file name is given.
        cmap : str, optional
            Colormap to show the bicoherence surfaces.
        '''

        if self.bispec1.bicoherence.shape == self.bispec2.bicoherence.shape:
            self._surface_distance = \
                np.linalg.norm(self.bispec1.bicoherence.ravel() -
                               self.bispec2.bicoherence.ravel())
        else:
            warn("Bicoherence surface must have equal shapes for the surface"
                 " distance metric.")
            self._surface_distance = np.NaN

        self._mean_distance = np.abs(self.bispec1.bicoherence.mean() -
                                     self.bispec2.bicoherence.mean())

        if verbose:
            import matplotlib.pyplot as plt

            fig = plt.gcf()

            if label1 is None:
                label1 = "1"
            if label2 is None:
                label2 = "2"

            fig = plt.gcf()

            ax1 = fig.add_subplot(121)
            ax1.set_title(label1)
            ax1.imshow(self.bispec1.bicoherence, origin="lower",
                       cmap=cmap,
                       interpolation="nearest", vmax=1.0, vmin=0.0)
            ax1.set_xlabel(r"$k_1$")
            ax1.set_ylabel(r"$k_2$")

            ax2 = plt.subplot(122)
            ax2.set_title(label2)
            im = plt.imshow(self.bispec2.bicoherence, origin="lower",
                            cmap=cmap,
                            interpolation="nearest", vmax=1.0, vmin=0.0)
            ax2.set_xlabel(r"$k_1$")
            ax2.set_yticklabels([])

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self


def BiSpectrum_Distance(*args, **kwargs):
    '''
    Old name for the Bispectrum class.
    '''

    warn("Use the new 'Bispectrum_Distance' class. 'BiSpectrum_Distance'"
         " is deprecated and will be removed in a future release.",
         Warning)

    return Bispectrum_Distance(*args, **kwargs)
