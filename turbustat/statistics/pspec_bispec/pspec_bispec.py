# Licensed under an MIT open source license - see LICENSE


import numpy as np
import numpy.random as ra
from numpy.fft import fftshift
import astropy.units as u

from ..rfft_to_fft import rfft_to_fft
from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data


class PowerSpectrum(BaseStatisticMixIn, StatisticBase_PSpec2D):

    """
    Compute the power spectrum of a given image. (e.g., Burkhart et al., 2010)

    Parameters
    ----------
    img : %(dtypes)s
        2D image.
    header : FITS header, optional
        The image header. Needed for the pixel scale.
    weights : %(dtypes)s
        Weights to be applied to the image.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None):
        super(PowerSpectrum, self).__init__()

        # Set data and header
        self.input_data_header(img, header)

        self.data[np.isnan(self.data)] = 0.0

        if weights is None:
            weights = np.ones(self.data.shape)
        else:
            # Get rid of all NaNs
            weights[np.isnan(weights)] = 0.0
            weights[np.isnan(self.data)] = 0.0
            self.data[np.isnan(self.data)] = 0.0

        self.weighted_data = self.data * weights

        self._ps1D_stddev = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        fft = fftshift(rfft_to_fft(self.weighted_data))

        self._ps2D = np.power(fft, 2.)

    def run(self, verbose=False, logspacing=True,
            return_stddev=True, low_cut=None, high_cut=0.5,
            ang_units=False, unit=u.deg):
        '''
        Full computation of the spatial power spectrum.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        low_cut : float, optional
            Low frequency cut off in frequencies used in the fitting.
        high_cut : float, optional
            High frequency cut off in frequencies used in the fitting.
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)

        self.fit_pspec(low_cut=low_cut, high_cut=high_cut,
                       large_scale=0.5)
        if verbose:
            print self.fit.summary()
            self.plot_fit(show=True, show_2D=True)
        return self


class PSpec_Distance(object):

    """

    Distance metric for the spatial power spectrum. A linear model with an
    interaction term is fit to the powerlaws. The distance is the
    t-statistic of the interaction term.

    Parameters
    ----------

    data1 : %(dtypes)s
        Data with an associated header.
    data2 : %(dtypes)s
        See data1.
    weights1 : %(dtypes)s, optional
        Weights to apply to data1
    weights2 : %(dtypes)s, optional
        Weights to apply to data2
    fiducial_model : PowerSpectrum
        Computed PowerSpectrum object. use to avoid recomputing.
    ang_units : bool, optional
        Convert the frequencies to angular units using the header.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data1, data2, weights1=None, weights2=None,
                 fiducial_model=None, ang_units=False):
        super(PSpec_Distance, self).__init__()

        self.ang_units = ang_units

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1, weights=weights1)
            self.pspec1.run()
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2, weights=weights2)
        self.pspec2.run()

        self.results = None
        self.distance = None

    def distance_metric(self, low_cut=None, high_cut=0.5, verbose=False,
                        label1=None, label2=None, ang_units=False, unit=u.deg):
        '''

        Implements the distance metric for 2 Power Spectrum transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        low_cut : float, optional
            Set the cut-off for low spatial frequencies. By default, this is
            set to the inverse of half of the smallest axis in the 2 images.
        high_cut : float, optional
            Set the cut-off for high spatial frequencies. Defaults to a
            frequency of 0.5; low pixel counts below this may lead to
            significant scatter.
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        self.distance = \
            np.abs((self.pspec1.slope - self.pspec2.slope) /
                   np.sqrt(self.pspec1.slope_err**2 +
                           self.pspec2.slope_err**2))

        if verbose:
            print self.pspec1.fit.summary()
            print self.pspec2.fit.summary()

            import matplotlib.pyplot as p
            self.pspec1.plot_fit(show=False, color='b',
                                 label=label1, symbol='D',
                                 ang_units=ang_units, unit=unit)
            self.pspec2.plot_fit(show=False, color='g',
                                 label=label2, symbol='o',
                                 ang_units=ang_units, unit=unit)
            p.legend(loc='best')
            p.show()

        return self


class BiSpectrum(BaseStatisticMixIn):

    """
    Computes the bispectrum (three-point correlation function) of the given
    image (Burkhart et al., 2010).
    The bispectrum and the bicoherence are returned. The bicoherence is
    a normalized version (real and to unity) of the bispectrum.

    Parameters
    ----------
    img : %(dtypes)s
        2D image.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img):
        super(BiSpectrum, self).__init__()

        self.need_header_flag = False
        self.header = None

        self.data = input_data(img, no_header=True)
        self.shape = self.data.shape

        # Set nans to min
        self.data[np.isnan(self.data)] = np.nanmin(self.data)

    def compute_bispectrum(self, nsamples=100, seed=1000,
                           mean_subract=False):
        '''
        Do the computation.

        Parameters
        ----------
        nsamples : int, optional
            Sets the number of samples to take at each vector
            magnitude.
        seed : int, optional
            Sets the seed for the distribution draws.
        '''

        if mean_subract:
            norm_data = self.data - self.data.mean()
        else:
            norm_data = self.data

        fftarr = np.fft.fft2(norm_data)
        conjfft = np.conj(fftarr)
        ra.seed(seed)

        bispec_shape = (int(self.shape[0] / 2.), int(self.shape[1] / 2.))

        self.bispectrum = np.zeros(bispec_shape, dtype=np.complex)
        self.bicoherence = np.zeros(bispec_shape, dtype=np.float)
        self.tracker = np.zeros(self.shape, dtype=np.int16)

        biconorm = np.ones_like(self.bispectrum, dtype=float)

        for k1mag in range(int(fftarr.shape[0] / 2.)):
            for k2mag in range(int(fftarr.shape[1] / 2.)):
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

                self.bispectrum[k1mag, k2mag] = np.sum(samps)

                biconorm[k1mag, k2mag] = np.sum(np.abs(samps))

                # Track where we're sampling from in fourier space
                self.tracker[k1x, k1y] += 1
                self.tracker[k2x, k2y] += 1
                self.tracker[k3x, k3y] += 1

        self.bicoherence = (np.abs(self.bispectrum) / biconorm)
        self.bispectrum_amp = np.log10(np.abs(self.bispectrum))

    def run(self, nsamples=100, verbose=False):
        '''
        Compute the bispectrum. Necessary to maintiain package standards.

        Parameters
        ----------
        nsamples : int, optional
            Sets the number of samples to take at each vector magnitude.
        verbose : bool, optional
            Enables plotting.
        '''

        self.compute_bispectrum(nsamples=nsamples)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(1, 2, 1)
            p.title("Bispectrum")
            p.imshow(
                self.bispectrum_amp, origin="lower", interpolation="nearest")
            p.colorbar()
            p.contour(self.bispectrum_amp, colors="k")
            p.xlabel("k1")
            p.ylabel("k2")

            p.subplot(1, 2, 2)
            p.title("Bicoherence")
            p.imshow(self.bicoherence, origin="lower", interpolation="nearest")
            p.colorbar()
            p.xlabel("k1")
            p.ylabel("k2")

            p.show()

        return self


class BiSpectrum_Distance(object):

    '''
    Calculate the distance between two images based on their bicoherence.
    The distance is the L2 norm between the bicoherence surfaces.

    Parameters
    ----------
    data1 : %(dtypes)s
        Contains the data and header of the image.
    data2 : %(dtypes)s
        Contains the data and header of the image.
    nsamples : int, optional
        Sets the number of samples to take at each vector magnitude.
    fiducial_model : Bispectrum
        Computed Bispectrum object. use to avoid recomputing.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data1, data2, nsamples=100, fiducial_model=None):
        super(BiSpectrum_Distance, self).__init__()

        if fiducial_model is not None:
            self.bispec1 = fiducial_model
        else:
            self.bispec1 = BiSpectrum(data1)
            self.bispec1.run(nsamples=nsamples)

        self.bispec2 = BiSpectrum(data2)
        self.bispec2.run(nsamples=nsamples)

        self.distance = None

    def distance_metric(self, metric='average', verbose=False, label1=None,
                        label2=None):
        '''
        verbose : bool, optional
            Enable plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        '''

        if metric is 'surface':
            self.distance = np.linalg.norm(self.bispec1.bicoherence.ravel() -
                                           self.bispec2.bicoherence.ravel())
        elif metric is 'average':
            self.distance = np.abs(self.bispec1.bicoherence.mean() -
                                   self.bispec2.bicoherence.mean())
        else:
            raise ValueError("metric must be 'surface' or 'average'.")

        if verbose:
            import matplotlib.pyplot as p

            fig = p.figure()
            ax1 = fig.add_subplot(121)
            ax1.set_title(label1)
            ax1.imshow(
                self.bispec1.bicoherence, origin="lower",
                interpolation="nearest", vmax=1.0, vmin=0.0)
            ax1.set_xlabel(r"$k_1$")
            ax1.set_ylabel(r"$k_2$")

            ax2 = fig.add_subplot(122)
            ax2.set_title(label2)
            im = p.imshow(
                self.bispec2.bicoherence, origin="lower",
                interpolation="nearest", vmax=1.0, vmin=0.0)
            ax2.set_xlabel(r"$k_1$")
            # ax2.set_ylabel(r"$k_2$")
            ax2.set_yticklabels([])

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax)

            # p.tight_layout()
            p.show()

        return self


def clip_func(arr, low, high):
    return np.logical_and(arr > low, arr < high)
