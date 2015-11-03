# Licensed under an MIT open source license - see LICENSE


import numpy as np
import numpy.random as ra
from ..psds import pspec
from ..rfft_to_fft import rfft_to_fft
import statsmodels.formula.api as sm
from pandas import Series, DataFrame
from numpy.fft import fft2, fftshift


class PowerSpectrum(object):

    """
    Compute the power spectrum of a given image. (Burkhart et al., 2010)

    Parameters
    ----------
    img : numpy.ndarray
        2D image.
    header : FITS header
        The image header. Needed for the pixel scale.

    """

    def __init__(self, img, header, weights=None):
        super(PowerSpectrum, self).__init__()
        self.img = img
        # Get rid of nans
        self.img[np.isnan(self.img)] = 0.0

        self.header = header
        self.degperpix = np.abs(header["CDELT2"])

        if weights is None:
            weights = np.ones(img.shape)
        else:
            # Get rid of all NaNs
            weights[np.isnan(weights)] = 0.0
            weights[np.isnan(self.img)] = 0.0
            self.img[np.isnan(self.img)] = 0.0

        self.weighted_img = self.img * weights

        self._ps1D_stddev = None

    @property
    def ps2D(self):
        return self._ps2D

    @property
    def ps1D(self):
        return self._ps1D

    @property
    def ps1D_stddev(self):
        if not self._stddev_flag:
            Warning("scf_spectrum_stddev is only calculated when return_stddev"
                    " is enabled.")

        return self._ps1D_stddev

    @property
    def freqs(self):
        return self._freqs

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        fft = fftshift(rfft_to_fft(self.weighted_img))

        self._ps2D = np.power(fft, 2.)

        return self

    def compute_radial_pspec(self, return_stddev=False, logspacing=True,
                             **kwargs):
        '''
        Computes the radially averaged power spectrum.

        Parameters
        ----------
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        kwargs : passed to pspec
        '''

        if return_stddev:
            self._freqs, self._ps1D, self._ps1D_stddev = \
                pspec(self.ps2D, logspacing=logspacing,
                      return_stddev=return_stddev, **kwargs)
            self._stddev_flag = True
        else:
            self._freqs, self._ps1D = \
                pspec(self.ps2D, logspacing=logspacing,
                      return_stddev=return_stddev, **kwargs)
            self._stddev_flag = False

        return self

    def run(self, phys_units=False, verbose=False, return_stddev=False,
            logspacing=True):
        '''
        Full computation of the Spatial Power Spectrum.

        Parameters
        ----------
        phys_units : bool, optional
            Sets frequency scale to physical units.
        verbose: bool, optional
            Enables plotting.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)

        if phys_units:
            self._freqs *= self.degperpix ** -1

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.imshow(
                np.log10(self.ps2D), origin="lower", interpolation="nearest")
            p.colorbar()
            ax = p.subplot(122)
            if self._stddev_flag:
                ax.errorbar(self.freqs, self.ps1D, yerr=self.ps1D_stddev,
                            fmt='D-', color='b', alpha=0.5, markersize=5)
                ax.set_xscale("log", nonposy='clip')
                ax.set_yscale("log", nonposy='clip')
            else:
                p.loglog(self.freqs, self.ps1D, "bD", alpha=0.5,
                         markersize=5)
            if phys_units:
                ax.set_xlabel(r"log K/deg$^{-1}$")
            else:
                ax.set_xlabel(r"log K/pixel$^{-1}$")

            p.ylabel("Power (K)")

            p.tight_layout()
            p.show()

        return self


class PSpec_Distance(object):

    """

    Distance metric for the spatial power spectrum. A linear model with an
    interaction term is fit to the powerlaws. The distance is the
    t-statistic of the interaction term.

    Parameters
    ----------

    data1 : dict
        Dictionary containing necessary property arrays.
    data2 : dict
        Dictionary containing necessary property arrays.
    fiducial_model : PowerSpectrum
        Computed PowerSpectrum object. use to avoid recomputing.

    """

    def __init__(self, data1, data2, weights1=None, weights2=None,
                 fiducial_model=None):
        super(PSpec_Distance, self).__init__()

        self.shape1 = data1[0].shape
        self.shape2 = data2[0].shape

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1[0],
                                        data1[1],
                                        weights=weights1)
            self.pspec1.run()
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2[0],
                                    data2[1],
                                    weights=weights2)
        self.pspec2.run()

        self.results = None
        self.distance = None

    def distance_metric(self, low_cut=2.0, high_cut=64.0, verbose=False):
        '''

        Implements the distance metric for 2 Power Spectrum transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        low_cut : int or float, optional
            Set the cut-off for low spatial frequencies. Visually, below ~2
            deviates from the power law (for the simulation set).
        high_cut : int or float, optional
            Set the cut-off for high spatial frequencies. Values beyond the
            size of the root grid are found to have no meaningful contribution
        verbose : bool, optional
            Enables plotting.
        '''

        clip_freq1 = \
            self.pspec1.freqs[clip_func(self.pspec1.freqs, low_cut, high_cut)]
        clip_ps1D1 = \
            self.pspec1.ps1D[clip_func(self.pspec1.freqs, low_cut, high_cut)]

        clip_freq2 = \
            self.pspec2.freqs[clip_func(self.pspec2.freqs, low_cut, high_cut)]
        clip_ps1D2 = \
            self.pspec2.ps1D[clip_func(self.pspec2.freqs, low_cut, high_cut)]

        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(
            x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(
            formula="log_ps1D ~ dummy + scales + regressor", data=df)

        self.results = model.fit()

        self.distance = np.abs(self.results.tvalues["regressor"])

        if verbose:

            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_freq1), np.log10(clip_ps1D1), "bD",
                   np.log10(clip_freq2), np.log10(clip_ps1D2), "gD")
            p.plot(df["scales"][:len(clip_freq1)],
                   self.results.fittedvalues[:len(clip_freq1)], "b",
                   df["scales"][-len(clip_freq2):],
                   self.results.fittedvalues[-len(clip_freq2):], "g")
            p.grid(True)
            p.xlabel("log K")
            p.ylabel("Power (K)")
            p.show()

        return self


class BiSpectrum(object):

    """
    Computes the bispectrum (three-point correlation function) of the given
    image (Burkhart et al., 2010).
    The bispectrum and the bicoherence are returned. The bicoherence is
    a normalized version (real and to unity) of the bispectrum.

    Parameters
    ----------
    img : numpy.ndarray
        2D image.

    """

    def __init__(self, img):
        super(BiSpectrum, self).__init__()
        self.img = img
        self.shape = img.shape

        # Set nans to min
        self.img[np.isnan(self.img)] = np.nanmin(self.img)

        self.kx = np.arange(0., self.shape[0] / 2., 1)
        self.ky = np.arange(0., self.shape[1] / 2., 1)

        self.bispectrum = np.zeros(
            (int(self.shape[0] / 2.) - 1, int(self.shape[1] / 2.) - 1),
            dtype=np.complex)
        self.bicoherence = np.zeros(
            (int(self.shape[0] / 2.) - 1, int(self.shape[1] / 2.) - 1))
        self.bispectrum_amp = None
        self.accumulator = np.zeros(
            (int(self.shape[0] / 2.) - 1, int(self.shape[1] / 2.) - 1))
        self.tracker = np.zeros(self.shape)

    def compute_bispectrum(self, nsamples=100, seed=1000):
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

        fft = np.fft.fft2(self.img)
        conjfft = np.conj(fft)
        ra.seed(seed)

        for k1mag in range(int(fft.shape[0] / 2.)):
            for k2mag in range(int(fft.shape[1] / 2.)):
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

                k3x = k1x + k2x
                k3y = k1y + k2y

                x = np.asarray([int(np.sqrt(i ** 2. + j ** 2.))
                                for i, j in zip(k1x, k1y)])
                y = np.asarray([int(np.sqrt(i ** 2. + j ** 2.))
                                for i, j in zip(k2x, k2y)])

                for n in range(nsamples):
                    self.bispectrum[x[n], y[n]] +=\
                        fft[k1x[n], k1y[n]] *\
                        fft[k2x[n], k2y[n]] *\
                        conjfft[k3x[n], k3y[n]]
                    self.bicoherence[x[n], y[n]] +=\
                        np.abs(fft[k1x[n], k1y[n]] *
                               fft[k2x[n], k2y[n]] *
                               conjfft[k3x[n], k3y[n]])
                    self.accumulator[x[n], y[n]] += 1.

                # Track where we're sampling from in fourier space
                    self.tracker[k1x[n], k1y[n]] += 1
                    self.tracker[k2x[n], k2y[n]] += 1
                    self.tracker[k3x[n], k3y[n]] += 1

        self.bicoherence = (np.abs(self.bispectrum) / self.bicoherence)
        self.bispectrum /= self.accumulator
        self.bispectrum_amp = np.log10(np.abs(self.bispectrum) ** 2.)

        return self

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


class BiSpectrum_Distance(object):

    '''
    Calculate the distance between two images based on their bicoherence.
    The distance is the L2 norm between the bicoherence surfaces.

    Parameters
    ----------
    data1 : FITS hdu
        Contains the data and header of the image.
    data2 : FITS hdu
        Contains the data and header of the image.
    nsamples : int, optional
        Sets the number of samples to take at each vector magnitude.
    fiducial_model : Bispectrum
        Computed Bispectrum object. use to avoid recomputing.

    '''

    def __init__(self, data1, data2, nsamples=100, fiducial_model=None):
        super(BiSpectrum_Distance, self).__init__()

        if fiducial_model is not None:
            self.bispec1 = fiducial_model
        else:
            self.bispec1 = BiSpectrum(data1[0])
            self.bispec1.run()

        self.bispec2 = BiSpectrum(data2[0])
        self.bispec2.run()

        self.distance = None

    def distance_metric(self, verbose=False):

        self.distance = np.linalg.norm(self.bispec1.bicoherence.ravel() -
                                       self.bispec2.bicoherence.ravel())

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(1, 3, 1)
            p.title("Bicoherence 1")
            p.imshow(
                self.bispec1.bicoherence, origin="lower",
                interpolation="nearest")
            p.colorbar()
            p.xlabel("k1")
            p.ylabel("k2")

            p.subplot(1, 3, 2)
            p.title("Bicoherence 2")
            p.imshow(
                self.bispec2.bicoherence, origin="lower",
                interpolation="nearest")
            p.colorbar()
            p.xlabel("k1")
            p.ylabel("k2")

            p.subplot(1, 3, 3)
            p.title("Difference")
            p.imshow(np.abs(self.bispec1.bicoherence - self.bispec2.bicoherence),
                     origin="lower", interpolation="nearest",
                     vmax=1.0, vmin=0.0)
            p.colorbar()
            p.xlabel("k1")
            p.ylabel("k2")

            p.show()

        return self


def clip_func(arr, low, high):
    return np.logical_and(arr > low, arr < high)
