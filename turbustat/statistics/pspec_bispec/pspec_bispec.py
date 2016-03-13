# Licensed under an MIT open source license - see LICENSE


import numpy as np
import numpy.random as ra
from ..psds import pspec
from ..rfft_to_fft import rfft_to_fft
import statsmodels.formula.api as sm
from pandas import Series, DataFrame
from numpy.fft import fftshift


class PowerSpectrum(object):

    """
    Compute the power spectrum of a given image. (e.g., Burkhart et al., 2010)

    Parameters
    ----------
    img : numpy.ndarray
        2D image.
    header : FITS header
        The image header. Needed for the pixel scale.
    weights : numpy.ndarray
        Weights to be applied to the image.
    ang_units : bool, optional
        Convert frequencies into angular units using the given header.
    """

    def __init__(self, img, header, weights=None, ang_units=False):
        super(PowerSpectrum, self).__init__()
        self.img = img
        # Get rid of nans
        self.img[np.isnan(self.img)] = 0.0

        self.header = header
        self.degperpix = np.abs(header["CDELT2"])
        self.ang_units = ang_units

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
            Warning("ps1D_stddev is only calculated when return_stddev"
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

    def compute_radial_pspec(self, return_stddev=True, logspacing=True,
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

        if self.ang_units:
            self._freqs *= self.degperpix ** -1

    def run(self, verbose=False, return_stddev=True,
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

            if self.ang_units:
                ax.set_xlabel(r"log k/deg$^{-1}$")
            else:
                ax.set_xlabel(r"log k/pixel$^{-1}$")

            p.ylabel("Power")

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
        List containing the integrated intensity image and its header.
    data2 : dict
        List containing the integrated intensity image and its header.
    weights1 : numpy.ndarray, optional
        Weights to apply to data1
    weights2 : numpy.ndarray, optional
        Weights to apply to data2
    fiducial_model : PowerSpectrum
        Computed PowerSpectrum object. use to avoid recomputing.
    ang_units : bool, optional
        Convert the frequencies to angular units using the header.
    """

    def __init__(self, data1, data2, weights1=None, weights2=None,
                 fiducial_model=None, ang_units=False):
        super(PSpec_Distance, self).__init__()

        self.ang_units = ang_units

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1[0],
                                        data1[1],
                                        weights=weights1, ang_units=ang_units)
            self.pspec1.run()
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2[0],
                                    data2[1],
                                    weights=weights2, ang_units=ang_units)
        self.pspec2.run()

        self.results = None
        self.distance = None

    def distance_metric(self, low_cut=None, high_cut=0.5, verbose=False,
                        label1=None, label2=None):
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
        '''

        if low_cut is None:
            # Default to a frequency of 1/2 the smallest axis in the images.
            low_cut = 2. / float(min(min(self.pspec1.ps2D.shape),
                                     min(self.pspec2.ps2D.shape)))

        keep_freqs1 = clip_func(self.pspec1.freqs, low_cut, high_cut)
        clip_freq1 = \
            self.pspec1.freqs[keep_freqs1]
        clip_ps1D1 = \
            self.pspec1.ps1D[keep_freqs1]
        clip_errors1 = \
            (0.434*self.pspec1.ps1D_stddev[keep_freqs1]/clip_ps1D1)

        keep_freqs2 = clip_func(self.pspec2.freqs, low_cut, high_cut)
        clip_freq2 = \
            self.pspec2.freqs[keep_freqs2]
        clip_ps1D2 = \
            self.pspec2.ps1D[keep_freqs2]
        clip_errors2 = \
            (0.434*self.pspec2.ps1D_stddev[keep_freqs2]/clip_ps1D2)

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

            fit_index = self.results.fittedvalues.index
            one_index = fit_index < len(clip_freq1)
            two_index = fit_index >= len(clip_freq1)

            import matplotlib.pyplot as p
            p.plot(df["scales"][fit_index[one_index]],
                   self.results.fittedvalues[one_index], "b",
                   df["scales"][fit_index[two_index]],
                   self.results.fittedvalues[two_index], "g")
            p.errorbar(np.log10(clip_freq1), np.log10(clip_ps1D1),
                       yerr=clip_errors1, color="b", fmt="D", markersize=5,
                       alpha=0.5, label=label1)
            p.errorbar(np.log10(clip_freq2), np.log10(clip_ps1D2),
                       yerr=clip_errors2, color="g", fmt="o", markersize=5,
                       alpha=0.5, label=label2)
            p.grid(True)
            p.ylabel("Power")

            if self.ang_units:
                p.xlabel(r"log k/deg$^{-1}$")
            else:
                p.xlabel(r"log k/pixel$^{-1}$")

            p.ylabel("Power")
            p.legend(loc='best')
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

        fftarr = np.fft.fft2(self.img)
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

                k3x = np.asarray([int(k1mag * np.cos(angle) +
                                      k2mag * np.cos(angle))
                                  for ang1, ang2 in zip(phi1, phi2)])
                k3y = np.asarray([int(k1mag * np.sin(angle) +
                                      k2mag * np.sin(angle))
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

    def distance_metric(self, verbose=False, label1=None, label2=None):
        '''
        verbose : bool, optional
            Enable plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        '''

        self.distance = np.linalg.norm(self.bispec1.bicoherence.ravel() -
                                       self.bispec2.bicoherence.ravel())

        if verbose:
            import matplotlib.pyplot as p

            ax1 = p.subplot(221)
            ax1.set_title(label1)
            ax1.imshow(
                self.bispec1.bicoherence, origin="lower",
                interpolation="nearest", vmax=1.0, vmin=0.0)
            ax1.set_xlabel(r"$k_1$")
            ax1.set_ylabel(r"$k_2$")

            ax2 = p.subplot(223)
            ax2.set_title(label2)
            ax2.imshow(
                self.bispec2.bicoherence, origin="lower",
                interpolation="nearest", vmax=1.0, vmin=0.0)
            ax2.set_xlabel(r"$k_1$")
            ax2.set_ylabel(r"$k_2$")

            ax3 = p.subplot(122)
            ax3.set_title("Difference")
            p.imshow(np.abs(self.bispec1.bicoherence -
                            self.bispec2.bicoherence),
                     origin="lower", interpolation="nearest",
                     vmax=1.0, vmin=0.0)
            cbar = p.colorbar(ax=ax3)
            ax3.set_xlabel(r"$k_1$")
            ax3.set_ylabel(r"$k_2$")

            p.tight_layout()

            p.show()

        return self


def clip_func(arr, low, high):
    return np.logical_and(arr > low, arr < high)
