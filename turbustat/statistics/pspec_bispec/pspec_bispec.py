# Licensed under an MIT open source license - see LICENSE


import numpy as np
import numpy.random as ra
from ..psds import pspec
import statsmodels.formula.api as sm
from pandas import Series, DataFrame

try:
    from scipy.fftpack import fft2, fftshift
except ImportError:
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

        self.ps2D = None
        self.ps1D = None
        self.freq = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        fft = fftshift(fft2(self.weighted_img.astype("f8")))

        self.ps2D = np.abs(fft) ** 2.

        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False,
                             return_stddev=False, azbins=1,
                             binsize=1.0, view=False, **kwargs):
        '''

        Computes the radially averaged power spectrum.
        This uses Adam Ginsburg's code (see https://github.com/keflavich/agpy).
        See the above url for parameter explanations.
        '''

        self.freq, self.ps1D = pspec(self.ps2D, return_index=return_index,
                                     wavenumber=wavenumber,
                                     return_stddev=return_stddev,
                                     azbins=azbins, binsize=binsize,
                                     view=view, **kwargs)

        return self

    def run(self, phys_units=False, verbose=False):
        '''
        Full computation of the Spatial Power Spectrum.

        Parameters
        ----------
        phys_units : bool, optional
            Sets frequency scale to physical units.
        verbose: bool, optional
            Enables plotting.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=True)

        if phys_units:
            self.freq *= self.degperpix ** -1

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.imshow(
                np.log10(self.ps2D), origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.loglog(self.freq, self.ps1D, "bD")
            p.xlabel("log K")
            p.ylabel("Power (K)")
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

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 Power Spectrum transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        verbose : bool, optional

        '''

        # Clipping from 8 pixels to half the box size
        # Noise effects dominate outside this region
        clip_mask1 = np.zeros((self.pspec1.freq.shape))
        for i, x in enumerate(self.pspec1.freq):
            if x > 8.0 and x < self.shape1[0] / 2.:
                clip_mask1[i] = 1
        clip_freq1 = self.pspec1.freq[np.where(clip_mask1 == 1)]
        clip_ps1D1 = self.pspec1.ps1D[np.where(clip_mask1 == 1)]

        clip_mask2 = np.zeros((self.pspec2.freq.shape))
        for i, x in enumerate(self.pspec2.freq):
            if x > 8.0 and x < self.shape2[0] / 2.:
                clip_mask2[i] = 1
        clip_freq2 = self.pspec2.freq[np.where(clip_mask2 == 1)]
        clip_ps1D2 = self.pspec2.ps1D[np.where(clip_mask2 == 1)]

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

        fft = np.fft.fft2(self.img.astype("f8"))
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
