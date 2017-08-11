# Licensed under an MIT open source license - see LICENSE


import numpy as np
import numpy.random as ra
from numpy.fft import fftshift
import astropy.units as u

from ..rfft_to_fft import rfft_to_fft
from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, input_data
from ..fitting_utils import check_fit_limits


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
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, distance=None):
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

        if distance is not None:
            self.distance = distance

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        fft = fftshift(rfft_to_fft(self.weighted_data))

        self._ps2D = np.power(fft, 2.)

    def run(self, verbose=False, logspacing=False,
            return_stddev=True, low_cut=None, high_cut=None,
            fit_2D=True, fit_2D_kwargs={},
            xunit=u.pix**-1, save_name=None,
            use_wavenumber=False, **fit_kwargs):
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
        low_cut : `~astropy.units.Quantity`, optional
            Low frequency cut off in frequencies used in the fitting.
        high_cut : `~astropy.units.Quantity`, optional
            High frequency cut off in frequencies used in the fitting.
        fit_2D : bool, optional
            Fit an elliptical power-law model to the 2D power spectrum.
        fit_2D_kwargs : dict, optional
            Keyword arguments for `PowerSpectrum.fit_2Dpspec`. Use the
            `low_cut` and `high_cut` keywords to provide fit limits.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis to in the plot.
        save_name : str,optional
            Save the figure when a file name is given.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        fit_kwargs : Passed to `~PowerSpectrum.fit_pspec`.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)

        self.fit_pspec(low_cut=low_cut, high_cut=high_cut, **fit_kwargs)

        if fit_2D:
            self.fit_2Dpspec(low_cut=low_cut, high_cut=high_cut,
                             **fit_2D_kwargs)

        if verbose:
            print(self.fit.summary())
            self.plot_fit(show=True, show_2D=True,
                          xunit=xunit, save_name=save_name,
                          use_wavenumber=use_wavenumber)
            if save_name is not None:
                import matplotlib.pyplot as plt
                plt.close()

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
    low_cut : float or np.ndarray, optional
        The lower frequency fitting limit. An array with 2 elements can be
        passed to give separate lower limits for the datasets.
    high_cut : float or np.ndarray, optional
        The upper frequency fitting limit. See `low_cut` above. Defaults to
        0.5.
    logspacing : bool, optional
        Enable to use logarithmically-spaced bins.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data1, data2, weights1=None, weights2=None,
                 breaks=None, fiducial_model=None, low_cut=None,
                 high_cut=0.5 / u.pix, logspacing=False, phys_distance=None):
        super(PSpec_Distance, self).__init__()

        low_cut, high_cut = check_fit_limits(low_cut, high_cut)

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1, weights=weights1,
                                        distance=phys_distance)
            self.pspec1.run(low_cut=low_cut[0], high_cut=high_cut[0],
                            logspacing=logspacing, fit_2D=False)
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2, weights=weights2,
                                    distance=phys_distance)
        self.pspec2.run(low_cut=low_cut[1], high_cut=high_cut[1],
                        logspacing=logspacing, fit_2D=False)

        self.results = None
        self.distance = None

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        xunit=u.pix**-1, save_name=None,
                        use_wavenumber=False):
        '''

        Implements the distance metric for 2 Power Spectrum transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis to in the plot.
        save_name : str,optional
            Save the figure when a file name is given.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        '''

        self.distance = \
            np.abs((self.pspec1.slope - self.pspec2.slope) /
                   np.sqrt(self.pspec1.slope_err**2 +
                           self.pspec2.slope_err**2))

        if verbose:
            print(self.pspec1.fit.summary())
            print(self.pspec2.fit.summary())

            import matplotlib.pyplot as p

            self.pspec1.plot_fit(show=False, color='b',
                                 label=label1, symbol='D',
                                 xunit=xunit,
                                 use_wavenumber=use_wavenumber)
            self.pspec2.plot_fit(show=False, color='g',
                                 label=label2, symbol='o',
                                 xunit=xunit,
                                 use_wavenumber=use_wavenumber)
            p.legend(loc='best')

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
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


    Example
    -------
    >>> from turbustat.statistics import BiSpectrum
    >>> from astropy.io import fits
    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits") # doctest: +SKIP
    >>> bispec = BiSpectrum(moment0) # doctest: +SKIP
    >>> bispec.run(verbose=True, nsamples=1000) # doctest: +SKIP

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
                           mean_subtract=False):
        '''
        Do the computation.

        Parameters
        ----------
        nsamples : int, optional
            Sets the number of samples to take at each vector
            magnitude.
        seed : int, optional
            Sets the seed for the distribution draws.
        mean_subtract : bool, optional
            Subtract the mean from the data before computing. This removes the
            "zero frequency" (i.e., constant) portion of the power, resulting
            in a loss of phase coherence along the k_1=k_2 line.
        '''

        if mean_subtract:
            norm_data = self.data - self.data.mean()
        else:
            norm_data = self.data

        fftarr = np.fft.fft2(norm_data)
        conjfft = np.conj(fftarr)
        ra.seed(seed)

        bispec_shape = (int(self.shape[0] / 2.), int(self.shape[1] / 2.))

        self._bispectrum = np.zeros(bispec_shape, dtype=np.complex)
        self._bicoherence = np.zeros(bispec_shape, dtype=np.float)
        self._tracker = np.zeros(self.shape, dtype=np.int16)

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

                self._bispectrum[k1mag, k2mag] = np.sum(samps)

                biconorm[k1mag, k2mag] = np.sum(np.abs(samps))

                # Track where we're sampling from in fourier space
                self._tracker[k1x, k1y] += 1
                self._tracker[k2x, k2y] += 1
                self._tracker[k3x, k3y] += 1

        self._bicoherence = (np.abs(self.bispectrum) / biconorm)
        self._bispectrum_amp = np.log10(np.abs(self.bispectrum))

    @property
    def bispectrum(self):
        '''
        Bispectrum array.
        '''
        return self._bispectrum

    @property
    def bispectrum_amp(self):
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

    def run(self, nsamples=100, seed=1000, mean_subtract=False, verbose=False,
            save_name=None):
        '''
        Compute the bispectrum. Necessary to maintain package standards.

        Parameters
        ----------
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
        '''

        self.compute_bispectrum(nsamples=nsamples, mean_subtract=mean_subtract,
                                seed=seed)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(1, 2, 1)
            p.title("Bispectrum")
            p.imshow(
                self.bispectrum_amp, origin="lower", interpolation="nearest")
            p.colorbar()
            p.contour(self.bispectrum_amp, colors="k")
            p.xlabel(r"$k_1$")
            p.ylabel(r"$k_2$")

            p.subplot(1, 2, 2)
            p.title("Bicoherence")
            p.imshow(self.bicoherence, origin="lower", interpolation="nearest")
            p.colorbar()
            p.xlabel(r"$k_1$")
            p.ylabel(r"$k_2$")

            p.tight_layout()

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
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
                        label2=None, save_name=None):
        '''
        verbose : bool, optional
            Enable plotting.
        label1 : str, optional
            Object or region name for data1
        label2 : str, optional
            Object or region name for data2
        save_name : str,optional
            Save the figure when a file name is given.
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
            p.ion()

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
            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()

        return self
