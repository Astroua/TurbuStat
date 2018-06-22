# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from numpy.fft import fftshift
import astropy.units as u


from ..rfft_to_fft import rfft_to_fft
from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types
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
    beam : `radio_beam.Beam`, optional
        Beam object for correcting for the effect of a finite beam.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, img, header=None, weights=None, distance=None,
                 beam=None):
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

        self.load_beam(beam=beam)

        if distance is not None:
            self.distance = distance

    def compute_pspec(self, beam_correct=False,
                      apodize_kernel=None, alpha=0.3, beta=0.0,
                      use_pyfftw=False, threads=1, **pyfftw_kwargs):
        '''
        Compute the 2D power spectrum.

        Parameters
        ----------
        beam_correct : bool, optional
            If a beam object was given, divide the 2D FFT by the beam response.
        apodize_kernel : None or 'splitcosinebell', 'hanning', 'tukey', 'cosinebell', 'tophat'
            If None, no apodization kernel is applied. Otherwise, the type of
            apodizing kernel is given.
        alpha : float, optional
            alpha shape parameter of the apodization kernel. See
            `~turbustat.apodizing_kernel` for more information.
        beta : float, optional
            beta shape parameter of the apodization kernel. See
            `~turbustat.apodizing_kernel` for more information.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfftw_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <http://hgomersall.github.io/pyFFTW/pyfftw/builders/builders.html>`__
            for a list of accepted kwargs.
        '''

        if apodize_kernel is not None:
            apod_kernel = self.apodizing_kernel(kernel_type=apodize_kernel,
                                                alpha=alpha,
                                                beta=beta)
            data = self.weighted_data * apod_kernel
        else:
            data = self.weighted_data

        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')

        fft = fftshift(rfft_to_fft(data, use_pyfftw=use_pyfftw,
                                   threads=threads,
                                   **pyfftw_kwargs))

        if beam_correct:
            if not hasattr(self, '_beam'):
                raise AttributeError("Beam correction cannot be applied since"
                                     " no beam object was given.")

            beam_kern = self._beam.as_kernel(self._wcs.wcs.cdelt[0] * u.deg,
                                             y_size=self.data.shape[0],
                                             x_size=self.data.shape[1])

            beam_fft = fftshift(rfft_to_fft(beam_kern.array))

            self._beam_pow = np.abs(beam_fft**2)

        self._ps2D = np.power(fft, 2.)

        if beam_correct:
            self._ps2D /= self._beam_pow

    def run(self, verbose=False, beam_correct=False,
            apodize_kernel=None, alpha=0.2, beta=0.0,
            use_pyfftw=False, threads=1,
            pyfftw_kwargs={}, return_stddev=True,
            low_cut=None, high_cut=None,
            fit_2D=True, radial_pspec_kwargs={}, fit_kwargs={},
            fit_2D_kwargs={},
            xunit=u.pix**-1, save_name=None,
            use_wavenumber=False):
        '''
        Full computation of the spatial power spectrum.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        beam_correct : bool, optional
            If a beam object was given, divide the 2D FFT by the beam response.
        apodize_kernel : None or 'splitcosinebell', 'hanning', 'tukey', 'cosinebell', 'tophat'
            If None, no apodization kernel is applied. Otherwise, the type of
            apodizing kernel is given.
        alpha : float, optional
            alpha shape parameter of the apodization kernel. See
            `~turbustat.apodizing_kernel` for more information.
        beta : float, optional
            beta shape parameter of the apodization kernel. See
            `~turbustat.apodizing_kernel` for more information.
        use_pyfftw : bool, optional
            Enable to use pyfftw, if it is installed.
        threads : int, optional
            Number of threads to use in FFT when using pyfftw.
        pyfft_kwargs : Passed to
            `~turbustat.statistics.rfft_to_fft.rfft_to_fft`. See
            `here <https://hgomersall.github.io/pyFFTW/pyfftw/interfaces/interfaces.html#interfaces-additional-args>`_
            for a list of accepted kwargs.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        low_cut : `~astropy.units.Quantity`, optional
            Low frequency cut off in frequencies used in the fitting.
        high_cut : `~astropy.units.Quantity`, optional
            High frequency cut off in frequencies used in the fitting.
        fit_2D : bool, optional
            Fit an elliptical power-law model to the 2D power spectrum.
        radial_pspec_kwargs : dict, optional
            Passed to `~PowerSpectrum.compute_radial_pspec`.
        fit_kwargs : dict, optional
            Passed to `~PowerSpectrum.fit_pspec`.
        fit_2D_kwargs : dict, optional
            Keyword arguments for `PowerSpectrum.fit_2Dpspec`. Use the
            `low_cut` and `high_cut` keywords to provide fit limits.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis to in the plot.
        save_name : str,optional
            Save the figure when a file name is given.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        '''

        # Remove threads if in dict
        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')

        self.compute_pspec(apodize_kernel=apodize_kernel,
                           alpha=alpha, beta=beta,
                           beam_correct=beam_correct,
                           use_pyfftw=use_pyfftw, threads=threads,
                           **pyfftw_kwargs)

        self.compute_radial_pspec(return_stddev=return_stddev,
                                  **radial_pspec_kwargs)

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
    breaks : `~astropy.units.Quantity`, list or array, optional
        Specify where the break point is with appropriate units.
        If none is given, no break point will be used in the fit.
    fiducial_model : PowerSpectrum
        Computed PowerSpectrum object. use to avoid recomputing.
    low_cut : `~astropy.units.Quantity` or np.ndarray, optional
        The lower frequency fitting limit. An array with 2 elements can be
        passed to give separate lower limits for the datasets.
    high_cut : `~astropy.units.Quantity` or np.ndarray, optional
        The upper frequency fitting limit. See `low_cut` above. Defaults to
        0.5.
    logspacing : bool, optional
        Enable to use logarithmically-spaced bins.
    phys_distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data1, data2, weights1=None, weights2=None,
                 breaks=None, fiducial_model=None, low_cut=None,
                 high_cut=0.5 / u.pix, logspacing=False, phys_distance=None):
        super(PSpec_Distance, self).__init__()

        low_cut, high_cut = check_fit_limits(low_cut, high_cut)

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1, weights=weights1,
                                        distance=phys_distance)
            self.pspec1.run(low_cut=low_cut[0], high_cut=high_cut[0],
                            radial_pspec_kwargs={"logspacing": logspacing},
                            fit_kwargs={'brk': breaks[0]},
                            fit_2D=False)
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2, weights=weights2,
                                    distance=phys_distance)
        self.pspec2.run(low_cut=low_cut[1], high_cut=high_cut[1],
                        fit_kwargs={'brk': breaks[1]},
                        radial_pspec_kwargs={"logspacing": logspacing},
                        fit_2D=False)

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
