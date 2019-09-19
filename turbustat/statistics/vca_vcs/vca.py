# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from numpy.fft import fftshift
import astropy.units as u

from ..rfft_to_fft import rfft_to_fft
from .slice_thickness import spectral_regrid_cube
from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types
from ...io.input_base import to_spectral_cube
from ..fitting_utils import check_fit_limits


class VCA(BaseStatisticMixIn, StatisticBase_PSpec2D):

    '''
    The VCA technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header : FITS header, optional
        Corresponding FITS header.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    beam : `radio_beam.Beam`, optional
        Beam object for correcting for the effect of a finite beam.
    channel_width : `~astropy.units.Quantity` or int, optional
        Set the width of channels to compute the VCA with. The channel width
        in the data is used by default. Given widths will be used to
        spectrally down-sample the data before calculating the VCA. Given an
        integer value, the spectral axis will be downsampled by the given
        number of spectral channels. Up-sampling to smaller channel sizes
        than the original is not supported.
    downsample_kwargs : dict, optional
        Passed to `~turbustat.statistics.vca_vca.slice_thickness.spectral_regrid_cube`.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, distance=None, beam=None,
                 channel_width=None, downsample_kwargs={}):
        super(VCA, self).__init__()

        self.input_data_header(cube, header)

        # Regrid the data when channel_width is given
        if channel_width is not None:
            sc_cube = to_spectral_cube(self.data, self.header)

            reg_cube = spectral_regrid_cube(sc_cube, channel_width,
                                            **downsample_kwargs)

            # Don't pass the header. It will read the new one in reg_cube
            self.input_data_header(reg_cube, None)

        isnan = np.isnan(self.data)
        if isnan.any():
            self.data = self.data.copy()
            self.data[isnan] = 0.0

        if distance is not None:
            self.distance = distance

        self._ps1D_stddev = None

        self.load_beam(beam=beam)

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
            data = self.data * apod_kernel
        else:
            data = self.data

        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')

        fft = fftshift(rfft_to_fft(data, use_pyfftw=use_pyfftw,
                                   threads=threads,
                                   **pyfftw_kwargs))

        if beam_correct:
            if not hasattr(self, '_beam'):
                raise AttributeError("Beam correction cannot be applied since"
                                     " no beam object was given.")

            beam_kern = self._beam.as_kernel(self._ang_size,
                                             y_size=self.data.shape[1],
                                             x_size=self.data.shape[2])

            beam_fft = fftshift(rfft_to_fft(beam_kern.array))

            self._beam_pow = np.abs(beam_fft**2)

        self._ps2D = np.power(fft, 2.).sum(axis=0)

        if beam_correct:
            self._ps2D /= self._beam_pow

    def run(self, verbose=False, beam_correct=False,
            apodize_kernel=None, alpha=0.2, beta=0.0,
            use_pyfftw=False, threads=1,
            pyfftw_kwargs={},
            radial_pspec_kwargs={},
            low_cut=None, high_cut=None,
            fit_2D=True, fit_kwargs={}, fit_2D_kwargs={},
            save_name=None, xunit=u.pix**-1, use_wavenumber=False):
        '''
        Full computation of VCA.

        Parameters
        ----------
        verbose : bool, optional
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
        radial_pspec_kwargs : dict, optional
            Passed to `~PowerSpectrum.compute_radial_pspec`.
        low_cut : `~astropy.units.Quantity`, optional
            Low frequency cut off in frequencies used in the fitting.
        high_cut : `~astropy.units.Quantity`, optional
            High frequency cut off in frequencies used in the fitting.
        fit_2D : bool, optional
            Fit an elliptical power-law model to the 2D power spectrum.
        fit_kwargs : dict, optional
            Passed to `~PowerSpectrum.fit_pspec`.
        fit_2D_kwargs : dict, optional
            Keyword arguments for `~VCA.fit_2Dpspec`. Use the
            `low_cut` and `high_cut` keywords to provide fit limits.
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis in the plot to.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        fit_kwargs : Passed to `~VCA.fit_pspec`.
        '''

        # Remove threads if in dict
        if pyfftw_kwargs.get('threads') is not None:
            pyfftw_kwargs.pop('threads')

        self.compute_pspec(apodize_kernel=apodize_kernel,
                           alpha=alpha, beta=beta,
                           beam_correct=beam_correct,
                           use_pyfftw=use_pyfftw, threads=threads,
                           **pyfftw_kwargs)

        self.compute_radial_pspec(**radial_pspec_kwargs)
        self.fit_pspec(low_cut=low_cut, high_cut=high_cut, **fit_kwargs)

        if fit_2D:
            self.fit_2Dpspec(low_cut=low_cut, high_cut=high_cut,
                             **fit_2D_kwargs)

        if verbose:

            print(self.fit.summary())
            if self._bootstrap_flag:
                print("Bootstrapping used to find stderrs! "
                      "Errors may not equal those shown above.")

            self.plot_fit(show_2D=True,
                          xunit=xunit, save_name=save_name,
                          use_wavenumber=use_wavenumber)
            if save_name is not None:
                import matplotlib.pyplot as p
                p.close()

        return self


class VCA_Distance(object):

    '''
    Calculate the distance between two cubes using VCA. The 1D power spectrum
    is modeled by a linear model. The distance is the t-statistic of the
    interaction between the two slopes.

    .. note:: When a pre-computed `VCA` class is given for `cube1` or `cube2`,
              if the channel widths are not equal to `channel_width` given to
              `VCA_Distance`, a ValueError will be raised.

    Parameters
    ----------
    cube1 : %(dtypes)s or `~VCA`
        Data cube. Or a `~VCA` class can be passed which may be pre-computed.
    cube2 : %(dtypes)s or `~VCA`
        See `cube1`.
    channel_width : `~astropy.units.Quantity`, optional
        Spectral resolution to degrade the cube to.
    breaks : `~astropy.units.Quantity`, list or array, optional
        Specify where the break point is with appropriate units.
        If none is given, no break point will be used in the fit.
    low_cut : `~astropy.units.Quantity` or np.ndarray, optional
        The lower frequency fitting limit. An array with 2 elements can be
        passed to give separate lower limits for the datasets.
    high_cut : `~astropy.units.Quantity` or np.ndarray, optional
        The upper frequency fitting limit. See `low_cut` above. Defaults to
        0.5.
    pspec_kwargs : dict, optional
        Passed to `radial_pspec_kwargs` in `~PowerSpectrum.run`.
    pspec2_kwargs : dict or None, optional
        Passed to `radial_pspec_kwargs` in `~PowerSpectrum.run` for `cube2`.
        When `None` is given, setting from `pspec_kwargs` are used for `cube2`.
    phys_distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, channel_width=None, breaks=None,
                 low_cut=None, high_cut=None,
                 radial_pspec_kwargs={}, radial_pspec_kwargs2=None):
        super(VCA_Distance, self).__init__()

        low_cut, high_cut = check_fit_limits(low_cut, high_cut)

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if radial_pspec_kwargs2 is None:
            radial_pspec_kwargs2 = radial_pspec_kwargs

        if isinstance(cube1, VCA):
            self.vca1 = cube1
            needs_run = False

            if not hasattr(self.vca1, '_slope'):
                warnings.warn("VCA class given as `cube1` does not have a"
                              " fitted slope. Re-running VCA.")
                needs_run = True
        else:
            self.vca1 = VCA(cube1, channel_width=channel_width)
            needs_run = True

        if needs_run:
            self.vca1.run(fit_kwargs={'brk': breaks[0]},
                          low_cut=low_cut[0],
                          high_cut=high_cut[0],
                          radial_pspec_kwargs=radial_pspec_kwargs,
                          fit_2D=False)

        if isinstance(cube2, VCA):
            self.vca2 = cube2
            needs_run = False

            if not hasattr(self.vca2, '_slope'):
                warnings.warn("VCA class given as `cube2` does not have a"
                              " fitted slope. Re-running VCA.")
                needs_run = True
        else:
            self.vca2 = VCA(cube2, channel_width=channel_width)
            needs_run = True

        if needs_run:
            self.vca2.run(fit_kwargs={'brk': breaks[1]},
                          low_cut=low_cut[1],
                          high_cut=high_cut[1],
                          radial_pspec_kwargs=radial_pspec_kwargs,
                          fit_2D=False)

    def distance_metric(self, verbose=False, xunit=u.pix**-1,
                        save_name=None, plot_kwargs1={},
                        plot_kwargs2={},
                        use_wavenumber=False):
        '''

        Implements the distance metric for 2 VCA transforms, each with the
        same channel width. We fit the linear portion of the transform to
        represent the powerlaw.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        xunit : `~astropy.units.Unit`, optional
            Unit of the x-axis in the plot in pixel, angular, or
            physical units.
        save_name : str, optional
            Name of the save file. Enables saving the figure.
        plot_kwargs1 : dict, optional
            Pass kwargs to `~turbustat.statistics.VCA.plot_fit`
            for `data1`.
        plot_kwargs2 : dict, optional
            Pass kwargs to `~turbustat.statistics.VCA.plot_fit`
            for `data2`.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.vca1.slope - self.vca2.slope) /
                   np.sqrt(self.vca1.slope_err**2 +
                           self.vca2.slope_err**2))

        if verbose:

            print(self.vca1.fit.summary())
            print(self.vca2.fit.summary())

            import matplotlib.pyplot as plt

            defaults1 = {'color': 'b', 'symbol': 'D', 'label': '1'}
            defaults2 = {'color': 'g', 'symbol': 'o', 'label': '2'}

            for key in defaults1:
                if key not in plot_kwargs1:
                    plot_kwargs1[key] = defaults1[key]

            for key in defaults2:
                if key not in plot_kwargs2:
                    plot_kwargs2[key] = defaults2[key]

            if 'xunit' in plot_kwargs1:
                del plot_kwargs1['xunit']
            if 'xunit' in plot_kwargs2:
                del plot_kwargs2['xunit']

            self.vca1.plot_fit(xunit=xunit,
                               use_wavenumber=use_wavenumber,
                               **plot_kwargs1)
            self.vca2.plot_fit(xunit=xunit,
                               use_wavenumber=use_wavenumber,
                               **plot_kwargs2)
            axes = plt.gcf().get_axes()
            axes[0].legend(loc='best', frameon=True)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()
        return self
