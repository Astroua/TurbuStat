# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from numpy.fft import fftfreq
from astropy import units as u

from ..lm_seg import Lm_Seg
from ..rfft_to_fft import rfft_to_fft
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types
from ...io.input_base import to_spectral_cube
from ..fitting_utils import clip_func
from .slice_thickness import spectral_regrid_cube


class VCS(BaseStatisticMixIn):

    '''
    The VCS technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header : FITS header, optional
        Corresponding FITS header.
    vel_units : bool, optional
        Convert frequencies to the spectral unit in the header.
    channel_width : `~astropy.units.Quantity`, optional
        Set the width of channels to compute the VCA with. The channel width
        in the data is used by default. Given widths will be used to
        spectrally down-sample the data before calculating the VCA. Up-sampling
        to smaller channel sizes than the original is not supported.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, channel_width=None):
        super(VCS, self).__init__()

        self.input_data_header(cube, header)

        if channel_width is not None:
            sc_cube = to_spectral_cube(self.data, self.header)

            reg_cube = spectral_regrid_cube(sc_cube, channel_width)

            # Don't pass the header. It will read the new one in reg_cube
            self.input_data_header(reg_cube, None)

        self._has_nan_flag = False
        if np.isnan(self.data).any():
            self.data[np.isnan(self.data)] = 0
            self._has_nan_flag = True

        self.vel_channels = np.arange(1, self.data.shape[0], 1)

        self.freqs = \
            np.abs(fftfreq(self.data.shape[0])) / u.pix

    def compute_pspec(self):
        '''
        Take the FFT of each spectrum in the velocity dimension and average.
        '''

        if self._has_nan_flag:
            # Is this the best way to be averaging the data?
            good_pixel_count = np.sum(self.data.max(axis=0) != 0)
        else:
            good_pixel_count = \
                float(self.data.shape[1] * self.data.shape[2])

        ps3D = np.power(rfft_to_fft(self.data), 2.)
        self._ps1D = np.nansum(ps3D, axis=(1, 2)) /\
            good_pixel_count

    @property
    def ps1D(self):
        '''
        The 1D VCS spectrum.
        '''
        return self._ps1D

    def fit_pspec(self, breaks=None, log_break=True, low_cut=None,
                  high_cut=None, fit_verbose=False):
        '''
        Fit the 1D Power spectrum using a segmented linear model. Note that
        the current implementation allows for only 1 break point in the
        model. If the break point is estimated via a spline, the breaks are
        tested, starting from the largest, until the model finds a good fit.

        Parameters
        ----------
        breaks : float or None, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error. If None, a spline is used to estimate the breaks.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        low_cut : `~astropy.units.Quantity`, optional
            Lowest frequency to consider in the fit.
        high_cut : `~astropy.units.Quantity`, optional
            Highest frequency to consider in the fit.
        fit_verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

        # Make the data to fit to
        if low_cut is None:
            # Default to the largest frequency, since this is just 1 pixel
            # But cut out fitting the total power point.
            self.low_cut = 0.98 / (float(self.data.shape[0]) * u.pix)
        else:
            self.low_cut = \
                self._spectral_freq_unit_conversion(low_cut, u.pix**-1)

        if high_cut is None:
            # Set to something larger than
            self.high_cut = (self.freqs.max().value * 1.01) / u.pix
        else:
            self.high_cut = \
                self._spectral_freq_unit_conversion(high_cut, u.pix**-1)

        # We keep both the real and imag frequencies, but we only need the real
        # component when fitting. We'll cut off the total power frequency too
        # in case it causes an extra break point (which seems to happen).
        shape = self.freqs.size
        rfreqs = self.freqs[1:shape // 2].value
        ps1D = self.ps1D[1:shape // 2]

        y = np.log10(ps1D[clip_func(rfreqs, self.low_cut.value,
                                    self.high_cut.value)])
        x = np.log10(rfreqs[clip_func(rfreqs, self.low_cut.value,
                                      self.high_cut.value)])

        if breaks is None:
            from scipy.interpolate import UnivariateSpline

            # Need to order the points
            spline = UnivariateSpline(x, y, k=1, s=1)

            # The first and last are just the min and max of x
            breaks = spline.get_knots()[1:-1]

            if fit_verbose:
                print("Breaks found from spline are: " + str(breaks))

            # Take the number according to max_breaks starting at the
            # largest x.
            breaks = breaks[::-1]

            # Ensure a break doesn't fall at the max or min.
            if breaks.size > 0:
                if breaks[0] == x.max():
                    breaks = breaks[1:]
            if breaks.size > 0:
                if breaks[-1] == x.min():
                    breaks = breaks[:-1]

            if x.size <= 3 or y.size <= 3:
                raise Warning("There are no points to fit to. Try lowering "
                              "'lg_scale_cut'.")

            # If no breaks, set to half-way before last point
            if breaks.size == 0:
                x_copy = np.sort(x.copy())
                breaks = np.array([0.5 * (x_copy[-1] + x_copy[-3])])

            # Now try these breaks until a good fit including the break is
            # found. If none are found, it accepts that there wasn't a good
            # break and continues on.
            i = 0
            while True:
                self.fit = \
                    Lm_Seg(x, y, breaks[i])
                self.fit.fit_model(verbose=fit_verbose)

                if self.fit.params.size == 5:
                    # Success!
                    break
                i += 1
                if i >= breaks.size:
                    warnings.warn("No good break point found. Returned fit\
                                   does not include a break!")
                    break

            return self

        if not log_break:
            breaks = np.log10(breaks)

        # Fit the final model with whichever breaks were passed.
        self.fit = Lm_Seg(x, y, breaks)
        self.fit.fit_model(verbose=fit_verbose)

    @property
    def slope(self):
        '''
        Power spectrum slope(s).
        '''
        return self.fit.slopes

    @property
    def slope_err(self):
        '''
        1-sigma error on the power spectrum slope(s).
        '''
        return self.fit.slope_errs

    @property
    def brk(self):
        '''
        Fitted break point.
        '''
        return self.fit.brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point.
        '''
        return self.fit.brk_err

    def run(self, verbose=False, save_name=None, xunit=u.pix**-1,
            **fit_kwargs):
        '''
        Run the entire computation.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis in the plot to.
        fit_kwargs : Passed to `~VCS.fit_pspec`.

        '''
        self.compute_pspec()
        self.fit_pspec(**fit_kwargs)

        if verbose:
            # Print the final fitted model when fit_verbose is not enabled.
            if not fit_kwargs.get("fit_verbose"):
                print(self.fit.fit.summary())

            import matplotlib.pyplot as plt

            xlab = r"log $( k_v / ($" + str(xunit**-1) + \
                "$)^{-1})$"

            good_interval = clip_func(self.freqs.value, self.low_cut.value,
                                      self.high_cut.value)

            freq = self._spectral_freq_unit_conversion(self.freqs, xunit)

            y_fit = \
                10**self.fit.model(np.log10(self.freqs.value[good_interval]))

            plt.loglog(freq, self.ps1D, "rD", label='Data', alpha=0.5)
            plt.loglog(freq[good_interval], y_fit, 'r',
                       label='Fit', linewidth=2)
            plt.xlabel(xlab)
            plt.ylabel(r"log P$_{1}$(k$_{v}$)")
            plt.axvline(self._spectral_freq_unit_conversion(self.low_cut,
                                                            xunit).value,
                        linestyle="--", color='r', alpha=0.5)
            plt.axvline(self._spectral_freq_unit_conversion(self.high_cut,
                                                            xunit).value,
                        linestyle="--", color='r', alpha=0.5)
            plt.grid(True)
            plt.legend(loc='best', frameon=True)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self


class VCS_Distance(object):

    '''
    Calculate the distance between two cubes using VCS. The 1D power spectrum
    is modeled by a broked linear model to account for the density and
    velocity dominated scales. The distance is the sum of  the t-statistics
    for each model.

    Parameters
    ----------
    cube1 : %(dtypes)s
        Data cube.
    cube2 : %(dtypes)s
        Data cube.
    slice_size : float, optional
        Slice to degrade the cube to.
    breaks : float, list or array, optional
        Specify where the break point is. If None, attempts to find using
        spline.
    fiducial_model : VCS
        Computed VCS object. use to avoid recomputing.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, breaks=None, fiducial_model=None,
                 channel_width=None, **fit_kwargs):
        super(VCS_Distance, self).__init__()

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is not None:
            self.vcs1 = fiducial_model
        else:
            self.vcs1 = VCS(cube1,
                            channel_width=channel_width).run(breaks=breaks[0],
                                                             **fit_kwargs)

        self.vcs2 = VCS(cube2,
                        channel_width=channel_width).run(breaks=breaks[1],
                                                         **fit_kwargs)

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        save_name=None, xunit=u.pix**-1):
        '''

        Implements the distance metric for 2 VCS transforms.
        This distance is the t-statistic of the difference
        in the slopes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Now construct the t-statistics for each portion

        # There should always be the velocity distance
        self.large_scale_distance = \
            np.abs((self.vcs1.slope[0] - self.vcs2.slope[0]) /
                   np.sqrt(self.vcs1.slope_err[0]**2 +
                           self.vcs2.slope_err[0]**2))

        # A density distance is only found if a break was found
        if self.vcs1.slope.size == 1 or self.vcs2.slope.size == 1:
            self.small_scale_distance = np.NaN
            self.break_distance = np.NaN
        else:
            self.small_scale_distance = \
                np.abs((self.vcs1.slope[1] - self.vcs2.slope[1]) /
                       np.sqrt(self.vcs1.slope_err[1]**2 +
                               self.vcs2.slope_err[1]**2))

            self.break_distance = \
                np.abs((self.vcs1.brk - self.vcs2.brk) /
                       np.sqrt(self.vcs1.brk_err**2 +
                               self.vcs2.brk_err**2))

        # The overall distance is the sum from the two models
        self.distance = \
            np.nansum([self.large_scale_distance, self.small_scale_distance])

        if verbose:

            print("Fit 1")
            print(self.vcs1.fit.fit.summary())
            print("Fit 2")
            print(self.vcs2.fit.fit.summary())

            xlab = r"log $\left( k_v / ({})^{-1} \right)$".format(1 / xunit)

            import matplotlib.pyplot as plt

            plt.plot(self.vcs1.fit.x, self.vcs1.fit.y, 'bD', alpha=0.5,
                     label=label1)
            plt.plot(self.vcs1.fit.x, self.vcs1.fit.model(self.vcs1.fit.x), 'b')
            plt.plot(self.vcs2.fit.x, self.vcs2.fit.y, 'go', alpha=0.5,
                     label=label2)
            plt.plot(self.vcs2.fit.x, self.vcs2.fit.model(self.vcs2.fit.x), 'g')
            plt.grid(True)
            plt.legend()
            plt.xlabel(xlab)
            plt.ylabel(r"$P_{1}(k_v)$")

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
