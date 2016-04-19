# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from numpy.fft import fftfreq
from astropy import units as u

from ..lm_seg import Lm_Seg
from ..rfft_to_fft import rfft_to_fft
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types


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
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, vel_units=False):
        super(VCS, self).__init__()

        self.input_data_header(cube, header)

        self.vel_units = vel_units

        if np.isnan(self.data).any():
            self.data[np.isnan(self.data)] = 0
            # Feel like this should be more specific
            self.good_pixel_count = np.sum(self.data.max(axis=0) != 0)
        else:
            self.good_pixel_count = float(
                self.data.shape[1] * self.data.shape[2])

        if vel_units:
            try:
                spec_unit = u.Unit(self.header["CUNIT3"])
                self.vel_to_pix = (np.abs(self.header["CDELT3"]) *
                                   spec_unit).to(u.km/u.s).value
            except (KeyError, u.UnitsError) as e:
                print("Spectral unit not in the header or it cannot be parsed "
                      "by astropy.units. Using pixel units.")
                print(e)
                self.vel_to_pix = 1.0
        else:
            self.vel_to_pix = 1.0

        self.vel_channels = np.arange(1, self.data.shape[0], 1)

        self.vel_freqs = \
            np.abs(fftfreq(self.data.shape[0])) / self.vel_to_pix

    def compute_pspec(self):
        '''
        Take the FFT of each spectrum in velocity dimension.
        '''

        ps3D = np.power(rfft_to_fft(self.data), 2.)
        self.ps1D = np.nansum(
            np.nansum(ps3D, axis=2), axis=1) /\
            self.good_pixel_count

    def fit_pspec(self, breaks=None, log_break=True, lg_scale_cut=2,
                  verbose=False):
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
        lg_scale_cut : int, optional
            Cuts off largest scales, which deviate from the powerlaw.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

        if breaks is None:
            from scipy.interpolate import UnivariateSpline

            # Need to order the points
            shape = self.vel_freqs.size
            spline_y = np.log10(self.ps1D[1:shape/2])
            spline_x = np.log10(self.vel_freqs[1:shape/2])

            spline = UnivariateSpline(spline_x, spline_y, k=1, s=1)

            # The first and last are just the min and max of x
            breaks = spline.get_knots()[1:-1]

            if verbose:
                print "Breaks found from spline are: " + str(breaks)

            # Take the number according to max_breaks starting at the
            # largest x.
            breaks = breaks[::-1]

            x = np.log10(self.vel_freqs[lg_scale_cut+1:-lg_scale_cut])
            y = np.log10(self.ps1D[lg_scale_cut+1:-lg_scale_cut])

            # Now try these breaks until a good fit including the break is
            # found. If none are found, it accept that there wasn't a good
            # break and continues on.
            i = 0
            while True:
                self.fit = \
                    Lm_Seg(x, y, breaks[i])
                self.fit.fit_model(verbose=verbose)

                if self.fit.params.size == 5:
                    break
                i += 1
                if i >= breaks.shape:
                    warnings.warn("No good break point found. Returned fit\
                                   does not include a break!")
                    break

            return self

        if not log_break:
            breaks = np.log10(breaks)

        self.fit = \
            Lm_Seg(np.log10(self.vel_freqs[lg_scale_cut+1:-lg_scale_cut]),
                   np.log10(self.ps1D[lg_scale_cut+1:-lg_scale_cut]), breaks)
        self.fit.fit_model(verbose=verbose)

    @property
    def slopes(self):
        return self.fit.slopes

    @property
    def slope_errs(self):
        return self.fit.slope_errs

    @property
    def brk(self):
        return self.fit.brk

    @property
    def brk_err(self):
        return self.fit.brk_err

    def run(self, verbose=False, breaks=None):
        '''
        Run the entire computation.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        breaks : float, optional
            Specify where the break point is. If None, attempts to find using
            spline.
        '''
        self.compute_pspec()
        self.fit_pspec(verbose=verbose, breaks=breaks)

        if verbose:
            import matplotlib.pyplot as p

            if self.vel_units:
                xlab = r"log $\left( k_v / (\mathrm{km}/\mathrm{s})^{-1} \right)$"
            else:
                xlab = r"log $\left( k_v / \mathrm{pixel}^{-1} \right)$"

            p.loglog(self.vel_freqs, self.ps1D, "bD", label='Data')
            p.loglog(10**self.fit.x, 10**self.fit.model(self.fit.x), 'r',
                     label='Fit', linewidth=2)
            p.xlabel(xlab)
            p.ylabel(r"log P$_{1}$(k$_{v}$)")
            p.grid(True)
            p.legend(loc='best')
            p.show()

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
    vel_units : bool, optional
        Convert frequencies to the spectral unit in the headers.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, breaks=None, fiducial_model=None,
                 vel_units=False):
        super(VCS_Distance, self).__init__()

        self.vel_units = vel_units

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is not None:
            self.vcs1 = fiducial_model
        else:
            self.vcs1 = VCS(cube1,
                            vel_units=vel_units).run(breaks=breaks[0])

        self.vcs2 = VCS(cube2,
                        vel_units=vel_units).run(breaks=breaks[1])

    def distance_metric(self, verbose=False, label1=None, label2=None):
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
        '''

        # Now construct the t-statistics for each portion

        # There should always be the velocity distance
        self.large_scale_distance = \
            np.abs((self.vcs1.slopes[0] - self.vcs2.slopes[0]) /
                   np.sqrt(self.vcs1.slope_errs[0]**2 +
                           self.vcs2.slope_errs[0]**2))

        # A density distance is only found if a break was found
        if self.vcs1.slopes.size == 1 or self.vcs2.slopes.size == 1:
            self.small_scale_distance = np.NaN
            self.break_distance = np.NaN
        else:
            self.small_scale_distance = \
                np.abs((self.vcs1.slopes[1] - self.vcs2.slopes[1]) /
                       np.sqrt(self.vcs1.slope_errs[1]**2 +
                               self.vcs2.slope_errs[1]**2))

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

            if self.vel_units:
                xlab = r"log $\left( k_v / (\mathrm{km}/\mathrm{s})^{-1} \right)$"
            else:
                xlab = r"log $\left( k_v / \mathrm{pixel}^{-1} \right)$"

            import matplotlib.pyplot as p
            p.plot(self.vcs1.fit.x, self.vcs1.fit.y, 'bD', alpha=0.5,
                   label=label1)
            p.plot(self.vcs1.fit.x, self.vcs1.fit.model(self.vcs1.fit.x), 'b')
            p.plot(self.vcs2.fit.x, self.vcs2.fit.y, 'go', alpha=0.5,
                   label=label2)
            p.plot(self.vcs2.fit.x, self.vcs2.fit.model(self.vcs2.fit.x), 'g')
            p.grid(True)
            p.legend()
            p.xlabel(xlab)
            p.ylabel(r"$P_{1}(k_v)$")
            p.show()

        return self
