# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from numpy.fft import fftshift
import astropy.units as u

from ..rfft_to_fft import rfft_to_fft
from slice_thickness import spectral_regrid_cube
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
    channel_width : `~astropy.units.Quantity`, optional
        Set the width of channels to compute the VCA with. The channel width
        in the data is used by default. Given widths will be used to
        spectrally down-sample the data before calculating the VCA. Up-sampling
        to smaller channel sizes than the original is not supported.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, channel_width=None, distance=None):
        super(VCA, self).__init__()

        self.input_data_header(cube, header)

        # Regrid the data when channel_width is given
        if channel_width is not None:
            sc_cube = to_spectral_cube(self.data, self.header)

            reg_cube = spectral_regrid_cube(sc_cube, channel_width)

            # Don't pass the header. It will read the new one in reg_cube
            self.input_data_header(reg_cube, None)

        if np.isnan(self.data).any():
            self.data[np.isnan(self.data)] = 0

        if distance is not None:
            self.distance = distance

        self._ps1D_stddev = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        vca_fft = fftshift(rfft_to_fft(self.data))

        self._ps2D = np.power(vca_fft, 2.).sum(axis=0)

    def run(self, verbose=False, save_name=None, return_stddev=True,
            logspacing=False, low_cut=None, high_cut=None,
            xunit=u.pix**-1, use_wavenumber=False, **fit_kwargs):
        '''
        Full computation of VCA.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        save_name : str,optional
            Save the figure when a file name is given.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        xunit : u.Unit, optional
            Choose the unit to convert the x-axis in the plot to.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        fit_kwargs : Passed to `~VCA.fit_pspec`.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(return_stddev=return_stddev,
                                  logspacing=logspacing)
        self.fit_pspec(low_cut=low_cut, high_cut=high_cut, **fit_kwargs)

        if verbose:

            print(self.fit.summary())

            self.plot_fit(show=True, show_2D=True,
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
        spline. If not specified, no break point will be used.
    fiducial_model : VCA
        Computed VCA object. use to avoid recomputing.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, channel_width=None, breaks=None,
                 fiducial_model=None, logspacing=False, low_cut=None,
                 high_cut=None, distance=None):
        super(VCA_Distance, self).__init__()

        low_cut, high_cut = check_fit_limits(low_cut, high_cut)

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is not None:
            self.vca1 = fiducial_model
        else:
            self.vca1 = VCA(cube1, channel_width=channel_width)
            self.vca1.run(brk=breaks[0], low_cut=low_cut[0],
                          high_cut=high_cut[0], logspacing=logspacing)

        self.vca2 = \
            VCA(cube2, channel_width=channel_width)
        self.vca2.run(brk=breaks[1], low_cut=low_cut[1], high_cut=high_cut[1],
                      logspacing=logspacing)

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        xunit=u.pix**-1, save_name=None,
                        use_wavenumber=False):
        '''

        Implements the distance metric for 2 VCA transforms, each with the
        same channel width. We fit the linear portion of the transform to
        represent the powerlaw.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        use_wavenumber : bool, optional
            Plot the x-axis as the wavenumber rather than spatial frequency.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.vca1.slope - self.vca2.slope) /
                   np.sqrt(self.vca1.slope_err**2 +
                           self.vca2.slope_err**2))

        if verbose:

            print("Fit to %s" % (label1))
            print(self.vca1.fit.summary())
            print("Fit to %s" % (label2))
            print(self.vca2.fit.summary())

            import matplotlib.pyplot as p

            self.vca1.plot_fit(show=False, color='b', label=label1, symbol='D',
                               xunit=xunit,
                               use_wavenumber=use_wavenumber)
            self.vca2.plot_fit(show=False, color='g', label=label2, symbol='o',
                               xunit=xunit,
                               use_wavenumber=use_wavenumber)
            p.legend(loc='upper right')

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()
        return self
