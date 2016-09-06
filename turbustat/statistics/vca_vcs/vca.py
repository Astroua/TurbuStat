# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from numpy.fft import fftshift
import astropy.units as u

from ..rfft_to_fft import rfft_to_fft
from slice_thickness import change_slice_thickness
from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types


class VCA(BaseStatisticMixIn, StatisticBase_PSpec2D):

    '''
    The VCA technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header : FITS header, optional
        Corresponding FITS header.
    slice_sizes : float or int, optional
        Slices to degrade the cube to.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, slice_size=None):
        super(VCA, self).__init__()

        self.input_data_header(cube, header)

        if np.isnan(self.data).any():
            self.data[np.isnan(self.data)] = 0

        if slice_size is None:
            self.slice_size = 1.0

        if slice_size != 1.0:
            self.data = \
                change_slice_thickness(self.data,
                                       slice_thickness=self.slice_size)

        self._ps1D_stddev = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        vca_fft = fftshift(rfft_to_fft(self.data))

        self._ps2D = np.power(vca_fft, 2.).sum(axis=0)

    def run(self, verbose=False, brk=None, return_stddev=True,
            logspacing=True, ang_units=False, unit=u.deg):
        '''
        Full computation of VCA.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        brk : float, optional
            Initial guess for the break point.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(return_stddev=return_stddev, max_bin=0.5)
        self.fit_pspec(brk=brk)

        if verbose:

            print self.fit.summary()

            self.plot_fit(show=True, show_2D=True, ang_units=ang_units,
                          unit=unit)

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

    def __init__(self, cube1, cube2, slice_size=1.0, breaks=None,
                 fiducial_model=None):
        super(VCA_Distance, self).__init__()

        assert isinstance(slice_size, float)

        if not isinstance(breaks, list) and not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is not None:
            self.vca1 = fiducial_model
        else:
            self.vca1 = \
                VCA(cube1, slice_size=slice_size).run(brk=breaks[0])

        self.vca2 = \
            VCA(cube2, slice_size=slice_size).run(brk=breaks[1])

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        ang_units=False, unit=u.deg):
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
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.vca1.slope - self.vca2.slope) /
                   np.sqrt(self.vca1.slope_err**2 +
                           self.vca2.slope_err**2))

        if verbose:

            print "Fit to %s" % (label1)
            print self.vca1.fit.summary()
            print "Fit to %s" % (label2)
            print self.vca2.fit.summary()

            import matplotlib.pyplot as p
            self.vca1.plot_fit(show=False, color='b', label=label1, symbol='D',
                               ang_units=ang_units, unit=unit)
            self.vca2.plot_fit(show=False, color='g', label=label2, symbol='o',
                               ang_units=ang_units, unit=unit)
            p.legend(loc='upper right')
            p.show()

        return self
