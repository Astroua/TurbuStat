# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from numpy.fft import fftshift

from ..rfft_to_fft import rfft_to_fft
from slice_thickness import change_slice_thickness
from ..base_pspec2 import StatisticBase_PSpec2D


class VCA(StatisticBase_PSpec2D):

    '''
    The VCA technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : numpy.ndarray
        Data cube.
    header : FITS header
        Corresponding FITS header.
    slice_sizes : float or int, optional
        Slices to degrade the cube to.
    ang_units : bool, optional
        Convert frequencies to angular units using the given header.
    '''

    def __init__(self, cube, header, slice_size=None, ang_units=False):
        super(VCA, self).__init__()

        self.cube = cube.astype("float64")
        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
        self.header = header
        self.shape = self.cube.shape

        if slice_size is None:
            self.slice_size = 1.0

        if slice_size != 1.0:
            self.cube = \
                change_slice_thickness(self.cube,
                                       slice_thickness=self.slice_size)

        self.ang_units = ang_units

        self._ps1D_stddev = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        vca_fft = fftshift(rfft_to_fft(self.cube))

        self._ps2D = np.power(vca_fft, 2.).sum(axis=0)

    def run(self, verbose=False, brk=None, return_stddev=True,
            logspacing=True):
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
        '''

        self.compute_pspec()
        self.compute_radial_pspec(return_stddev=return_stddev)
        self.fit_pspec(brk=brk)

        if verbose:

            print self.fit.summary()

            self.plot_fit(show=True, show_2D=True)

        return self


class VCA_Distance(object):

    '''
    Calculate the distance between two cubes using VCA. The 1D power spectrum
    is modeled by a linear model. The distance is the t-statistic of the
    interaction between the two slopes.

    Parameters
    ----------
    cube1 : FITS hdu
        Data cube.
    cube2 : FITS hdu
        Data cube.
    slice_size : float, optional
        Slice to degrade the cube to.
    breaks : float, list or array, optional
        Specify where the break point is. If None, attempts to find using
        spline. If not specified, no break point will be used.
    fiducial_model : VCA
        Computed VCA object. use to avoid recomputing.
    ang_units : bool, optional
        Convert frequencies to angular units using the given header.
    '''

    def __init__(self, cube1, cube2, slice_size=1.0, breaks=None,
                 fiducial_model=None, ang_units=False):
        super(VCA_Distance, self).__init__()
        cube1, header1 = cube1
        cube2, header2 = cube2

        self.ang_units = ang_units

        assert isinstance(slice_size, float)

        if not isinstance(breaks, list) or not isinstance(breaks, np.ndarray):
            breaks = [breaks] * 2

        if fiducial_model is not None:
            self.vca1 = fiducial_model
        else:
            self.vca1 = \
                VCA(cube1, header1, slice_size=slice_size,
                    ang_units=ang_units).run(brk=breaks[0])

        self.vca2 = \
            VCA(cube2, header2, slice_size=slice_size,
                ang_units=ang_units).run(brk=breaks[1])

    def distance_metric(self, verbose=False, label1=None, label2=None):
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
            self.vca1.plot_fit(show=False, color='b', label=label1, symbol='D')
            self.vca2.plot_fit(show=False, color='g', label=label2, symbol='o')
            p.legend(loc='upper right')
            p.show()

        return self
