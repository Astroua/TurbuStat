# Licensed under an MIT open source license - see LICENSE


import numpy as np
from numpy.fft import fft2, fftshift
import astropy.units as u

from ..base_pspec2 import StatisticBase_PSpec2D
from ..base_statistic import BaseStatisticMixIn
from ...io import input_data, common_types, twod_types


class MVC(BaseStatisticMixIn, StatisticBase_PSpec2D):

    """
    Implementation of Modified Velocity Centroids (Lazarian & Esquivel, 03)

    Parameters
    ----------
    centroid : %(dtypes)s
        Normalized first moment array.
    moment0 : %(dtypes)s
        Moment 0 array.
    linewidth : %(dtypes)s
        Normalized second moment array
    header : FITS header
        Header of any of the arrays. Used only to get the
        spatial scale.
    """

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, centroid, moment0, linewidth, header):

        # data property not used here
        self.no_data_flag = True
        self.data = None

        self.header = header

        self.centroid = input_data(centroid, no_header=True)
        self.moment0 = input_data(moment0, no_header=True)
        self.linewidth = input_data(linewidth, no_header=True)

        # Get rid of nans.
        self.centroid[np.isnan(self.centroid)] = np.nanmin(self.centroid)
        self.moment0[np.isnan(self.moment0)] = np.nanmin(self.moment0)
        self.linewidth[np.isnan(self.linewidth)] = np.nanmin(self.linewidth)
        self.degperpix = np.abs(header["CDELT2"])

        assert self.centroid.shape == self.moment0.shape
        assert self.centroid.shape == self.linewidth.shape
        self.shape = self.centroid.shape

        self._ps1D_stddev = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.

        The quantity calculated here is the same as Equation 3 in Lazarian &
        Esquivel (2003), but the inputted arrays are not in the same form as
        described. We can, however, adjust for the use of normalized Centroids
        and the linewidth.

        An unnormalized centroid can be constructed by multiplying the centroid
        array by the moment0. Velocity dispersion is the square of the
        linewidth subtracted by the square of the normalized centroid.
        '''

        term1 = fft2(self.centroid*self.moment0)

        term2 = np.power(self.linewidth, 2) + np.power(self.centroid, 2)

        mvc_fft = term1 - term2 * fft2(self.moment0)

        # Shift to the center
        mvc_fft = fftshift(mvc_fft)

        self._ps2D = np.abs(mvc_fft) ** 2.

    def run(self, verbose=False, logspacing=True,
            return_stddev=True, low_cut=None, high_cut=0.5,
            ang_units=False, unit=u.deg):
        '''
        Full computation of MVC.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        low_cut : float, optional
            Low frequency cut off in frequencies used in the fitting.
        high_cut : float, optional
            High frequency cut off in frequencies used in the fitting.
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)

        self.fit_pspec(low_cut=low_cut, high_cut=high_cut,
                       large_scale=0.5)

        if verbose:

            print self.fit.summary()

            self.plot_fit(show=True, show_2D=True, ang_units=ang_units,
                          unit=unit)

        return self


class MVC_Distance(object):
    """
    Distance metric for MVC.

    Parameters
    ----------
    data1 : dict
        A dictionary containing the centroid, moment 0, and linewidth arrays
        of a spectral cube. This output is created by Mask_and_Moments.to_dict.
        The minimum expected keys are 'centroid', 'moment0' and 'linewidth'. If
        weight_by_error is enabled, the dictionary should also contain the
        error arrays, where the keys are the three above with '_error'
        appended to the end.
    data2 : dict
        See data1.
    fiducial_model : MVC
        Computed MVC object. use to avoid recomputing.
    weight_by_error : bool, optional
        When enabled, the property arrays are weighted by the inverse
        squared of the error arrays.
    """

    def __init__(self, data1, data2, fiducial_model=None,
                 weight_by_error=False):

        # Create weighted or non-weighted versions
        if weight_by_error:
            centroid1 = data1["centroid"][0] * \
                data1["centroid_error"][0] ** -2.
            moment01 = data1["moment0"][0] * \
                data1["moment0_error"][0] ** -2.
            linewidth1 = data1["linewidth"][0] * \
                data1["linewidth_error"][0] ** -2.
            centroid2 = data2["centroid"][0] * \
                data2["centroid_error"][0] ** -2.
            moment02 = data2["moment0"][0] * \
                data2["moment0_error"][0] ** -2.
            linewidth2 = data2["linewidth"][0] * \
                data2["linewidth_error"][0] ** -2.
        else:
            centroid1 = data1["centroid"][0]
            moment01 = data1["moment0"][0]
            linewidth1 = data1["linewidth"][0]
            centroid2 = data2["centroid"][0]
            moment02 = data2["moment0"][0]
            linewidth2 = data2["linewidth"][0]

        if fiducial_model is not None:
            self.mvc1 = fiducial_model
        else:
            self.mvc1 = MVC(centroid1, moment01, linewidth1,
                            data1["centroid"][1])
            self.mvc1.run()

        self.mvc2 = MVC(centroid2, moment02, linewidth2,
                        data2["centroid"][1])
        self.mvc2.run()

    def distance_metric(self, low_cut=None, high_cut=0.5, verbose=False,
                        label1=None, label2=None, ang_units=False, unit=u.deg):
        '''

        Implements the distance metric for 2 MVC transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A linear model with an interaction term is fit to the two powerlaws.
        The distance is the t-statistic of the interaction.

        Parameters
        ----------
        low_cut : int or float, optional
            Set the cut-off for low spatial frequencies. Visually, below ~2
            deviates from the power law (for the simulation set).
        high_cut : int or float, optional
            Set the cut-off for high spatial frequencies. Values beyond the
            size of the root grid are found to have no meaningful contribution
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for dataset1
        label2 : str, optional
            Object or region name for dataset2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.mvc1.slope - self.mvc2.slope) /
                   np.sqrt(self.mvc1.slope_err**2 +
                           self.mvc2.slope_err**2))

        if verbose:

            print self.mvc1.fit.summary()
            print self.mvc2.fit.summary()

            import matplotlib.pyplot as p
            self.mvc1.plot_fit(show=False, color='b', label=label1, symbol='D',
                               ang_units=ang_units, unit=unit)
            self.mvc2.plot_fit(show=False, color='g', label=label2, symbol='o',
                               ang_units=ang_units, unit=unit)
            p.legend(loc='best')
            p.show()

        return self
