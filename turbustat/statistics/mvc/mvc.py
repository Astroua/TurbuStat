# Licensed under an MIT open source license - see LICENSE


import numpy as np
from numpy.fft import fft2, fftshift

from ..base_pspec2 import StatisticBase_PSpec2D


class MVC(StatisticBase_PSpec2D):

    """
    Implementation of Modified Velocity Centroids (Lazarian & Esquivel, 03)

    Parameters
    ----------
    centroid : numpy.ndarray
        Normalized first moment array.
    moment0 : numpy.ndarray
        Moment 0 array.
    linewidth : numpy.ndarray
        Normalized second moment array
    header : FITS header
        Header of any of the arrays. Used only to get the
        spatial scale.

    """

    def __init__(self, centroid, moment0, linewidth, header, ang_units=False):
        self.centroid = centroid
        self.moment0 = moment0
        self.linewidth = linewidth
        self.ang_units = ang_units

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

        return self

    def run(self, verbose=False, logspacing=True,
            return_stddev=True, low_cut=None, high_cut=0.5):
        '''
        Full computation of MVC.

        Parameters
        ----------
        phys_units : bool, optional
            Sets frequency scale to physical units.
        verbose: bool, optional
            Enables plotting.
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=logspacing,
                                  return_stddev=return_stddev)
        self.fit_pspec(low_cut=low_cut, high_cut=high_cut,
                       large_scale=0.5)

        if verbose:

            print self.fit.summary()

            self.plot_fit(show=True, show_2D=True)

        return self


class MVC_distance(object):

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
        # super(mvc_distance, self).__init__()

        self.shape1 = data1["centroid"][0].shape
        self.shape2 = data2["centroid"][0].shape

        low_cut = 2. / float(min(min(self.shape1),
                                 min(self.shape2)))

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
            self.mvc1.run(low_cut=low_cut)

        self.mvc2 = MVC(centroid2, moment02, linewidth2,
                        data2["centroid"][1])
        self.mvc2.run(low_cut=low_cut)

        self.results = None
        self.distance = None

    def distance_metric(self, low_cut=None, high_cut=0.5, verbose=False,
                        labels=None):
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
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.mvc1.slope - self.mvc2.slope) /
                   np.sqrt(self.mvc1.slope_err**2 +
                           self.mvc2.slope_err**2))

        if verbose:
            if labels is None:
                labels = ['1', '2']

            print "Fit to %s" % (labels[0])
            print self.mvc1.fit.summary()
            print "Fit to %s" % (labels[1])
            print self.mvc2.fit.summary()

            import matplotlib.pyplot as p
            self.mvc1.plot_fit(show=False, color='b', label=labels[0])
            self.mvc2.plot_fit(show=False, color='r', label=labels[1])
            p.legend(loc='best')
            p.show()

        return self
