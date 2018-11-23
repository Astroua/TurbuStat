# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import math
from scipy.optimize import leastsq
import astropy.wcs as wcs


def hellinger(data1, data2, bin_width=1.0):
    '''
    Calculate the Hellinger Distance between two datasets.

    Parameters
    ----------
    data1 : numpy.ndarray
        1D array.
    data2 : numpy.ndarray
        1D array.

    Returns
    -------
    distance : float
        Distance value.
    '''
    distance = (bin_width / np.sqrt(2)) * \
        np.sqrt(np.nansum((np.sqrt(data1) -
                           np.sqrt(data2)) ** 2.))
    return distance


def standardize(x, dtype=np.float64):
    '''
    Center and divide by standard deviation (i.e., z-scores).
    '''
    return (x - np.nanmean(x.astype(dtype))) / np.nanstd(x.astype(dtype))


def normalize_by_mean(x):
    '''
    Normalize by the mean.
    '''
    return x / np.nanmean(x)


def center(x):
    '''
    Centre data on zero by subtracting the mean.
    '''
    return x - np.nanmean(x)


def normalize(x):
    '''
    Force the data to have a range of 0 to 1.
    '''
    return (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))


def data_normalization(x, norm_type="standardize"):
    '''
    Apply the specified form to normalize the data.

    Parameters
    ----------
    x : numpy.ndarray
        Input data.
    norm_type : {"standardize", "center", "normalize", "normalize_by_mean"},
                 optional
        Normalization scheme to use.
    '''

    all_norm_types = ["standardize", "center", "normalize",
                      "normalize_by_mean"]

    if norm_type == "standardize":
        return standardize(x)
    elif norm_type == "center":
        return center(x)
    elif norm_type == "normalize":
        return normalize(x)
    elif norm_type == "normalize_by_mean":
        return normalize_by_mean(x)
    else:
        raise ValueError("norm_type {0} is not an accepted input. Must be "
                         "one of {1}".format(norm_type, all_norm_types))


def kl_divergence(P, Q):
    '''
    Kullback Leidler Divergence

    Parameters
    ----------

    P,Q : numpy.ndarray
        Two Discrete Probability distributions

    Returns
    -------

    kl_divergence : float
    '''
    P = P[~np.isnan(P)]
    Q = Q[~np.isnan(Q)]
    P = P[np.isfinite(P)]
    Q = Q[np.isfinite(Q)]
    return np.nansum(np.where(Q != 0, P * np.log(P / Q), 0))


def common_histogram_bins(dataset1, dataset2, nbins=None, logscale=False,
                          return_centered=False, min_val=None, max_val=None):
    '''
    Returns bins appropriate for both given datasets. If nbins is not
    specified, the number is set by the square root of the average
    number of elements in the datasets. This assumes that there are many
    (~>100) elements in each dataset.

    Parameters
    ----------
    dataset1 : 1D numpy.ndarray
        Dataset to use in finding matching set of bins.
    dataset2 : 1D numpy.ndarray
        Same as above.
    nbins : int, optional
        Specify the number of bins to use.
    logscale : bool, optional
        Return logarithmically spaced bins.
    return_centered : bool, optional
        Return the centers of the bins along the the usual edge output.
    min_val : float, optional
        Minimum value to use for the bins.
    max_val : float, optional
        Maximum value to use for the bins.
    '''

    if dataset1.ndim > 1 or dataset2.ndim > 1:
        raise ValueError("dataset1 and dataset2 should be 1D arrays.")

    global_min = min(np.nanmin(dataset1), np.nanmin(dataset2))
    if min_val is not None:
        global_min = max(global_min, min_val)
    global_max = max(np.nanmax(dataset1), np.nanmax(dataset2))
    if max_val is not None:
        global_max = min(global_max, max_val)

    if nbins is None:
        avg_num = np.sqrt((dataset1.size + dataset2.size) / 2.)
        nbins = np.floor(avg_num).astype(int)

    if logscale:
        bins = np.logspace(np.log10(global_min),
                           np.log10(global_max), num=nbins)
    else:
        bins = np.linspace(global_min, global_max, num=nbins)

    if return_centered:
        center_bins = (bins[:-1] + bins[1:]) / 2
        return bins, center_bins

    return bins


class EllipseModel(object):
    """Total least squares estimator for 2D ellipses.
    The functional model of the ellipse is::
        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)
    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.
    The estimator is based on a least squares minimization. The optimal
    solution is computed directly, no iterations are required. This leads
    to a simple, stable and robust fitting method.
    The ``params`` attribute contains the parameters in the following order::
        xc, yc, a, b, theta
    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`, `b`,
        `theta`.
    Examples
    --------
    >>> xy = EllipseModel().predict_xy(np.linspace(0, 2 * np.pi, 25),
    ...                                params=(10, 15, 4, 8, np.deg2rad(30)))
    >>> ellipse = EllipseModel()
    >>> ellipse.estimate(xy)
    True
    >>> np.round(ellipse.params, 2)  # doctest: +SKIP
    array([ 10.  ,  15.  ,   4.  ,   8.  ,   0.52])
    >>> np.round(abs(ellipse.residuals(xy)), 5)  # doctest: +SKIP
    array([ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
            0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])

    Copyright (C) 2011, the scikit-image team
    All rights reserved.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.
     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in
        the documentation and/or other materials provided with the
        distribution.
     3. Neither the name of skimage nor the names of its contributors may be
        used to endorse or promote products derived from this software without
        specific prior written permission.

    THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
    IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
    WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
    INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
    (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
    HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
    STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
    IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.
    """

    def __init__(self):
        self.params = None

    def estimate(self, data):
        """Estimate circle model from data using total least squares.
        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        success : bool
            True, if model estimation succeeds.
        References
        ----------
        .. [1] Halir, R.; Flusser, J. "Numerically stable direct least squares
               fitting of ellipses". In Proc. 6th International Conference in
               Central Europe on Computer Graphics and Visualization.
               WSCG (Vol. 98, pp. 125-132).
        """
        # Original Implementation: Ben Hammel, Nick Sullivan-Molina
        # another REFERENCE: [2] http://mathworld.wolfram.com/Ellipse.html
        # _check_data_dim(data, dim=2)

        x = data[:, 0]
        y = data[:, 1]

        # Quadratic part of design matrix [eqn. 15] from [1]
        D1 = np.vstack([x ** 2, x * y, y ** 2]).T
        # Linear part of design matrix [eqn. 16] from [1]
        D2 = np.vstack([x, y, np.ones(len(x))]).T

        # forming scatter matrix [eqn. 17] from [1]
        S1 = np.dot(D1.T, D1)
        S2 = np.dot(D1.T, D2)
        S3 = np.dot(D2.T, D2)

        # Constraint matrix [eqn. 18]
        C1 = np.array([[0., 0., 2.], [0., -1., 0.], [2., 0., 0.]])

        try:
            # Reduced scatter matrix [eqn. 29]
            M = np.linalg.inv(C1).dot(
                S1 - np.dot(S2, np.linalg.inv(S3)).dot(S2.T))
        except np.linalg.LinAlgError:  # LinAlgError: Singular matrix
            return False

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors
        # from this equation [eqn. 28]
        eig_vals, eig_vecs = np.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * np.multiply(eig_vecs[0, :], eig_vecs[2, :]) \
               - np.power(eig_vecs[1, :], 2)
        a1 = eig_vecs[:, (cond > 0)]
        # seeks for empty matrix
        if 0 in a1.shape or len(a1.ravel()) != 3:
            return False
        a, b, c = a1.ravel()

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = np.dot(-np.linalg.inv(S3), S2.T).dot(a1)
        d, f, g = a2.ravel()

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 (eqn. 15) from [2]
        b /= 2.
        d /= 2.
        f /= 2.

        # finding center of ellipse [eqn.19 and 20] from [2]
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from [2]
        numerator = a * f ** 2 + c * d ** 2 + g * b ** 2 \
                    - 2 * b * d * f - a * c * g
        term = np.sqrt((a - c) ** 2 + 4 * b ** 2)
        denominator1 = (b ** 2 - a * c) * (term - (a + c))
        denominator2 = (b ** 2 - a * c) * (- term - (a + c))
        width = np.sqrt(2 * numerator / denominator1)
        height = np.sqrt(2 * numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse
        # to x-axis [eqn. 23] from [2].
        phi = 0.5 * np.arctan((2. * b) / (a - c))
        if a > c:
            phi += 0.5 * np.pi

        self.params = np.nan_to_num([x0, y0, width, height, phi])

        return True

    def residuals(self, data):
        """
        Determine residuals of data to model.
        For each point the shortest distance to the ellipse is returned.

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        Returns
        -------
        residuals : (N, ) array
            Residual for each data point.
        """

        # _check_data_dim(data, dim=2)

        xc, yc, a, b, theta = self.params

        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = math.cos(t)
            st = math.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt) ** 2 + (yi - yt) ** 2

        # def Dfun(t, xi, yi):
        #     ct = math.cos(t)
        #     st = math.sin(t)
        #     xt = xc + a * ctheta * ct - b * stheta * st
        #     yt = yc + a * stheta * ct + b * ctheta * st
        #     dfx_t = - 2 * (xi - xt) * (- a * ctheta * st
        #                                - b * stheta * ct)
        #     dfy_t = - 2 * (yi - yt) * (- a * stheta * st
        #                                + b * ctheta * ct)
        #     return [dfx_t + dfy_t]

        residuals = np.empty((N, ), dtype=np.double)

        # initial guess for parameter t of closest point on ellipse
        t0 = np.arctan2(y - yc, x - xc) - theta

        # determine shortest distance to ellipse for each point
        for i in range(N):
            xi = x[i]
            yi = y[i]
            # faster without Dfun, because of the python overhead
            t, _ = leastsq(fun, t0[i], args=(xi, yi))
            residuals[i] = np.sqrt(fun(t, xi, yi))

        return residuals

    def predict_xy(self, t, params=None):
        """Predict x- and y-coordinates using the estimated model.
        Parameters
        ----------
        t : array
            Angles in circle in radians. Angles start to count from positive
            x-axis to positive y-axis in a right-handed system.
        params : (5, ) array, optional
            Optional custom parameter set.
        Returns
        -------
        xy : (..., 2) array
            Predicted x- and y-coordinates.
        """

        if params is None:
            params = self.params

        xc, yc, a, b, theta = params

        ct = np.cos(t)
        st = np.sin(t)
        ctheta = math.cos(theta)
        stheta = math.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)

    def estimate_stderrs(self, data, niters=100, alpha=0.6827, debug=False):
        '''
        Use residual bootstrapping to estimate the uncertainty on each
        parameter. *Not part of scikit-image.*

        Parameters
        ----------
        data : (N, 2) array
            N points with ``(x, y)`` coordinates, respectively.
        niters : int, optional
            Number of bootstrap iterations. Defaults to 100.
        alpha : float, optional

        '''

        if not hasattr(self, "params"):
            raise AttributeError("Run EllipseModel.estimate first.")

        if self.params is None:
            raise ValueError("No parameters set. Run fit first.")

        if alpha < 0 or alpha >= 1.:
            raise ValueError("alpha must be between 0 and 1.")

        niters = int(niters)
        params = np.empty((5, niters))

        resid = self.residuals(data)

        for i in range(niters):
            boot_fit = EllipseModel()

            resamp_resid = resid[np.random.permutation(resid.size)]

            # Now we need to add the residuals to the x and y values.
            # The residuals themselves are distances from the ellipse
            # Assume a dirichlet prior of equal weight when adding the
            # residuals to the x and y data, which will preserve the overall
            # residual distance
            prior_weights = np.random.dirichlet((1, 1), size=resid.size)
            # We also need to randomly sample to add or subtract that distance
            prior_dirn = np.random.choice([-1, 1], size=(resid.size, 2))

            resamp_resid = np.tile(resamp_resid, (2, 1)).T * prior_weights * \
                prior_dirn

            resamp_y = data + resamp_resid

            boot_fit.estimate(resamp_y)

            params[:, i] = boot_fit.params

        if debug:
            import matplotlib.pyplot as plt

            plt.subplot(231)
            _ = plt.hist(params[0], bins=10)
            plt.axvline(self.params[0])
            plt.subplot(232)
            _ = plt.hist(params[1], bins=10)
            plt.axvline(self.params[1])
            plt.subplot(233)
            _ = plt.hist(params[2], bins=10)
            plt.axvline(self.params[2])
            plt.subplot(234)
            _ = plt.hist(params[3], bins=10)
            plt.axvline(self.params[3])
            plt.subplot(235)
            _ = plt.hist(params[4], bins=10)
            plt.axvline(self.params[4])

        self.percentiles = np.percentile(params,
                                         [100 * (0.5 - alpha / 2.),
                                          100 * (0.5 + alpha / 2.)],
                                         axis=1)

        # We're going to ASSUME the percentile regions are symmetric enough
        # to do this. Testing on a number of data sets shows this isn't a bad
        # assumption.
        self.param_errs = 0.5 * (self.percentiles[1] - self.percentiles[0])


def common_scale(wcs1, wcs2, tol=1e-5):
    '''
    Return the factor to make the pixel scales in the WCS objects the same.

    Assumes pixels a near to being square and distortions between the grids
    are minimal. If they are distorted an error is raised.

    For laziness, the celestial scales should be the same (so pixels are
    squares). Otherwise this approach of finding a common scale will not work
    and a reprojection would be a better approach before running any
    comparisons.

    Parameters
    ----------
    wcs1 : astropy.wcs.WCS
        WCS Object to match to.
    wcs2 : astropy.wcs.WCS
        WCS Object.

    Returns
    -------
    scale : float
        Factor between the pixel scales.
    '''

    if wcs.utils.is_proj_plane_distorted(wcs1):
        raise wcs.WcsError("First WCS object is distorted.")

    if wcs.utils.is_proj_plane_distorted(wcs2):
        raise wcs.WcsError("Second WCS object is distorted.")

    scales1 = np.abs(wcs.utils.proj_plane_pixel_scales(wcs1.celestial))
    scales2 = np.abs(wcs.utils.proj_plane_pixel_scales(wcs2.celestial))

    # Forcing near square pixels
    if np.abs(scales1[0] - scales1[1]) > tol:
        raise ValueError("Pixels in first WCS are not square. Recommend "
                         "reprojecting to the same grid.")

    if np.abs(scales2[0] - scales2[1]) > tol:
        raise ValueError("Pixels in second WCS are not square. Recommend "
                         "reprojecting to the same grid.")

    scale = scales2[0] / scales1[0]

    return scale


def fourier_shift(x, shift, axis=0):
    '''
    Shift a spectrum by a given number of pixels.

    Parameters
    ----------
    x : np.ndarray
        Array to be shifted
    shift : int or float
        Number of pixels to shift.
    axis : int, optional
        Axis to shift along.

    Returns
    -------
    x2 : np.ndarray
        Shifted array.
    '''
    mask = ~np.isfinite(x)
    nonan = x.copy()
    nonan[mask] = 0.0

    nonan_shift = _shifter(nonan, shift, axis)
    mask_shift = _shifter(mask, shift, axis) > 0.5

    nonan_shift[mask_shift] = np.NaN

    return nonan_shift


def _shifter(x, shift, axis):
    ftx = np.fft.fft(x, axis=axis)
    m = np.fft.fftfreq(x.shape[axis])
    m_shape = [1] * len(x.shape)
    m_shape[axis] = m.shape[0]
    m = m.reshape(m_shape)
    phase = np.exp(-2 * np.pi * m * 1j * shift)
    x2 = np.real(np.fft.ifft(ftx * phase, axis=axis))
    return x2


def pixel_shift(x, shift, axis=0):
    '''
    Shift a spectrum by an integer number of pixels. Much quicker than the
    FFT method, when it can be avoided!

    Parameters
    ----------
    x : np.ndarray
        Array to be shifted
    shift : int or float
        Number of pixels to shift.
    axis : int, optional
        Axis to shift along.

    Returns
    -------
    x2 : np.ndarray
        Shifted array.
    '''

    if not isinstance(shift, int):
        shift = int(shift)

    return np.roll(x, shift, axis=axis)


def padwithzeros(vector, pad_width, iaxis, kwargs):
    '''
    Pad array with zeros.
    '''
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def padwithnans(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = np.NaN
    vector[-pad_width[1]:] = np.NaN
    return vector
