
import numpy as np
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


def standardize(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


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
                          return_centered=False):
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
    '''

    if dataset1.ndim > 1 or dataset2.ndim > 1:
        raise ValueError("dataset1 and dataset2 should be 1D arrays.")

    global_min = min(np.nanmin(dataset1), np.nanmin(dataset2))
    global_max = max(np.nanmax(dataset1), np.nanmax(dataset2))

    if nbins is None:
        avg_num = np.sqrt((dataset1.size + dataset2.size)/2.)
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


class EllipseModel():

    """

    From scikit-image (https://github.com/scikit-image/scikit-image). A copy
    of the license is below.

    Total least squares estimator for 2D ellipses.
    The functional model of the ellipse is::
        xt = xc + a*cos(theta)*cos(t) - b*sin(theta)*sin(t)
        yt = yc + a*sin(theta)*cos(t) + b*cos(theta)*sin(t)
        d = sqrt((x - xt)**2 + (y - yt)**2)
    where ``(xt, yt)`` is the closest point on the ellipse to ``(x, y)``. Thus
    d is the shortest distance from the point to the ellipse.
    This estimator minimizes the squared distances from all points to the
    ellipse::
        min{ sum(d_i**2) } = min{ sum((x_i - xt)**2 + (y_i - yt)**2) }
    Thus you have ``2 * N`` equations (x_i, y_i) for ``N + 5`` unknowns (t_i,
    xc, yc, a, b, theta), which gives you an effective redundancy of ``N - 5``.
    The ``params`` attribute contains the parameters in the following order::
        xc, yc, a, b, theta
    A minimum number of 5 points is required to solve for the parameters.

    Attributes
    ----------
    params : tuple
        Ellipse model parameters in the following order `xc`, `yc`, `a`,
        `b`, `theta`.

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
        """

        if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError('Input data must have shape (N, 2).')

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        # pre-allocate jacobian for all iterations
        A = np.zeros((N + 5, 2 * N), dtype=np.double)
        # same for all iterations: xc, yc
        A[0, :N] = -1
        A[1, N:] = -1

        diag_idxs = np.diag_indices(N)

        def fun(params):
            xyt = self.predict_xy(params[5:], params[:5])
            fx = x - xyt[:, 0]
            fy = y - xyt[:, 1]
            return np.append(fx, fy)

        def Dfun(params):
            xc, yc, a, b, theta = params[:5]
            t = params[5:]

            ct = np.cos(t)
            st = np.sin(t)
            ctheta = np.cos(theta)
            stheta = np.sin(theta)

            # derivatives for fx, fy in the following order:
            #       xc, yc, a, b, theta, t_i

            # fx
            A[2, :N] = - ctheta * ct
            A[3, :N] = stheta * st
            A[4, :N] = a * stheta * ct + b * ctheta * st
            A[5:, :N][diag_idxs] = a * ctheta * st + b * stheta * ct
            # fy
            A[2, N:] = - stheta * ct
            A[3, N:] = - ctheta * st
            A[4, N:] = - a * ctheta * ct + b * stheta * st
            A[5:, N:][diag_idxs] = a * stheta * st - b * ctheta * ct

            return A

        # initial guess of parameters using a circle model
        params0 = np.empty((N + 5, ), dtype=np.double)
        xc0 = x.mean()
        yc0 = y.mean()
        r0 = np.sqrt((x - xc0)**2 + (y - yc0)**2).mean()
        params0[:5] = (xc0, yc0, r0, 0, 0)
        params0[5:] = np.arctan2(y - yc0, x - xc0)

        params, pcov = leastsq(fun, params0, Dfun=Dfun, col_deriv=True,
                               full_output=True)[:2]

        self.params = params[:5]

        resids = self.residuals(data)

        dof = N - 5

        if dof > 0 and pcov is not None:
            s_sq = (resids**2).sum() / dof
            pcov = pcov * s_sq
        else:
            pcov = np.zeros((5, 5)) * np.NaN

        errors = np.sqrt(np.abs(pcov.diagonal()))

        self.param_errs = errors

        return True

    def residuals(self, data):
        """Determine residuals of data to model.
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

        if data.ndim != 2 or data.shape[1] != 2:
                raise ValueError('Input data must have shape (N, 2).')

        xc, yc, a, b, theta = self.params

        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        x = data[:, 0]
        y = data[:, 1]

        N = data.shape[0]

        def fun(t, xi, yi):
            ct = np.cos(t)
            st = np.sin(t)
            xt = xc + a * ctheta * ct - b * stheta * st
            yt = yc + a * stheta * ct + b * ctheta * st
            return (xi - xt)**2 + (yi - yt)**2

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
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        x = xc + a * ctheta * ct - b * stheta * st
        y = yc + a * stheta * ct + b * ctheta * st

        return np.concatenate((x[..., None], y[..., None]), axis=t.ndim)


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
