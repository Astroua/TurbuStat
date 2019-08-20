# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

from warnings import warn
import numpy as np
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline, interp1d
from astropy.modeling import fitting, models
from astropy.modeling import models as astropy_models
from scipy.signal import argrelmin
import astropy.units as u
from matplotlib.path import Path
from skimage.measure import find_contours
from scipy.ndimage import map_coordinates

from ..stats_utils import EllipseModel


def WidthEstimate2D(inList, method='contour', noise_ACF=0,
                    diagnosticplots=False, brunt_beamcorrect=True,
                    beam_fwhm=None, spatial_cdelt=None, **fit_kwargs):
    """
    Estimate spatial widths from a set of autocorrelation images.

    .. warning:: Error estimation is not implemented for `interpolate` or
      `xinterpolate`.

    Parameters
    ----------
    inList: {list of 2D `~numpy.ndarray`s, 3D `~numpy.ndarray}
        The list of autocorrelation images.
    method: {'contour', 'fit', 'interpolate', 'xinterpolate'}, optional
        The width estimation method to use. `contour` fits an ellipse to the
        1/e contour about the peak. `fit` fits a 2D Gaussian to the peak.
        `interpolate` and `xinterpolate` both estimate the 1/e level from
        interpolating the data onto a finer grid near the center.
        `xinterpolate` first fits a 2D Gaussian to estimate the radial
        distances about the peak.
    noise_ACF: {float, 2D `~numpy.ndarray`}, optional
        The noise autocorrelation function to subtract from the autocorrelation
        images. This is typically produced by the last few eigenimages, whose
        structure should consistent of irreducible noise.
    diagnosticplots: bool, optional
        Show diagnostic plots for the first 9 autocorrelation images showing
        the goodness of fit (for the gaussian estimator) or ??? (presently
        nothing) for the others.
    brunt_beamcorrect : bool, optional
        Apply the beam correction. When enabled, the beam size must be given.
    beam_fwhm : None or astropy.units.Quantity
        The FWHM beam width in angular units. Must be given when using
        `brunt_beamcorrect`.
    spatial_cdelt : {None, astropy.units.Quantity}, optional
        The angular scale of a pixel in the given data. Must be given when
        using brunt_beamcorrect.
    fit_kwargs : dict, optional
        Used when method is 'contour'. Passed to
        `turbustat.statistics.stats_utils.EllipseModel.estimate_stderrs`.

    Returns
    -------
    scales : array
        The array of estimated scales with length len(inList) or the 0th
        dimension size if `inList` is a 3D array.
    scale_errors : array
        Uncertainty estimations on the scales.

    """

    allowed_methods = ['fit', 'interpolate', 'xinterpolate', 'contour']
    if method not in allowed_methods:
        raise ValueError("Method must be 'fit', 'interpolate', 'xinterpolate'"
                         " or 'contour'.")

    y_scales = np.zeros(len(inList))
    x_scales = np.zeros(len(inList))
    y_scale_errors = np.zeros(len(inList))
    x_scale_errors = np.zeros(len(inList))

    # set up the x/y grid just once
    z = inList[0]
    # NOTE: previous versions were dividing by an extra factor of 2!
    x = fft.fftfreq(z.shape[0]) * z.shape[0]
    y = fft.fftfreq(z.shape[1]) * z.shape[1]
    xmat, ymat = np.meshgrid(x, y, indexing='ij')
    xmat = np.fft.fftshift(xmat)
    ymat = np.fft.fftshift(ymat)
    rmat = (xmat**2 + ymat**2)**0.5

    for idx, zraw in enumerate(inList):
        z = zraw - noise_ACF

        if method == 'fit':
            output, cov = fit_2D_gaussian(xmat, ymat, z)
            y_scales[idx] = output.y_stddev_0.value
            x_scales[idx] = output.x_stddev_0.value

            errs = np.sqrt(np.abs(cov.diagonal()))
            # Order in the cov matrix is given by the order of parameters in
            # model.param_names. But amplitude and the means are fixed, so in
            # this case they are the first 2.
            y_scale_errors[idx] = errs[1]
            x_scale_errors[idx] = errs[0]

            if diagnosticplots and idx < 9:
                import matplotlib.pyplot as plt
                ax = plt.subplot(3, 3, idx + 1)
                ax.imshow(z, cmap='afmhot')
                ax.contour(output(xmat, ymat),
                           levels=np.array([0.25, 0.5, 0.75, 1.0]) * z.max(),
                           colors=['c'] * 3)
                # ax.show()

        elif method == 'interpolate':
            warn("Error estimation not implemented for interpolation!")
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = int(len(zvec) / 100.)
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz:-dz:dz])

            x_scales[idx] = spl(np.exp(-1)) / np.sqrt(2)
            y_scales[idx] = spl(np.exp(-1)) / np.sqrt(2)

            # Need to implement some error estimation
            x_scale_errors[idx] = 0.0
            y_scale_errors[idx] = 0.0

        elif method == 'xinterpolate':
            warn("Error estimation not implemented for interpolation!")
            output, cov = fit_2D_gaussian(xmat, ymat, z)
            try:
                aspect = output.y_stddev_0.value[0] / output.x_stddev_0.value[0]
                theta = output.theta_0.value[0]
            except IndexError:  # raised with astropy >v3.3 (current dev.)
                aspect = output.y_stddev_0.value / output.x_stddev_0.value
                theta = output.theta_0.value

            rmat = ((xmat * np.cos(theta) + ymat * np.sin(theta))**2 +
                    (-xmat * np.sin(theta) + ymat * np.cos(theta))**2 *
                    aspect**2)**0.5
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = int(len(zvec) / 100.)
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz:-dz:dz])

            x_scales[idx] = spl(np.exp(-1)) / np.sqrt(2)
            y_scales[idx] = spl(np.exp(-1)) / np.sqrt(2)

            # Need to implement some error estimation
            x_scale_errors[idx] = 0.0
            y_scale_errors[idx] = 0.0

        elif method == 'contour':
            znorm = z
            znorm /= znorm.max()

            level = np.exp(-1)
            paths = get_contour_path(xmat, ymat, znorm, level, idx)

            # Only points that contain the origin

            if len(paths) > 0:
                pidx = np.where([p.contains_point((0, 0)) for p in paths])[0]
                if pidx.shape[0] > 0:
                    good_path = paths[pidx[0]]

                    output = fit_2D_ellipse(good_path.vertices, **fit_kwargs)
                    (y_scales[idx], x_scales[idx], y_scale_errors[idx],
                     x_scale_errors[idx], ellip) = output

                else:
                    y_scales[idx] = np.nan
                    x_scales[idx] = np.nan
                    y_scale_errors[idx] = np.nan
                    x_scale_errors[idx] = np.nan
                    ellip = None
            else:
                y_scales[idx] = np.nan
                x_scales[idx] = np.nan
                y_scale_errors[idx] = np.nan
                x_scale_errors[idx] = np.nan
                ellip = None

            if diagnosticplots and idx < 9 and ellip is not None:
                import matplotlib.pyplot as plt
                ax = plt.subplot(3, 3, idx + 1)
                ax.imshow(z, cmap='afmhot')
                ax.contour(z, levels=np.array([np.exp(-1)]) * z.max(),
                           colors='c')
                full_params = np.array([0, 0,
                                        ellip.params[2],
                                        ellip.params[3],
                                        ellip.params[-1]])
                pts = ellip.predict_xy(np.linspace(0, 2 * np.pi),
                                       params=full_params)
                ax.plot(pts[:, 1] + z.shape[1] // 2,
                        pts[:, 0] + z.shape[0] // 2, "g--")
                ax.set_yticks([])
                ax.set_xticks([])
                ax.set_title("{}".format(idx + 1))

    if diagnosticplots:
        plt.tight_layout()

    if brunt_beamcorrect:
        if beam_fwhm is None or spatial_cdelt is None:
            raise ValueError("beam_fwhm and spatial_cdelt must be"
                             " given when 'brunt_beamcorrect' is "
                             "enabled.")

        # Quantities must be in angular units
        try:
            beam_fwhm = beam_fwhm.to(u.deg)
        except u.UnitConversionError:
            raise u.UnitConversionError("beam_fwhm must be in angular"
                                        " units.")
        try:
            spatial_cdelt = np.abs(spatial_cdelt.to(u.deg))
        except u.UnitConversionError:
            raise u.UnitConversionError("spatial_cdelt must be in "
                                        "angular units.")

        # We need the number of pixels across 1 FWHM
        # Since it's just being used in the formula below, I don't
        # think rounding to the nearest int is needed.
        pix_per_beam = beam_fwhm.value / spatial_cdelt.value / \
            np.sqrt(8 * np.log(2)) / np.sqrt(2)

        # Using the definition from Chris Brunt's thesis for a gaussian
        # beam. Note that the IDL code has:
        # e=(3./((kappa+2)*(kappa+3.)))^(1./kappa)
        # deltay[i]=(p[i,0]^kappa-(e*1.0)^kappa)^(1./kappa)
        # deltaz[i]=(p[i,1]^kappa-(e*1.0)^kappa)^(1./kappa)
        # I think the (e * 1.0) term is where the beam size should be
        # used, which is what is used here.

        # For an exponential ACF
        kappa = 1.

        e = np.power(3. / ((kappa + 2.) * (kappa + 3.)), 1 / kappa)
        # This is the other form that appears in the thesis.
        # e = 0.65 + 0.1 * kappa

        y_term1 = np.power(y_scales, kappa)
        x_term1 = np.power(x_scales, kappa)
        term2 = np.power(e * pix_per_beam, kappa)

        y_scale_errors = \
            np.abs(np.power(y_term1 - term2, (1 / kappa) - 1) *
                   np.power(y_scales, kappa - 1)) * y_scale_errors
        x_scale_errors = \
            np.abs(np.power(x_term1 - term2, (1 / kappa) - 1) *
                   np.power(x_scales, kappa - 1)) * x_scale_errors

        y_scales = np.power(y_term1 - term2, 1 / kappa)
        x_scales = np.power(x_term1 - term2, 1 / kappa)

    scales = np.sqrt(y_scales**2 +
                     x_scales**2)

    scale_errors = \
        np.sqrt((x_scales * x_scale_errors)**2 +
                (y_scales * y_scale_errors)**2) / scales

    return scales, scale_errors


def WidthEstimate1D(inList, method='walk-down'):
    '''
    Find widths from spectral eigenvectors. These eigenvectors should already
    be normalized. Widths are defined by the location where 1/e of the maximum
    occurs.

    .. note:: If the spectral dimension is small in the given eigenvectors
      (i.e., their length), the 1/e level might not be reached. If this is the
      case, try padding the initial data cube with zeros in the spectral
      dimension. The effect on the results should be minimal, as the additional
      eigenvalues from the padding will be zero. This is especially important
      when using `walk-down`.

    .. warning:: Error estimation is not implemented for `interpolate`.

    Parameters
    ----------
    inList: {list of 1D `~numpy.ndarray`s, 2D `~numpy.ndarray}
        List of normalized eigenvectors, or a 2D array with eigenvectors
        along the 2nd axis.
    method : {'walk-down', 'fit', 'interpolate'}, optional
        The width estimation method to use. The options are 'fit',
        'interpolate', or 'walk-down'.  `walk-down` starts at the peak, and
        uses a bisector to estimate where the 1/e level lies between the two
        nearest points. `fit` fits a Gaussian to data before the first local
        minimum. `interpolate` estimates the 1/e level before the first local
        minimum.

    Returns
    -------
    scales : array
        The array of estimated scales with length len(inList)
    scale_errors : array
        Uncertainty estimations on the scales.

    '''
    scales = np.zeros((inList.shape[1],))
    scale_errors = np.zeros((inList.shape[1],))
    for idx, y in enumerate(inList.T):
        x = np.fft.fftfreq(len(y)) * len(y)
        if method == 'interpolate':
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                warn("Error estimation not implemented for interpolation!")
                interpolator = interp1d(y[0:minima[0] + 1], x[0:minima[0] + 1])
                try:
                    scales[idx] = interpolator(np.exp(-1))
                except ValueError:
                    warn("Interpolation failed.")
                    scales[idx] = np.NaN
                # scale_errors[idx] = ??
        elif method == 'fit':
            g = models.Gaussian1D(amplitude=y[0], mean=[0], stddev=[10],
                                  fixed={'amplitude': True, 'mean': True})
            fit_g = fitting.LevMarLSQFitter()
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                xtrans = np.abs(x)[0:minima[0]]
                yfit = y[0:minima[0]]
            else:
                xtrans = np.abs(x)
                yfit = y
            output = fit_g(g, xtrans, yfit)
            # Pull out errors from cov matrix. Stddev is the last parameter in
            # the list
            # If the fit failed, param_cov will be None. If this occurs, fill
            # in NaNs.
            if fit_g.fit_info['param_cov'] is None:
                warn("Fitting failed.")
                scales[idx] = np.NaN
                scale_errors[idx] = np.NaN
                continue

            errors = np.sqrt(np.abs(fit_g.fit_info['param_cov'].diagonal()))
            try:
                scales[idx] = np.abs(output.stddev.value[0]) * np.sqrt(2)
            except IndexError:  # raised with astropy >v3.3 (current dev.)
                scales[idx] = np.abs(output.stddev.value) * np.sqrt(2)
            scale_errors[idx] = errors[-1] * np.sqrt(2)
        elif method == "walk-down":
            y /= y.max()
            # Starting from the first point, start walking down the curve until
            # 1/e is reached
            for i, val in enumerate(y):
                if val < np.exp(-1):
                    diff = val - y[i - 1]
                    scales[idx] = x[i - 1] + ((np.exp(-1) - y[i - 1]) / diff)
                    # Following Heyer & Brunt
                    scale_errors[idx] = 0.5
                    break

                if i == y.size - 1:
                    warn("Cannot find width where the 1/e level is"
                         " reached. Ensure the eigenspectra are "
                         "normalized!")
                    scale_errors[idx] = np.NaN
                    scales[idx] = np.NaN

        else:
            raise ValueError("method must be 'walk-down', 'interpolate' or"
                             " 'fit'.")

    return scales, scale_errors


def fit_2D_ellipse(pts, **bootstrap_kwargs):
    '''
    Return ellipse widths
    '''

    ellip = EllipseModel()
    try:
        ellip.estimate(pts)
        ellip.estimate_stderrs(pts, **bootstrap_kwargs)

        xwidth = ellip.params[2] / np.sqrt(2)
        ywidth = ellip.params[3] / np.sqrt(2)

        xwidth_err = ellip.param_errs[2] / np.sqrt(2)
        ywidth_err = ellip.param_errs[3] / np.sqrt(2)
    except ValueError:
        ywidth = xwidth = ywidth_err = xwidth_err = np.NaN

    return ywidth, xwidth, ywidth_err, xwidth_err, ellip


def fit_2D_gaussian(xmat, ymat, z):
    '''
    Return fitted model parameters
    '''

    g = astropy_models.Gaussian2D(x_mean=[0], y_mean=[0], x_stddev=[1],
                                  y_stddev=[1], amplitude=z.max(),
                                  theta=[0],
                                  fixed={'amplitude': True,
                                         'x_mean': True,
                                         'y_mean': True}) + \
        astropy_models.Const2D(amplitude=[np.percentile(z, 10)])

    fit_g = fitting.LevMarLSQFitter()
    output = fit_g(g, xmat, ymat, z)
    cov = fit_g.fit_info['param_cov']

    if cov is None:
        warn("Fitting failed.")
        cov = np.zeros((4, 4)) * np.NaN

    return output, cov


def get_contour_path(xmat, ymat, z, level, idx=0):
    '''
    Return the contour path using matplotlib._cntr for versions <2.2.
    Otherwise use skimage.measure.find_contours.

    '''
    # Get the contour in the array coordinates then map to the given xmat, ymat

    contours = find_contours(z, level)

    if len(contours) == 0:
        return []

    # Map onto the given coordinate grids
    mapped_contours = []
    for contour in contours:
        x_proj = map_coordinates(xmat, contour.T)
        y_proj = map_coordinates(ymat, contour.T)

        mapped_contours.append(np.vstack([x_proj, y_proj]).T)

    paths = [Path(verts) for verts in mapped_contours]

    return paths
