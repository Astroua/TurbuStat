
from warnings import warn
import numpy as np
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline, interp1d
from astropy.modeling import fitting, models
from astropy.modeling import models as astropy_models
from scipy.signal import argrelmin
from ..stats_utils import EllipseModel
import astropy.units as u
import matplotlib.pyplot as plt


def WidthEstimate2D(inList, method='contour', noise_ACF=0,
                    diagnosticplots=False, brunt_beamcorrect=True,
                    beam_fwhm=None, spatial_cdelt=None):
    """
    Parameters
    ----------
    inList: list of 2d arrays
        The list of autocorrelation images from which widths will be estimated
    method: 'contour', 'fit', 'interpolate', or 'xinterpolate'
        The width estimation method to use
    noise_ACF: float or 2darray
        The noise autocorrelation function to subtract from the autocorrelation
        images
    diagnosticsplots: bool
        Show diagnostic plots for the first 9 autocorrelation images showing
        the goodness of fit (for the gaussian estimator) or ??? (presently
        nothing) for the others
    brunt_beamcorrect : bool, optional
        Apply the beam correction. When enabled, the beam size must be given.
    beam_fwhm : None or astropy.units.Quantity
        The FWHM beam width in angular units. Must be given when using
        brunt_beamcorrect.
    spatial_cdelt : None or astropy.units.Quantity
        The angular scale of a pixel in the given data. Must be given when
        using brunt_beamcorrect.

    Returns
    -------
    scales : array
        The array of estimated scales with length len(inList)
    scale_errors : array
        Uncertainty estimations on the scales.

    """
    scales = np.zeros(len(inList))
    scale_errors = np.zeros(len(inList))

    # set up the x/y grid just once
    z = inList[0]
    x = fft.fftfreq(z.shape[0]) * z.shape[0] / 2.0
    y = fft.fftfreq(z.shape[1]) * z.shape[1] / 2.0
    xmat, ymat = np.meshgrid(x, y, indexing='ij')
    xmat = np.fft.fftshift(xmat)
    ymat = np.fft.fftshift(ymat)
    rmat = (xmat**2 + ymat**2)**0.5

    for idx, zraw in enumerate(inList):
        z = zraw - noise_ACF

        if method == 'fit':
            output, cov = fit_2D_gaussian(xmat, ymat, z)
            scales[idx] = np.sqrt(output.x_stddev_0.value[0]**2 +
                                  output.y_stddev_0.value[0]**2)
            errs = np.sqrt(np.abs(cov.diagonal()))
            # Order in the cov matrix is given by the order of parameters in
            # model.param_names. But amplitude and the means are fixed, so in
            # this case they are the first 2.
            scale_errors[idx] = (np.abs(output.x_stddev_0 * errs[0]) +
                                 np.abs(output.y_stddev_0 * errs[1])) / \
                scales[idx]

            if diagnosticplots and idx < 9:
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
            dz = len(zvec) / 100.
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz:-dz:dz])
            scales[idx] = spl(np.exp(-1))

            # Need to implement some error estimation
            scale_errors[idx] = 0.0

        elif method == 'xinterpolate':
            warn("Error estimation not implemented for interpolation!")
            output, cov = fit_2D_gaussian(xmat, ymat, z)
            aspect = output.y_stddev_0.value[0] / output.x_stddev_0.value[0]
            theta = output.theta_0.value[0]
            rmat = ((xmat * np.cos(theta) + ymat * np.sin(theta))**2 +
                    (-xmat * np.sin(theta) + ymat * np.cos(theta))**2 *
                    aspect**2)**0.5
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec) / 100.
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz:-dz:dz])
            scales[idx] = spl(np.exp(-1))

            # Need to implement some error estimation
            scale_errors[idx] = 0.0

        if method == 'contour':
            znorm = z
            znorm /= znorm.max()
            return_interactive = False
            if plt.isinteractive():
                plt.ioff()
                return_interactive = True

            try:
                cs = plt.contour(xmat, ymat, znorm, levels=[np.exp(-1)])
            except ValueError as e:
                raise e("Contour level not found in autocorrelation image " +
                        str(idx))
            paths = cs.collections[0].get_paths()
            plt.close()

            if return_interactive:
                plt.ion()

            # Only points that contain the origin

            if len(paths) > 0:
                pidx = np.where([p.contains_point((0, 0)) for p in paths])[0]
                if pidx.shape[0] > 0:
                    good_path = paths[pidx[0]]
                    scales[idx], scale_errors[idx], model = \
                        fit_2D_ellipse(good_path.vertices)
                else:
                    scales[idx] = np.nan
            else:
                scales[idx] = np.nan

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
                pix_per_beam = beam_fwhm.value / spatial_cdelt.value

                # Using the definition from Chris Brunt's thesis for a gaussian
                # beam. Note that the IDL code has:
                # e=(3./((kappa+2)*(kappa+3.)))^(1./kappa)
                # deltay[i]=(p[i,0]^kappa-(e*1.0)^kappa)^(1./kappa)
                # deltaz[i]=(p[i,1]^kappa-(e*1.0)^kappa)^(1./kappa)
                # I think the (e * 1.0) term is where the beam size should be
                # used, which is what is used here.
                kappa = 0.8
                e = np.power(3. / ((kappa + 2.) * (kappa + 3.)), 1 / kappa)

                term1 = np.power(scales, kappa)
                term2 = np.power(e * pix_per_beam, kappa)

                scale_errors = \
                    np.abs(np.power(term1 - term2, (1 / kappa) - 1) *
                           np.power(scales, kappa - 1)) * scale_errors

                scales = np.power(term1 - term2, 1 / kappa)

    return scales, scale_errors


def WidthEstimate1D(inList, method='interpolate'):
    '''
    Find widths from spectral eigenvectors. These eigenvectors should already
    be normalized.

    Parameters
    ----------
    inList : list of arrays or array
        List of normalized eigenvectors, or a 2D array with eigenvectors
        along the 2nd axis.
    method : str, optional
        The width estimation method to use.

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
        x = fft.fftfreq(len(y)) * len(y) / 2.0
        if method == 'interpolate':
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                warn("Error estimation not implemented for interpolation!")
                interpolator = interp1d(y[0:minima[0] + 1], x[0:minima[0] + 1])
                scales[idx] = interpolator(np.exp(-1))
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
            errors = np.sqrt(np.abs(fit_g.fit_info['param_cov'].diagonal()))
            scales[idx] = np.abs(output.stddev.value[0]) * np.sqrt(2)
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
                    raise Warning("Cannot find width where the 1/e level is"
                                  " reached. Ensure the eigenspectra are "
                                  "normalized!")

        else:
            raise ValueError("method must be 'walk-down', 'interpolate' or"
                             " 'fit'.")

    return scales, scale_errors


def fit_2D_ellipse(pts):
    '''
    Return ellipse widths
    '''

    ellip = EllipseModel()
    ellip.estimate(pts)

    width = np.sqrt((ellip.params[2]**2 + ellip.params[3]**2) / 2.)
    width_err = (np.abs(ellip.params[2] * ellip.param_errs[2]) +
                 np.abs(ellip.params[3] * ellip.param_errs[3])) / (2 * width)

    return width, width_err, ellip


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
        astropy_models.Const2D(amplitude=[z.mean()])

    fit_g = fitting.LevMarLSQFitter()
    output = fit_g(g, xmat, ymat, z)
    cov = fit_g.fit_info['param_cov']

    return output, cov


def plot_stuff(raw, fit, residual, n_eigs):
    pass
