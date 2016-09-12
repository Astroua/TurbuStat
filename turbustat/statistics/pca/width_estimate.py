
import numpy as np
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline, interp1d
from astropy.modeling import fitting, models
from astropy.modeling import models as astropy_models
from scipy.signal import argrelmin
# from ..stats_utils import EllipseModel
from turbustat.statistics.stats_utils import EllipseModel
import matplotlib.pyplot as plt


def WidthEstimate2D(inList, method='contour', noise_ACF=0,
                    diagnosticplots=False):
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


    Returns
    -------
    scales : array
        The array of estimated scales with length len(inList)

    """
    scales = np.zeros(len(inList))

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
            g = astropy_models.Gaussian2D(x_mean=[0], y_mean=[0],
                                          x_stddev=[1], y_stddev=[1],
                                          amplitude=z.max(), theta=[0],
                                          fixed={'amplitude': True,
                                                 'x_mean': True,
                                                 'y_mean': True}) + \
                astropy_models.Const2D(amplitude=[z.mean()])

            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g, xmat, ymat, z)
            scales[idx] = np.sqrt(2 * (output.x_stddev_0.value[0]**2 +
                                       output.y_stddev_0.value[0]**2))
            if diagnosticplots and idx < 9:
                ax = plt.subplot(3, 3, idx + 1)
                ax.imshow(z, cmap='afmhot')
                ax.contour(output(xmat, ymat),
                           levels=np.array([0.25, 0.5, 0.75, 1.0]) * z.max(),
                           colors=['c'] * 3)
                # ax.show()

        elif method == 'interpolate':
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec) / 100.
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz:-dz:dz])
            scales[idx] = spl(np.exp(-1)) * np.sqrt(2)
        elif method == 'xinterpolate':
            g = astropy_models.Gaussian2D(x_mean=[0], y_mean=[0], x_stddev=[1],
                                          y_stddev=[1], amplitude=z.max(),
                                          theta=[0],
                                          fixed={'amplitude': True,
                                                 'x_mean': True,
                                                 'y_mean': True}) + \
                astropy_models.Const2D(amplitude=[z.mean()])

            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g, xmat, ymat, z)
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
            scales[idx] = spl(np.exp(-1)) * np.sqrt(2)
            # plt.plot((((xmat**2) + (ymat**2))**0.5).ravel(), zvec, 'b,')
            # plt.plot(rmat.ravel(), zvec, 'r,')
            # plt.vlines(scales[idx], zvec.min(), zvec.max())
            # plt.draw()
            # raw_input("Continue??")
            # plt.show()
            # pdb.set_trace()
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
                    scales[idx], model = fit_2D_ellipse(good_path.vertices)
                else:
                    scales[idx] = np.nan
            else:
                scales[idx] = np.nan

    return scales


def WidthEstimate1D(inList, method='interpolate'):
    scales = np.zeros((inList.shape[1], ))
    for idx, y in enumerate(inList.T):
        x = fft.fftfreq(len(y)) * len(y) / 2.0
        if method == 'interpolate':
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                interpolator = interp1d(y[0:minima[0] + 1], x[0:minima[0] + 1])
                scales[idx] = interpolator(np.exp(-1))
        elif method == 'fit':
            g = models.Gaussian1D(amplitude=y[0], mean=[0], stddev=[10],
                                  fixed={'amplitude': True, 'mean': True})
            fit_g = fitting.LevMarLSQFitter()
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                xtrans = (np.abs(x)**0.5)[0:minima[0]]
                yfit = y[0:minima[0]]
            else:
                xtrans = np.abs(x)**0.5
                yfit = y
            output = fit_g(g, xtrans, yfit)
            scales[idx] = np.abs(output.stddev.value[0])
        else:
            raise ValueError("method must be 'interpolate' or 'fit'.")
    return scales


def fit_2D_ellipse(pts):
    '''
    Return ellipse widths
    '''

    ellip = EllipseModel()
    ellip.estimate(pts)

    return np.sqrt(ellip.params[2]**2 + ellip.params[3]**2), ellip


def plot_stuff(raw, fit, residual, n_eigs):
    pass
