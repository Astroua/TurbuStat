
import numpy as np
import numpy.fft as fft
from scipy.interpolate import LSQUnivariateSpline, interp1d
from astropy.modeling import models, fitting
from scipy.signal import argrelmin
from skimage.measure import EllipseModel, find_contours
import matplotlib.pyplot as plt


def WidthEstimate2D(inList, method='contour', noise_ACF=0):
    scales = np.zeros(len(inList))
    models = []

    x = fft.fftfreq(inList[0].shape[0]) * inList[0].shape[0] / 2.0
    y = fft.fftfreq(inList[0].shape[1]) * inList[0].shape[1] / 2.0
    xmat, ymat = np.meshgrid(x, y, indexing='ij')
    # z = np.roll(z, z.shape[0] / 2, axis=0)
    # z = np.roll(z, z.shape[1] / 2, axis=1)
    xmat = np.roll(xmat, xmat.shape[0] / 2, axis=0)
    xmat = np.roll(xmat, xmat.shape[1] / 2, axis=1)
    ymat = np.roll(ymat, ymat.shape[0] / 2, axis=0)
    ymat = np.roll(ymat, ymat.shape[1] / 2, axis=1)
    rmat = (xmat**2 + ymat**2)**0.5

    for idx, zraw in enumerate(inList):
        z = zraw - noise_ACF

        if method == 'fit':
            g = models.Gaussian2D(x_mean=[0], y_mean=[0],
                                  x_stddev=[1], y_stddev=[1],
                                  amplitude=z[0, 0],
                                  theta=[0],
                                  fixed={'amplitude': True,
                                         'x_mean': True,
                                         'y_mean': True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g, np.abs(xmat)**0.5, np.abs(ymat)**0.5, z)
            scales[idx] = 2**0.5 * np.sqrt(output.x_stddev.value[0]**2 +
                                           output.y_stddev.value[0]**2)
        elif method == 'interpolate':
            rvec = rmat.ravel()
            zvec = z.ravel()
            zvec /= zvec.max()
            sortidx = np.argsort(zvec)
            rvec = rvec[sortidx]
            zvec = zvec[sortidx]
            dz = len(zvec) / 100.
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz::dz])
            scales[idx] = spl(np.exp(-1))
        elif method == 'xinterpolate':
            g = models.Gaussian2D(x_mean=[0], y_mean=[0],
                                  x_stddev=[1], y_stddev=[1],
                                  amplitude=z[0, 0],
                                  theta=[0],
                                  fixed={'amplitude': True,
                                         'x_mean': True,
                                         'y_mean': True})
            fit_g = fitting.LevMarLSQFitter()
            output = fit_g(g, xmat, ymat, z)
            aspect = 1 / (output.x_stddev.value[0] / output.y_stddev.value[0])
            theta = output.theta.value[0]
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
            spl = LSQUnivariateSpline(zvec, rvec, zvec[dz::dz])
            scales[idx] = spl(np.exp(-1))
            plt.plot((((xmat**2) + (ymat**2))**0.5).ravel(), z.ravel(), 'b,')
            plt.plot(rmat.ravel(), z.ravel(), 'r,')
            plt.vlines(scales[idx], zvec.min(), zvec.max())
            plt.show()
            pdb.set_trace()
        if method == 'contour':
            znorm = z
            znorm /= znorm.max()
            try:
                cs = plt.contour(xmat, ymat, znorm, levels=[np.exp(-1)])
            except ValueError as e:
                raise e("Contour level not found in autocorrelation image " +
                        str(idx))
            paths = cs.collections[0].get_paths()
            plt.close()

            # Only points that contain the origin

            if len(paths) > 1:
                pidx = np.where([p.contains_point((0, 0)) for p in paths])[0]
                if pidx.shape[0] > 0:
                    good_path = paths[pidx[0]]
                    scales[idx], model = fit_2D_ellipse(good_path.vertices)
                else:
                    scales[idx] = np.nan
                    model = np.nan
            elif len(paths) == 1:
                good_path = paths[0]
                scales[idx], model = fit_2D_ellipse(good_path.vertices)
            else:
                scales[idx] = np.nan
                model = np.nan

            models.append(model)

    return scales, models


def WidthEstimate1D(inList, method='interpolate'):
    scales = np.zeros(len(inList))
    for idx, y in enumerate(inList):
        x = fft.fftfreq(len(y)) * len(y) / 2.0
        if method == 'interpolate':
            minima = argrelmin(y)[0]
            if minima[0] > 1:
                interpolator = interp1d(y[0:minima[0]], x[0:minima[0]])
                print minima
                print np.max(y[0:minima[0]])
                print np.min(y[0:minima[0]])
                plt.plot(y, 'r-')
                plt.plot(y[0:minima[0]], 'bD')
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
            scales[idx] = np.abs(output.stddev.value[0]) * (2**0.5)
    return scales


def fit_2D_ellipse(pts):
    '''
    Return ellipse widths
    '''

    ellip = EllipseModel()
    ellip.estimate(pts)

    return np.mean(ellip.params[2:4]), ellip


def plot_stuff(raw, fit, residual, n_eigs):
    pass
