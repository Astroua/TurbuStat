# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import astropy.units as u
from astropy.utils import NumpyRNGContext

'''
Routines for fitting a line with errors in both variables.
'''


def leastsq_linear(x, y, x_err, y_err, verbose=False):
    '''
    Fit a line with errors in both variables using least squares fitting.
    Specifically, this uses orthogonal distance regression (in scipy) since
    there is no clear "independent" variable here and the covariance is
    unknown (or at least difficult to estimate).

    Parameters
    ----------
    x : `~numpy.ndarray`
        x data.
    y : `~numpy.ndarray`
        y data.
    x_err : `~numpy.ndarray`
        x errors.
    y_err : `~numpy.ndarray`
        y errors.
    verbose : bool, optional
        Plot the resulting fit.

    Returns
    -------
    params : `~numpy.ndarray`
        Fit parameters (slope, intercept)
    errors : `~numpy.ndarray`
        1-sigma errors from the covariance matrix (slope, intercept).
    '''

    import scipy.odr as odr
    from scipy.stats import linregress

    def fit_func(B, x):
        return B[0] * x + B[1]

    linear = odr.Model(fit_func)

    mydata = odr.Data(x, y, wd=np.power(x_err, -2), we=np.power(y_err, -2))

    # Set the initial guess for ODR from a normal linear regression
    beta0 = linregress(x, y)[:2]

    # beta sets the initial parameters
    myodr = odr.ODR(mydata, linear, beta0=beta0)

    output = myodr.run()

    params = output.beta
    errors = output.sd_beta

    # found a source saying this equivalent to reduced chi-square. Not sure if
    # this is true... Bootstrapping is likely a better way to go.
    # gof = output.res_var

    if verbose:
        output.pprint()

        import matplotlib.pyplot as plt

        fig = plt.figure()

        ax = plt.subplot2grid((4, 1), (0, 0), colspan=1, rowspan=3)
        ax_r = plt.subplot2grid((4, 1), (3, 0), colspan=1,
                                rowspan=1,
                                sharex=ax)

        ax.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='k')
        ax.set_ylabel("log Spectral Length")
        xvals = np.linspace(x.min(), x.max(), x.size * 10)
        ax.plot(xvals, params[0] * xvals + params[1], 'r-')
        ax.fill_between(xvals,
                        (params[0] - errors[0]) * xvals +
                        (params[1] - errors[1]),
                        (params[0] + errors[0]) * xvals +
                        (params[1] + errors[1]),
                        facecolor='red', interpolate=True, alpha=0.4)
        # Some very large error bars makes it difficult to see the model
        y_range = np.ptp(y)
        x_range = np.ptp(x)
        ax.set_ylim([y.min() - y_range / 4, y.max() + y_range / 4])
        ax.set_xlim([x.min() - x_range / 4, x.max() + x_range / 4])

        ax_r.errorbar(x, y - (params[0] * x + params[1]),
                      xerr=x_err, yerr=y_err, fmt='o', color='k')
        ax_r.axhline(0., color='red', linestyle='-', alpha=0.7)
        ax_r.set_ylabel("Residuals")

        ax_r.set_xlabel("log Spatial Length")

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        plt.show()

    return params, errors


def bayes_linear(x, y, x_err, y_err, nWalkers=10, nBurn=100, nSample=1000,
                 conf_interval=[15.9, 84.1], verbose=False,
                 return_samples=False):
    '''
    Fit a line with errors in both variables using MCMC.

    Original version of this function is Erik Rosolowsky's:
    https://github.com/low-sky/py-low-sky/blob/master/BayesLinear.py

    Parameters
    ----------
    x : `~numpy.ndarray`
        x data.
    y : `~numpy.ndarray`
        y data.
    x_err : `~numpy.ndarray`
        x errors.
    y_err : `~numpy.ndarray`
        y errors.
    nWalkers : int, optional
        Number of walkers in the sampler (>2 required). Defaults to 10.
    nBurn : int, optional
        Number of steps to burn chain in for. Default is 100.
    nSample : int, optional
        Number of steps to sample chain with. Default is 1000.
    conf_interval : list, optional
        Upper and lower percentiles to estimate the bounds on the parameters.
        Defaults to the 1-sigma intervals (34.1% about the median).
    verbose : bool, optional
        Plot the resulting fit.
    return_samples : bool, optional
        Returns the entire chain of samples, when enabled.

    Returns
    -------
    params : `~numpy.ndarray`
        Fit parameters (slope, intercept)
    errors : `~numpy.ndarray`
        Confidence interval defined by values given in `conf_interval`
        (slope, intercept).
    samples : `~numpy.ndarray`
        Samples from the chain. Returned only when `return_samples` is enabled.

    '''

    try:
        import emcee
    except ImportError:
        raise ImportError("emcee must be installed to use Bayesian fitting.")

    def _logprob(p, x, y, x_err, y_err):
        theta, b = p[0], p[1]
        if np.abs(theta - np.pi / 4) > np.pi / 4:
            return -np.inf
        Delta = (np.cos(theta) * y - np.sin(theta) * x - b * np.cos(theta))**2
        Sigma = (np.sin(theta))**2 * x_err**2 + (np.cos(theta))**2 * y_err**2
        lp = -0.5 * np.nansum(Delta / Sigma) - 0.5 * np.nansum(np.log(Sigma))

        return lp

    ndim = 2
    p0 = np.zeros((nWalkers, ndim))
    p0[:, 0] = np.pi / 4 + np.random.randn(nWalkers) * 0.1
    p0[:, 1] = np.random.randn(nWalkers) * y.std() + y.mean()

    sampler = emcee.EnsembleSampler(nWalkers, ndim, _logprob,
                                    args=[x, y, x_err, y_err])
    pos, prob, state = sampler.run_mcmc(p0, nBurn)
    sampler.reset()
    sampler.run_mcmc(pos, nSample)

    slopes = np.tan(sampler.flatchain[:, 0])
    intercepts = sampler.flatchain[:, 1]

    slope = np.median(slopes)
    intercept = np.median(intercepts)

    params = np.array([slope, intercept])

    # Use the percentiles given in conf_interval
    error_intervals = np.empty((2, 2))
    error_intervals[0] = np.percentile(slopes, conf_interval)
    error_intervals[1] = np.percentile(intercepts, conf_interval)

    if verbose:
        # Make some trace plots, PDFs and a plot of the range of solutions

        import matplotlib.pyplot as plt
        from astropy.visualization import hist

        fig = plt.figure(figsize=(9.9, 4.8))

        ax = plt.subplot2grid((4, 4), (0, 0), colspan=1, rowspan=2)
        ax.plot(slopes, 'k', linewidth=0.5)
        ax.set_ylabel("Slope")
        # ax.set_xlabel("Iteration")
        ax.get_xaxis().set_ticklabels([])

        ax2 = plt.subplot2grid((4, 4), (0, 1), colspan=1, rowspan=2)
        ax2.plot(intercepts, 'k', linewidth=0.5)
        ax2.set_ylabel("Intercept")
        # ax2.set_xlabel("Iteration")
        ax2.get_xaxis().set_ticklabels([])

        ax3 = plt.subplot2grid((4, 4), (2, 0), colspan=1, rowspan=2)
        hist(slopes, bins='knuth', color='k', alpha=0.6, ax=ax3)
        ax3.axvline(slope, color='r', linestyle='-')
        ax3.axvline(error_intervals[0][0], color='r', linestyle='--')
        ax3.axvline(error_intervals[0][1], color='r', linestyle='--')
        ax3.set_xlabel("Slope")

        ax4 = plt.subplot2grid((4, 4), (2, 1), colspan=1, rowspan=2)
        hist(intercepts, bins='knuth', color='k', alpha=0.6, ax=ax4)
        ax4.axvline(intercept, color='r', linestyle='-')
        ax4.axvline(error_intervals[1][0], color='r', linestyle='--')
        ax4.axvline(error_intervals[1][1], color='r', linestyle='--')
        ax4.set_xlabel("Intercept")

        ax5 = plt.subplot2grid((4, 4), (0, 2), colspan=2, rowspan=3)
        ax_r = plt.subplot2grid((4, 4), (3, 2), colspan=2,
                                rowspan=1,
                                sharex=ax5)

        ax5.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='k')
        ax5.set_ylabel("log Spectral Length")
        xvals = np.linspace(x.min(), x.max(), x.size * 10)
        ax5.plot(xvals, slope * xvals + intercept, 'r-')
        ax5.fill_between(xvals,
                         error_intervals[0, 0] * xvals + error_intervals[1, 0],
                         error_intervals[0, 1] * xvals + error_intervals[1, 1],
                         facecolor='red', interpolate=True, alpha=0.4)
        # ax5.get_xaxis().set_ticklabels([])

        # Some very large error bars makes it difficult to see the model
        y_range = np.ptp(y)
        x_range = np.ptp(x)
        ax5.set_ylim([y.min() - y_range / 4, y.max() + y_range / 4])
        ax5.set_xlim([x.min() - x_range / 4, x.max() + x_range / 4])

        ax_r.errorbar(x, y - (slope * x + intercept),
                      xerr=x_err, yerr=y_err, fmt='o', color='k')

        ax_r.axhline(0., color='red', linestyle='--', alpha=0.7)
        ax_r.set_ylabel("Residuals")
        ax_r.set_xlabel("log Spatial Length")

        print("Slope: {0} ({1}, {2})".format(slope, error_intervals[0, 0],
                                             error_intervals[0, 1]))
        print("Intercept: {0} ({1}, {2})".format(intercept,
                                                 error_intervals[1, 0],
                                                 error_intervals[1, 1]))

        plt.tight_layout()

        fig.subplots_adjust(hspace=0.1)

        plt.show()

    if return_samples:
        return params, error_intervals, np.vstack([slopes, intercepts])

    return params, error_intervals


def check_fit_limits(xlow, xhigh):
    '''
    Check that the inputs are floats (or ints), or a 2-element array for
    passing separate limits.

    Parameters
    ----------
    xlow : float or np.ndarray, optional
        The lower lag fitting limit. An array with 2 elements can be passed to
        give separate lower limits for the datasets.
    xhigh : float or np.ndarray, optional
        The upper lag fitting limit. See `xlow` above.

    '''

    if xlow is None:
        xlow = np.array([xlow] * 2)
    elif isinstance(xlow, u.Quantity):
        if xlow.isscalar:
            xlow = u.Quantity([xlow] * 2)

    if not len(xlow) == 2:
        raise ValueError("xlow must be a 2-element array when giving "
                         "separate fitting limits for each dataset.")

    if xhigh is None:
        xhigh = np.array([xhigh] * 2)
    elif isinstance(xhigh, u.Quantity):
        if xhigh.isscalar:
            xhigh = u.Quantity([xhigh] * 2)

    if not len(xhigh) == 2:
        raise ValueError("xhigh must be a 2-element array when giving "
                         "separate fitting limits for each dataset.")

    return xlow, xhigh


def clip_func(arr, low, high):
    return np.logical_and(arr > low, arr <= high)


def residual_bootstrap(fit_model, nboot=1000, seed=38574895,
                       return_samps=False, debug=False,
                       **fit_kwargs):
    '''
    Bootstrap with residual resampling.
    '''

    y = fit_model.model.wendog
    y_res = fit_model.wresid

    resamps = []

    if debug:
        import matplotlib.pyplot as plt

    with NumpyRNGContext(seed):

        for _ in range(nboot):

            y_resamp = y + y_res[np.random.choice(y_res.size - 1, y_res.size)]

            resamp_mod = fit_model.model.__class__(y_resamp,
                                                   fit_model.model.exog)
            resamp_fit = resamp_mod.fit(**fit_kwargs)

            if debug:
                plt.plot(fit_model.model.exog[:, 1], y, label='Data')
                plt.plot(fit_model.model.exog[:, 1], y_resamp, label='Resamp')
                plt.plot(resamp_fit.model.exog[:, 1], resamp_fit.model.endog,
                         label='Resamp Model')
                plt.legend()
                plt.draw()

                print(resamp_fit.params)

                input("?")
                plt.clf()

            resamps.append(resamp_fit.params)

    resamps = np.array(resamps).squeeze()

    if return_samps:
        return resamps

    return np.std(resamps, axis=0)
