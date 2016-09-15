
import numpy as np

'''
Routines for fitting linear models with errors in both variables.
'''


def leastsq_linear(x, y, x_err, y_err, verbose=False):
    '''
    Fit a line with errors in both variables using least squares fitting.
    Specifically, this uses orthogonal distance regression (in scipy) since
    there is no clear "independent" variable here and the covariance is
    unknown (or at least difficult to estimate).

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

        import matplotlib.pyplot as p

        p.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='k')
        p.ylabel("log Spectral Length")
        p.xlabel("log Spatial Length")
        xvals = np.linspace(x.min(), x.max(), x.size * 10)
        p.plot(xvals, params[0] * xvals + params[1], 'r-')
        p.fill_between(xvals,
                       (params[0] - errors[0]) * xvals +
                       (params[1] - errors[1]),
                       (params[0] + errors[0]) * xvals +
                       (params[1] + errors[1]),
                       facecolor='red', interpolate=True, alpha=0.4)
        # Some very large error bars makes it difficult to see the model
        y_range = np.ptp(y)
        x_range = np.ptp(x)
        p.ylim([y.min() - y_range / 4, y.max() + y_range / 4])
        p.xlim([x.min() - x_range / 4, x.max() + x_range / 4])

        p.tight_layout()
        p.show()

    return params, errors


def bayes_linear(x, y, x_err, y_err, nWalkers=10, nBurn=100, nSample=1000,
                 conf_interval=[15, 85], verbose=False, return_samples=False):
    '''
    Fit a line with errors in both variables using MCMC.

    Original version of this function is Erik Rosolowsky's:
    https://github.com/low-sky/py-low-sky/blob/master/BayesLinear.py

    Parameters
    ----------
    X,Y -- Data vectors (same length, can be length 1)
    Xerror, Yerror -- 1 sigma errors for X,Y
    nWakers = 10 (default), number of walkers in the sampler (>2 required)
    nBurn = 100 (default), number of steps to burn in chain for
    nSample = 1000 (default), number of steps to sample chain with

    Returns:
    samples = a numpy array of samples of the ratio of the data

    Examples
    --------
    samples = BayesRatio(X, Y, Xerror, Yerror, nWalkers=100,
                         nBurn=100,nSample=1000)

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

        import matplotlib.pyplot as p
        from astropy.visualization import hist

        p.subplot(2, 3, 1)
        p.plot(slopes, 'k', linewidth=0.5)
        p.title("Slope Values")
        p.xlabel("Iteration")\

        p.subplot(2, 3, 2)
        p.plot(intercepts, 'k', linewidth=0.5)
        p.title("Intercept Values")
        p.xlabel("Iteration")

        ax3 = p.subplot(2, 3, 4)
        hist(slopes, bins='knuth', color='k', alpha=0.6)
        ylow, yhigh = ax3.get_ylim()
        p.vlines(slope, ylow, yhigh, colors='r', linestyles='-')
        p.vlines(error_intervals[0], ylow, yhigh, colors='r', linestyles='--')
        p.xlabel("Slope")

        ax4 = p.subplot(2, 3, 5)
        hist(intercepts, bins='knuth', color='k', alpha=0.6)
        ylow, yhigh = ax4.get_ylim()
        p.vlines(intercept, ylow, yhigh, colors='r', linestyles='-')
        p.vlines(error_intervals[1], ylow, yhigh, colors='r', linestyles='--')
        p.xlabel("Intercept")

        p.subplot(1, 3, 3)
        p.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', color='k')
        p.ylabel("log Spectral Length")
        p.xlabel("log Spatial Length")
        xvals = np.linspace(x.min(), x.max(), x.size * 10)
        p.plot(xvals, slope * xvals + intercept, 'r-')
        p.fill_between(xvals,
                       error_intervals[0, 0] * xvals + error_intervals[1, 0],
                       error_intervals[0, 1] * xvals + error_intervals[1, 1],
                       facecolor='red', interpolate=True, alpha=0.4)
        # Some very large error bars makes it difficult to see the model
        y_range = np.ptp(y)
        x_range = np.ptp(x)
        p.ylim([y.min() - y_range / 4, y.max() + y_range / 4])
        p.xlim([x.min() - x_range / 4, x.max() + x_range / 4])

        print("Slope: {0} ({1}, {2})").format(slope, error_intervals[0, 0],
                                              error_intervals[0, 1])
        print("Intercept: {0} ({1}, {2})").format(intercept,
                                                  error_intervals[1, 0],
                                                  error_intervals[1, 1])

        p.tight_layout()
        p.show()

    if return_samples:
        return params, error_intervals, np.vstack([slopes, intercepts])

    return params, error_intervals
