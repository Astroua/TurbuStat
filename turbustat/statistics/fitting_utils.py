
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

    if verbose:
        output.pprint()

    params = output.beta
    errors = output.sd_beta

    # found a source saying this equivalent to reduced chi-square. Not sure if
    # this is true... Bootstrapping is likely a better way to go.
    # gof = output.res_var

    return params, errors


def bayes_linear(x, y, x_err, y_err, nWalkers=10, nBurn=100, nSample=1000,
                 conf_interval=[15, 85], verbose=False):
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

    return params, error_intervals
