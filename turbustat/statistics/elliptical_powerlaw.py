# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
from astropy.modeling import Fittable2DModel, Parameter, fitting
from warnings import warn


def fit_elliptical_powerlaw(values, x, y, p0, fit_method='LevMarq',
                            bootstrap=False, niters=100, alpha=0.6827,
                            debug=False, radial_weighting=False,
                            fix_ellip_params=False):
    '''
    General function for fitting the 2D elliptical power-law model.

    Gradient-based minimizers will have difficulty altering theta if the
    initial guess requires crossing a zero-point in either sin or cos.
    We fix this issue by fitting twice, where the second fit uses a theta
    offset by pi / 2 from the original guess. Whichever fit has the lowest
    residuals is returned.

    Parameters
    ----------
    values : `~numpy.ndarray`
        Values to be fitted. Should be in log10 scale.
    x : `~numpy.ndarray`
        x-position values.
    y : `~numpy.ndarray`
        y-position values.
    p0 : tuple
        Initial parameter estimates. Should contain:
        (log amplitude, ellip_transf, theta, slope).
    fit_method : {"LevMarq"}, optional
        Fitting method. The Levenberg-Marquardt fitter is the only option
        currently implemented.
    bootstrap : bool, optional
        Estimate parameter standard errors using bootstrap residual sampling.
        This method is only valid if the model is correct for the data, and
        the errors are uncorrelated.
    niters : int, optional
        Number of bootstrap iterations.
    alpha : float, optional
        Two-sided confidence interval for estimating standard errors from the
        bootstrap. Must be within 0 and 1. The default corresponds to a 1-sigma
        interval (~68%).
    debug : bool, optional
        Enable plotting histograms of the bootstrap parameter distributions.
    radial_weighting : bool, optional
        To account for the increasing number of samples at greater radii, the
        fit can be weighted by :math:`1/{\rm radius}` to emphasize the points
        at small radii. DO NOT enabled weighting when the field is elliptical!
        This will bias the fit parameters! Default is False.
    fix_ellip_params : bool, optional
        If the field is expected to be isotropic, the ellipticity and theta
        parameters can be fixed in the fit. This will help the fit since
        the isotropic case sits at the edge of the ellipticity parameter space
        and can be difficult to correctly converge to.

    Returns
    -------
    params : `~numpy.ndarray`
        Fitted parameters.
    stderrs : `~numpy.ndarray`
        Standard errors from the covariance matrix, or from bootstrapping.
     fit_model : `~LogEllipticalPowerLaw2D`
        The fitted power-law model.
     fitter : `~astropy.modeling.fitting.LevMarLSQFitter`
        The astropy fitter class.

    .. note:: `ellip_transf` is the ellipticity transformed by
    :math:`1 / (1 + exp(-ellip))`, which transforms the ellipticity from the
    bounded interval of [0, 1] onto [0, inf].

    '''

    # All values must be finite.
    if not np.isfinite(values).all():
        raise ValueError("values contains a non-finite value.")

    if fit_method == 'LevMarq':

        if radial_weighting:
            weights = 1 / np.sqrt(x**2 + y**2)
        else:
            weights = None

        if radial_weighting and not fix_ellip_params:
            warn("Radial weighting with the elliptical parameters left free "
                 "can bias the fit! Check the fit results carefully!")

        p0 = list(p0)
        if fix_ellip_params:
            p0[1] = np.inf

        model = LogEllipticalPowerLaw2D(*p0)

        if fix_ellip_params:
            model.ellip_transf.fixed = True
            model.theta.fixed = True

        fitter = fitting.LevMarLSQFitter()
        fit_model = fitter(model, x, y, values, weights=weights)

        resids = np.sum(np.abs(values - fit_model(x, y)))

        if not fix_ellip_params:
            # Fit again w/ theta offset by pi / 2
            # This is the dumbest way I found to get a good fit in theta
            p0_f = list(p0)
            p0_f[2] = (p0[2] + np.pi / 2.) % np.pi
            model_f = LogEllipticalPowerLaw2D(*p0_f)

            fitter_f = fitting.LevMarLSQFitter()
            fit_model_f = fitter(model_f, x, y, values, weights=weights)

            resids_f = np.sum(np.abs(values - fit_model_f(x, y)))

            if resids > resids_f:
                if debug:
                    print("Using theta flipped model fit!")
                fitter = fitter_f
                fit_model = fit_model_f

        # Use bootstrap re-sampling of the model residuals to estimate
        # 1-sigma CIs.
        if bootstrap:
            niters = int(niters)
            params = np.empty((4, niters))

            resid = values - fit_model(x, y)

            for i in range(niters):
                boot_fit = fitting.LevMarLSQFitter()

                resamp_vals = values + resid[np.random.permutation(resid.size)]

                boot_model = boot_fit(fit_model, x, y, resamp_vals)

                params[:, i] = boot_model.parameters

            percentiles = np.percentile(params,
                                        [100 * (0.5 - alpha / 2.),
                                         100 * (0.5 + alpha / 2.)],
                                        axis=1)

            if debug:
                import matplotlib.pyplot as plt

                plt.subplot(221)
                plt.title("log Amp")
                _ = plt.hist(params[0], bins='auto')
                plt.axvline(fit_model.parameters[0])
                plt.subplot(222)
                plt.title("trans ellip")
                _ = plt.hist(params[1], bins='auto')
                plt.axvline(fit_model.parameters[1])
                plt.subplot(223)
                plt.title("theta")
                _ = plt.hist(params[2], bins='auto')
                plt.axvline(fit_model.parameters[2])
                plt.subplot(224)
                plt.title("gamma")
                _ = plt.hist(params[3], bins='auto')
                plt.axvline(fit_model.parameters[3])

            stderrs = 0.5 * (percentiles[1] - percentiles[0])

        # Otherwise use the covariance matrix to get standard errors.
        # These WILL be underestimated! In many cases Lev-Marq won't return
        # the covariance matrix at all (though the fit is usually correct).
        else:
            # Try extracting the covariance matrix
            cov_matrix = fitter.fit_info.get('param_cov')
            # cov_matrix = None
            if cov_matrix is None:
                warn("Covariance matrix calculation failed. Check results "
                     "carefully.")
                stderrs = np.zeros((4,)) * np.NaN
            else:
                stderrs = np.sqrt(np.abs(np.diag(cov_matrix)))

        params = fit_model.parameters
    else:
        raise ValueError("'LevMarq' is the only implemented fitting method. "
                         "'fit_method' must be set to this.")

    return params, stderrs, fit_model, fitter


class LogEllipticalPowerLaw2D(Fittable2DModel):
    """
    Two-dimensional elliptical power-law fit in log-log space.

    Adapted from http://adsabs.harvard.edu/abs/2015A&A...580A..79T.

    The major axis is rotated onto the x-axis, then re-scaled to match the
    minor dimension. A circular power-law is applied to the rotated and
    scaled axes. These operations create a weird parameter space to search
    as the importance of the rotation depends on how close the ellipticity
    is to being circular (as it approaches 1). This introduces a constant
    gradient in theta, tending towards the circular case, and requires
    good initial parameters in both theta and ellip. We find that this
    can be improved by re-parameterizing ellip such that it is defined
    on the real line, instead of being bounded between 0 and 1. The
    transform is defined as :math:`log({\rm ellip}) - log(1 - {\rm ellip})`.
    The transformed value is the expected value (`ellip_transf`) for the model,
    not the ellipticity. The inverse transform is
    :math:`1 / (1 + {\rm e}^{-{\rm ellip_transf}}`.

    Parameters
    ----------
    x : np.ndarray
        x values.
    y : np.ndarray
        y values.
    logamplitude : float
        log10 amplitude of the model.
    ellip_transf : float
        Transformed ellipticity of the model, such that it is valid on the
        real line. The ellipticity is bounded between 0 (zero width in minor
        direction) and 1 (circular).
    theta : float
        Position angle of the major axis. Bounded between 0 and :math:`2 \pi`.
    gamma : float,
        Power-law index.
    """

    logamplitude = Parameter(default=0, bounds=(-10, None))
    ellip_transf = Parameter(default=1)
    theta = Parameter(default=0)
    gamma = Parameter(default=-1)

    @classmethod
    def evaluate(cls, x, y, logamplitude, ellip_transf, theta, gamma):

        # Transform the ellipticity parameter to its actual value
        ellip = 1. / (1 + np.exp(-ellip_transf))

        costhet = np.cos(theta)
        sinthet = np.sin(theta)

        q = ellip

        term1 = (q * costhet)**2 + sinthet**2
        term2 = 2 * (1 - q**2) * sinthet * costhet
        term3 = (q * sinthet)**2 + costhet**2

        model = logamplitude + 0.5 * gamma * \
            np.log10(x**2 * term1 + x * y * term2 + y**2 * term3)

        # The center is the zero-frequency term and will be excluded from the
        # fit.
        model[~np.isfinite(model)] = 0.0

        return model


def interval_transform(x, a, b):

    return np.log(x - a) - np.log(b - x)


def inverse_interval_transform(x_trans, a, b):

    return (b - a) / (1 + np.exp(-x_trans)) + a


def interval_transform_stderr(dx, x, a, b):
    '''
    Error propagation to transformed variable.
    '''

    return np.abs(1 / (x - a) + 1 / (b - x)) * dx


def inverse_interval_transform_stderr(dx_trans, x_trans, a, b):
    '''
    Error propagation to transformed variable.
    '''

    deriv = (b - a) * np.exp(-x_trans) / (1 + np.exp(-x_trans))**2

    return np.abs(deriv) * dx_trans
