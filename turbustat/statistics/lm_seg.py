
'''
Port of the R seg.lm.fit function.
From the pysegmented package: https://github.com/e-koch/pysegmented

The MIT License (MIT)

Copyright (c) 2014 Eric Koch

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import statsmodels.api as sm
import numpy as np
import warnings
from copy import copy


class Lm_Seg(object):
    """

    """
    def __init__(self, x, y, brk):
        super(Lm_Seg, self).__init__()
        self.x = x
        self.y = y
        self.brk = brk

        if not np.isfinite(self.brk):
            raise ValueError("brk must be a finite value.")

        # Make sure the starting break point is in range of the data
        if not (self.x > self.brk).any():
            raise ValueError("brk is outside the range.")

        # Check for nans, infs...
        if not np.isfinite(x).all():
            self.y = self.y[np.isfinite(self.x)]
            self.x = self.x[np.isfinite(self.x)]

        if not np.isfinite(y).all():
            self.x = self.x[np.isfinite(self.y)]
            self.y = self.y[np.isfinite(self.y)]

    def fit_model(self, tol=1e-3, iter_max=100, h_step=2.0, epsil_0=10,
                  constant=True, verbose=True):
        '''
        '''
        # Fit a normal linear model to the data
        if constant:
            x_const = sm.add_constant(self.x)
            model = sm.OLS(self.y, x_const)
        else:
            model = sm.OLS(self.y, self.x)
        init_lm = model.fit()

        if verbose:
            print init_lm.summary()

        epsil = epsil_0

        # Before we get into the loop, make sure that this was a bad fit
        if epsil_0 < tol:
            warnings.warning('Initial epsilon is smaller than tolerance. \
                             The tolerance should be set smaller.')
            return init_lm

        # Sum of residuals
        dev_0 = np.sum(init_lm.resid**2.)

        # Catch cases where a break isn't necessary
        self.break_fail_flag = False

        # Count
        it = 0

        # Now loop through and minimize the residuals by changing where the
        # breaking point is.
        while np.abs(epsil) > tol:
            U = (self.x - self.brk) * (self.x > self.brk)
            V = deriv_max(self.x, self.brk)

            X_all = np.vstack([self.x, U, V]).T
            if constant:
                X_all = sm.add_constant(X_all)

            model = sm.OLS(self.y, X_all)
            fit = model.fit()

            beta = fit.params[2]  # Get coef
            gamma = fit.params[3]  # Get coef

            # Adjust the break point
            new_brk = copy(self.brk)
            new_brk += (h_step * gamma) / beta

            # If the new break point is outside of the allowed range, reset
            # the step size to half of the original, then try stepping again
            h_it = 0
            if not (self.x > new_brk).any() or (self.x > new_brk).all():
                while True:
                    # Remove step taken
                    new_brk -= (h_step * gamma) / beta
                    # Now half the step and try again.
                    h_step /= 2.0
                    new_brk += (h_step * gamma) / beta
                    h_it += 1
                    if (self.x > new_brk).any() and not (self.x > new_brk).all():
                        self.brk = new_brk
                        break
                    if h_it >= 5:
                        self.break_fail_flag = True
                        it = iter_max + 1
                        warnings.warn("Cannot find good step-size, assuming\
                                       break not needed")
                        break
            else:
                self.brk = new_brk

            dev_1 = np.sum(fit.resid**2.)

            epsil = (dev_1 - dev_0) / (dev_0 + 1e-3)

            dev_0 = dev_1

            if verbose:
                print "Iteration: %s/%s" % (it+1, iter_max)
                print fit.summary()
                print "Break Point: " + str(self.brk)
                print "Epsilon: " + str(epsil)

            it += 1

            if it > iter_max:
                warnings.warn("Max iterations reached. \
                               Result may not be minimized.")
                break

        if self.break_fail_flag:
            self.brk = self.x.max()

            X_all = sm.add_constant(self.x)
        else:
            # With the break point hopefully found, do a final good fit
            U = (self.x - self.brk) * (self.x > self.brk)
            V = deriv_max(self.x, self.brk)

            X_all = np.vstack([self.x, U, V]).T
            X_all = sm.add_constant(X_all)

        model = sm.OLS(self.y, X_all)
        self.fit = model.fit()
        self._params = self.fit.params
        self._errs = self.fit.bse

        if not self.break_fail_flag:
            self.brk_err = brk_errs(self.params, fit.cov_params())
        else:
            self.brk_err = 0.0

        self.get_slopes()

        return self

    def model(self, x=None, model_return=False):
        p = self.params

        trans_pt = np.abs(self.x-self.brk).argmin()

        mod_eqn = lambda k: p[0] + p[1]*k*(k < self.brk) + \
            ((p[1]+p[2])*k + (-p[2])*k[trans_pt])*(k >= self.brk)

        if model_return or x is None:
            return mod_eqn

        return mod_eqn(x)

    def get_slopes(self):
        '''
        '''
        # Deal with non-break case
        if self.break_fail_flag:
            self._slopes = np.asarray([self.params[1]])
            self._slope_errs = np.asarray([self.param_errs[1]])

            return self

        n_slopes = self.params[1:-2].shape[0]
        self._slopes = np.empty(n_slopes)
        self._slope_errs = np.empty(n_slopes)

        for s in range(n_slopes):
            if s == 0:
                self._slopes[s] = self.params[s+1]
                self._slope_errs[s] = self.param_errs[s+1]
            else:
                self._slopes[s] = self.params[s+1] + self._slopes[:s]
                self._slope_errs[s] = \
                    np.sqrt(self.param_errs[s+1]**2 + self._slope_errs[:s]**2)

        return self

    @property
    def slopes(self):
        return self._slopes

    @property
    def slope_errs(self):
        return self._slope_errs

    @property
    def params(self):
        return np.append(self._params, self.brk)

    @property
    def param_errs(self):
        return np.append(self._errs, self.brk_err)

    def plot(self, x, show_data=True):
        '''
        '''
        import matplotlib.pyplot as p

        if show_data:
            p.plot(self.x, self.y, 'bD')

        p.plot(x, self.model(x), 'g')

        p.grid(True)

        p.show()


def deriv_max(a, b, pow=1):
    if pow == 1:
        dum = -1 * np.ones(a.shape)
        dum[a < b] = 0
        return dum
    else:
        return -pow * np.max(a - b, axis=0) ** (pow-1)


def brk_errs(params, cov):
    '''
    Given the covariance matrix of the fits, calculate the standard
    error on the break.
    '''

    # Var gamma
    term1 = cov[3, 3]

    # Var beta * (beta/gamma)^2`
    term2 = cov[2, 2] * (params[3]/params[2])**2.

    # Correlation b/w gamma and beta
    term3 = 2 * cov[3, 2] * (params[3]/params[2])

    return np.sqrt(term1 + term2 + term3)
