# Licensed under an MIT open source license - see LICENSE

import numpy as np
from scipy.stats import ks_2samp, lognorm  # , anderson_ksamp
from statsmodels.distributions.empirical_distribution import ECDF
from statsmodels.base.model import GenericLikelihoodModel
from warnings import warn

from ..stats_utils import hellinger, common_histogram_bins, data_normalization
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types, threed_types, input_data


class PDF(BaseStatisticMixIn):
    '''
    Create the PDF of a given array.

    Parameters
    ----------
    img : %(dtypes)s
        A 1-3D array.
    min_val : float, optional
        Minimum value to keep in the given image.
    bins : list or numpy.ndarray or int, optional
        Bins to compute the PDF from.
    weights : %(dtypes)s, optional
        Weights to apply to the image. Must have the same shape as the image.
    use_standardized : bool, optional
        Enable to standardize the data before computing the PDF and ECDF.
    normalization_type : {"standardize", "center", "normalize",
                          "normalize_by_mean"}, optional
        See `~turbustat.statistics.stat_utils.data_normalization`.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types +
                                      threed_types)}

    def __init__(self, img, min_val=-np.inf, bins=None, weights=None,
                 normalization_type=None):
        super(PDF, self).__init__()

        self.need_header_flag = False
        self.header = None

        output_data = input_data(img, no_header=True)

        self.img = output_data

        # We want to remove NaNs and value below the threshold.
        keep_values = np.logical_and(np.isfinite(output_data),
                                     output_data > min_val)
        self.data = output_data[keep_values]

        # Do the same for the weights, then apply weights to the data.
        if weights is not None:
            output_weights = input_data(weights, no_header=True)

            self.weights = output_weights[keep_values]

            isfinite = np.isfinite(self.weights)

            self.data = self.data[isfinite] * self.weights[isfinite]

        if normalization_type is not None:
            self._normalization_type = normalization_type
            self.data = data_normalization(self.data,
                                           norm_type=normalization_type)
        else:
            self._normalization_type = "None"

        self._bins = bins

        self._pdf = None
        self._ecdf = None

    def make_pdf(self, bins=None):
        '''
        Create the PDF.

        Parameters
        ----------
        bins : list or numpy.ndarray or int, optional
            Bins to compute the PDF from. Overrides initial bin input.
        '''

        if bins is not None:
            self._bins = bins

        # If the number of bins is not given, use sqrt of data length.
        if self.bins is None:
            self._bins = np.sqrt(self.data.shape[0])
            self._bins = int(np.round(self.bins))

        # norm_weights = np.ones_like(self.data) / self.data.shape[0]

        self._pdf, bin_edges = np.histogram(self.data, bins=self.bins,
                                            density=True)#,
                                            # weights=norm_weights)

        self._bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    @property
    def normalization_type(self):
        return self._normalization_type

    @property
    def pdf(self):
        return self._pdf

    @property
    def bins(self):
        return self._bins

    def make_ecdf(self):
        '''
        Create the ECDF.
        '''

        if self.pdf is None:
            self.make_pdf()

        self._ecdf_function = ECDF(self.data)

        self._ecdf = self._ecdf_function(self.bins)

    @property
    def ecdf(self):
        return self._ecdf

    def find_percentile(self, values):
        '''
        Return the percentiles of given values from the
        data distribution.

        Parameters
        ----------
        values : float or np.ndarray
            Value or array of values.
        '''

        if self.ecdf is None:
            self.make_ecdf()

        return self._ecdf_function(values) * 100.

    def find_at_percentile(self, percentiles):
        '''
        Return the values at the given percentiles.

        Parameters
        ----------
        percentiles : float or np.ndarray
            Percentile or array of percentiles. Must be between 0 and 100.
        '''

        if np.any(np.logical_or(percentiles > 100, percentiles < 0.)):
            raise ValueError("Percentiles must be between 0 and 100.")

        return np.percentile(self.data, percentiles)

    def fit_pdf(self, model=lognorm, verbose=False,
                fit_type='mle', **kwargs):
        '''
        Fit a model to the PDF. Use statsmodel's generalized likelihood
        setup to get uncertainty estimates and such.

        Parameters
        ----------
        model : scipy.stats distribution, optional
            Pass any scipy
        verbose : bool, optional
            Enable printing of the fit results.
        fit_type : {'mle', 'mcmc'}, optional
            Type of fitting to use. By default Maximum Likelihood Estimation
             ('mle') is used. An MCMC approach ('mcmc') may also be used. This
             requires the optional `emcee` to be installed. kwargs can be
             passed to adjust various properties of the MCMC chain.
        kwargs : Passed to `emcee.XXX`
        '''

        if fit_type not in ['mle', 'mcmc']:
            raise ValueError("fit_type must be 'mle' or 'mcmc'.")


        class Likelihood(GenericLikelihoodModel):

            # Get the number of parameters from shapes.
            # Add one for scales, since we're assuming loc is frozen.
            # Keeping loc=0 is appropriate for log-normal models.
            nparams = 1 if model.shapes is None else \
                len(model.shapes.split(",")) + 1

            def loglike(self, params):
                if np.isnan(params).any():
                    return - np.inf

                loglikes = \
                    model.logpdf(self.endog, *params[:-1],
                                 scale=params[-1],
                                 loc=0)

                if not np.isfinite(loglikes).all():
                    return - np.inf

                else:
                    return loglikes.sum()

        def emcee_fit(model, init_params, burnin=200, steps=2000, thin=10):

            try:
                import emcee
            except ImportError:
                raise ImportError("emcee must be installed for MCMC fitting.")

            ndim = len(init_params)
            nwalkers = ndim * 10
            p0 = np.zeros((nwalkers, ndim))
            for i, val in enumerate(init_params):
                p0[:, i] = np.random.randn(nwalkers) * 0.1 + val
            sampler = emcee.EnsembleSampler(nwalkers,
                                            ndim,
                                            model.loglike,
                                            args=[])
            pos, prob, state = sampler.run_mcmc(p0, burnin)
            sampler.reset()
            pos, prob, state = sampler.run_mcmc(pos, steps, thin=thin)

            return sampler

        # Do an initial fit with the scipy model
        init_params = model.fit(self.data, floc=0.0)
        init_params = np.append(init_params[:-2], init_params[-1])

        self._model = Likelihood(self.data)

        if fit_type == 'mle':
            fitted_model = \
                self._model.fit(start_params=init_params, method='nm')
            self._mle_fit = fitted_model
            fitted_model.df_model = len(init_params)
            fitted_model.df_resid = len(self.data) - len(init_params)

            self._model_params = fitted_model.params.copy()
            try:
                self._model_stderrs = fitted_model.bse.copy()
                cov_calc_failed = False
            except ValueError:
                warn("Variance calculation failed.")
                self._model_stderrs = np.ones_like(self.model_params) * np.NaN
                cov_calc_failed = True
        elif fit_type == 'mcmc':
            chain = emcee_fit(self._model,
                              init_params.copy(),
                              **kwargs)
            self._model_params = np.mean(chain.flatchain, axis=0)
            self._model_stderrs = np.percentile(chain.flatchain, [15, 85],
                                                axis=0)
            self._mcmc_chain = chain

        if verbose:
            if fit_type == 'mle':
                if cov_calc_failed:
                    print("Fitted parameters: {}".format(self.model_params))
                    print("Covariance calculation failed.")
                else:
                    print(fitted_model.summary())
            else:
                print("Ran chain for {0} iterations".format(chain.iterations))
                print("Used {} walkers".format(chain.acceptance_fraction.size))
                print("Mean acceptance fraction of {}"
                      .format(np.mean(chain.acceptance_fraction)))
                print("Parameter values: {}".format(self.model_params))
                print("15th to 85th percentile ranges: {}"
                      .format(self.model_stderrs[1] - self.model_stderrs[0]))

    @property
    def model_params(self):
        if hasattr(self, "_model_params"):
            return self._model_params
        raise Exception("Not model has been fit. Run `fit_pdf` first.")

    @property
    def model_stderrs(self):
        if hasattr(self, "_model_stderrs"):
            return self._model_stderrs
        raise Exception("Not model has been fit. Run `fit_pdf` first.")

    def corner_plot(self, **kwargs):
        '''
        Create a corner plot from the MCMC. Requires the 'corner' package.
        '''

        if not hasattr(self, "_mcmc_chain"):
            raise Exception("Must run MCMC fitting first.")

        try:
            import corner
        except ImportError:
            raise ImportError("The optional package 'corner' is not "
                              "installed.")

        corner.corner(self._mcmc_chain.flatchain, **kwargs)

    def run(self, verbose=False, bins=None, do_fit=True,
            model=lognorm, **kwargs):
        '''
        Compute the PDF and ECDF. Enabling verbose provides
        a summary plot.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting of the results.
        bins : list or numpy.ndarray or int, optional
            Bins to compute the PDF from. Overrides initial bin input.
        '''

        self.make_pdf(bins=bins)
        self.make_ecdf()

        if do_fit:
            self.fit_pdf(model=model, verbose=verbose, **kwargs)

        if verbose:

            import matplotlib.pyplot as p

            if self.normalization_type == "standardize":
                xlabel = r"z-score"
            elif self.normalization_type == "center":
                xlabel = r"$I - \bar{I}$"
            elif self.normalization_type == "normalize_by_mean":
                xlabel = r"$I/\bar{I}$"
            else:
                xlabel = r"Intensity"

            # PDF
            p.subplot(121)
            p.semilogy(self.bins, self.pdf, 'b-', label='Data')
            if do_fit:
                # Plot the fitted model.
                vals = np.linspace(self.bins[0], self.bins[-1], 1000)
                p.semilogy(vals,
                           model.pdf(vals, *self.model_params[:-1],
                                     scale=self.model_params[-1],
                                     loc=0), 'r--', label='Fit')
                p.legend(loc='best')
            # else:
            #     p.loglog(self.bins, self.pdf, 'b-')
            p.grid(True)
            p.xlabel(xlabel)
            p.ylabel("PDF")

            # ECDF
            ax2 = p.subplot(122)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            if self.normalization_type != "None":
                ax2.plot(self.bins, self.ecdf, 'b-')
                if do_fit:
                    ax2.plot(vals,
                             model.cdf(vals, *self.model_params[:-1],
                                       scale=self.model_params[-1],
                                       loc=0), 'r--')
            else:
                ax2.semilogx(self.bins, self.ecdf, 'b-')
                if do_fit:
                    ax2.semilogx(vals,
                                 model.cdf(vals, *self.model_params[:-1],
                                           scale=self.model_params[-1],
                                           loc=0), 'r--')
            p.grid(True)
            p.xlabel(xlabel)
            p.ylabel("ECDF")

            p.tight_layout()
            p.show()

        return self


class PDF_Distance(object):
    '''
    Calculate the distance between two arrays using their PDFs.

    Parameters
    ----------
    img1 : %(dtypes)s
        Array (1-3D).
    img2 : %(dtypes)s
        Array (1-3D).
    min_val1 : float, optional
        Minimum value to keep in img1
    min_val2 : float, optional
        Minimum value to keep in img2
    weights1 : %(dtypes)s, optional
        Weights to be used with img1
    weights2 : %(dtypes)s, optional
        Weights to be used with img2
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types +
                                      threed_types)}

    def __init__(self, img1, img2, min_val1=-np.inf, min_val2=-np.inf,
                 normalization_type="standardize", nbins=None,
                 weights1=None, weights2=None):
        super(PDF_Distance, self).__init__()

        self.PDF1 = PDF(img1, min_val=min_val1,
                        normalization_type=normalization_type,
                        weights=weights1)

        self.PDF2 = PDF(img2, min_val=min_val2,
                        normalization_type=normalization_type,
                        weights=weights2)

        self.bins, self.bin_centers = \
            common_histogram_bins(self.PDF1.data, self.PDF2.data,
                                  return_centered=True, nbins=nbins)

        # Feed the common set of bins to be used in the PDFs
        self.PDF1.run(verbose=False, bins=self.bins)
        self.PDF2.run(verbose=False, bins=self.bins)

    def compute_hellinger_distance(self):
        '''
        Computes the Hellinger Distance between the two PDFs.
        '''

        self.hellinger_distance = hellinger(self.PDF1.pdf, self.PDF2.pdf)

    def compute_ks_distance(self):
        '''
        Compute the distance using the KS Test.
        '''

        D, p = ks_2samp(self.PDF1.data, self.PDF2.data)

        self.ks_distance = D
        self.ks_pval = p

    def compute_ad_distance(self):
        '''
        Compute the distance using the Anderson-Darling Test.
        '''

        raise NotImplementedError(
            "Use of the Anderson-Darling test has been disabled"
            " due to occurence of overflow errors.")

        # D, _, p = anderson_ksamp([self.PDF1.data, self.PDF2.data])

        # self.ad_distance = D
        # self.ad_pval = p

    def distance_metric(self, statistic='all', verbose=False,
                        label1=None, label2=None,
                        show_data=True):
        '''
        Calculate the distance.
        *NOTE:* The data are standardized before comparing to ensure the
        distance is calculated on the same scales.

        Parameters
        ----------
        statistic : 'all', 'hellinger', 'ks'
            Which measure of distance to use.
        labels : tuple, optional
            Sets the labels in the output plot.
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for img1
        label2 : str, optional
            Object or region name for img2
        show_data : bool, optional
            Plot the moment0, image, or 1D data.
        '''

        if statistic is 'all':
            self.compute_hellinger_distance()
            self.compute_ks_distance()
            # self.compute_ad_distance()
        elif statistic is 'hellinger':
            self.compute_hellinger_distance()
        elif statistic is 'ks':
            self.compute_ks_distance()
        # elif statistic is 'ad':
        #     self.compute_ad_distance()
        else:
            raise TypeError("statistic must be 'all',"
                            "'hellinger', or 'ks'.")
                            # "'hellinger', 'ks' or 'ad'.")

        if verbose:
            import matplotlib.pyplot as p
            # PDF
            p.subplot(121)
            p.plot(self.bin_centers,
                   self.PDF1.pdf, 'b-', label=label1)
            p.plot(self.bin_centers,
                   self.PDF2.pdf, 'g-', label=label2)
            p.legend(loc="best")
            p.grid(True)
            p.xlabel(r"z-score")
            p.ylabel("PDF")

            # ECDF
            ax2 = p.subplot(122)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            ax2.plot(self.bin_centers, self.PDF1.ecdf, 'b-')
            ax2.plot(self.bin_centers, self.PDF2.ecdf, 'g-')
            p.grid(True)
            p.xlabel(r"z-score")
            p.ylabel("ECDF")

            p.show()

        return self
