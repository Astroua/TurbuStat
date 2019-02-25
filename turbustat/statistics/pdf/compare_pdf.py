# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

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
    normalization_type : {"standardize", "center", "normalize", "normalize_by_mean"}, optional
        See `~turbustat.statistics.stat_utils.data_normalization`.

    Examples
    --------
    >>> from turbustat.statistics import PDF
    >>> from astropy.io import fits
    >>> moment0 = fits.open("Design4_21_0_0_flatrho_0021_13co.moment0.fits")[0]  # doctest: +SKIP
    >>> pdf_mom0 = PDF(moment0).run(verbose=True)  # doctest: +SKIP

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

        self._do_fit = False

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
                                            density=True)
                                            # weights=norm_weights)

        self._bins = (bin_edges[:-1] + bin_edges[1:]) / 2

    @property
    def normalization_type(self):
        return self._normalization_type

    @property
    def pdf(self):
        '''
        PDF values in `~PDF.bins`.
        '''
        return self._pdf

    @property
    def bins(self):
        '''
        Bin centers.
        '''
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
        '''
        ECDF values in `~PDF.bins`.
        '''
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
                fit_type='mle', floc=True, loc=0.0, fscale=False, scale=1.0,
                **kwargs):
        '''
        Fit a model to the PDF. Use statsmodel's generalized likelihood
        setup to get uncertainty estimates and such.

        Parameters
        ----------
        model : scipy.stats distribution, optional
            Pass any scipy distribution. NOTE: All fits assume `loc` can be
            fixed to 0. This is reasonable for all realistic PDF forms in the
            ISM.
        verbose : bool, optional
            Enable printing of the fit results.
        fit_type : {'mle', 'mcmc'}, optional
            Type of fitting to use. By default Maximum Likelihood Estimation
             ('mle') is used. An MCMC approach ('mcmc') may also be used. This
             requires the optional `emcee` to be installed. kwargs can be
             passed to adjust various properties of the MCMC chain.
        floc : bool, optional
            Fix the `loc` parameter when fitting.
        loc : float, optional
            Value to set `loc` to when fixed.
        fscale : bool, optional
            Fix the `scale` parameter when fitting.
        scale : float, optional
            Value to set `scale` to when fixed.
        kwargs : Passed to `~emcee.EnsembleSampler`.
        '''

        if fit_type not in ['mle', 'mcmc']:
            raise ValueError("fit_type must be 'mle' or 'mcmc'.")

        self._fit_fixes = {"loc": [floc, loc], "scale": [fscale, scale]}

        self._do_fit = True

        class Likelihood(GenericLikelihoodModel):

            # Get the number of parameters from shapes.
            # Add one for scales, since we're assuming loc is frozen.
            # Keeping loc=0 is appropriate for log-normal models.
            nparams = 1 if model.shapes is None else \
                len(model.shapes.split(",")) + 1

            _loc = loc
            _scale = scale

            def loglike(self, params):
                if np.isnan(params).any():
                    return - np.inf

                if not floc and not fscale:
                    loc = params[-2]
                    scale = params[-1]
                    cut = -2
                elif not floc:
                    loc = params[-1]
                    scale = self._scale
                    cut = -1
                elif not fscale:
                    scale = params[-1]
                    loc = self._loc
                    cut = -1

                loglikes = \
                    model.logpdf(self.endog, *params[:cut],
                                 scale=scale,
                                 loc=loc)

                if not np.isfinite(loglikes).all():
                    return -np.inf

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
        if floc and fscale:
            init_params = model.fit(self.data)
        elif floc:
            init_params = model.fit(self.data, floc=loc)
            # Remove loc from the params
            init_params = np.append(init_params[:-2], init_params[-1])
        elif fscale:
            init_params = model.fit(self.data, fscale=scale)
            # Remove scale from the params
            init_params = np.append(init_params[:-2], init_params[-2])
        else:
            init_params = model.fit(self.data)

        init_params = np.array(init_params)

        self._model = Likelihood(self.data)

        self._scipy_model = model

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
        '''
        Parameters of the fitted model.
        '''
        if hasattr(self, "_model_params"):
            return self._model_params
        raise Exception("No model has been fit. Run `fit_pdf` first.")

    @property
    def model_stderrs(self):
        '''
        Standard errors of the fitted model. If using an MCMC, the 15th and
        85th percentiles are returned.
        '''
        if hasattr(self, "_model_stderrs"):
            return self._model_stderrs
        raise Exception("No model has been fit. Run `fit_pdf` first.")

    def corner_plot(self, **kwargs):
        '''
        Create a corner plot from the MCMC. Requires the 'corner' package.

        Parameters
        ----------
        kwargs : Passed to `~corner.corner`.
        '''

        if not hasattr(self, "_mcmc_chain"):
            raise Exception("Must run MCMC fitting first.")

        try:
            import corner
        except ImportError:
            raise ImportError("The optional package 'corner' is not "
                              "installed.")

        corner.corner(self._mcmc_chain.flatchain, **kwargs)

    def trace_plot(self, **kwargs):
        '''
        Create a trace plot from the MCMC.

        Parameters
        ----------
        kwargs : Passed to `~matplotlib.pyplot.plot`.
        '''

        if not hasattr(self, "_mcmc_chain"):
            raise Exception("Must run MCMC fitting first.")

        npars = self._mcmc_chain.flatchain.shape[1]

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(npars, 1, sharex=True)
        for i, ax in enumerate(axes.ravel()):
            ax.plot(self._mcmc_chain.flatchain[:, i], **kwargs)
            ax.set_ylabel("par{}".format(i + 1))

        axes.ravel()[-1].set_xlabel("Iterations")

        plt.tight_layout()

    def plot_distrib(self, save_name=None, color='r', fit_color='k',
                     show_ecdf=True):
        '''
        Plot the PDF distribution and (if fitted) the best fit model.
        Optionally show the ECDF and fit ECDF, too.

        Parameters
        ----------
        save_name : str,optional
            Save the figure when a file name is given.
        color : {str, RGB tuple}, optional
            Color to show the Genus curves in.
        fit_color : {str, RGB tuple}, optional
            Color of the fitted line. Defaults to `color` when no input is
            given.
        show_ecdf : bool, optional
            Plot the ECDF when enabled.
        '''

        import matplotlib.pyplot as plt

        if self.normalization_type == "standardize":
            xlabel = r"z-score"
        elif self.normalization_type == "center":
            xlabel = r"$I - \bar{I}$"
        elif self.normalization_type == "normalize_by_mean":
            xlabel = r"$I/\bar{I}$"
        else:
            xlabel = r"Intensity"

        if fit_color is None:
            fit_color = color

        # PDF
        if show_ecdf:
            plt.subplot(121)
        else:
            plt.subplot(111)

        plt.semilogy(self.bins, self.pdf, '-', color=color, label='Data')

        if self._do_fit:
            # Plot the fitted model.
            vals = np.linspace(self.bins[0], self.bins[-1], 1000)

            # Check which of the parameters were kept fixed
            if self._fit_fixes['loc'][0] and self._fit_fixes['scale'][0]:
                loc = self._fit_fixes['loc'][1]
                scale = self._fit_fixes['scale'][1]
                params = self.model_params
            elif self._fit_fixes['loc'][0]:
                loc = self._fit_fixes['loc'][1]
                scale = self.model_params[-1]
                params = self.model_params[:-1]
            elif self._fit_fixes['scale'][0]:
                loc = self.model_params[-1]
                scale = self._fit_fixes['scale'][1]
                params = self.model_params[:-1]
            else:
                loc = self.model_params[-2]
                scale = self.model_params[-1]
                params = self.model_params[:-2]

            plt.semilogy(vals,
                         self._scipy_model.pdf(vals, *params,
                                               scale=scale,
                                               loc=loc),
                         '--', color=fit_color, label='Fit')
            plt.legend(loc='best')

        plt.grid(True)
        plt.xlabel(xlabel)
        plt.ylabel("PDF")

        # ECDF
        if show_ecdf:
            ax2 = plt.subplot(122)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            if self.normalization_type != "None":
                ax2.plot(self.bins, self.ecdf, '-', color=color)
                if self._do_fit:
                    ax2.plot(vals,
                             self._scipy_model.cdf(vals, *params,
                                                   scale=scale,
                                                   loc=loc),
                             '--', color=fit_color)
            else:
                ax2.semilogx(self.bins, self.ecdf, '-', color=color)
                if self._do_fit:
                    ax2.semilogx(vals,
                                 self._scipy_model.cdf(vals, *params,
                                                       scale=scale,
                                                       loc=0),
                                 '--', color=fit_color)
            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel("ECDF")

        plt.tight_layout()

        if save_name is not None:
            plt.savefig(save_name)
            plt.close()
        else:
            plt.show()

    def run(self, verbose=False, save_name=None, bins=None, do_fit=True,
            model=lognorm, color=None, **kwargs):
        '''
        Compute the PDF and ECDF. Enabling verbose provides
        a summary plot.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting of the results.
        save_name : str,optional
            Save the figure when a file name is given.
        bins : list or numpy.ndarray or int, optional
            Bins to compute the PDF from. Overrides initial bin input.
        do_fit : bool, optional
            Enables (by default) fitting a given model.
        model : scipy.stats distribution, optional
            Pass any scipy distribution. See `~PDF.fit_pdf`.
        color : {str, RGB tuple}, optional
            Color to show the Genus curves in when `verbose=True`.
        kwargs : Passed to `~PDF.fit_pdf`.
        '''

        self.make_pdf(bins=bins)
        self.make_ecdf()

        if do_fit:
            self.fit_pdf(model=model, verbose=verbose, **kwargs)

        if verbose:
            self.plot_distrib(save_name=save_name, color=color)

        return self


class PDF_Distance(object):
    '''
    Calculate the distance between two arrays using their PDFs.

    .. note:: Pre-computed `~PDF` classes cannot be passed to `~PDF_Distance`
              as the data need to be normalized and the PDFs should use the
              same set of histogram bins.

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
    do_fit : bool, optional
        Enables fitting a lognormal distribution to each data set.
    normalization_type : {"normalize", "normalize_by_mean"}, optional
        See `~turbustat.statistics.stat_utils.data_normalization`.
    nbins : int, optional
        Manually set the number of bins to use for creating the PDFs.
    weights1 : %(dtypes)s, optional
        Weights to be used with img1
    weights2 : %(dtypes)s, optional
        Weights to be used with img2
    bin_min : float, optional
        Minimum value to use for the histogram bins *after* normalization is
        applied.
    bin_max : float, optional
        Maximum value to use for the histogram bins *after* normalization is
        applied.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types +
                                      threed_types)}

    def __init__(self, img1, img2, min_val1=-np.inf, min_val2=-np.inf,
                 do_fit=True, normalization_type=None,
                 nbins=None, weights1=None, weights2=None,
                 bin_min=None, bin_max=None):
        super(PDF_Distance, self).__init__()

        if do_fit:
            if normalization_type in ["standardize", "center"]:
                raise Exception("Cannot perform lognormal fit when using"
                                " 'standardize' or 'center'.")

        self.normalization_type = normalization_type

        self.PDF1 = PDF(img1, min_val=min_val1,
                        normalization_type=normalization_type,
                        weights=weights1)

        self.PDF2 = PDF(img2, min_val=min_val2,
                        normalization_type=normalization_type,
                        weights=weights2)

        self.bins, self.bin_centers = \
            common_histogram_bins(self.PDF1.data, self.PDF2.data,
                                  return_centered=True, nbins=nbins,
                                  min_val=bin_min, max_val=bin_max)

        # Feed the common set of bins to be used in the PDFs
        self._do_fit = do_fit
        self.PDF1.run(verbose=False, bins=self.bins, do_fit=do_fit)
        self.PDF2.run(verbose=False, bins=self.bins, do_fit=do_fit)

    def compute_hellinger_distance(self):
        '''
        Computes the Hellinger Distance between the two PDFs.
        '''

        # We're using the same bins, so normalize each to unity to keep the
        # distance normalized.
        self.hellinger_distance = \
            hellinger(self.PDF1.pdf / self.PDF1.pdf.sum(),
                      self.PDF2.pdf / self.PDF2.pdf.sum())

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

    def compute_lognormal_distance(self):
        '''
        Compute the combined t-statistic for the difference in the widths of
        a lognormal distribution.
        '''

        try:
            self.PDF1.model_params
            self.PDF2.model_params
        except AttributeError:
            raise Exception("Fitting has not been performed. 'do_fit' must "
                            "first be enabled.")

        diff = np.abs(self.PDF1.model_params[0] - self.PDF2.model_params[0])
        denom = np.sqrt(self.PDF1.model_stderrs[0]**2 +
                        self.PDF2.model_stderrs[0]**2)

        self.lognormal_distance = diff / denom

    def distance_metric(self, statistic='all', verbose=False,
                        plot_kwargs1={'color': 'b', 'marker': 'D',
                                      'label': '1'},
                        plot_kwargs2={'color': 'g', 'marker': 'o',
                                      'label': '2'},
                        save_name=None):
        '''
        Calculate the distance.
        *NOTE:* The data are standardized before comparing to ensure the
        distance is calculated on the same scales.

        Parameters
        ----------
        statistic : 'all', 'hellinger', 'ks', 'lognormal'
            Which measure of distance to use.
        labels : tuple, optional
            Sets the labels in the output plot.
        verbose : bool, optional
            Enables plotting.
        plot_kwargs1 : dict, optional
            Pass kwargs to `~matplotlib.pyplot.plot` for
            `dataset1`.
        plot_kwargs2 : dict, optional
            Pass kwargs to `~matplotlib.pyplot.plot` for
            `dataset2`.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        if statistic is 'all':
            self.compute_hellinger_distance()
            self.compute_ks_distance()
            # self.compute_ad_distance()
            if self._do_fit:
                self.compute_lognormal_distance()
        elif statistic is 'hellinger':
            self.compute_hellinger_distance()
        elif statistic is 'ks':
            self.compute_ks_distance()
        elif statistic is 'lognormal':
            if not self._do_fit:
                raise Exception("Fitting must be enabled to compute the"
                                " lognormal distance.")
            self.compute_lognormal_distance()
        # elif statistic is 'ad':
        #     self.compute_ad_distance()
        else:
            raise TypeError("statistic must be 'all',"
                            "'hellinger', 'ks', or 'lognormal'.")
                            # "'hellinger', 'ks' or 'ad'.")

        if verbose:

            import matplotlib.pyplot as plt

            defaults1 = {'color': 'b', 'marker': 'D', 'label': '1'}
            defaults2 = {'color': 'g', 'marker': 'o', 'label': '2'}

            for key in defaults1:
                if key not in plot_kwargs1:
                    plot_kwargs1[key] = defaults1[key]
            for key in defaults2:
                if key not in plot_kwargs2:
                    plot_kwargs2[key] = defaults2[key]

            if self.normalization_type == "standardize":
                xlabel = r"z-score"
            elif self.normalization_type == "center":
                xlabel = r"$I - \bar{I}$"
            elif self.normalization_type == "normalize_by_mean":
                xlabel = r"$I/\bar{I}$"
            else:
                xlabel = r"Intensity"

            # Print fit summaries if using fitting
            if self._do_fit:
                try:
                    print(self.PDF1._mle_fit.summary())
                except ValueError:
                    warn("Covariance calculation failed. Check the fit quality"
                         " for data set 1!")
                try:
                    print(self.PDF2._mle_fit.summary())
                except ValueError:
                    warn("Covariance calculation failed. Check the fit quality"
                         " for data set 2!")

            # PDF
            plt.subplot(121)
            plt.semilogy(self.bin_centers, self.PDF1.pdf,
                         color=plot_kwargs1['color'], linestyle='none',
                         marker=plot_kwargs1['marker'],
                         label=plot_kwargs1['label'])
            plt.semilogy(self.bin_centers, self.PDF2.pdf,
                         color=plot_kwargs2['color'], linestyle='none',
                         marker=plot_kwargs2['marker'],
                         label=plot_kwargs2['label'])
            if self._do_fit:
                # Plot the fitted model.
                vals = np.linspace(self.bin_centers[0], self.bin_centers[-1],
                                   1000)

                fit_params1 = self.PDF1.model_params
                plt.semilogy(vals,
                             lognorm.pdf(vals, *fit_params1[:-1],
                                         scale=fit_params1[-1],
                                         loc=0),
                             color=plot_kwargs1['color'], linestyle='-')

                fit_params2 = self.PDF2.model_params
                plt.semilogy(vals,
                             lognorm.pdf(vals, *fit_params2[:-1],
                                         scale=fit_params2[-1],
                                         loc=0),
                             color=plot_kwargs2['color'], linestyle='-')

            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel("PDF")
            plt.legend(frameon=True)

            # ECDF
            ax2 = plt.subplot(122)
            ax2.yaxis.tick_right()
            ax2.yaxis.set_label_position("right")
            if self.normalization_type is not None:
                ax2.plot(self.bin_centers, self.PDF1.ecdf,
                         color=plot_kwargs1['color'], linestyle='-',
                         marker=plot_kwargs1['marker'],
                         label=plot_kwargs1['label'])

                ax2.plot(self.bin_centers, self.PDF2.ecdf,
                         color=plot_kwargs2['color'], linestyle='-',
                         marker=plot_kwargs2['marker'],
                         label=plot_kwargs2['label'])

                if self._do_fit:
                    ax2.plot(vals,
                             lognorm.cdf(vals,
                                         *fit_params1[:-1],
                                         scale=fit_params1[-1],
                                         loc=0),
                             color=plot_kwargs1['color'], linestyle='-',)

                    ax2.plot(vals,
                             lognorm.cdf(vals,
                                         *fit_params2[:-1],
                                         scale=fit_params2[-1],
                                         loc=0),
                             color=plot_kwargs2['color'], linestyle='-',)

            else:
                ax2.semilogx(self.bin_centers, self.PDF1.ecdf,
                             color=plot_kwargs1['color'], linestyle='-',
                             marker=plot_kwargs1['marker'],
                             label=plot_kwargs1['label'])

                ax2.semilogx(self.bin_centers, self.PDF2.ecdf,
                             color=plot_kwargs2['color'], linestyle='-',
                             marker=plot_kwargs2['marker'],
                             label=plot_kwargs2['label'])

                if self._do_fit:
                    ax2.semilogx(vals,
                                 lognorm.cdf(vals, *fit_params1[:-1],
                                             scale=fit_params1[-1],
                                             loc=0),
                                 color=plot_kwargs1['color'], linestyle='-',)

                    ax2.semilogx(vals,
                                 lognorm.cdf(vals, *fit_params2[:-1],
                                             scale=fit_params2[-1],
                                             loc=0),
                                 color=plot_kwargs2['color'], linestyle='-',)

            plt.grid(True)
            plt.xlabel(xlabel)
            plt.ylabel("ECDF")

            plt.tight_layout()

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
