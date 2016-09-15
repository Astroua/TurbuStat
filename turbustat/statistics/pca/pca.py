# Licensed under an MIT open source license - see LICENSE


import numpy as np
import astropy.units as u
from warnings import warn

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data, find_beam_width

# PCA utilities
from ..threeD_to_twoD import var_cov_cube
from .width_estimate import WidthEstimate1D, WidthEstimate2D

# Fitting utilities
from ..fitting_utils import bayes_linear, leastsq_linear


class PCA(BaseStatisticMixIn):

    '''
    Implementation of Principal Component Analysis (Heyer & Brunt, 2002)

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute. Defaults to using all eigenvalues.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, n_eigs=-1, distance=None):
        super(PCA, self).__init__()

        self.data, self.header = input_data(cube)

        self.spectral_shape = self.data.shape[0]

        if n_eigs == -1:
            self.n_eigs = self.spectral_shape
        elif n_eigs < -1 or n_eigs > self.spectral_shape or n_eigs == 0:
            raise Warning("n_eigs must be less than the number of velocity"
                          " channels ({}) or -1 for"
                          " all".format(self.spectral_shape))
        else:
            self.n_eigs = n_eigs

        if distance is not None:
            self.distance = distance

    @property
    def n_eigs(self):
        return self._n_eigs

    @n_eigs.setter
    def n_eigs(self, value):
        if value <= 0:
            raise ValueError("n_eigs must be > 0.")

        self._n_eigs = value

    def compute_pca(self, mean_sub=False):
        '''
        Create the covariance matrix and its eigenvalues.

        If `mean_sub` is disabled, the first eigenvalue is dominated by the
        mean of the data, not the variance.

        Parameters
        ----------
        mean_sub : bool, optional
            When enabled, subtracts the means of the channels before
            calculating the covariance. By default, this is disabled to
            match the Heyer & Brunt method.
        '''

        self.cov_matrix = var_cov_cube(self.data, mean_sub=mean_sub)

        all_eigsvals, eigvecs = np.linalg.eig(self.cov_matrix)
        eigvecs = eigvecs[np.argsort(all_eigsvals)[::-1]]
        all_eigsvals = np.sort(all_eigsvals)[::-1]  # Sort by maximum

        if mean_sub:
            self._total_variance = np.sum(all_eigsvals)
            self._var_prop = np.sum(all_eigsvals[:self.n_eigs]) / \
                self.total_variance
        else:
            self._total_variance = np.sum(all_eigsvals[1:])
            self._var_prop = np.sum(all_eigsvals[1:self.n_eigs]) / \
                self.total_variance

        self._eigvals = all_eigsvals[:self.n_eigs]
        self._eigvecs = eigvecs[:, :self.n_eigs]

        self._mean_sub = mean_sub

    @property
    def var_proportion(self):
        return self._var_prop

    @property
    def total_variance(self):
        return self._total_variance

    @property
    def eigvals(self):
        return self._eigvals

    @property
    def eigvecs(self):
        return self._eigvecs

    def eigimages(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        if n_eigs > 0:
            iterat = xrange(n_eigs)
        elif n_eigs < 0:
            iterat = xrange(n_eigs, 0, 1)

        for ct, idx in enumerate(iterat):
            eigimg = np.zeros(self.data.shape[1:], dtype=float)
            for channel in range(self.data.shape[0]):
                if self._mean_sub:
                    mean_value = np.nanmean(self.data[channel])
                    eigimg += np.nan_to_num((self.data[channel] - mean_value) *
                                            self.eigvecs[channel, idx])
                else:
                    eigimg += np.nan_to_num(self.data[channel] *
                                            self.eigvecs[channel, idx])
            if ct == 0:
                eigimgs = eigimg
            else:
                eigimgs = np.dstack((eigimgs, eigimg))
        return eigimgs.swapaxes(0, 2)

    def autocorr_images(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        # Calculate the eigenimages
        eigimgs = self.eigimages(n_eigs=n_eigs)

        for idx, image in enumerate(eigimgs):
            fftx = np.fft.fft2(image)
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft2((fftx - fftx.mean()) * (fftxs - fftxs.mean()))
            acor = np.fft.fftshift(acor)

            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        return acors.swapaxes(0, 2)

    def autocorr_spec(self, n_eigs=None):

        if n_eigs is None:
            n_eigs = self.n_eigs

        for idx in range(n_eigs):
            fftx = np.fft.fft(self.eigvecs[:, idx])
            fftxs = np.conjugate(fftx)
            acor = np.fft.ifft((fftx - fftx.mean()) * (fftxs - fftxs.mean()))
            if idx == 0:
                acors = acor.real
            else:
                acors = np.dstack((acors, acor.real))

        return acors.swapaxes(0, 1).squeeze()

    def noise_ACF(self, n_eigs=-10):

        if n_eigs is None:
            n_eigs = self.n_eigs

        acors = self.autocorr_images(n_eigs=n_eigs)

        noise_ACF = np.nansum(acors, axis=0) / float(n_eigs)

        return noise_ACF

    def find_spatial_widths(self, n_eigs=None, method='contour',
                            brunt_beamcorrect=True, beam_fwhm=None,
                            physical_scales=True, distance=None):
        '''
        Derive the spatial widths using the autocorrelation of the
        eigenimages.

        Parameters
        ----------

        '''
        # Try reading beam width from the header is it is not given.
        if brunt_beamcorrect:
            if beam_fwhm is None:
                beam_fwhm = find_beam_width(self.header)

        if n_eigs is None:
            n_eigs = self.n_eigs

        acors = self.autocorr_images(n_eigs=n_eigs)
        noise_ACF = self.noise_ACF()

        self._spatial_width, self._spatial_width_error = \
            WidthEstimate2D(acors, noise_ACF=noise_ACF, method=method,
                            brunt_beamcorrect=brunt_beamcorrect,
                            beam_fwhm=beam_fwhm,
                            spatial_cdelt=self.header['CDELT2'] * u.deg)

        self._spatial_width = self._spatial_width * u.pix
        self._spatial_width_error = self._spatial_width_error * u.pix

        # If distance is given, convert to physical scale
        if distance is not None:
            self.distance = distance

        if physical_scales:
            if not hasattr(self, "_distance"):
                warn("Distance must be given to use physical units")
            else:
                self._spatial_width = self.to_physical(self._spatial_width)
                self._spatial_width_error = \
                    self.to_physical(self._spatial_width_error)

    @property
    def spatial_width(self):
        return self._spatial_width

    @property
    def spatial_width_error(self):
        return self._spatial_width_error

    def find_spectral_widths(self, n_eigs=None, method='walk-down',
                             physical_units=True):
        '''
        Calculate the spectral scales for the structure functions.
        '''

        if n_eigs is None:
            n_eigs = self.n_eigs

        acorr_spec = self.autocorr_spec(n_eigs=n_eigs)

        self._spectral_width, self._spectral_width_error = \
            WidthEstimate1D(acorr_spec, method=method)

        self._spectral_width = self._spectral_width * u.pix
        self._spectral_width_error = self._spectral_width_error * u.pix

        if physical_units:
            spec = self._wcs.wcs.spec
            spec_cdelt = np.abs(self._wcs.wcs.cdelt[spec]) * \
                u.Unit(self._wcs.wcs.cunit[spec])
            self._spectral_width = self._spectral_width.value * spec_cdelt
            self._spectral_width_error = self._spectral_width_error.value * \
                spec_cdelt

    @property
    def spectral_width(self):
        return self._spectral_width

    @property
    def spectral_width_error(self):
        return self._spectral_width_error

    def fit_plaw(self, fit_method='odr', verbose=False, **kwargs):
        '''
        Fit the size-linewidth relation.

        Parameters
        ----------
        fit_method : str, optional
            Set the type of fitting to perform. Options are 'odr'
            (orthogonal distance regression) or 'bayes' (MCMC). Note that
            'bayes' requires the emcee package to be installed.
        verbose : bool, optional
            Prints out additional information about the fitting.
        kwargs : Passed to bayes_linear when fit_method is bayes.
        '''

        if fit_method != 'odr' and fit_method != 'bayes':
            raise TypeError("fit_method must be 'odr' or 'bayes'.")

        # Only keep the width estimations that worked
        are_finite = np.isfinite(self.spectral_width) * \
            np.isfinite(self.spatial_width) * \
            np.isfinite(self.spatial_width_error) * \
            np.isfinite(self.spectral_width_error)

        y = np.log10(self.spectral_width[are_finite].value)
        x = np.log10(self.spatial_width[are_finite].value)

        y_err = 0.434 * self.spectral_width_error[are_finite].value / \
            self.spectral_width[are_finite].value
        x_err = 0.434 * self.spatial_width_error[are_finite].value / \
            self.spatial_width[are_finite].value

        if fit_method == 'odr':
            params, errors = leastsq_linear(x, y, x_err, y_err,
                                            verbose=verbose)
            # Turn into +/- range values, as would be returned by the MCMC fit
            errors = np.vstack([params - errors, params + errors]).T
        else:
            params, errors = bayes_linear(x, y, x_err, y_err, verbose=verbose,
                                          **kwargs)

        self._index = params[0]
        self._index_error_range = errors[0]

        # Take the intercept out of log scale
        self._intercept = 10 ** params[1]
        self._intercept_error_range = np.log(10) * (10 ** params[1]) * \
            errors[1]

    @property
    def index(self):
        return self._index

    @property
    def index_error_range(self):
        return self._index_error_range

    @property
    def gamma(self):
        '''
        These values are based off a broken linear fit in Section 3.3.1 from
        Chris Brunt's thesis. These are based off calibrating against uniform
        density field's with different indices.
        '''
        if self.index < 0.67:
            return (self.index - 0.32) / 0.59
        else:
            return (self.index - 0.03) / 1.07

    @property
    def gamma_error_range(self):
        '''
        See description in self.gamma
        '''
        if self.index < 0.67:
            return self.index_error_range / 0.59
        else:
            return self.index_error_range / 1.07

    @property
    def intercept(self):
        return self._intercept

    @property
    def intercept_error_range(self):
        return self._intercept_error_range

    def model(self, x):
        '''
        Model from the fitting procedure
        '''
        return self.index * x + np.log10(self.intercept)

    def sonic_length(self, T_k=10 * u.K, mu=1.36, use_gamma=True):
        '''
        Estimate of the sonic length based on a given temperature. Uses the
        intercept from the fit.

        Based on sonic.pro used in the Heyer & Brunt PCA implementation.

        Parameters
        ----------
        T_k : astropy.units.Quantity, optional
            Temperature given in units convertible to Kelvin.
        mu : float, optional
            Factor to multiply by m_H to account for He and metals.
        use_gamma : bool, optional
            Toggle whether to use gamma or the fit index.
        '''

        import astropy.constants as const

        try:
            T_k = T_k.to(u.K).value
        except u.UnitConversionError:
            raise u.UnitConversionError("Cannot convert T_k to Kelvin.")

        # Sound speed in m/s
        c_s = np.sqrt(const.k_B.decompose() * T_k / (mu * const.m_p)).value

        lambd = np.power(c_s / self.intercept, 1. / self.index)

        # Added in quadrature and simplified
        term1 = np.log(c_s / self.intercept) * \
            (self.index / self.index_error_range)
        term2 = self.intercept / self.intercept_error_range
        lambd_error_range = (lambd / self.index) * np.sqrt(term1**2 + term2**2)

        return lambd, lambd_error_range

    def run(self, verbose=False, mean_sub=False):
        '''
        Run method. Needed to maintain package standards.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        self.compute_pca(mean_sub=mean_sub)

        if verbose:
            import matplotlib.pyplot as p

            print 'Proportion of Variance kept: %s' % (self.var_proportion)

            p.subplot(121)
            p.imshow(self.cov_matrix, origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.bar(np.arange(1, self.n_eigs + 1), self.eigvals, 0.5, color='r')
            p.xlim([0, self.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel('Variance')
            p.show()

        return self


class PCA_Distance(object):

    '''
    Compare two data cubes based on the eigenvalues of the PCA decomposition.
    The distance is the Euclidean distance between the eigenvalues.

    Parameters
    ----------
    cube1 : %(dtypes)s
        Data cube.
    cube2 : %(dtypes)s
        Data cube.
    n_eigs : int
        Number of eigenvalues to compute.
    fiducial_model : PCA
        Computed PCA object. Use to avoid recomputing.
    mean_sub : bool, optional
        Subtracts the mean before computing the covariance matrix. Not
        subtracting the mean is done in the original Heyer & Brunt works.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, n_eigs=50, fiducial_model=None,
                 mean_sub=True):
        super(PCA_Distance, self).__init__()

        if fiducial_model is not None:
            self.pca1 = fiducial_model
        else:
            self.pca1 = PCA(cube1, n_eigs=n_eigs)
            self.pca1.run(mean_sub=mean_sub)

        self.pca2 = PCA(cube2, n_eigs=n_eigs)
        self.pca2.run(mean_sub=mean_sub)

        self._mean_sub = mean_sub

    def distance_metric(self, verbose=False, label1="Cube 1", label2="Cube 2"):
        '''
        Computes the distance between the cubes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        '''

        # The eigenvalues need to be normalized before being compared. If
        # mean_sub is False, the first eigenvalue is not used.
        if self._mean_sub:
            eigvals1 = self.pca1.eigvals / np.sum(self.pca1.eigvals)
            eigvals2 = self.pca2.eigvals / np.sum(self.pca2.eigvals)
        else:
            eigvals1 = self.pca1.eigvals[1:] / np.sum(self.pca1.eigvals[1:])
            eigvals2 = self.pca2.eigvals[1:] / np.sum(self.pca2.eigvals[1:])

        self.distance = np.linalg.norm(eigvals1 - eigvals2)

        if verbose:
            import matplotlib.pyplot as p

            print "Proportions of total variance: 1 - %0.3f, 2 - %0.3f" % \
                (self.pca1.var_proportion, self.pca2.var_proportion)

            p.subplot(2, 2, 1)
            p.imshow(
                self.pca1.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca1.cov_matrix))
            p.colorbar()
            p.title(label1)
            p.subplot(2, 2, 3)
            p.bar(np.arange(1, len(eigvals1) + 1), eigvals1, 0.5,
                  color='r')
            p.xlim([0, self.pca1.n_eigs + 1])
            p.xlabel('Eigenvalues')
            p.ylabel("Proportion of Variance")
            p.subplot(2, 2, 2)
            p.imshow(
                self.pca2.cov_matrix, origin="lower", interpolation="nearest",
                vmin=np.median(self.pca2.cov_matrix))
            p.colorbar()
            p.title(label2)
            p.subplot(2, 2, 4)
            p.bar(np.arange(1, len(eigvals2) + 1), eigvals2, 0.5,
                  color='r')
            p.xlim([0, self.pca2.n_eigs + 1])
            p.xlabel('Eigenvalues')

            p.tight_layout()
            p.show()

        return self
