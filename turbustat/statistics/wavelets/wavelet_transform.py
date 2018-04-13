# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import astropy.units as u
import statsmodels.api as sm

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types
from ..fitting_utils import check_fit_limits
from ..lm_seg import Lm_Seg


class Wavelet(BaseStatisticMixIn):
    '''
    Compute the wavelet transform of a 2D array.

    Parameters
    ----------
    array : %(dtypes)s
        2D data.
    header : FITS header, optional
        Header for the array.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int, optional
        Number of scales to compute the transform at.
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data, header=None, scales=None, num=50,
                 distance=None):

        self.input_data_header(data, header)

        # NOTE: can't use nan_interpolating from astropy
        # until the normalization for sum to zeros kernels is fixed!!!
        self.data[np.isnan(self.data)] = np.nanmin(self.data)

        if distance is not None:
            self.distance = distance

        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            a_max = min(self.data.shape) / 2.
            # Log spaces scales up to half of the smallest size of the array
            scales = np.logspace(np.log10(a_min), np.log10(a_max), num) * u.pix

        self.scales = scales

    @property
    def scales(self):
        '''
        Wavelet scales.
        '''
        return self._scales

    @scales.setter
    def scales(self, values):

        if not isinstance(values, u.Quantity):
            raise TypeError("scales must be given as a "
                            "astropy.units.Quantity.")

        # Now make sure that we can convert into pixels before setting.
        try:
            pix_scal = self._to_pixel(values)
        except Exception as e:
            raise e

        # The radius should be larger than a pixel
        if np.any(pix_scal.value < 1):
            raise ValueError("One of the chosen lags is smaller than one "
                             "pixel."
                             " Ensure that all lag values are larger than one "
                             "pixel.")

        # Finally, limit the radius to a maximum of half the image size.
        if np.any(pix_scal.value > min(self.data.shape) / 2.):
            raise ValueError("At least one of the lags is larger than half of "
                             "the image size (in the smallest dimension. "
                             "Ensure that all lag values are smaller than "
                             "this.")

        self._scales = values

    def compute_transform(self, scale_normalization=True):
        '''
        Compute the wavelet transform at each scale.

        Parameters
        ----------
        scale_normalization: bool, optional
            Compute the transform with the correct scale-invariant
            normalization.

        '''

        n0, m0 = self.data.shape
        A = len(self.scales)

        self._Wf = np.zeros((A, n0, m0), dtype=np.float)

        factor = 2
        if not scale_normalization:
            factor = 4
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

        pix_scales = self._to_pixel(self.scales).value

        for i, an in enumerate(pix_scales):
            psi = MexicanHat2DKernel(an)

            self._Wf[i] = \
                convolve_fft(self.data, psi, normalize_kernel=False).real * \
                an**factor

    @property
    def Wf(self):
        '''
        The wavelet transforms of the image. Each plane is the transform at
        different wavelet sizes.
        '''
        return self._Wf

    def make_1D_transform(self):
        '''
        Create the 1D transform.
        '''

        self._values = np.empty_like(self.scales.value)
        for i, plane in enumerate(self.Wf):
            self._values[i] = (plane[plane > 0]).mean()

    @property
    def values(self):
        '''
        The 1-dimensional wavelet transform.
        '''
        return self._values

    def fit_transform(self, xlow=None, xhigh=None, brk=None, min_fits_pts=3,
                      **fit_kwargs):
        '''
        Perform a fit to the transform in log-log space.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower scale value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper scale value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        fit_kwargs : Passed to `turbustat.statistics.Lm_Seg.fit_model`
        '''

        pix_scales = self._to_pixel(self.scales)
        x = np.log10(pix_scales.value)
        y = np.log10(self.values)

        if xlow is not None:
            xlow = self._to_pixel(xlow)

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.values, dtype=bool)
            xlow = pix_scales.min() * 0.99

        if xhigh is not None:
            xhigh = self._to_pixel(xhigh)

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.values, dtype=bool)
            xhigh = pix_scales.max() * 1.01

        self._fit_range = [xlow, xhigh]

        within_limits = np.logical_and(lower_limit, upper_limit)

        y = y[within_limits]
        x = x[within_limits]

        if brk is not None:
            # Try fitting a segmented model

            pix_brk = self._to_pixel(brk)

            if pix_brk < xlow or pix_brk > xhigh:
                raise ValueError("brk must be within xlow and xhigh.")

            model = Lm_Seg(x, y, np.log10(pix_brk.value))

            model.fit_model(**fit_kwargs)

            self.fit = model.fit

            if model.params.size == 5:

                # Check to make sure this leaves enough to fit to.
                if sum(x < model.brk) < min_fits_pts:
                    warnings.warn("Not enough points to fit to." +
                                  " Ignoring break.")

                    self._brk = None
                else:
                    good_pts = x.copy() < model.brk
                    x = x[good_pts]
                    y = y[good_pts]

                    self._brk = 10**model.brk * u.pix
                    self._brk_err = np.log(10) * self.brk.value * \
                        model.brk_err * u.pix

                    self._slope = model.slopes
                    self._slope_err = model.slope_errs

                    self.fit = model.fit

            else:
                self._brk = None
                # Break fit failed, revert to normal model
                warnings.warn("Model with break failed, reverting to model\
                               without break.")
        else:
            self._brk = None

        # Revert to model without break if none is given, or if the segmented
        # model failed.
        if self.brk is None:

            x = sm.add_constant(x)

            model = sm.OLS(y, x, missing='drop')

            self.fit = model.fit()

            self._slope = self.fit.params[1]
            self._slope_err = self.fit.bse[1]

        self._model = model

    @property
    def slope(self):
        '''
        Fitted slope.
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        Standard error on the fitted slope.
        '''
        return self._slope_err

    @property
    def brk(self):
        '''
        Break point in the segmented linear model.
        '''
        return self._brk

    @property
    def brk_err(self):
        '''
        1-sigma on the break point in the segmented linear model.
        '''
        return self._brk_err

    @property
    def fit_range(self):
        '''
        Range of scales used in the fit.
        '''
        return self._fit_range

    def fitted_model(self, xvals):
        '''
        Computes the fitted power-law in log-log space using the
        given x values.

        Parameters
        ----------
        xvals : `~numpy.ndarray`
            Values of log(lags) to compute the model at (base 10 log).

        Returns
        -------
        model_values : `~numpy.ndarray`
            Values of the model at the given values.
        '''

        if isinstance(self._model, Lm_Seg):
            return self._model.model(xvals)
        else:
            return self.fit.params[0] + self.fit.params[1] * xvals

    def plot_transform(self, xunit=u.pix, show=True,
                       color='b', symbol='D', label=None):
        '''
        Plot the transform and the fit.
        '''

        import matplotlib.pyplot as plt

        pix_scales = self._to_pixel(self.scales)
        scales = self._spatial_unit_conversion(pix_scales, xunit).value

        plt.loglog(scales, self.values, color + symbol)
        # Plot the fit within the fitting range.
        low_lim = \
            self._spatial_unit_conversion(self._fit_range[0], xunit).value
        high_lim = \
            self._spatial_unit_conversion(self._fit_range[1], xunit).value

        plt.loglog(scales, 10**self.fitted_model(np.log10(pix_scales.value)),
                   color + '--', linewidth=8, label='Fit', alpha=0.75)

        plt.axvline(low_lim, color=color, alpha=0.5, linestyle='-')
        plt.axvline(high_lim, color=color, alpha=0.5, linestyle='-')

        plt.ylabel(r"$T_g$")
        plt.xlabel("Scales ({})".format(xunit))

        plt.grid()

        if show:
            plt.show()

    def run(self, verbose=False, xunit=u.pix,
            xlow=None, xhigh=None, brk=None, scale_normalization=True,
            save_name=None, **plot_kwargs):
        '''
        Compute the Wavelet transform.

        Parameters
        ----------
        verbose : bool, optional
            Plot wavelet transform.
        xunit : u.Unit, optional
            Choose the unit to convert to when ang_units is enabled.
        xlow : `~astropy.units.Quantity`, optional
            Lower scale value to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper scale value to consider in the fit.
        brk : `~astropy.units.Quantity`, optional
            Give an initial guess for a break point. This enables fitting
            with a `turbustat.statistics.Lm_Seg`.
        scale_normalization: bool, optional
            Multiply the wavelet transform by the correct normalization
            factor.
        save_name : str,optional
            Save the figure when a file name is given.
        plot_kwargs : Passed to `~Wavelet.plot_transform`.
        '''
        self.compute_transform(scale_normalization=scale_normalization)
        self.make_1D_transform()
        self.fit_transform(xlow=xlow, xhigh=xhigh, brk=brk)

        if verbose:

            print(self.fit.summary())

            import matplotlib.pyplot as plt

            self.plot_transform(xunit=xunit, show=True, **plot_kwargs)

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self


class Wavelet_Distance(object):
    '''
    Compute the distance between the two cubes using the Wavelet transform.
    We fit a linear model to the two wavelet transforms. The distance is the
    t-statistic of the interaction term describing the difference in the
    slopes.

    Parameters
    ----------
    dataset1 : %(dtypes)s
        2D image.
    dataset2 : %(dtypes)s
        2D image.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int
        Number of scales to calculate the transform at.
    fiducial_model : wt2D
        Computed wt2D object. use to avoid recomputing.
    xlow : `astropy.units.Quantity`, optional
        The lower lag fitting limit. An array with 2 elements can be passed to
        give separate lower limits for the datasets.
    xhigh : `astropy.units.Quantity`, optional
        The upper lag fitting limit. See `xlow` above.

    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, dataset1, dataset2,
                 scales=None, num=50, xlow=None, xhigh=None,
                 fiducial_model=None):
        super(Wavelet_Distance, self).__init__()

        xlow, xhigh = check_fit_limits(xlow, xhigh)

        if fiducial_model is None:
            self.wt1 = Wavelet(dataset1, scales=scales)
            self.wt1.run(xlow=xlow[0], xhigh=xhigh[0])
        else:
            self.wt1 = fiducial_model

        self.wt2 = Wavelet(dataset2, scales=scales)
        self.wt2.run(xlow=xlow[1], xhigh=xhigh[1])

    def distance_metric(self, verbose=False, label1=None,
                        label2=None, xunit=u.deg,
                        save_name=None):
        '''
        Implements the distance metric for 2 wavelet transforms.
        We fit the linear portion of the transform to represent the powerlaw

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for dataset1
        label2 : str, optional
            Object or region name for dataset2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.wt1.slope - self.wt2.slope) /
                   np.sqrt(self.wt1.slope_err**2 +
                           self.wt2.slope_err**2))

        if verbose:

            print(self.wt1.fit.summary())
            print(self.wt2.fit.summary())

            import matplotlib.pyplot as plt

            self.wt1.plot_transform(xunit=xunit, show=False,
                                    color='b', symbol='D', label=label1)
            self.wt2.plot_transform(xunit=xunit, show=False,
                                    color='g', symbol='o', label=label1)
            plt.legend(loc='best')

            if save_name is not None:
                plt.savefig(save_name)
                plt.close()
            else:
                plt.show()

        return self
