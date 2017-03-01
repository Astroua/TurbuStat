# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import astropy.units as u
import statsmodels.api as sm

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types
from ..fitting_utils import check_fit_limits


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
    scale_normalization: bool, optional
        Compute the transform with the correct scale-invariant normalization.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, data, header=None, scales=None, num=50,
                 scale_normalization=True):

        self.input_data_header(data, header)

        # NOTE: can't use nan_interpolating from astropy
        # until the normalization for sum to zeros kernels is fixed!!!
        self.data[np.isnan(self.data)] = np.nanmin(self.data)

        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            a_max = min(self.data.shape) / 2.
            # Log spaces scales up to half of the smallest size of the array
            self.scales = \
                np.logspace(np.log10(a_min), np.log10(a_max), num) * u.pix
        else:
            self.scales = self.to_pixel(scales)

        self.scale_normalization = scale_normalization

        if not self.scale_normalization:
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

    def compute_transform(self):
        '''
        Compute the wavelet transform at each scale.
        '''

        n0, m0 = self.data.shape
        A = len(self.scales)

        self.Wf = np.zeros((A, n0, m0), dtype=np.float)

        factor = 2
        if not self.scale_normalization:
            factor = 4
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

        for i, an in enumerate(self.scales.value):
            psi = MexicanHat2DKernel(an)

            self.Wf[i] = \
                convolve_fft(self.data, psi).real * an**factor

    def make_1D_transform(self):
        '''
        Create the 1D transform.
        '''

        self.values = np.empty_like(self.scales.value)
        for i, plane in enumerate(self.Wf):
            self.values[i] = (plane[plane > 0]).mean()

    def fit_transform(self, xlow=None, xhigh=None):
        '''
        Perform a fit to the transform in log-log space.
        '''

        x = np.log10(self.scales.value)
        y = np.log10(self.values)

        if xlow is not None:
            lower_limit = x >= np.log10(xlow)
        else:
            lower_limit = \
                np.ones_like(self.scales, dtype=bool).value

        if xhigh is not None:
            upper_limit = x <= np.log10(xhigh)
        else:
            upper_limit = \
                np.ones_like(self.scales, dtype=bool).value

        self._fit_range = \
            [xlow if xlow is not None else self.scales.min().value,
             xhigh if xhigh is not None else self.scales.max().value]

        within_limits = np.logical_and(lower_limit, upper_limit)

        y = y[within_limits]
        x = x[within_limits]
        x = sm.add_constant(x)

        model = sm.OLS(y, x, missing='drop')

        self.fit = model.fit()

        self._slope = self.fit.params[1]
        self._slope_err = self.fit.bse[1]

    @property
    def slope(self):
        return self._slope

    @property
    def slope_err(self):
        return self._slope_err

    def plot_transform(self, ang_units=False, unit=u.deg, show=True,
                       color='b', symbol='D', label=None):
        '''
        Plot the transform and the fit.
        '''

        import matplotlib.pyplot as p

        if ang_units:
            scales = \
                self.scales.to(unit,
                               equivalencies=self.angular_equiv).value
        else:
            scales = self.scales.value

        p.loglog(scales, self.values, color + symbol)
        # Plot the fit within the fitting range.
        low_lim = self._fit_range[0]
        high_lim = self._fit_range[1]
        if ang_units:
            low_lim = (low_lim * self.scales.unit)
            low_lim = low_lim.to(unit, equivalencies=self.angular_equiv)
            low_lim = low_lim.value

            high_lim = (high_lim * self.scales.unit)
            high_lim = high_lim.to(unit, equivalencies=self.angular_equiv)
            high_lim = high_lim.value

        within_limits = np.logical_and(scales >= low_lim,
                                       scales <= high_lim)

        p.loglog(scales[within_limits], 10**self.fit.fittedvalues,
                 color + '--', label=label, linewidth=8, alpha=0.75)

        p.axvline(low_lim,
                  color=color, alpha=0.5, linestyle='-')
        p.axvline(high_lim,
                  color=color, alpha=0.5, linestyle='-')

        p.ylabel(r"$T_g$")
        if ang_units:
            p.xlabel("Scales (deg)")
        else:
            p.xlabel("Scales (pixels)")

        if show:
            p.show()

    def run(self, verbose=False, ang_units=False, unit=u.deg,
            xlow=None, xhigh=None):
        '''
        Compute the Wavelet transform.

        Parameters
        ----------
        verbose : bool, optional
            Plot wavelet transform.
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : u.Unit, optional
            Choose the angular unit to convert to when ang_units is enabled.
        '''
        self.compute_transform()
        self.make_1D_transform()
        self.fit_transform(xlow=xlow, xhigh=xhigh)

        if verbose:
            self.plot_transform(ang_units=ang_units, unit=unit, show=True)

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
    xlow : float or np.ndarray, optional
        The lower lag fitting limit. An array with 2 elements can be passed to
        give separate lower limits for the datasets.
    xhigh : float or np.ndarray, optional
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
                        label2=None, ang_units=False, unit=u.deg,
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

            import matplotlib.pyplot as p

            self.wt1.plot_transform(ang_units=ang_units, unit=unit, show=False,
                                    color='b', symbol='D', label=label1)
            self.wt2.plot_transform(ang_units=ang_units, unit=unit, show=False,
                                    color='g', symbol='o', label=label1)
            p.legend(loc='best')

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()

        return self
