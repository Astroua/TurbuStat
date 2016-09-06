# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import astropy.units as u
import statsmodels.api as sm

from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, twod_types


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
            a_max = min(self.data.shape)/2
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

    def fit_transform(self):
        '''
        Perform a fit to the transform in log-log space.
        '''

        x = np.log10(self.scales.value)
        y = np.log10(self.values)

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

    def run(self, verbose=False, ang_units=False, unit=u.deg):
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
        self.fit_transform()

        if verbose:
            import matplotlib.pyplot as p

            if ang_units:
                scales = \
                    self.scales.to(unit,
                                   equivalencies=self.angular_equiv).value
            else:
                scales = self.scales.value

            p.loglog(scales, self.values, 'bD')
            p.loglog(scales, 10**self.fit.fittedvalues, 'b-')

            p.ylabel(r"$T_g$")
            if ang_units:
                p.xlabel("Scales (deg)")
            else:
                p.xlabel("Scales (pixels)")

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
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + twod_types)}

    def __init__(self, dataset1, dataset2,
                 scales=None, num=50, fiducial_model=None):
        super(Wavelet_Distance, self).__init__()

        if fiducial_model is None:
            self.wt1 = Wavelet(dataset1, scales=scales)
            self.wt1.run()
        else:
            self.wt1 = fiducial_model

        self.wt2 = Wavelet(dataset2, scales=scales)
        self.wt2.run()

    def distance_metric(self, verbose=False, label1=None,
                        label2=None, ang_units=False, unit=u.deg):
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
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.wt1.slope - self.wt2.slope) /
                   np.sqrt(self.wt1.slope_err**2 +
                           self.wt2.slope_err**2))

        if verbose:

            print(self.wt1.fit.summary())
            print(self.wt2.fit.summary())

            if ang_units:
                scales1 = \
                    self.wt1.scales.to(unit, equivalencies=self.wt1.angular_equiv).value
                scales2 = \
                    self.wt2.scales.to(unit, equivalencies=self.wt2.angular_equiv).value
            else:
                scales1 = self.wt1.scales.value
                scales2 = self.wt2.scales.value

            scales1 = np.log10(scales1)
            values1 = np.log10(self.wt1.values)
            scales2 = np.log10(scales2)
            values2 = np.log10(self.wt2.values)

            import matplotlib.pyplot as p
            p.plot(scales1, values1, 'bD', label=label1)
            p.plot(scales2, values2, 'go', label=label2)
            p.plot(scales1,
                   self.wt1.fit.fittedvalues[:len(values1)], "b",
                   scales2,
                   self.wt2.fit.fittedvalues[-len(values2):], "g")
            p.grid(True)

            if ang_units:
                p.xlabel("log a ("+unit.to_string()+")")
            else:
                p.xlabel("log a (pixels)")

            p.ylabel(r"log $T_g$")
            p.legend(loc='best')
            p.show()

        return self


# def clip_to_linear(data, threshold=1.0, kernel_width=0.05, ends_clipped=0.05):
#     '''
#     Takes the second derivative of the data with a ricker wavelet.
#     Data is clipped to the linear portion (2nd derivative ~ 0)

#     Parameters
#     ----------

#     data : numpy.ndarray
#         x and y data.
#     threshold : float, optional
#         Acceptable value of the second derivative to be called linear.
#     kernel_width : float, optional
#         Kernel width set to this percentage of the data length
#     ends_clipped : float, optional
#         Percentage of data to clip off at the ends. End points have residual
#         effects from the convolution.

#     Returns
#     -------
#     data_clipped : numpy.ndarray
#         Linear portion of the data set returned.
#     '''

#     from scipy.signal import ricker

#     y = data[1, :]
#     x = data[0, :]

#     num_pts = len(y)

#     kernel = ricker(num_pts, num_pts * kernel_width)

#     sec_deriv = np.convolve(y, kernel, mode="same")

#     # Ends go back to being ~ linear, so clip them off
#     if ends_clipped > 0.0:
#         clipped_pts = int(num_pts * ends_clipped)

#         sec_deriv = sec_deriv[: num_pts - clipped_pts]
#         y = y[: num_pts - clipped_pts]
#         x = x[: num_pts - clipped_pts]

#     linear_pts = np.abs(sec_deriv) < threshold

#     data_clipped = np.empty((2, len(y[linear_pts])))
#     data_clipped[:, :] = x[linear_pts], y[linear_pts]

#     return data_clipped
