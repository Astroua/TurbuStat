# Licensed under an MIT open source license - see LICENSE


import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import statsmodels.api as sm
# from pandas import Series, DataFrame


class Wavelet(object):
    '''
    Compute the wavelet transform of a 2D array.

    Parameters
    ----------
    array : numpy.ndarray
        2D array.
    header : FITS header
        Header for the array.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int, optional
        Number of scales to compute the transform at.
    ang_units : bool, optional
        Convert scales to angular units using the given header.
    scale_normalization: bool, optional
        Compute the transform with the correct scale-invariant normalization.
    '''

    def __init__(self, array, header, scales=None, num=50, ang_units=False,
                 scale_normalization=True):

        self.array = array
        # NOTE: can't use nan_interpolating from astropy
        # until the normalization for sum to zeros kernels is fixed!!!
        self.array[np.isnan(self.array)] = np.nanmin(self.array)

        self.header = header
        self.ang_units = ang_units

        if ang_units:
            try:
                self.imgscale = np.abs(self.header["CDELT2"])
            except ValueError:
                warnings.warn("Header doesn't not contain the\
                               angular size. Reverting to pixel scales.")
                ang_units = False
        if not ang_units:
            self.imgscale = 1.0

        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            a_max = min(self.array.shape)/2
            # Log spaces scales up to half of the smallest size of the array
            self.scales = np.logspace(np.log10(a_min), np.log10(a_max), num) *\
                self.imgscale
        else:
            self.scales = scales

        self.scale_normalization = scale_normalization

        if not self.scale_normalization:
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

    def compute_transform(self):
        '''
        Compute the wavelet transform at each scale.
        '''

        n0, m0 = self.array.shape
        A = len(self.scales)

        self.Wf = np.zeros((A, n0, m0), dtype=np.float)

        factor = 2
        if not self.scale_normalization:
            factor = 4
            Warning("Transform values are only reliable with the proper scale"
                    " normalization. When disabled, the slope of the transform"
                    " CANNOT be used for physical interpretation.")

        for i, an in enumerate(self.scales):
            psi = MexicanHat2DKernel(an)

            self.Wf[i] = \
                convolve_fft(self.array, psi).real * an**factor

    def make_1D_transform(self):
        '''
        Create the 1D transform.
        '''

        self.values = np.empty_like(self.scales)
        for i, plane in enumerate(self.Wf):
            self.values[i] = (plane[plane > 0]).mean()

    def fit_transform(self):
        '''
        Perform a fit to the transform in log-log space.
        '''

        x = np.log10(self.scales)
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

    def run(self, verbose=False):
        '''
        Compute the Wavelet transform.
        '''
        self.compute_transform()
        self.make_1D_transform()
        self.fit_transform()

        if verbose:
            import matplotlib.pyplot as p

            p.loglog(self.scales, self.values, 'bD')
            p.loglog(self.scales, 10**self.fit.fittedvalues, 'b-')

            p.ylabel(r"$T_g$")
            if self.ang_units:
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
    dataset1 : FITS hdu
        2D image.
    dataset2 : FITS hdu
        2D image.
    ang_units : bool, optional
        Sets whether to use angular units.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int
        Number of scales to calculate the transform at.
    fiducial_model : wt2D
        Computed wt2D object. use to avoid recomputing.
    '''

    def __init__(self, dataset1, dataset2,
                 ang_units=False, scales=None, num=50, fiducial_model=None):
        super(Wavelet_Distance, self).__init__()

        array1 = dataset1[0]
        header1 = dataset1[1]
        array2 = dataset2[0]
        header2 = dataset2[1]

        self.ang_units = ang_units

        if fiducial_model is None:
            self.wt1 = Wavelet(array1, header1, scales=scales,
                               ang_units=ang_units)
            self.wt1.run()
        else:
            self.wt1 = fiducial_model

        self.wt2 = Wavelet(array2, header2, scales=scales, ang_units=ang_units)
        self.wt2.run()

    def distance_metric(self, verbose=False, label1=None,
                        label2=None):
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
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.wt1.slope - self.wt2.slope) /
                   np.sqrt(self.wt1.slope_err**2 +
                           self.wt2.slope_err**2))

        if verbose:

            print(self.wt1.fit.summary())
            print(self.wt2.fit.summary())

            scales1 = np.log10(self.wt1.scales)
            values1 = np.log10(self.wt1.values)
            scales2 = np.log10(self.wt2.scales)
            values2 = np.log10(self.wt2.values)

            import matplotlib.pyplot as p
            p.plot(scales1, values1, 'bD', label=label1)
            p.plot(scales2, values2, 'go', label=label2)
            p.plot(scales1,
                   self.wt1.fit.fittedvalues[:len(values1)], "b",
                   scales2,
                   self.wt2.fit.fittedvalues[-len(values2):], "g")
            p.grid(True)

            if self.ang_units:
                xunit = "deg"
            else:
                xunit = "pixels"
            p.xlabel("log a ("+xunit+")")
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
