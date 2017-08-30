# Licensed under an MIT open source license - see LICENSE
from __future__ import (print_function, absolute_import, division,
                        unicode_literals)

import numpy as np
from astropy.convolution import Gaussian2DKernel, Kernel2D
from astropy.convolution.kernels import _round_up_to_odd_integer
from astropy.modeling import Fittable2DModel, Parameter
from collections import OrderedDict
from astropy import units as u


def core_kernel(lag, x_size, y_size):
    '''
    Core Kernel for convolution.

    Parameters
    ----------

    lag : int
        Size of the lag. Set the kernel size.
    x_size : int
        Grid size to use in the x direction
    y_size_size : int
        Grid size to use in the y_size direction

    Returns
    -------

    kernel : numpy.ndarray
        Normalized kernel.
    '''

    return Gaussian2DKernel(lag / (2 * np.sqrt(2)))


def annulus_kernel(lag, diam_ratio, x_size, y_size):
    '''

    Annulus Kernel for convolution.

    Parameters
    ----------
    lag : int
        Size of the lag. Set the kernel size.
    diam_ratio : float
                 Ratio between kernel diameters.

    Returns
    -------

    kernel : numpy.ndarray
        Normalized kernel.
    '''

    diff_factor = 2 * np.sqrt(2)

    return AnnulusKernel(lag / diff_factor, diam_ratio)


class AnnulusKernel(Kernel2D):
    """
    Two-dimensional Annulus with a Gaussian profile.
    """
    _separable = True
    _is_bool = False

    def __init__(self, stddev, ratio,
                 support_scaling=8, **kwargs):

        amp = 1. / (2 * np.pi * stddev**2 * (ratio**2 - 1))
        self._model = GaussianAnnulus2D(amp, 0, 0, stddev, ratio)

        self._default_size = _round_up_to_odd_integer(8 * stddev)
        super(AnnulusKernel, self).__init__(**kwargs)
        self._truncation = np.abs(1. - self._array.sum())


class GaussianAnnulus2D(Fittable2DModel):
    r"""
    Two dimensional Annulus model with a Gaussian profile.

    Parameters
    ----------
    amplitude : float
        Amplitude of the Gaussian.
    x_mean : float
        Mean of the Gaussian in x.
    y_mean : float
        Mean of the Gaussian in y.
    stddev : float or None
        Standard deviation of the Gaussian.
    ratio : float, optional
        Ratio between the inner and outer widths. This sets the width of the
        annulus.

    Notes
    -----

    """

    amplitude = Parameter(default=1)
    x_mean = Parameter(default=0)
    y_mean = Parameter(default=0)
    stddev = Parameter(default=1)
    ratio = Parameter(default=1.0)

    def __init__(self, amplitude=amplitude.default, x_mean=x_mean.default,
                 y_mean=y_mean.default, stddev=None, ratio=None,
                 **kwargs):

        if stddev is None:
            stddev = self.__class__.stddev.default

        # kwargs.setdefault('bounds', {})
        # kwargs['bounds'].setdefault('stddev', (FLOAT_EPSILON, None))

        super(GaussianAnnulus2D, self).__init__(
            amplitude=amplitude, x_mean=x_mean, y_mean=y_mean,
            stddev=stddev, ratio=ratio, **kwargs)

    def bounding_box(self, factor=5.5):
        """
        Tuple defining the default ``bounding_box`` limits in each dimension,
        ``((y_low, y_high), (x_low, x_high))``

        The default offset from the mean is 5.5-sigma, corresponding
        to a relative error < 1e-7. The limits are adjusted for rotation.

        Parameters
        ----------
        factor : float, optional
            The multiple of `stddev` used to define the limits.
            The default is 5.5.

        """

        a = factor * self.stddev

        dy = dx = a

        return ((self.y_mean - dy, self.y_mean + dy),
                (self.x_mean - dx, self.x_mean + dx))

    @staticmethod
    def evaluate(x, y, amplitude, x_mean, y_mean, stddev, ratio):
        """Two dimensional Gaussian Annulus function"""

        inner = np.exp(-(x ** 2. + y ** 2.) / (2 * stddev**2))
        outer = np.exp(-(x ** 2. + y ** 2.) / (2 * (ratio * stddev)**2))

        return amplitude * (outer - inner)

    @property
    def input_units(self):
        if self.x_mean.unit is None and self.y_mean.unit is None:
            return None
        else:
            return {'x': self.x_mean.unit,
                    'y': self.y_mean.unit}

    def _parameter_units_for_data_units(self, inputs_unit, outputs_unit):
        # Note that here we need to make sure that x and y are in the same
        # units otherwise this can lead to issues since rotation is not well
        # defined.
        if inputs_unit['x'] != inputs_unit['y']:
            raise u.UnitsError("Units of 'x' and 'y' inputs should match")
        return OrderedDict([('x_mean', inputs_unit['x']),
                            ('y_mean', inputs_unit['x']),
                            ('stddev', inputs_unit['x']),
                            ('ratio', u.dimensionless_unscaled),
                            ('amplitude', outputs_unit['z'])])
