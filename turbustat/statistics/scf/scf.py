# Licensed under an MIT open source license - see LICENSE


import numpy as np
import cPickle as pickle
from copy import deepcopy
from astropy import units as u
from astropy.wcs import WCS
import statsmodels.api as sm

from ..psds import pspec
from ..base_statistic import BaseStatisticMixIn
from ...io import common_types, threed_types, input_data
from ..stats_utils import common_scale, fourier_shift, pixel_shift


class SCF(BaseStatisticMixIn):
    '''
    Computes the Spectral Correlation Function of a data cube
    (Rosolowsky et al, 1999).

    Parameters
    ----------
    cube : %(dtypes)s
        Data cube.
    header :  FITS header, optional
        Header for the cube.
    size : int, optional
        The total size of the lags used in one dimension in pixels. The maximum
        lag size will be (size - 1) / 2 in each direction.
    roll_lags : `~numpy.ndarray` or `~astropy.units.Quantity`, optional
        Pass a custom array of lag values. An odd number of lags, centered at
        0, must be given. If no units are given, it is
        assumed that the lags are in pixels. The lags should have
        symmetric positive and negative values (e.g., [-1, 0, 1]).
    distance : `~astropy.units.Quantity`, optional
        Physical distance to the region in the data.

    Example
    -------
    >>> from spectral_cube import SpectralCube
    >>> from turbustat.statistics import SCF
    >>> cube = SpectralCube.read("Design4.13co.fits")  # doctest: +SKIP
    >>> scf = SCF(cube)  # doctest: +SKIP
    >>> scf.run(verbose=True)  # doctest: +SKIP
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube, header=None, size=11, roll_lags=None,
                 distance=None):
        super(SCF, self).__init__()

        # Set data and header
        self.input_data_header(cube, header)

        if distance is not None:
            self.distance = distance

        if roll_lags is None:
            if size % 2 == 0:
                Warning("Size must be odd. Reducing size to next lowest odd"
                        " number.")
                size = size - 1
            self.roll_lags = (np.arange(size) - size / 2) * u.pix
        else:
            if roll_lags.size % 2 == 0:
                Warning("Size of roll_lags must be odd. Reducing size to next"
                        "lowest odd number.")
                roll_lags = roll_lags[: -1]

            if isinstance(roll_lags, u.Quantity):
                pass
            elif isinstance(roll_lags, np.ndarray):
                roll_lags = roll_lags * u.pix
            else:
                raise TypeError("roll_lags must be an astropy.units.Quantity"
                                " array or a numpy.ndarray.")

            self.roll_lags = roll_lags

            # Make sure that we can convert the lags
            self._to_pixel(self.roll_lags)

        self.size = self.roll_lags.size

        self._scf_surface = None
        self._scf_spectrum_stddev = None

    @property
    def scf_surface(self):
        '''
        SCF correlation array
        '''
        return self._scf_surface

    @property
    def scf_spectrum(self):
        '''
        Azimuthally averaged 1D SCF spectrum
        '''
        return self._scf_spectrum

    @property
    def scf_spectrum_stddev(self):
        '''
        Standard deviation of the `~SCF.scf_spectrum`
        '''
        if not self._stddev_flag:
            Warning("scf_spectrum_stddev is only calculated when return_stddev"
                    " is enabled.")
        return self._scf_spectrum_stddev

    @property
    def lags(self):
        '''
        Values of the lags, in pixels, to compute SCF at
        '''
        return self._lags

    def compute_surface(self, boundary='continuous'):
        '''
        Computes the SCF up to the given lag value.

        Parameters
        ----------
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        '''

        if boundary not in ["continuous", "cut"]:
            raise ValueError("boundary must be 'continuous' or 'cut'.")

        self._scf_surface = np.zeros((self.size, self.size))

        # Convert the lags into pixel units.
        pix_lags = self._to_pixel(self.roll_lags).value

        dx = pix_lags.copy()
        dy = pix_lags.copy()

        for i, x_shift in enumerate(dx):
            for j, y_shift in enumerate(dy):

                if x_shift == 0 and y_shift == 0:
                    self._scf_surface[i, j] = 1.

                if x_shift == 0:
                    tmp = self.data
                else:
                    if float(x_shift).is_integer():
                        shift_func = pixel_shift
                    else:
                        shift_func = fourier_shift
                    tmp = shift_func(self.data, x_shift, axis=1)

                if y_shift != 0:
                    if float(y_shift).is_integer():
                        shift_func = pixel_shift
                    else:
                        shift_func = fourier_shift
                    tmp = shift_func(tmp, y_shift, axis=2)

                if boundary is "cut":
                    # Always round up to the nearest integer.
                    x_shift = np.ceil(x_shift).astype(int)
                    y_shift = np.ceil(y_shift).astype(int)
                    if x_shift < 0:
                        x_slice_data = slice(None, tmp.shape[1] + x_shift)
                        x_slice_tmp = slice(-x_shift, None)
                    else:
                        x_slice_data = slice(x_shift, None)
                        x_slice_tmp = slice(None, tmp.shape[1] - x_shift)

                    if y_shift < 0:
                        y_slice_data = slice(None, tmp.shape[2] + y_shift)
                        y_slice_tmp = slice(-y_shift, None)
                    else:
                        y_slice_data = slice(y_shift, None)
                        y_slice_tmp = slice(None, tmp.shape[2] - y_shift)

                    data_slice = (slice(None), x_slice_data, y_slice_data)
                    tmp_slice = (slice(None), x_slice_tmp, y_slice_tmp)
                elif boundary is "continuous":
                    data_slice = (slice(None),) * 3
                    tmp_slice = (slice(None),) * 3

                values = \
                    np.nansum(((self.data[data_slice] - tmp[tmp_slice]) ** 2),
                              axis=0) / \
                    (np.nansum(self.data[data_slice] ** 2, axis=0) +
                     np.nansum(tmp[tmp_slice] ** 2, axis=0))

                scf_value = 1. - \
                    np.sqrt(np.nansum(values) / np.sum(np.isfinite(values)))
                self._scf_surface[i, j] = scf_value

    def compute_spectrum(self, logspacing=False, return_stddev=True,
                         **kwargs):
        '''
        Compute the 1D spectrum as a function of lag. Can optionally
        use log-spaced bins. kwargs are passed into the pspec function,
        which provides many options. The default settings are applicable in
        nearly all use cases.

        Parameters
        ----------
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins. Default is True.
        kwargs : passed to `turbustat.statistics.psds.pspec`
        '''

        # If scf_surface hasn't been computed, do it
        if self.scf_surface is None:
            self.compute_surface()

        if return_stddev:
            self._lags, self._scf_spectrum, self._scf_spectrum_stddev = \
                pspec(self.scf_surface, logspacing=logspacing,
                      return_stddev=return_stddev, return_freqs=False,
                      **kwargs)
            self._stddev_flag = True
        else:
            self._lags, self._scf_spectrum = \
                pspec(self.scf_surface, logspacing=logspacing,
                      return_freqs=False, **kwargs)
            self._stddev_flag = False

        roll_lag_diff = np.abs(self.roll_lags[1] - self.roll_lags[0])

        self._lags = self._lags * roll_lag_diff

    def fit_plaw(self, xlow=None, xhigh=None, verbose=False):
        '''
        Fit a power-law to the SCF spectrum.

        Parameters
        ----------
        xlow : `~astropy.units.Quantity`, optional
            Lower lag value limit to consider in the fit.
        xhigh : `~astropy.units.Quantity`, optional
            Upper lag value limit to consider in the fit.
        verbose : bool, optional
            Show fit summary when enabled.
        '''

        x = np.log10(self.lags.value)
        y = np.log10(self.scf_spectrum)

        if xlow is not None:
            if not isinstance(xlow, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")

            # Convert xlow into the same units as the lags
            xlow = self._spatial_unit_conversion(xlow, self.lags.unit)

            lower_limit = x >= np.log10(xlow.value)
        else:
            lower_limit = \
                np.ones_like(self.scf_spectrum, dtype=bool)

        if xhigh is not None:
            if not isinstance(xhigh, u.Quantity):
                raise TypeError("xlow must be an astropy.units.Quantity.")
            # Convert xhigh into the same units as the lags
            xhigh = self._spatial_unit_conversion(xhigh, self.lags.unit)

            upper_limit = x <= np.log10(xhigh.value)
        else:
            upper_limit = \
                np.ones_like(self.scf_spectrum, dtype=bool)

        within_limits = np.logical_and(lower_limit, upper_limit)

        if not within_limits.any():
            raise ValueError("Limits have removed all lag values. Make xlow"
                             " and xhigh less restrictive.")

        y = y[within_limits]
        x = x[within_limits]

        x = sm.add_constant(x)

        # If the std were computed, use them as weights
        if self._stddev_flag:

            # Converting to the log stds doesn't matter since the weights
            # remain proportional to 1/sigma^2, and an overal normalization is
            # applied in the fitting routine.
            weights = self.scf_spectrum_stddev[within_limits] ** -2

            model = sm.WLS(y, x, missing='drop', weights=weights)
        else:
            model = sm.OLS(y, x, missing='drop')

        self.fit = model.fit()

        if verbose:
            print(self.fit.summary())

        self._slope = self.fit.params[1]
        self._slope_err = self.fit.bse[1]

    @property
    def slope(self):
        '''
        SCF spectrum slope
        '''
        return self._slope

    @property
    def slope_err(self):
        '''
        1-sigma error on the SCF spectrum slope
        '''
        return self._slope_err

    def fitted_model(self, xvals):
        '''
        Computes the modelled power-law using the given x values.

        Parameters
        ----------
        xvals : `~astropy.Quantity`
            Values of lags to compute the model at.

        Returns
        -------
        model_values : `~numpy.ndarray`
            Values of the model at the given values. Equivalent to log values
            of the SCF spectrum.
        '''

        if not isinstance(xvals, u.Quantity):
            raise TypeError("xvals must be an astropy.units.Quantity.")

        # Convert into the lag units used for the fit
        xvals = self._spatial_unit_conversion(xvals, self.lags.unit).value

        model_values = \
            self.fit.params[0] + self.fit.params[1] * np.log10(xvals)

        return 10**model_values

    def save_results(self, output_name=None, keep_data=False):
        '''
        Save the results of the SCF to avoid re-computing.
        The pickled file will not include the data cube by default.

        Parameters
        ----------
        output_name : str, optional
            Name of the outputted pickle file.
        keep_data : bool, optional
            Save the data cube in the pickle file when enabled.
        '''

        if output_name is None:
            output_name = "scf_output.pkl"

        if output_name[-4:] != ".pkl":
            output_name += ".pkl"

        self_copy = deepcopy(self)

        # Don't keep the whole cube unless keep_data enabled.
        if not keep_data:
            self_copy._data = None

        with open(output_name, 'wb') as output:
                pickle.dump(self_copy, output, -1)

    @staticmethod
    def load_results(pickle_file):
        '''
        Load in a saved pickle file.

        Parameters
        ----------
        pickle_file : str
            Name of filename to load in.

        Returns
        -------
        self : SCF instance
            SCF instance with saved results.

        Examples
        --------
        Load saved results.
        >>> scf = SCF.load_results("scf_saved.pkl") # doctest: +SKIP

        '''

        with open(pickle_file, 'rb') as input:
                self = pickle.load(input)

        return self

    def run(self, logspacing=False, return_stddev=True, boundary='continuous',
            xlow=None, xhigh=None, save_results=False, output_name=None,
            verbose=False, xunit=u.pix, save_name=None):
        '''
        Computes all SCF outputs.

        Parameters
        ----------
        logspacing : bool, optional
            Return logarithmically spaced bins for the lags.
        return_stddev : bool, optional
            Return the standard deviation in the 1D bins.
        boundary : {"continuous", "cut"}
            Treat the boundary as continuous (wrap-around) or cut values
            beyond the edge (i.e., for most observational data).
        xlow : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        xhigh : `~astropy.Quantity`, optional
            See `~SCF.fit_plaw`.
        save_results : bool, optional
            Pickle the results.
        output_name : str, optional
            Name of the outputted pickle file.
        verbose : bool, optional
            Enables plotting.
        xunit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        self.compute_surface(boundary=boundary)
        self.compute_spectrum(logspacing=logspacing,
                              return_stddev=return_stddev)
        self.fit_plaw(verbose=verbose, xlow=xlow, xhigh=xhigh)

        if save_results:
            self.save_results(output_name=output_name)

        if verbose:
            import matplotlib.pyplot as p

            p.subplot(1, 2, 1)
            p.imshow(self.scf_surface, origin="lower", interpolation="nearest")
            cb = p.colorbar()
            cb.set_label("SCF Value")

            p.subplot(2, 2, 2)
            p.hist(self.scf_surface.ravel())
            p.xlabel("SCF Value")

            ax = p.subplot(2, 2, 4)
            lags = self._spatial_unit_conversion(self.lags, xunit).value

            if self._stddev_flag:
                ax.errorbar(lags, self.scf_spectrum,
                            yerr=self.scf_spectrum_stddev,
                            fmt='D', color='k', markersize=5, label="Data")
                ax.set_xscale("log", nonposy='clip')
                ax.set_yscale("log", nonposy='clip')
            else:
                p.loglog(self.lags, self.scf_spectrum, 'kD',
                         markersize=5, label="Data")

            ax.set_xlim(lags.min() * 0.75, lags.max() * 1.25)
            ax.set_ylim(self.scf_spectrum.min() * 0.75,
                        self.scf_spectrum.max() * 1.25)

            # Overlay the fit. Use points 5% lower than the min and max.
            xvals = np.linspace(lags.min() * 0.95,
                                lags.max() * 1.05, 50) * lags.unit
            p.loglog(xvals, self.fitted_model(xvals), 'r--', linewidth=2,
                     label='Fit')

            p.legend()

            ax.set_xlabel("Lag ({})".format(xunit))

            p.tight_layout()

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()
        return self


class SCF_Distance(object):

    '''
    Calculates the distance between two data cubes based on their SCF surfaces.
    The distance is the L2 norm between the surfaces. We weight the surface by
    1/r^2 where r is the distance from the centre.

    Parameters
    ----------
    cube1 : %(dtypes)s
        Data cube.
    cube2 : %(dtypes)s
        Data cube.
    size : int, optional
        Maximum size roll over which SCF will be calculated.
    boundary : {"continuous", "cut"}
        Treat the boundary as continuous (wrap-around) or cut values
        beyond the edge (i.e., for most observational data). A two element
        list can also be passed for treating the boundaries differently
        between the given cubes.
    fiducial_model : SCF
        Computed SCF object. Use to avoid recomputing.
    weighted : bool, optional
        Sets whether to apply the 1/r^2 weighting to the distance.
    '''

    __doc__ %= {"dtypes": " or ".join(common_types + threed_types)}

    def __init__(self, cube1, cube2, size=21 * u.pix, boundary='continuous',
                 fiducial_model=None, weighted=True):
        super(SCF_Distance, self).__init__()
        self.weighted = weighted

        dataset1 = input_data(cube1, no_header=False)
        dataset2 = input_data(cube2, no_header=False)

        # Create a default set of lags, in pixels
        if size % 2 == 0:
            Warning("Size must be odd. Reducing size to next lowest odd"
                    " number.")
            size = size - 1

        self.size = size
        roll_lags = np.arange(size) - size / 2

        # Now adjust the lags such they have a common scaling when the datasets
        # are not on a common grid.
        scale = common_scale(WCS(dataset1[1]), WCS(dataset2[1]))

        if scale == 1.0:
            roll_lags1 = roll_lags
            roll_lags2 = roll_lags
        elif scale > 1.0:
            roll_lags1 = scale * roll_lags
            roll_lags2 = roll_lags
        else:
            roll_lags1 = roll_lags
            roll_lags2 = roll_lags / float(scale)

        if not isinstance(boundary, basestring):
            if len(boundary) != 2:
                raise ValueError("If boundary is not a string, it must be a "
                                 "list or array of 2 string elements.")
        else:
            boundary = [boundary, boundary]

        if fiducial_model is not None:
            self.scf1 = fiducial_model
        else:
            self.scf1 = SCF(cube1, roll_lags=roll_lags1)
            self.scf1.run(return_stddev=True, boundary=boundary[0])

        self.scf2 = SCF(cube2, roll_lags=roll_lags2)
        self.scf2.run(return_stddev=True, boundary=boundary[1])

    def distance_metric(self, verbose=False, label1=None, label2=None,
                        ang_units=False, unit=u.deg, save_name=None):
        '''
        Compute the distance between the surfaces.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        label1 : str, optional
            Object or region name for cube1
        label2 : str, optional
            Object or region name for cube2
        ang_units : bool, optional
            Convert frequencies to angular units using the given header.
        unit : `~astropy.units.Unit`, optional
            Choose the angular unit to convert to when ang_units is enabled.
        save_name : str,optional
            Save the figure when a file name is given.
        '''

        # Since the angular scales are matched, we can assume that they will
        # have the same weights. So just use the shape of the lags to create
        # the weight surface.
        dx = np.arange(self.size) - self.size / 2
        dy = np.arange(self.size) - self.size / 2

        a, b = np.meshgrid(dx, dy)
        if self.weighted:
            # Centre pixel set to 1
            a[np.where(a == 0)] = 1.
            b[np.where(b == 0)] = 1.
            dist_weight = 1 / np.sqrt(a ** 2 + b ** 2)
        else:
            dist_weight = np.ones((self.size, self.size))

        difference = (self.scf1.scf_surface - self.scf2.scf_surface) ** 2. * \
            dist_weight
        self.distance = np.sqrt(np.sum(difference) / np.sum(dist_weight))

        if verbose:
            import matplotlib.pyplot as p

            # print "Distance: %s" % (self.distance)

            p.subplot(2, 2, 1)
            p.imshow(
                self.scf1.scf_surface, origin="lower", interpolation="nearest")
            p.title(label1)
            p.colorbar()
            p.subplot(2, 2, 2)
            p.imshow(
                self.scf2.scf_surface, origin="lower", interpolation="nearest",
                label=label2)
            p.title(label2)
            p.colorbar()
            p.subplot(2, 2, 3)
            p.imshow(difference, origin="lower", interpolation="nearest")
            p.title("Weighted Difference")
            p.colorbar()
            ax = p.subplot(2, 2, 4)
            if ang_units:
                lags1 = \
                    self.scf1.lags.to(unit,
                                      equivalencies=self.scf1.angular_equiv).value
                lags2 = \
                    self.scf2.lags.to(unit,
                                      equivalencies=self.scf2.angular_equiv).value
            else:
                lags1 = self.scf1.lags.value
                lags2 = self.scf2.lags.value

            ax.errorbar(lags1, self.scf1.scf_spectrum,
                        yerr=self.scf1.scf_spectrum_stddev,
                        fmt='D', color='b', markersize=5, label=label1)
            ax.errorbar(lags2, self.scf2.scf_spectrum,
                        yerr=self.scf2.scf_spectrum_stddev,
                        fmt='o', color='g', markersize=5, label=label2)
            ax.set_xscale("log", nonposy='clip')
            ax.set_yscale("log", nonposy='clip')

            ax.set_xlim(min(lags1.min(), lags2.min()) * 0.75,
                        max(lags1.max(), lags2.max()) * 1.25)
            ax.set_ylim(min(self.scf1.scf_spectrum.min(),
                            self.scf2.scf_spectrum.min()) * 0.75,
                        max(self.scf1.scf_spectrum.max(),
                            self.scf2.scf_spectrum.max()) * 1.25)

            # Overlay the fit. Use points 5% lower than the min and max.
            xvals = np.linspace(np.log10(min(lags1.min(),
                                             lags2.min()) * 0.95),
                                np.log10(max(lags1.max(),
                                             lags2.max()) * 1.05), 50)
            p.plot(10**xvals, 10**self.scf1.fitted_model(xvals), 'b--',
                   linewidth=2)
            p.plot(10**xvals, 10**self.scf2.fitted_model(xvals), 'g--',
                   linewidth=2)
            ax.legend(loc='upper right')
            p.tight_layout()

            if save_name is not None:
                p.savefig(save_name)
                p.close()
            else:
                p.show()
        return self
