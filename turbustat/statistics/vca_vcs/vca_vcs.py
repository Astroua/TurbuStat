# Licensed under an MIT open source license - see LICENSE


import numpy as np
import statsmodels.api as sm
import warnings

try:
    from scipy.fftpack import fftn, fftfreq, fftshift
except ImportError:
    from numpy.fft import fftn, fftfreq, fftshift

from ..lm_seg import Lm_Seg
from ..psds import pspec
from slice_thickness import change_slice_thickness


class VCA(object):

    '''
    The VCA technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : numpy.ndarray
        Data cube.
    header : FITS header
        Corresponding FITS header.
    slice_sizes : float or int, optional
        Slices to degrade the cube to.
    phys_units : bool, optional
        Sets whether physical scales can be used.
    '''

    def __init__(self, cube, header, slice_size=None, phys_units=False):
        super(VCA, self).__init__()

        self.cube = cube.astype("float64")
        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            # Feel like this should be more specific
            self.good_channel_count = np.sum(self.cube.max(axis=0) != 0)
        self.header = header
        self.shape = self.cube.shape

        if slice_size is None:
            self.slice_size = 1.0

        if slice_size != 1.0:
            self.cube = \
                change_slice_thickness(self.cube.copy(),
                                       slice_thickness=self.slice_size)

        self.phys_units_flag = False
        if phys_units:
            self.phys_units_flag = True

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        vca_fft = fftshift(fftn(self.cube.astype("f8")))

        self.ps2D = (np.abs(vca_fft) ** 2.).sum(axis=0)

        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False,
                             return_stddev=False, azbins=1,
                             binsize=1.0, view=False, **kwargs):
        '''
        Computes the radially averaged power spectrum
        This uses Adam Ginsburg's code (see https://github.com/keflavich/agpy).
        See the above url for parameter explanations.
        '''

        self.freqs, self.ps1D = \
            pspec(self.ps2D, return_index=return_index,
                  wavenumber=wavenumber,
                  return_stddev=return_stddev,
                  azbins=azbins, binsize=binsize,
                  view=view, **kwargs)

        if self.phys_units_flag:
            self.freqs *= np.abs(self.header["CDELT2"]) ** -1.

        return self

    def fit_pspec(self, brk=None, log_break=True, low_cut=np.sqrt(2),
                  verbose=False):
        '''
        Fit the 1D Power spectrum using a segmented linear model. Note that
        the current implementation allows for only 1 break point in the
        model. If the break point is estimated via a spline, the breaks are
        tested, starting from the largest, until the model finds a good fit.

        Parameters
        ----------
        brk : float or None, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error. If None, a spline is used to estimate the breaks.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        lg_scale_cut : int, optional
            Cuts off largest scales, which deviate from the powerlaw.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

        # Make the data to fit to
        self.low_cut = low_cut
        x = np.log10(self.freqs[self.freqs > low_cut])
        y = np.log10(self.ps1D[self.freqs > low_cut])

        if brk is not None:
            # Try the fit with a break in it.
            if not log_break:
                brk = np.log10(brk)

            brk_fit = \
                Lm_Seg(x, y, brk)
            brk_fit.fit_model(verbose=verbose)

            if brk_fit.params.size == 5:

                x = x[x < brk_fit.brk]
                y = y[x < brk_fit.brk]

                self.high_cut = 10**brk_fit.brk

            else:
                self.high_cut = self.freqs.max()
                # Break fit failed, revert to normal model
                warnings.warn("Model with break failed, reverting to model\
                               without break.")
        else:
            self.high_cut = self.freqs.max()

        x = sm.add_constant(x)

        model = sm.OLS(y, x)

        self.fit = model.fit()

        self._slope = self.fit.params[1]

        cov_matrix = self.fit.cov_params()
        self._slope_err = np.sqrt(cov_matrix[1, 1])

        return self

    @property
    def slope(self):
        return self._slope

    @property
    def slope_err(self):
        return self._slope_err

    def plot_fit(self, show=True, show_2D=False, color='r', label=None):
        '''
        Plot the fitted model.
        '''

        import matplotlib.pyplot as p

        if self.phys_units_flag:
            xlab = r"log K"
        else:
            xlab = r"K (pixel$^{-1}$)"

        # 2D Spectrum is shown alongside 1D. Otherwise only 1D is returned.
        if show_2D:
            p.subplot(122)
            p.imshow(np.log10(self.ps2D), interpolation="nearest",
                     origin="lower")
            p.colorbar()

            p.subplot(121)

        good_interval = np.logical_and(self.freqs > self.low_cut,
                                       self.freqs <= self.high_cut)

        p.loglog(self.freqs[good_interval],
                 self.ps1D[good_interval], color+"D")

        y_fit = self.fit.fittedvalues
        p.loglog(self.freqs[good_interval], 10**y_fit, color+'-',
                 label=label)
        p.xlabel(xlab)
        p.ylabel(r"P$_2(K)$")
        p.grid(True)

        if show:
            p.show()

    def run(self, verbose=False, brk=None, **kwargs):
        '''
        Full computation of VCA.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        kwargs : passed to pspec.
        '''

        self.compute_pspec()
        self.compute_radial_pspec(**kwargs)
        self.fit_pspec(brk=brk)

        if verbose:
            self.plot_fit(show=True, show_2D=True)

        return self


class VCS(object):

    '''
    The VCS technique (Lazarian & Pogosyan, 2004).

    Parameters
    ----------
    cube : numpy.ndarray
        Data cube.
    header : FITS header
        Corresponding FITS header.
    phys_units : bool, optional
        Sets whether physical scales can be used.
    '''

    def __init__(self, cube, header, phys_units=False):
        super(VCS, self).__init__()

        self.header = header
        self.cube = cube
        self.fftcube = None
        self.correlated_cube = None
        self.ps1D = None
        self.phys_units = phys_units

        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            # Feel like this should be more specific
            self.good_pixel_count = np.sum(self.cube.max(axis=0) != 0)
        else:
            self.good_pixel_count = float(
                self.cube.shape[1] * self.cube.shape[2])

        # Lazy check to make sure we have units of km/s
        if np.abs(self.header["CDELT3"]) > 1:
            self.vel_to_pix = np.abs(self.header["CDELT3"]) / 1000.
        else:
            self.vel_to_pix = np.abs(self.header["CDELT3"])

        self.vel_channels = np.arange(1, self.cube.shape[0], 1)

        if self.phys_units:
            self.vel_freqs = np.abs(
                fftfreq(self.cube.shape[0])) / self.vel_to_pix
        else:
            self.vel_freqs = np.abs(fftfreq(self.cube.shape[0]))

    def compute_fft(self):
        '''
        Take the FFT of each spectrum in velocity dimension.
        '''

        self.fftcube = fftn(self.cube.astype("f8"))
        self.correlated_cube = np.abs(self.fftcube) ** 2.

        return self

    def make_ps1D(self):
        '''
        Create a 1D power spectrum by averaging the correlation cube over
        all pixels.
        '''

        self.ps1D = np.nansum(
            np.nansum(self.correlated_cube, axis=2), axis=1) /\
            self.good_pixel_count

        return self

    def fit_pspec(self, breaks=None, log_break=True, lg_scale_cut=2,
                  verbose=False):
        '''
        Fit the 1D Power spectrum using a segmented linear model. Note that
        the current implementation allows for only 1 break point in the
        model. If the break point is estimated via a spline, the breaks are
        tested, starting from the largest, until the model finds a good fit.

        Parameters
        ----------
        breaks : float or None, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error. If None, a spline is used to estimate the breaks.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        lg_scale_cut : int, optional
            Cuts off largest scales, which deviate from the powerlaw.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

        if breaks is None:
            from scipy.interpolate import UnivariateSpline

            # Need to order the points
            shape = self.vel_freqs.size
            spline_y = np.log10(self.ps1D[1:shape/2])
            spline_x = np.log10(self.vel_freqs[1:shape/2])

            spline = UnivariateSpline(spline_x, spline_y, k=1, s=1)

            # The first and last are just the min and max of x
            breaks = spline.get_knots()[1:-1]

            if verbose:
                print "Breaks found from spline are: " + str(breaks)

            # Take the number according to max_breaks starting at the
            # largest x.
            breaks = breaks[::-1]

            x = np.log10(self.vel_freqs[lg_scale_cut+1:-lg_scale_cut])
            y = np.log10(self.ps1D[lg_scale_cut+1:-lg_scale_cut])

            # Now try these breaks until a good fit including the break is
            # found. If none are found, it accept that there wasn't a good
            # break and continues on.
            i = 0
            while True:
                self.fit = \
                    Lm_Seg(x, y, breaks[i])
                self.fit.fit_model(verbose=verbose)

                if self.fit.params.size == 5:
                    break
                i += 1
                if i >= breaks.shape:
                    warnings.warn("No good break point found. Returned fit\
                                   does not include a break!")
                    break

            return self

        if not log_break:
            breaks = np.log10(breaks)

        self.fit = \
            Lm_Seg(np.log10(self.vel_freqs[lg_scale_cut+1:-lg_scale_cut]),
                   np.log10(self.ps1D[lg_scale_cut+1:-lg_scale_cut]), breaks)
        self.fit.fit_model(verbose=verbose)

        return self

    @property
    def slopes(self):
        return self.fit.slopes

    @property
    def slope_errs(self):
        return self.fit.slope_errs

    @property
    def brk(self):
        return self.fit.brk

    @property
    def brk_err(self):
        return self.fit.brk_err

    def run(self, verbose=False):
        '''
        Run the entire computation.

        Parameters
        ----------
        verbose: bool, optional
            Enables plotting.
        '''
        self.compute_fft()
        self.make_ps1D()
        self.fit_pspec(verbose=verbose)

        if verbose:
            import matplotlib.pyplot as p

            if self.phys_units:
                xlab = r"log k$_v$ $(km^{-1}s)$"
            else:
                xlab = r"log k (pixel$^{-1}$)"

            p.loglog(self.vel_freqs, self.ps1D, "bD", label='Data')
            p.loglog(10**self.fit.x, 10**self.fit.model(self.fit.x), 'r',
                     label='Fit', linewidth=2)
            p.xlabel(xlab)
            p.ylabel(r"log P$_{1}$(k$_{v}$)")
            p.grid(True)
            p.legend(loc='best')
            p.show()

        return self


class VCA_Distance(object):

    '''
    Calculate the distance between two cubes using VCA. The 1D power spectrum
    is modeled by a linear model. The distance is the t-statistic of the
    interaction between the two slopes.

    Parameters
    ----------
    cube1 : FITS hdu
        Data cube.
    cube2 : FITS hdu
        Data cube.
    slice_size : float, optional
        Slice to degrade the cube to.
    fiducial_model : VCA
        Computed VCA object. use to avoid recomputing.
    '''

    def __init__(self, cube1, cube2, slice_size=1.0, brk=None,
                 fiducial_model=None):
        super(VCA_Distance, self).__init__()
        cube1, header1 = cube1
        cube2, header2 = cube2
        self.shape1 = cube1.shape[1:]  # Shape of the plane
        self.shape2 = cube2.shape[1:]

        assert isinstance(slice_size, float)

        if fiducial_model is not None:
            self.vca1 = fiducial_model
        else:
            self.vca1 = VCA(cube1, header1, slice_size=slice_size).run(brk=brk, verbose=True)

        self.vca2 = VCA(cube2, header2, slice_size=slice_size).run(brk=brk)

    def distance_metric(self, labels=None, verbose=False):
        '''

        Implements the distance metric for 2 VCA transforms, each with the
        same channel width. We fit the linear portion of the transform to
        represent the powerlaw.

        Parameters
        ----------
        labels : list, optional
            Contains names of datacubes given in order.
        verbose : bool, optional
            Enables plotting.
        '''

        # Construct t-statistic
        self.distance = \
            np.abs((self.vca1.slope - self.vca2.slope) /
                   np.sqrt(self.vca1.slope_err**2 +
                           self.vca2.slope_err**2))

        if verbose:
            if labels is None:
                labels = ['1', '2']
            import matplotlib.pyplot as p
            self.vca1.plot_fit(show=False, color='b', label=labels[0])
            self.vca2.plot_fit(show=False, color='r', label=labels[1])
            p.show()

        return self


class VCS_Distance(object):

    '''
    Calculate the distance between two cubes using VCS. The 1D power spectrum
    is modeled by a broked linear model to account for the density and
    velocity dominated scales. The distance is the sum of  the t-statistics
    for each model.

    Parameters
    ----------
    cube1 : FITS hdu
        Data cube.
    cube2 : FITS hdu
        Data cube.
    slice_size : float, optional
        Slice to degrade the cube to.
    fiducial_model : VCA
        Computed VCA object. use to avoid recomputing.
    '''

    def __init__(self, cube1, cube2, slice_size=1.0, fiducial_model=None):
        super(VCS_Distance, self).__init__()
        self.cube1, self.header1 = cube1
        self.cube2, self.header2 = cube2

        assert isinstance(slice_size, float)

        if fiducial_model is not None:
            self.vcs1 = fiducial_model
        else:
            self.vcs1 = VCS(self.cube1, self.header1).run()

        self.vcs2 = VCS(self.cube2, self.header2).run()

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 VCS transforms.
        This distance is the t-statistic of the difference
        in the slopes.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        # Now construct the t-statistics for each portion

        # There should always be the velocity distance
        self.velocity_distance = \
            np.abs((self.vcs1.slopes[0] - self.vcs2.slopes[0]) /
                   np.sqrt(self.vcs1.slope_errs[0]**2 +
                           self.vcs2.slope_errs[0]**2))

        # A density distance is only found if a break was found
        if self.vcs1.slopes.size == 1 or self.vcs2.slopes.size == 1:
            self.density_distance = np.NaN
            self.break_distance = np.NaN
        else:
            self.density_distance = \
                np.abs((self.vcs1.slopes[1] - self.vcs2.slopes[1]) /
                       np.sqrt(self.vcs1.slope_errs[1]**2 +
                               self.vcs2.slope_errs[1]**2))

            self.break_distance = \
                np.abs((self.vcs1.brk - self.vcs2.brk) /
                       np.sqrt(self.vcs1.brk_err**2 +
                               self.vcs2.brk_err**2))

        # The overall distance is the sum from the two models
        self.distance = \
            np.nansum([self.velocity_distance, self.density_distance])

        if verbose:

            print "Fit 1"
            print self.vcs1.fit.fit.summary()
            print "Fit 2"
            print self.vcs2.fit.fit.summary()

            import matplotlib.pyplot as p
            p.plot(self.vcs1.fit.x, self.vcs1.fit.y, 'bD', alpha=0.3)
            p.plot(self.vcs1.fit.x, self.vcs1.fit.model(self.vcs1.fit.x), 'g',
                   label='Fit 1')
            p.plot(self.vcs2.fit.x, self.vcs2.fit.y, 'mD', alpha=0.3)
            p.plot(self.vcs2.fit.x, self.vcs2.fit.model(self.vcs2.fit.x), 'r',
                   label='Fit 2')
            p.grid(True)
            p.legend()
            p.xlabel(r"log k$_v$")
            p.ylabel(r"$P_{1}(k_v)$")
            p.show()

        return self
