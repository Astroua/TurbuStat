# Licensed under an MIT open source license - see LICENSE


import numpy as np
from ..psds import pspec
import statsmodels.formula.api as sm
from pandas import Series, DataFrame
from ..lm_seg import Lm_Seg

try:
    from scipy.fftpack import fftn, fftfreq, fftshift
except ImportError:
    from numpy.fft import fftn, fftfreq, fftshift


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
    slice_sizes : list or numpy.ndarray, optional
        Slices to degrade the cube to.
    phys_units : bool, optional
        Sets whether physical scales can be used.
    '''

    def __init__(self, cube, header, slice_sizes=None, phys_units=False):
        super(VCA, self).__init__()

        self.cube = cube.astype("float64")
        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            # Feel like this should be more specific
            self.good_channel_count = np.sum(self.cube.max(axis=0) != 0)
        self.header = header
        self.shape = self.cube.shape

        if slice_sizes is None:
            self.slice_sizes = [1.0, 5.0, 10.0, 20.0]
        else:
            self.slice_sizes = slice_sizes

        self.degraded_cubes = []
        for size in self.slice_sizes:
            self.degraded_cubes.append(
                change_slice_thickness(self.cube, slice_thickness=size))

        self.phys_units_flag = False
        if phys_units:
            self.phys_units_flag = True

        self.ps2D = []
        self.ps1D = []
        self.freq = None

    def compute_pspec(self):
        '''
        Compute the 2D power spectrum.
        '''

        for cube in self.degraded_cubes:
            vca_fft = fftshift(fftn(cube.astype("f8")))

            self.ps2D.append((np.abs(vca_fft) ** 2.).sum(axis=0))

        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False,
                             return_stddev=False, azbins=1,
                             binsize=1.0, view=False, **kwargs):
        '''
        Computes the radially averaged power spectrum
        This uses Adam Ginsburg's code (see https://github.com/keflavich/agpy).
        See the above url for parameter explanations.
        '''

        for ps in self.ps2D:
            self.freq, ps1D = pspec(ps, return_index=return_index,
                                    wavenumber=wavenumber,
                                    return_stddev=return_stddev,
                                    azbins=azbins, binsize=binsize,
                                    view=view, **kwargs)
            self.ps1D.append(ps1D)

        if self.phys_units_flag:
            self.freq *= np.abs(self.header["CDELT2"]) ** -1.

        return self

    def run(self, verbose=False, **kwargs):
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

        if verbose:
            import matplotlib.pyplot as p

            num = len(self.slice_sizes)
            if num <= 4:
                width = 2
                length = num
            else:
                width = 4
                length = num / 2

            if self.phys_units_flag:
                xlab = r"log K"
            else:
                xlab = r"K (pixel$^{-1}$)"

            for i in range(1, num + 1):
                p.subplot(length, width, 2 * i - 1)
                p.loglog(self.freq, self.ps1D[i - 1], "kD")
                p.title(
                    "".join(["VCA with Thickness: ", str(self.slice_sizes[i - 1])]))
                p.xlabel(xlab)
                p.ylabel(r"P$_2(K)$")
                p.grid(True)
                p.subplot(length, width, 2 * i)
                p.imshow(
                    np.log10(self.ps2D)[i - 1], interpolation="nearest",
                    origin="lower")
                p.colorbar()
            p.show()

        if len(self.slice_sizes) == 1:
            self.ps1D = self.ps1D[0]
            self.ps2D = self.ps2D[0]

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

    def fit_pspec(self, breaks=-1.0, log_break=True, lg_scale_cut=2,
                  verbose=False):
        '''
        Fit the 1D Power spectrum using a segmented linear model.

        Parameters
        ----------
        breaks : float or list, optional
            Guesses for the break points. If given as a list, the length of
            the list sets the number of break points to be fit. If a choice is
            outside of the allowed range from the data, Lm_Seg will raise an
            error.
        log_break : bool, optional
            Sets whether the provided break estimates are log-ed values.
        lg_scale_cut : int, optional
            Cuts off largest scales, which deviate from the powerlaw.
        verbose : bool, optional
            Enables verbose mode in Lm_Seg.
        '''

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

    def __init__(self, cube1, cube2, slice_size=1.0, fiducial_model=None):
        super(VCA_Distance, self).__init__()
        cube1, header1 = cube1
        cube2, header2 = cube2
        self.shape1 = cube1.shape[1:]  # Shape of the plane
        self.shape2 = cube2.shape[1:]

        assert isinstance(slice_size, float)

        if fiducial_model is not None:
            self.vca1 = fiducial_model
        else:
            self.vca1 = VCA(cube1, header1, slice_sizes=[slice_size]).run()

        self.vca2 = VCA(cube2, header2, slice_sizes=[slice_size]).run()

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 VCA transforms, each with the
        same channel width. We fit the linear portion of the transform to
        represent the powerlaw.

        Parameters
        ----------
        verbose : bool, optional
            Enables plotting.
        '''

        # Clipping from 8 pixels to half the box size
        # Noise effects dominate outside this region
        clip_mask1 = np.zeros((self.vca1.freq.shape))
        for i, x in enumerate(self.vca1.freq):
            if x > 8.0 and x < self.shape1[0] / 2.:
                clip_mask1[i] = 1
        clip_freq1 = self.vca1.freq[np.where(clip_mask1 == 1)]
        clip_ps1D1 = self.vca1.ps1D[np.where(clip_mask1 == 1)]

        clip_mask2 = np.zeros((self.vca2.freq.shape))
        for i, x in enumerate(self.vca2.freq):
            if x > 8.0 and x < self.shape2[0] / 2.:
                clip_mask2[i] = 1
        clip_freq2 = self.vca2.freq[np.where(clip_mask2 == 1)]
        clip_ps1D2 = self.vca2.ps1D[np.where(clip_mask2 == 1)]

        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(
            x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(
            formula="log_ps1D ~ dummy + scales + regressor", data=df)

        self.results = model.fit()

        self.distance = np.abs(self.results.tvalues["regressor"])

        if verbose:

            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_freq1), np.log10(clip_ps1D1), "bD",
                   np.log10(clip_freq2), np.log10(clip_ps1D2), "gD")
            p.plot(df["scales"][:len(clip_freq1)],
                   self.results.fittedvalues[:len(clip_freq1)], "b",
                   df["scales"][-len(clip_freq2):],
                   self.results.fittedvalues[-len(clip_freq2):], "g")
            p.grid(True)
            p.xlabel("log K")
            p.ylabel(r"$P_{2}(K)$")
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

        self.velocity_distance = \
            np.abs((self.vcs1.slopes[0] - self.vcs2.slopes[0]) /
                   np.sqrt(self.vcs1.slope_errs[0]**2 +
                           self.vcs2.slope_errs[0]**2))

        self.density_distance = \
            np.abs((self.vcs1.slopes[1] - self.vcs2.slopes[1]) /
                   np.sqrt(self.vcs1.slope_errs[1]**2 +
                           self.vcs2.slope_errs[1]**2))

        self.break_distance = \
            np.abs((self.vcs1.brk - self.vcs2.brk) /
                   np.sqrt(self.vcs1.brk_err**2 + self.vcs2.brk_err**2))

        # The overall distance is the sum from the two models
        self.distance = self.velocity_distance + self.density_distance

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
