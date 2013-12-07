
'''

Implementation of the VCA & VCS techniques (Lazarian & Pogosyan)

'''

import numpy as np
import scipy.ndimage as nd

from slice_thickness import change_slice_thickness

class VCA(object):
    """

    VCA technique

    INPUTS
    ------

    cube - array
           PPV data cube

    header - dictionary
             corresponding FITS header

    distances - list
                List of distances vector lengths
                If None, VCA is computed using the distances in the distance array



    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, cube, header, slice_sizes=None, phys_units=False):
        super(VCA, self).__init__()


        self.cube = cube.astype("float64")
        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            self.good_channel_count = np.sum(self.cube[0,:,:]!=0) ## Feel like this should be more specific
        self.header = header
        self.shape = self.cube.shape


        if slice_sizes is None:
            self.slice_sizes = [1.0, 5.0, 10.0, 20.0]
        else:
            self.slice_sizes = slice_sizes

        self.degraded_cubes = []
        for size in self.slice_sizes:
            self.degraded_cubes.append(change_slice_thickness(self.cube, slice_thickness=size))

        self.phys_units_flag = False
        if phys_units:
            self.phys_units_flag = True

        self.ps2D = []
        self.ps1D = []
        self.freq = None


    def compute_pspec(self):
        '''
        Compute the power spectrum of
        '''

        from scipy.fftpack import fftn

        for cube in self.degraded_cubes:
            mvc_fft = np.fft.fftshift(fftn(cube.astype("f8")))

            self.ps2D.append((np.abs(mvc_fft)**2.).sum(axis=0))

        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False, return_stddev=False, azbins=1, \
                             binsize=1.0, view=False, **kwargs):
        '''

        Computes the radially averaged power spectrum
        Based on Adam Ginsburg's code

        '''

        from psds import pspec

        for ps in self.ps2D:
            self.freq, ps1D = pspec(ps, return_index=return_index, wavenumber=wavenumber, \
                                      return_stddev=return_stddev, azbins=azbins, binsize=binsize,\
                                      view=view, **kwargs)
            self.ps1D.append(ps1D)

        if self.phys_units_flag:
            self.freq *= np.abs(self.header["CDELT2"])**-1.

        return self

    def run(self, verbose=False, **kwargs):
        '''
        Full computation of VCA
        '''

        self.compute_pspec()
        self.compute_radial_pspec(**kwargs)

        if verbose:
            import matplotlib.pyplot as p

            num = len(self.slice_sizes)
            if num <=4:
                width=2
                length = num
            else:
                width=4
                length = num/2

            if self.phys_units_flag:
                xlab = r"log K"
            else:
                xlab = r"K (pixel$^{-1}$)"

            for i in range(1,num+1):
                p.subplot(length,width,2*i-1)
                p.loglog(self.freq, self.ps1D[i-1], "kD")
                p.title("".join(["VCA with Thickness: ",str(self.slice_sizes[i-1])]))
                p.xlabel(xlab)
                p.ylabel(r"P$_2(K)$")
                p.grid(True)
                p.subplot(length,width,2*i)
                p.imshow(np.log10(self.ps2D)[i-1],interpolation="nearest",origin="lower")
                p.colorbar()
            p.show()

        if len(self.slice_sizes)==1:
            self.ps1D = self.ps1D[0]
            self.ps2D = self.ps2D[0]

        return self


class VCS(object):
    """

    VCS technique

    INPUTS
    ------

    cube - array
           3D array of data

    header - dictionary
             corresponding FITS header

    slice_thickness - float
                      velocity slice size for use in VCS in pixels. Minimum is 1.

    FUNCTIONS
    ---------

    OUTPUTS
    -------


    """

    def __init__(self, cube, header, phys_units=True):
        super(VCS, self).__init__()

        self.header = header
        self.cube = cube
        self.fftcube = None
        self.correlated_cube = None
        self.ps1D = None
        self.phys_units = phys_units

        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            self.good_channel_count = np.sum(self.cube[0,:,:]!=0) ## Feel like this should be more specific
        else:
            self.good_channel_count = float(self.cube.shape[1] * self.cube.shape[2])


        if np.abs(self.header["CDELT3"])> 1: ## Lazy check to make sure we have units of km/s
            self.vel_to_pix = np.abs(self.header["CDELT3"])/1000.
        else:
            self.vel_to_pix = np.abs(self.header["CDELT3"])

        self.vel_channels = np.arange(1, self.cube.shape[0], 1)

        from numpy.fft import fftfreq
        if self.phys_units:
            self.vel_freqs = np.abs(fftfreq(self.cube.shape[0]))/self.vel_to_pix
        else:
            self.vel_freqs = np.abs(fftfreq(self.cube.shape[0]))



    def compute_fft(self):
        '''

        Take the FFT of each spectrum in velocity dimension

        '''

        import scipy.fftpack as fft

        self.fftcube = fft.fftn(self.cube.astype("f8"))
        self.correlated_cube = np.abs(self.fftcube)**2.

        return self


    def make_ps1D(self):
        '''

        Create a 1D power spectrum by averaging the correlation cube over all pixels

        '''

        self.ps1D = np.nansum(np.nansum(self.correlated_cube, axis=2), axis=1)/self.good_channel_count

        return self

    def run(self, verbose=False):

        self.compute_fft()
        self.make_ps1D()

        if verbose:
            import matplotlib.pyplot as p

            if self.phys_units:
                xlab = r"log k$_v$ $(km^{-1}s)$"
            else:
                xlab = r"log k (pixel$^{-1}$)"

            p.loglog(self.vel_freqs, self.ps1D, "bD-")
            p.xlabel(xlab)
            p.ylabel(r"log P$_{1}$(k$_{v}$)")
            p.grid(True)
            p.show()

        return self



class VCA_Distance(object):
    """docstring for VCA_Distance"""
    def __init__(self, cube1, cube2, slice_size=1.0):
        super(VCA_Distance, self).__init__()
        self.cube1, self.header1 = cube1
        self.cube2, self.header2 = cube2
        self.shape1 = self.cube1.shape[1:] # Shape of the plane
        self.shape2 = self.cube2.shape[1:]

        assert isinstance(slice_size, float)
        self.vca1 = VCA(self.cube1, self.header1, slice_sizes=[slice_size]).run()
        self.vca2 = VCA(self.cube2, self.header2, slice_sizes=[slice_size]).run()

    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 VCA transforms, each with the same channel width.
        We fit the linear portion of the transform to represent the powerlaw
        A statistical comparison is used on the powerlaw indexes.

        '''

        import statsmodels.formula.api as sm
        from pandas import Series, DataFrame

        ## Clipping from 8 pixels to half the box size
        ## Noise effects dominate outside this region
        clip_mask1 = np.zeros((self.vca1.freq.shape))
        for i,x in enumerate(self.vca1.freq):
            if x>8.0 and x<self.shape1[0]/2.:
                clip_mask1[i] = 1
        clip_freq1 = self.vca1.freq[np.where(clip_mask1==1)]
        clip_ps1D1 = self.vca1.ps1D[np.where(clip_mask1==1)]

        clip_mask2 = np.zeros((self.vca2.freq.shape))
        for i,x in enumerate(self.vca2.freq):
            if x>8.0 and x<self.shape2[0]/2.:
                clip_mask2[i] = 1
        clip_freq2 = self.vca2.freq[np.where(clip_mask2==1)]
        clip_ps1D2 = self.vca2.ps1D[np.where(clip_mask2==1)]


        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy
        constant = np.array([[1] * (len(clip_freq1) + len(clip_freq2))])

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(formula = "log_ps1D ~ dummy + scales + regressor", data = df)

        self.results = model.fit()

        self.distance = self.results.tvalues["regressor"]

        if verbose:

            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_freq1), np.log10(clip_ps1D1), "bD", np.log10(clip_freq2), np.log10(clip_ps1D2), "gD")
            p.plot(df["scales"][:len(clip_freq1)], self.results.fittedvalues[:len(clip_freq1)], "b", \
                   df["scales"][-len(clip_freq2):], self.results.fittedvalues[-len(clip_freq2):], "g")
            p.grid(True)
            p.xlabel("log K")
            p.ylabel(r"$P_{2}(K)$")
            p.show()

        return self

class VCS_Distance(object):
    """docstring for VCS_Distance"""
    def __init__(self, arg):
        super(VCS_Distance, self).__init__()
        self.arg = arg
        raise NotImplementedError("")