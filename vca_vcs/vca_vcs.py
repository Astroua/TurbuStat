
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
    def __init__(self, cube, header, slice_sizes=None, phys_units=True):
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
                p.xlabel(xlab)
                p.ylabel(r"P$_2(K)$")
                p.grid(True)
                p.subplot(length,width,2*i)
                p.imshow(np.log10(self.ps2D)[i-1],interpolation="nearest",origin="lower")
                p.colorbar()
            p.show()

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

    def __init__(self, cube, header, slice_sizes=None, phys_units=True):
        super(VCS, self).__init__()

        self.header = header
        self.cube = cube
        self.fftcubes = []
        self.correlated_cubes = []
        self.ps1D = []

        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
            self.good_channel_count = np.sum(self.cube[0,:,:]!=0) ## Feel like this should be more specific
        else:
            self.good_channel_count = float(self.cube.shape[1] * self.cube.shape[2])

        self.phys_units_flag = False
        if phys_units:
            self.phys_units_flag = True

        if np.abs(self.header["CDELT3"])> 1: ## Lazy check to make sure we have units of km/s
            self.vel_to_pix = np.abs(self.header["CDELT3"])/1000.
        else:
            self.vel_to_pix = np.abs(self.header["CDELT3"])

        if slice_sizes is None:
            self.slice_sizes = [1.0, 5.0, 10.0, 25.0]
        else:
            self.slice_sizes = slice_sizes

        self.degraded_cubes = []
        self.vel_channels = []
        self.vel_freqs = []

        for size in self.slice_sizes:
            deg_cube = change_slice_thickness(self.cube, slice_thickness=size)
            self.degraded_cubes.append(deg_cube)
            self.vel_channels.append(np.arange(1, deg_cube.shape[0], 1))



    def compute_fft(self):
        '''

        Take the FFT of each spectrum in velocity dimension

        '''

        import scipy.fftpack as fft
        from numpy.fft import fftfreq

        for deg_cube in self.degraded_cubes:
            fftcube = fft.fftn(deg_cube.astype("f8"))
            corr_cube = np.abs(fftcube)**2.

            self.fftcubes.append(fftcube)
            self.correlated_cubes.append(corr_cube)

            if self.phys_units_flag:
                freqs = np.abs(fftfreq(corr_cube.shape[0]))/self.vel_to_pix ## units of s/km
            else:
                freqs = np.abs(fftfreq(corr_cube.shape[0]))

            self.vel_freqs.append(freqs)

        return self


    def make_ps1D(self):
        '''

        Create a 1D power spectrum by averaging the correlation cube over all pixels

        '''

        for corr_cube in self.correlated_cubes:
            self.ps1D.append(np.nansum(np.nansum(corr_cube, axis=2), axis=1)/self.good_channel_count)

        return self

    def run(self, verbose=False):

        self.compute_fft()
        self.make_ps1D()

        if verbose:
            import matplotlib.pyplot as p

            if self.phys_units_flag:
                xlab = r"log k$_v$ $(km^{-1}s)$"
            else:
                xlab = r"log k (pixel$^{-1}$)"

            num = len(self.slice_sizes)
            for i in range(1,num+1):
                p.subplot(num/2,2,i)
                p.loglog(self.vel_freqs[i-1], self.ps1D[i-1], "kD")
                p.xlabel(xlab )
                p.ylabel(r"log P$_{1}$(k$_{v}$)")
                p.grid(True)
            p.show()

        return self



class VCA_VCS_Distance(object):
    """


    """
    def __init__(self):
        super(VCA_VCS_Distance, self).__init__()

        raise NotImplementedError("Working on it...")