
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
                p.title("".join(["VCA with Thickness: ",str(self.slice_sizes[i-1])]))
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
    def __init__(self, cube1, cube2, slice_size):
        super(VCA_Distance, self).__init__()
        self.cube1, self.header1 = cube1
        self.cube2, self.header2 = cube2

    def distance(self, verbose=False):

        if verbose:
            import matplotlib.pyplot as p
            pass

class VCS_Distance(object):
    """docstring for VCS_Distance"""
    def __init__(self, arg):
        super(VCS_Distance, self).__init__()
        self.arg = arg
        raise NotImplementedError("")