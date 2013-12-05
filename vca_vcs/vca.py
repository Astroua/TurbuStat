
'''

Implementation of VCA technique (Lazarian & Pogosyan)

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
    def __init__(self, cube, header, distances=None, slice_sizes=None):
        super(VCA, self).__init__()


        self.cube = cube.astype("float64")
        if np.isnan(self.cube).any():
            self.cube[np.isnan(self.cube)] = 0
        self.header = header
        self.distances = distances
        self.shape = self.cube.shape

        self.center_pixel = []
        for i_shape in self.shape:
            if i_shape % 2. != 0:
                self.center_pixel.append((i_shape-1)/2.)
            else:
                self.center_pixel.append(i_shape/2.)

        self.center_pixel = tuple(self.center_pixel)
        centering_array = np.ones(self.shape)
        centering_array[self.center_pixel] = 0

        self.distance_array = nd.distance_transform_edt(centering_array)

        if slice_sizes is None:
            self.slice_sizes = [1.0, 5.0, 10.0, 20.0]
        else:
            self.slice_sizes = slice_sizes

        self.degraded_cubes = []
        for size in self.slice_sizes:
            self.degraded_cubes.append(change_slice_thickness(self.cube, slice_thickness=size))


        if not distances:
            self.distances = np.unique(self.distance_array[np.nonzero(self.distance_array)])
        else:
            assert isinstance(distances, list)
            self.distances = distances

        self.correlation_array = np.ones(self.shape)
        self.correlation_spectrum = np.zeros((1,len(self.distances)))
        self.ps2D = []
        self.ps1D = []
        self.freq = None


    def compute_pspec(self):
        '''
        Compute the power spectrum of MVC of a single slice
        '''

        from scipy.fftpack import fftn

        for cube in self.degraded_cubes:
            mvc_fft = np.fft.fftshift(fftn(cube.astype("f8")))

            self.ps2D.append(np.abs(mvc_fft).sum(axis=0))

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

        return self

    def run(self, distances=None, view=True):
        '''
        Full computation of MVC
        '''

        if not distances:
            pass
        else:
            assert isinstance(distances, list)
            self.distances = distances

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=True)

        # av_ps1D = [np.nanmean(ps) for ps in zip(*self.ps1D)]

        if view:
            import matplotlib.pyplot as p

            num = len(self.slice_sizes)
            for i in range(1,num+1):
                p.subplot(num,2,2*i-1)
                p.loglog(self.freq, self.ps1D[i-1], "kD")
                p.xlabel(r"k (pixel$^{-1}$)")
                p.ylabel(r"P$_2(k)$")
                p.grid(True)
                p.subplot(num,2,2*i)
                p.imshow(np.log10(self.ps2D)[i-1],interpolation="nearest",origin="lower")
                p.colorbar()
            p.show()

        return self
