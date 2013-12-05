
'''

Implementation of the VCS method (Lazarian & Pogosyan)

'''

import numpy as np

from slice_thickness import change_slice_thickness

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

    def __init__(self, cube, header, slice_sizes=None):
        super(VCS, self).__init__()

        self.header = header
        self.cube = cube
        self.fftcubes = []
        self.correlated_cubes = []
        self.ps1D = []

        if np.isnan(self.cube).any():
            self.nanflag = True
        else:
            self.nanflag = False


        if np.abs(self.header["CDELT3"])> 1: ## Lazy check to make sure we have units of km/s
            vel_pix_division = np.abs(self.header["CDELT3"])/1000.
            reference_velocity = self.header["CRVAL3"]/1000.
        else:
            vel_pix_division = np.abs(self.header["CDELT3"])
            reference_velocity = self.header["CRVAL3"]

        if slice_sizes is None:
            self.slice_sizes = [1.0, 5.0, 10.0, 25.0]
        else:
            self.slice_sizes = slice_sizes

        self.degraded_cubes = []
        for size in self.slice_sizes:
            self.degraded_cubes.append(change_slice_thickness(self.cube, slice_thickness=size))

        # self.vel_channels = (np.arange(1,self.cube.shape[0]+1))# * vel_pix_division) + \
                                            #reference_velocity - (vel_pix_division * self.header["CRPIX3"])

        # self.vel_channels = np.concatenate([np.arange(self.cube.shape[0]/2.,0,-1),np.arange(0,self.cube.shape[0]/2.)])
        self.vel_freqs, self.vel_channels = find_velocity_freq(self.cube.shape[0], self.slice_sizes, vel_pix_division, \
                                            reference_velocity, self.header["CRPIX3"], phys_units=False)



    def compute_fft(self):
        '''

        Take the FFT of each spectrum in velocity dimension

        '''

        for deg_cube in self.degraded_cubes:
            fftcube, corr_cube = fft_over_channels(deg_cube, nanflag=self.nanflag)

            self.fftcubes.append(fftcube)
            self.correlated_cubes.append(corr_cube)



        return self


    def make_ps1D(self):
        '''

        Create a 1D power spectrum by averaging the correlation cube over all pixels

        '''
        from scipy.stats import nanmean

        for corr_cube in self.correlated_cubes:
            self.ps1D.append(nanmean(nanmean(corr_cube, axis=2), axis=1))

        return self

    def run(self, view=True):

        self.compute_fft()
        self.make_ps1D()

        if view:
            import matplotlib.pyplot as p

            num = len(self.slice_sizes)
            for i in range(1,num+1):
                p.subplot(num/2,2,i)
                p.loglog(self.vel_freqs[i-1], self.ps1D[i-1], "kD")
                p.xlabel(r" log(k$_v$)")
                p.ylabel(r" P$_{1}$(k)")
                p.grid(True)
            p.show()

        return self


def fft_over_channels(cube, nanflag=False):
    '''

    Take the FFT of each spectrum in velocity dimension

    '''

    from scipy.fftpack import fft, ifft
    from numpy.fft import fftshift
    if nanflag:
        from astropy.convolution import convolve_fft

    fftcube = np.empty(cube.shape, dtype=np.complex64)
    uniform_kernel = (-1) * np.ones(cube.shape[0])

    x_ind, y_ind = np.arange(0,cube.shape[1]), np.arange(0,cube.shape[2])

    for i in x_ind:
        for j in y_ind:
            if nanflag:
                ## This doesn't quite work
                fftcube[:,i,j] = convolve_fft(cube[:,i,j], ~np.isnan(cube[:,i,j]), interpolate_nan=True, \
                                          normalize_kernel=True, fftn=fft, ifftn=ifft)
            else:
                fftcube[:,i,j] = fftshift(fft(cube[:,i,j], n=cube.shape[0]))\
                                                      # [0.5*self.cube.shape[0]:1.5*self.cube.shape[0]])

    correlated_cube = np.abs(fftcube)


    return fftcube, correlated_cube

def find_velocity_freq(num_channels, slice_sizes, vel_pix_division, ref_velocity, ref_pixel, phys_units=False):
    '''

    Returns the new velocity indices of the degraded cubes

    INPUTS
    ------

    vel_channels - float
                   original number of velocity channels

    slice_sizes - list
                  list of slice sizes

    header - dictionary
             cube's FITS header

    phys_units - bool
                 returns velocities in physical units if True. False gives pixel units.

    '''

    if not isinstance(slice_sizes, list):
        slice_sizes = [slice_sizes]

    vel_freqs = []
    vel_channels = []

    ## velocity channels aranged to reflect symmetry about the zero point
    zero_pixel = round(ref_pixel - (ref_velocity/ vel_pix_division))-1
    print zero_pixel

    for size in slice_sizes:
        size = float(size)

        if phys_units:
            new_channels = (np.linspace(1,num_channels+1, (num_channels/size)) * vel_pix_division ) + \
                                (ref_velocity - (vel_pix_division * ref_pixel))
        else:
            if zero_pixel < 0.0:
                new_channels = np.arange(0, num_channels/size, 1)
            elif zero_pixel > num_channels:
                new_channels = np.arange(num_channels/size, 0, -1)
            else:
                new_channels = np.concatenate([np.linspace(0,zero_pixel, num_channels/(2*size)),\
                                                np.linspace(num_channels-zero_pixel,0, num_channels/(2*size))])

        vel_channels.append(new_channels)
        vel_freqs.append(1/new_channels)


    return vel_freqs, vel_channels