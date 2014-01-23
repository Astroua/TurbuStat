'''

Implementation of Spatial Power Spectrum and Bispectrum as described in Burkhart et al. (2010)

'''

import numpy as np

class PowerSpectrum(object):
    """
    PowerSpectrum

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, img, header):
        super(PowerSpectrum, self).__init__()
        self.img = img
        self.header = header
        self.degperpix = np.abs(header["CDELT2"])

        self.ps2D = None
        self.ps1D = None
        self.freq = None

    def compute_pspec(self):
        '''
        Compute the power spectrum of MVC
        '''

        from scipy.fftpack import fft2, fftshift

        mvc_fft = fftshift(fft2(self.img.astype("f8")))

        self.ps2D = np.abs(mvc_fft)**2.


        return self

    def compute_radial_pspec(self, return_index=True, wavenumber=False, return_stddev=False, azbins=1, \
                             binsize=1.0, view=False, **kwargs):
        '''

        Computes the radially averaged power spectrum
        Based on Adam Ginsburg's code

        '''

        from psds import pspec

        self.freq, self.ps1D = pspec(self.ps2D, return_index=return_index, wavenumber=wavenumber, \
                                      return_stddev=return_stddev, azbins=azbins, binsize=binsize,\
                                      view=view, **kwargs)

        return self

    def run(self, phys_units=False, verbose=False):
        '''
        Full computation of the Spatial Power Spectrum
        '''

        self.compute_pspec()
        self.compute_radial_pspec(logspacing=True)

        if phys_units:
            self.freq *= self.degperpix**-1

        if verbose:
            import matplotlib.pyplot as p
            p.subplot(121)
            p.imshow(np.log10(self.ps2D), origin="lower", interpolation="nearest")
            p.colorbar()
            p.subplot(122)
            p.loglog(self.freq, self.ps1D, "bD")
            p.xlabel("log K")
            p.ylabel("Power (K)")
            p.show()


        return self

class PSpec_Distance(object):
    """

    Distance metric for the spatial power spectrum and wrapper for whole analysis

    INPUTS
    ------

    data1 - dictionary
            dictionary containing necessary property arrays

    data2 - dictionary
            dictionary containing necessary property arrays



    FUNCTIONS
    ---------

    mvc_distance - computes the distance metric for 2 datasets

    OUTPUTS
    -------

    distance - float
               value of distance metric

    results  - statsmodels class
               results of the linear fit

    """

    def __init__(self, data1, data2, fiducial_model=None):
        super(PSpec_Distance, self).__init__()

        self.shape1 = data1["integrated_intensity"][0].shape
        self.shape2 = data2["integrated_intensity"][0].shape

        if fiducial_model is None:
            self.pspec1 = PowerSpectrum(data1["integrated_intensity"][0]*data1["integrated_intensity_error"][0]**2., data1["integrated_intensity"][1])
            self.pspec1.run()
        else:
            self.pspec1 = fiducial_model

        self.pspec2 = PowerSpectrum(data2["integrated_intensity"][0]*data2["integrated_intensity_error"][0]**2., data2["integrated_intensity"][1])
        self.pspec2.run()

        self.results = None
        self.distance = None



    def distance_metric(self, verbose=False):
        '''

        Implements the distance metric for 2 Power Spectrum transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A statistical comparison is used on the powerlaw indexes.

        '''

        import statsmodels.formula.api as sm
        from pandas import Series, DataFrame

        ## Clipping from 8 pixels to half the box size
        ## Noise effects dominate outside this region
        clip_mask1 = np.zeros((self.pspec1.freq.shape))
        for i,x in enumerate(self.pspec1.freq):
            if x>8.0 and x<self.shape1[0]/2.:
                clip_mask1[i] = 1
        clip_freq1 = self.pspec1.freq[np.where(clip_mask1==1)]
        clip_ps1D1 = self.pspec1.ps1D[np.where(clip_mask1==1)]

        clip_mask2 = np.zeros((self.pspec2.freq.shape))
        for i,x in enumerate(self.pspec2.freq):
            if x>8.0 and x<self.shape2[0]/2.:
                clip_mask2[i] = 1
        clip_freq2 = self.pspec2.freq[np.where(clip_mask2==1)]
        clip_ps1D2 = self.pspec2.ps1D[np.where(clip_mask2==1)]


        dummy = [0] * len(clip_freq1) + [1] * len(clip_freq2)
        x = np.concatenate((np.log10(clip_freq1), np.log10(clip_freq2)))
        regressor = x.T * dummy
        constant = np.array([[1] * (len(clip_freq1) + len(clip_freq2))])

        log_ps1D = np.concatenate((np.log10(clip_ps1D1), np.log10(clip_ps1D2)))

        d = {"dummy": Series(dummy), "scales": Series(x), "log_ps1D": Series(log_ps1D), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(formula = "log_ps1D ~ dummy + scales + regressor", data = df)

        self.results = model.fit()

        self.distance = np.abs(self.results.tvalues["regressor"])

        if verbose:

            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(np.log10(clip_freq1), np.log10(clip_ps1D1), "bD", np.log10(clip_freq2), np.log10(clip_ps1D2), "gD")
            p.plot(df["scales"][:len(clip_freq1)], self.results.fittedvalues[:len(clip_freq1)], "b", \
                   df["scales"][-len(clip_freq2):], self.results.fittedvalues[-len(clip_freq2):], "g")
            p.grid(True)
            p.xlabel("log K")
            p.ylabel("Power (K)")
            p.show()

        return self


class BiSpectrum(object):
    """
    BiSpectrum

    INPUTS
    ------

    FUNCTIONS
    ---------

    OUTPUTS
    -------

    """
    def __init__(self, img, header):
        super(BiSpectrum, self).__init__()
        self.img = img
        self.header = header
        self.shape = img.shape

        # self.wavevectors = np.fft.fftfreq(min(img.shape)/2.)
        self.kx = np.arange(0., self.shape[0]/2.,1)
        self.ky = np.arange(0., self.shape[1]/2.,1)

        self.bispectrum = np.zeros((int(self.shape[0]/2.)-1, int(self.shape[1]/2.)-1), dtype=np.complex)
        self.bicoherence = np.zeros((int(self.shape[0]/2.)-1, int(self.shape[1]/2.)-1))
        self.bispectrum_amp = None
        self.accumulator = np.zeros((int(self.shape[0]/2.)-1, int(self.shape[1]/2.)-1))
        self.tracker = np.zeros(self.shape)

    def compute_bispectrum(self, nsamples=None, seed=1000):

        import numpy.random as ra
        fft = np.fft.fft2(self.img.astype("f8"))
        conjfft = np.conj(fft)
        ra.seed(seed)

        if nsamples is None:
            nsamples = 100#self.shape[0]*self.shape[1]

        # norm_samples = ra.normal(0, 1, size=(4,nsamples)) # Normally distributed points in 4D
        # norm_samples = np.abs(norm_samples) / np.sqrt((norm_samples**2).sum(axis=0)) # Normalize onto the surface of a 4D sphere and confine to the positive quadrant
        # samples = np.empty(norm_samples.shape)
        # samples[:2,:] = (self.shape[0]/2.) * ra.uniform(0,1,nsamples) * norm_samples[:2,:]
        # samples[2:,:] = (self.shape[1]/2.) * ra.uniform(0,1,nsamples) * norm_samples[2:,:]

        # for i in range(samples.shape[1]):
        #     k1x, k2x, k1y, k2y = [int(round(u)) for u in samples[:,i]]
        #     k3x = k1x + k2x
        #     k3y = k1y + k2y
        #     if k3x>self.shape[0] or k3y>self.shape[1]:
        #         print "BLAH"
        #     x = int(np.sqrt(k1x**2. + k1y**2.))
        #     y = int(np.sqrt(k2x**2. + k2y**2.))

        #     self.bispectrum[x,y] = fft[k1x,k1y] * fft[k2x,k2y] * conjfft[k3x,k3y]
        #     self.accumulator[x,y] += 1.

        for k1mag in range(int(fft.shape[0]/2.)):
           for k2mag in range(int(fft.shape[1]/2.)):
               phi1 = ra.uniform(0, 2*np.pi, nsamples)
               phi2  = ra.uniform(0,2*np.pi, nsamples)

               k1x =np.asarray([int(k1mag * np.cos(angle)) for angle in phi1])
               k2x =np.asarray([int(k2mag * np.cos(angle)) for angle in phi2])
               k1y =np.asarray([int(k1mag * np.sin(angle)) for angle in phi1])
               k2y =np.asarray([int(k2mag * np.sin(angle)) for angle in phi2])

               k3x = k1x + k2x
               k3y = k1y + k2y

               x = np.asarray([int(np.sqrt(i**2. + j**2.)) for i,j in zip(k1x,k1y)])
               y = np.asarray([int(np.sqrt(i**2. + j**2.)) for i,j in zip(k2x,k2y)])

               for n in range(nsamples):
                    self.bispectrum[x[n],y[n]] += fft[k1x[n],k1y[n]] * fft[k2x[n],k2y[n]] * conjfft[k3x[n],k3y[n]]
                    self.bicoherence[x[n],y[n]] += np.abs(fft[k1x[n],k1y[n]] * fft[k2x[n],k2y[n]] * conjfft[k3x[n],k3y[n]])
                    self.accumulator[x[n],y[n]] += 1.

               # Track where we're sampling from in fourier space
                    self.tracker[k1x[n],k1y[n]] += 1
                    self.tracker[k2x[n],k2y[n]] += 1
                    self.tracker[k3x[n],k3y[n]] += 1

        self.bicoherence = (np.abs(self.bispectrum)/self.bicoherence)
        self.bispectrum /= self.accumulator
        self.bispectrum_amp = np.log10(np.abs(self.bispectrum)**2.)

        return self






class BiSpec_Distance(object):
    """docstring for BiSpec_Distance"""
    def __init__(self, arg):
        super(BiSpec_Distance, self).__init__()
        self.arg = arg
        raise NotImplementedError("")