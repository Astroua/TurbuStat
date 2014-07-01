
import numpy as np
import warnings
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import statsmodels.formula.api as sm
from pandas import Series, DataFrame

try:
    from scipy.fftpack import fftn, ifftn, fftfreq
except ImportError:
    from numpy.fft import fftn, ifftn, fftfreq


class Mexican_hat():
    '''
    Implements the Mexican hat wavelet class.
    Code is from kPyWavelet.
    '''

    name = 'Mexican hat'

    def __init__(self):
        # Reconstruction factor $C_{\psi, \delta}$
        self.cpsi = 1.  # pi

    def psi_ft(self, k, l):
        '''
        Fourier transform of the Mexican hat wavelet as in Wang and
        Lu (2010), equation [15].
        '''
        K, L = np.meshgrid(k, l)
        return (K ** 2. + L ** 2.) * np.exp(-0.5 * (K ** 2. + L ** 2.))

    def psi(self, x, y):
        '''
        Mexican hat wavelet as in Wang and Lu (2010), equation [14].
        '''
        X, Y = np.meshgrid(x, y)
        return (2. - (X ** 2. + Y ** 2.)) * np.exp(-0.5 * (X ** 2. + Y ** 2.))


class wt2D(object):
    '''
    Compute the wavelet transform of a 2D array.

    Parameters
    ----------
    array : numpy.ndarray
        2D array.
    header : FITS header
        Header for the array.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    dx : float, optional
        Spacing in the x-direction.
    dy : float, optional
        Spacing in the y-direction.
    wavelet : wavelet class
        The wavelet class to use.
    '''

    def __init__(self, array, header, scales, dx=0.25, dy=0.25,
                 wavelet=Mexican_hat(), ang_units=True):
        super(wt2D, self).__init__()
        self.array = array.astype("f8")
        self.header = header
        self.scales = scales
        self.wavelet = wavelet

        ### NOTE: can't use nan_interpolating from astropy
        ### until the normalization for sum to zeros kernels is fixed!!!
        self.array[np.isnan(self.array)] = np.nanmin(self.array)

        self.nan_flag = False
        if np.isnan(self.array).any():
            self.nan_flag = True

        if ang_units:
            try:
                self.imgscale = np.abs(self.header["CDELT2"])
            except ValueError:
                warnings.warn("Header doesn't not contain the\
                               angular size. Reverting to pixel scales.")
                ang_units = False
        if not ang_units:
            self.imgscale = 1.0

        a_min = 5 / 3.  # Minimum scale size given by Gill and Henriksen (90)
        self.dx = dx * a_min
        self.dy = dy * a_min

        self.Wf = None
        self.iWf = None

    def cwt2d(self, dx=None, dy=None):
        '''
        Bi-dimensional continuous wavelet transform of the signal at
        specified scale a.

        Parameters
        ----------
        dx : float, optional
            Spacing in the x-direction.
        dy : float, optional
            Spacing in the y-direction.
        '''
        if dx is not None:
            assert isinstance(dx, list)
            self.dx = dx
        if dy is not None:
            assert isinstance(dy, list)
            self.dx = dy

        # Determines the shape of the arrays and the discrete scales.
        n0, m0 = self.array.shape
        N, M = 2 ** int(np.ceil(np.log2(n0))), 2 ** int(np.ceil(np.log2(m0)))
        if self.scales is None:
            self.scales = 2 ** np.arange(int(np.floor(np.log2(min(n0, m0)))))
        A = len(self.scales)
        # Calculates the zonal and meridional wave numbers.
        l, k = fftfreq(N, self.dy), fftfreq(M, self.dx)

        # Calculates the Fourier transform of the input signal.
        f_ft = fftn(self.array, shape=(N, M))
        # Creates empty wavelet transform array and fills it for every discrete
        # scale using the convolution theorem.
        self.Wf = np.zeros((A, N, M), 'complex')
        for i, an in enumerate(self.scales):
            psi_ft_bar = an * self.wavelet.psi_ft(an * k, an * l)
            self.Wf[i, :, :] = ifftn(f_ft * psi_ft_bar, shape=(N, M))

        self.Wf = self.Wf[:, :n0, :m0]

        return self

    def astropy_cwt2d(self, dx=None, dy=None):
        '''
        Same as cwt2D except it uses astropy.convolve_fft's ability
        to interpolate over NaNs.

        Parameters
        ----------
        dx : float, optional
            Spacing in the x-direction.
        dy : float, optional
            Spacing in the y-direction.
        '''

        if dx is not None:
            assert isinstance(dx, list)
            self.dx = dx
        if dy is not None:
            assert isinstance(dy, list)
            self.dx = dy

        n0, m0 = self.array.shape
        N, M = 2 ** int(np.ceil(np.log2(n0))), 2 ** int(np.ceil(np.log2(m0)))
        if self.scales is None:
            self.scales = 2 ** np.arange(int(np.floor(np.log2(min(n0, m0)))))
        A = len(self.scales)

        self.Wf = np.zeros((A, N, M), 'complex')

        for i, an in enumerate(self.scales):
            psi = MexicanHat2DKernel(an, x_size=n0, y_size=m0)
            self.Wf[i, :, :] = convolve_fft(self.array, psi,
                                            interpolate_nan=True,
                                            normalize_kernel=True,
                                            fftn=fftn, ifftn=ifftn)

        self.Wf = self.Wf[:, :n0, :m0]

        return self

    def icwt2d(self, da=0.25):
        '''
        Inverse bi-dimensional continuous wavelet transform as in Wang and
        Lu (2010), equation [5].

        Parameters
        ----------
        da : float, optional
            Spacing in the frequency axis.
        '''
        if self.Wf is None:
            raise TypeError("Run cwt2D before icwt2D")
        m0, l0, k0 = self.Wf.shape

        if m0 != self.scales.size:
            raise Warning('Scale parameter array shape does not match\
                           wavelet transform array shape.')
        # Calculates the zonal and meridional wave numters.
        L, K = 2 ** int(np.ceil(np.log2(l0))), 2 ** int(np.ceil(np.log2(k0)))
        # Calculates the zonal and meridional wave numbers.
        l, k = fftfreq(L, self.dy), fftfreq(K, self.dx)
        # Creates empty inverse wavelet transform array and fills it for every
        # discrete scale using the convolution theorem.
        self.iWf = np.zeros((m0, L, K), 'complex')
        for i, an in enumerate(self.scales):
            psi_ft_bar = an * self.wavelet.psi_ft(an * k, an * l)
            W_ft = fftn(self.Wf[i, :, :], s=(L, K))
            self.iWf[i, :, :] = ifftn(W_ft * psi_ft_bar, s=(L, K)) *\
                da / an ** 2.

        self.iWf = self.iWf[:, :l0, :k0].real.sum(axis=0) / self.wavelet.cpsi

        return self

    def make_1D_transform(self):
        self.curve = transform((self.Wf, self.scales), self.imgscale)

    def run(self):
        '''
        Compute the Wavelet transform.
        '''
        if self.nan_flag:
            self.astropy_cwt2d()
        else:
            self.cwt2d()
        self.make_1D_transform()


class Wavelet_Distance(object):
    '''
    Compute the distance between the two cubes using the Wavelet transform.
    We fit a linear model to the two wavelet transforms. The distance is the
    t-statistic of the interaction term describing the difference in the
    slopes.

    Parameters
    ----------
    dataset1 : FITS hdu
        2D image.
    dataset2 : FITS hdu
        2D image.
    wavelet : class
        Wavelet class. Only Mexican_hat() is implemented.
    ang_units : bool, optional
        Sets whether to use angular units.
    scales : numpy.ndarray or list
        The scales where the transform is calculated.
    num : int
        Number of scales to calculate the transform at.
    dx : float, optional
        Spacing in the x-direction.
    dy : float, optional
        Spacing in the y-direction.
    fiducial_model : wt2D
        Computed wt2D object. use to avoid recomputing.
    '''

    def __init__(self, dataset1, dataset2, wavelet=Mexican_hat(),
                 ang_units=True, scales=None, num=50, dx=0.25, dy=0.25,
                 fiducial_model=None):
        super(Wavelet_Distance, self).__init__()

        array1 = dataset1[0]
        header1 = dataset1[1]
        array2 = dataset2[0]
        header2 = dataset2[1]
        self.wavelet = wavelet
        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            self.scales1 = np.logspace(
                np.log10(a_min), np.log10(min(array1.shape)), num)
            self.scales2 = np.logspace(
                np.log10(a_min), np.log10(min(array2.shape)), num)
        else:
            self.scales1 = scales
            self.scales2 = scales

        if fiducial_model is None:
            self.wt1 = wt2D(array1, header1, self.scales1, wavelet=wavelet,
                            ang_units=ang_units)
            self.wt1.run()
        else:
            self.wt1 = fiducial_model

        self.wt2 = wt2D(array2, header2, self.scales2, wavelet=wavelet,
                        ang_units=ang_units)
        self.wt2.run()

        self.curve1 = self.wt1.curve
        self.curve2 = self.wt2.curve

        self.results = None
        self.distance = None

    def distance_metric(self, non_linear=True, verbose=False):
        '''
        Implements the distance metric for 2 wavelet transforms.
        We fit the linear portion of the transform to represent the powerlaw

        Parameters
        ----------
        non_linear : bool, optional
            Enables clipping of non-linear portions of the transform.
        verbose : bool, optional
            Enables plotting.
        '''

        if non_linear:
            self.curve1 = clip_to_linear(self.curve1)
            self.curve2 = clip_to_linear(self.curve2)

        dummy = [0] * len(self.curve1[0, :]) + [1] * len(self.curve2[0, :])
        x = np.concatenate((self.curve1[0, :], self.curve2[0, :]))
        regressor = x.T * dummy

        log_T_g = np.concatenate((self.curve1[1, :], self.curve2[1, :]))

        d = {"dummy": Series(dummy), "scales": Series(
            x), "log_T_g": Series(log_T_g), "regressor": Series(regressor)}

        df = DataFrame(d)

        model = sm.ols(formula="log_T_g ~ dummy + scales + regressor", data=df)

        self.results = model.fit()

        self.distance = np.abs(self.results.tvalues["regressor"])

        if verbose:
            print self.results.summary()

            import matplotlib.pyplot as p
            p.plot(self.curve1[0, :], self.curve1[1, :], 'bD',
                   self.curve2[0, :], self.curve2[1, :], 'gD')
            p.plot(self.curve1[0, :],
                   self.results.fittedvalues[:len(self.curve1[1, :])], "b",
                   self.curve2[0, :],
                   self.results.fittedvalues[-len(self.curve2[1, :]):], "g")
            p.grid(True)
            p.xlabel("log a")
            p.ylabel(r"log $T_g$")
            p.show()

        return self


def clip_to_linear(data, threshold=1.0, kernel_width=0.1, ends_clipped=0.05):
    '''
    Takes the second derivative of the data with a ricker wavelet.
    Data is clipped to the linear portion (2nd derivative ~ 0)

    Parameters
    ----------

    data : numpy.ndarray
        x and y data.
    threshold : float, optional
        Acceptable value of the second derivative to be called linear.
    kernel_width : float, optional
        Kernel width set to this percentage of the data length
    ends_clipped : float, optional
        Percentage of data to clip off at the ends. End points have residual
        effects from the convolution.

    Returns
    -------
    data_clipped : numpy.ndarray
        Linear portion of the data set returned.
    '''

    from scipy.signal import ricker

    y = data[1, :]
    x = data[0, :]

    num_pts = len(y)

    kernel = ricker(num_pts, num_pts * kernel_width)

    sec_deriv = np.convolve(y, kernel, mode="same")

    # Ends go back to being ~ linear, so clip them off
    if ends_clipped > 0.0:
        clipped_pts = int(num_pts * ends_clipped)

        sec_deriv = sec_deriv[clipped_pts: num_pts - clipped_pts]
        y = y[clipped_pts: num_pts - clipped_pts]
        x = x[clipped_pts: num_pts - clipped_pts]

    linear_pts = np.abs(sec_deriv) < threshold

    data_clipped = np.empty((2, len(y[linear_pts])))
    data_clipped[:, :] = x[linear_pts], y[linear_pts]

    return data_clipped


def transform(data, imgscale):
    '''
    Put output of the wavelet transform into the mean of the nonzero components
    This reduces the dataset to 1D.

    Parameters
    ----------
    data : tuple
        Contains N arrays and scales from the transform.

    Returns
    -------
    data_1D - numpy.ndarray
        Scales in the first column and log <T_g> in the second.
    '''

    wav_arrays = data[0]
    scales = data[1]

    log_av_T_g = []
    for i in range(len(scales)):
        average_Tg_i = np.log10(np.abs(wav_arrays[i, :, :]
                                [wav_arrays[i, :, :] > 0]).mean())
        log_av_T_g.append(average_Tg_i)

    physical_scales = np.log10(scales * imgscale)

    data_1D = np.array([physical_scales, log_av_T_g])

    return data_1D
