
'''

Applying 2D Continuous Wavelet Transform to data curve properties as shown in
Gill and Henriksen, 1990

Based on code from kPyWavelet and astropy.nddata.convolution.convolve

'''

import numpy as np
from astropy.convolution import convolve_fft, MexicanHat2DKernel
import statsmodels.formula.api as sm
from pandas import Series, DataFrame

try:
    from scipy.fftpack import fftn, ifftn, fftfreq
except ImportError:
    from numpy.fft import fftn, ifftn, fftfreq


class Mexican_hat():

    """

    Implements the Mexican hat wavelet class.
    From kPyWavelet

    """

    name = 'Mexican hat'

    def __init__(self):
        # Reconstruction factor $C_{\psi, \delta}$
        self.cpsi = 1.  # pi

    def psi_ft(self, k, l):
        """
        Fourier transform of the Mexican hat wavelet as in Wang and
        Lu (2010), equation [15].

        """
        K, L = np.meshgrid(k, l)
        return (K ** 2. + L ** 2.) * np.exp(-0.5 * (K ** 2. + L ** 2.))

    def psi(self, x, y):
        """Mexican hat wavelet as in Wang and Lu (2010), equation [14]."""
        X, Y = np.meshgrid(x, y)
        return (2. - (X ** 2. + Y ** 2.)) * np.exp(-0.5 * (X ** 2. + Y ** 2.))


class wt2D(object):

    """docstring for wt2D"""

    def __init__(self, array, scales, dx=0.25, dy=0.25, wavelet=Mexican_hat()):
        super(wt2D, self).__init__()
        self.array = array.astype("f8")
        self.scales = scales
        self.wavelet = wavelet

        ### NOTE: can't use nan_interpolating from astropy until the normalization
        ### for sum to zeros kernels is fixed!!!
        self.array[np.isnan(self.array)] = np.nanmin(self.array)

        self.nan_flag = False
        if np.isnan(self.array).any():
            self.nan_flag = True

        a_min = 5 / 3.  # Minimum scale size given by Gill and Henriksen (90)
        self.dx = dx * a_min
        self.dy = dy * a_min

        self.Wf = None
        self.iWf = None

    def cwt2d(self, dx=None, dy=None):
        """
        Bi-dimensional continuous wavelet transform of the signal at
        specified scale a.

        PARAMETERS
            f (array like):
                Input signal array.
            dx, dy (float):
                Sample spacing for each dimension.
            a (array like, optional):
                Scale parameter array.
            wavelet (class, optional) :
                Mother wavelet class. Default is Mexican_hat()

        RETURNS

        EXAMPLE

        """

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
        """
        Inverse bi-dimensional continuous wavelet transform as in Wang and
        Lu (2010), equation [5].

        PARAMETERS
            W (array like):
                Wavelet transform, the result of the cwt2d function.
            scales (array like, optional):
                Scale parameter array.

        RETURNS
            iW (array like) :
                Inverse wavelet transform.

        EXAMPLE

        """
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

    def run(self):
        if self.nan_flag:
            self.astropy_cwt2d()
        else:
            self.cwt2d()


class Wavelet_Distance(object):

    """

    docstring for Wavelet_Distance


    INPUTS
    ------

    dataset1 - tuple
               Contains FITS image [0] and FITS header [1].

    dataset2 - tuple
               See above.

    wavelet - class
              Wavelet class. Only Mexican_hat() is implemented.

    distance - float OR list
               If float, the distance to the two datasets are the same.
               If list, it contains the two distances for each dataset.
               If no distance is provided, pixel units are used.
    """

    def __init__(self, dataset1, dataset2, wavelet=Mexican_hat(),
                 distance=None, scales=None, num=50, dx=0.25, dy=0.25,
                 fiducial_model=None):
        super(Wavelet_Distance, self).__init__()

        self.array1 = dataset1[0]
        self.array2 = dataset2[0]
        self.wavelet = wavelet
        if scales is None:
            a_min = round((5. / 3.), 3)  # Smallest scale given by paper
            self.scales1 = np.logspace(
                np.log10(a_min), np.log10(min(self.array1.shape)), num)
            self.scales2 = np.logspace(
                np.log10(a_min), np.log10(min(self.array2.shape)), num)
        else:
            self.scales1 = scales
            self.scales2 = scales

        if distance is None:
            self.imgscale1 = 1.0
            self.imgscale2 = 1.0
        else:
            if isinstance(distance, list):
                self.imgscale1 = np.abs(
                    dataset1[1]["CDELT2"]) * (np.pi / 180.0) * distance[0]
                self.imgscale2 = np.abs(
                    dataset2[1]["CDELT2"]) * (np.pi / 180.0) * distance[1]
            else:
                self.imgscale1 = np.abs(
                    dataset1[1]["CDELT2"]) * (np.pi / 180.0) * distance
                self.imgscale2 = np.abs(
                    dataset2[1]["CDELT2"]) * (np.pi / 180.0) * distance

        if fiducial_model is None:
            self.wt1 = wt2D(self.array1, self.scales1, wavelet=wavelet)
            self.wt1.run()
        else:
            self.wt1 = fiducial_model

        self.wt2 = wt2D(self.array2, self.scales2, wavelet=wavelet)
        self.wt2.run()

        self.curve1 = None
        self.curve2 = None

        self.results = None
        self.distance = None

    def distance_metric(self, non_linear=True, verbose=False):
        '''

        Implements the distance metric for 2 wavelet transforms.
        We fit the linear portion of the transform to represent the powerlaw
        A statistical comparison is used on the powerlaw indexes.

        INPUTS
        ------
        curve1 - array
                 results of the wavelet transform
                 column 1 is log10 T_g (transform)
                 column 0 is log10 a (scales)

        curve2 - array
                 comparator to curve1 - same form

        non_linear - bool
                     flag if portion of data is non-linear
                     runs the clip_to_linear function to only use the linear
                     portion in the model

        OUTPUTS
        -------
        distance - float
                   result of the distance metric

        '''

        self.curve1 = transform((self.wt1.Wf, self.scales1), self.imgscale1)
        self.curve2 = transform((self.wt2.Wf, self.scales2), self.imgscale2)

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

    INPUTS
    ------

    data      - array
              - x and y data

    threshold - float
                acceptable value of the second derivative to be called linear

    kernel_width - float
                   kernel width set to this percentage of the data length

    ends_clipped - float
                   Percentage of data to clip off at the ends.
                   End points have residual effects from the convolution.

    OUTPUTS
    -------

    data_clipped - array
                   Linear portion of the data set returned

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
    This reduces the dataset to 1D

    INPUTS
    ------

    data   - tuple
           [0] - N arrays from the transform
           [1] - scales of the transform

    OUTPUTS
    -------

    data_1D - array
            scales in first column, log <T_g> in the second

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
