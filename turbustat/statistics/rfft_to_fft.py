
import numpy as np

'''
Reconstruct FFT output from RFFT in order to save memory
'''


def rfft_to_fft(image):
    '''

    '''

    ndim = len(image.shape)

    if ndim < 2 or ndim > 3:
        raise TypeError("Dimension of image must be 2D or 3D.")

    last_dim = image.shape[-1]

    fft_abs = np.abs(np.fft.rfftn(image))

    if ndim == 2:
        if last_dim % 2 == 0:
            fftstar_abs = fft_abs.copy()[:, -2:0:-1]
        else:
            fftstar_abs = fft_abs.copy()[:, -1:0:-1]

        fftstar_abs[1::, :] = fftstar_abs[:0:-1, :]

        return np.concatenate((fft_abs, fftstar_abs), axis=1)

    elif ndim == 3:
        if last_dim % 2 == 0:
            fftstar_abs = fft_abs.copy()[:, :, -2:0:-1]
        else:
            fftstar_abs = fft_abs.copy()[:, :, -1:0:-1]

        fftstar_abs[1::, :, :] = fftstar_abs[:0:-1, :, :]
        fftstar_abs[:, 1::, :] = fftstar_abs[:, :0:-1, :]

        return np.concatenate((fft_abs, fftstar_abs), axis=2)
