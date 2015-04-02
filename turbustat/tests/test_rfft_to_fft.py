
from turbustat.statistics.rfft_to_fft import rfft_to_fft
from ._testing_data import dataset1


import numpy as np
import numpy.testing as npt
from unittest import TestCase


class testRFFT(TestCase):
    """docstring for testRFFT"""
    def __init__(self):
        self.dataset1 = dataset1

        self.comp_rfft = rfft_to_fft(self.dataset1)

    def rfft_to_rfft(self):
        test_rfft = np.abs(np.fft.rfftn(self.dataset1))

        shape2 = test_rfft.shape[-1]

        npt.assert_allclose(test_rfft, self.comp_rfft[:, :, :shape2+1])

    def fft_to_rfft(self):
        test_fft = np.abs(np.fft.fftn(self.dataset1))

        npt.assert_allclose(test_fft, self.comp_rfft)
