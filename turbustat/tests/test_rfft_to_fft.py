# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

from ..statistics.rfft_to_fft import rfft_to_fft
from ._testing_data import dataset1


import numpy as np
import numpy.testing as npt

try:
    import pyfftw
    PYFFTW_INSTALLED = True
except ImportError:
    PYFFTW_INSTALLED = False


def test_rfft_to_rfft():

    comp_rfft = rfft_to_fft(dataset1['moment0'][0])

    test_rfft = np.abs(np.fft.rfftn(dataset1['moment0'][0]))

    shape2 = test_rfft.shape[-1]

    npt.assert_allclose(test_rfft, comp_rfft[:, :shape2])


def test_fft_to_rfft():
    comp_rfft = rfft_to_fft(dataset1['moment0'][0])

    test_fft = np.abs(np.fft.fftn(dataset1['moment0'][0]))

    npt.assert_allclose(test_fft, comp_rfft)


@pytest.mark.skipif("not PYFFTW_INSTALLED")
def test_fftw():
    comp_rfft = rfft_to_fft(dataset1['moment0'][0])

    comp_rfft_fftw = rfft_to_fft(dataset1['moment0'][0], use_pyfftw=True,
                                 threads=1)

    test_fft = np.abs(np.fft.fftn(dataset1['moment0'][0]))

    npt.assert_allclose(test_fft, comp_rfft_fftw)
    npt.assert_allclose(comp_rfft, comp_rfft_fftw)
