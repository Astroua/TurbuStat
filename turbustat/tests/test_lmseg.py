# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest
import numpy as np
import numpy.testing as npt
from astropy.utils.misc import NumpyRNGContext


from ..statistics.lm_seg import Lm_Seg


def test_lmseg():
    x = np.linspace(0, 10, 200)

    with NumpyRNGContext(12129):
        y = 2 + 2 * x * (x < 5) + (5 * x - 15) * (x >= 5) + \
            np.random.normal(0, 0.1, 200)

    model = Lm_Seg(x, y, 3)
    model.fit_model(tol=1e-3, verbose=False)

    npt.assert_approx_equal(5.0, model.brk, significant=2)
    npt.assert_allclose([2.0, 5.0], model.slopes, rtol=0.1)


def test_lmseg_weighted():
    x = np.linspace(0, 10, 200)

    with NumpyRNGContext(12129):
        yerr = np.random.normal(0, 0.1, 200)
        y = 2 + 2 * x * (x < 5) + (5 * x - 15) * (x >= 5) + \
            yerr

    model = Lm_Seg(x, y, 3, weights=yerr**-2)
    model.fit_model(tol=1e-3, verbose=True)

    npt.assert_approx_equal(5.0, model.brk, significant=2)
    npt.assert_allclose([2.0, 5.0], model.slopes, rtol=0.1)
