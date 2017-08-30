# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import numpy as np
import numpy.testing as npt
import astropy.units as u

from ..statistics import Wavelet, Wavelet_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_Wavelet_method():
    tester = Wavelet(dataset1["moment0"])
    tester.run()
    npt.assert_almost_equal(tester.values, computed_data['wavelet_val'])

    npt.assert_almost_equal(tester.slope, computed_data['wavelet_slope'])


def test_Wavelet_method_fitlimits():

    distance = 250 * u.pc

    xlow = 2 * u.pix
    xhigh = 10 * u.pix

    tester = Wavelet(dataset1["moment0"])
    tester.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.value * dataset1['moment0'][1]['CDELT2'] * u.deg
    xhigh = xhigh.value * dataset1['moment0'][1]['CDELT2'] * u.deg

    tester2 = Wavelet(dataset1["moment0"])
    tester2.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = Wavelet(dataset1["moment0"], distance=distance)
    tester3.run(xlow=xlow, xhigh=xhigh)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_Wavelet_method_customscales():

    distance = 250 * u.pc

    scales = np.arange(1, 11, 2) * u.pix

    tester = Wavelet(dataset1["moment0"], scales=scales)
    tester.run()

    scales = scales.value * dataset1['moment0'][1]['CDELT2'] * u.deg

    tester2 = Wavelet(dataset1["moment0"], scales=scales)
    tester2.run()

    scales = scales.to(u.rad).value * distance

    tester3 = Wavelet(dataset1["moment0"], scales=scales, distance=distance)
    tester3.run()

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_Wavelet_distance():
    tester_dist = \
        Wavelet_Distance(dataset1["moment0"],
                         dataset2["moment0"]).distance_metric()
    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['wavelet_distance'])
