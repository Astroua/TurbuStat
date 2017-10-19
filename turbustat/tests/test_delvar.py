# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division


import numpy.testing as npt
import astropy.units as u

from ..statistics import DeltaVariance, DeltaVariance_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_DelVar_method():
    # There is a difference in the convolution of astropy 1.x and 2.x on the
    # large-scales. Restrict the fitting region to where the convolution
    # agrees.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run(xhigh=11. * u.pix)
    # The slice is again to restrict where the convolution functions both give
    # the same value
    npt.assert_allclose(tester.delta_var[:-7],
                        computed_data['delvar_val'][:-7])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope'])


def test_DelVar_method_fill():
    # There is a difference in the convolution of astropy 1.x and 2.x on the
    # large-scales. Restrict the fitting region to where the convolution
    # agrees.
    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1["moment0_error"][0])
    tester.run(boundary='fill', xhigh=11. * u.pix)
    # The slice is again to restrict where the convolution functions both give
    # the same value
    npt.assert_allclose(tester.delta_var[:-7],
                        computed_data['delvar_fill_val'][:-7])
    npt.assert_almost_equal(tester.slope, computed_data['delvar_fill_slope'])


def test_DelVar_method_fitlimits():

    distance = 250 * u.pc

    xlow = 4 * u.pix
    xhigh = 30 * u.pix

    tester = DeltaVariance(dataset1["moment0"])
    tester.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.value * (dataset1['moment0'][1]['CDELT2'] * u.deg)
    xhigh = xhigh.value * (dataset1['moment0'][1]['CDELT2'] * u.deg)

    tester2 = DeltaVariance(dataset1["moment0"])
    tester2.run(xlow=xlow, xhigh=xhigh)

    xlow = xlow.to(u.rad).value * distance
    xhigh = xhigh.to(u.rad).value * distance

    tester3 = DeltaVariance(dataset1["moment0"], distance=distance)
    tester3.run(xlow=xlow, xhigh=xhigh)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_DelVar_method_wbrk():

    tester = \
        DeltaVariance(dataset1["moment0"],
                      weights=dataset1['moment0_error'][0])
    tester.run(xhigh=11 * u.pix, brk=6 * u.pix)

    npt.assert_almost_equal(tester.slope, computed_data['delvar_slope_wbrk'])
    npt.assert_almost_equal(tester.brk.value, computed_data['delvar_brk'])


def test_DelVar_distance():
    tester_dist = \
        DeltaVariance_Distance(dataset1["moment0"],
                               dataset2["moment0"],
                               weights1=dataset1["moment0_error"][0],
                               weights2=dataset2["moment0_error"][0],
                               xhigh=11 * u.pix)
    tester_dist.distance_metric()
    npt.assert_almost_equal(tester_dist.curve_distance,
                            computed_distances['delvar_curve_distance'],
                            decimal=3)
    npt.assert_almost_equal(tester_dist.slope_distance,
                            computed_distances['delvar_slope_distance'],
                            decimal=3)
