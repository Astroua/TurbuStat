# Licensed under an MIT open source license - see LICENSE


'''
Test functions for PSpec
'''

import numpy.testing as npt
import astropy.units as u

from ..statistics import PowerSpectrum, PSpec_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_PSpec_method():
    tester = \
        PowerSpectrum(dataset1["moment0"])
    tester.run()
    npt.assert_allclose(tester.ps1D, computed_data['pspec_val'])


def test_Pspec_method_fitlimits():

    distance = 250 * u.pc

    low_cut = 0.02 / u.pix
    high_cut = 0.1 / u.pix

    tester = PowerSpectrum(dataset1["moment0"])
    tester.run(low_cut=low_cut, high_cut=high_cut)

    low_cut = low_cut.value / (dataset1['moment0'][1]['CDELT2'] * u.deg)
    high_cut = high_cut.value / (dataset1['moment0'][1]['CDELT2'] * u.deg)

    tester2 = PowerSpectrum(dataset1["moment0"])
    tester2.run(low_cut=low_cut, high_cut=high_cut)

    low_cut = low_cut.to(u.rad**-1).value / distance
    high_cut = high_cut.to(u.rad**-1).value / distance

    tester3 = PowerSpectrum(dataset1["moment0"], distance=distance)
    tester3.run(low_cut=low_cut, high_cut=high_cut)

    npt.assert_allclose(tester.slope, tester2.slope)
    npt.assert_allclose(tester.slope, tester3.slope)


def test_PSpec_distance():
    tester_dist = \
        PSpec_Distance(dataset1["moment0"],
                       dataset2["moment0"])
    tester_dist.distance_metric()

    npt.assert_almost_equal(tester_dist.distance,
                            computed_distances['pspec_distance'])


def test_pspec_nonequal_shape():

    mom0_sliced = dataset1["moment0"][0][:16, :]
    mom0_hdr = dataset1["moment0"][1]

    test = PowerSpectrum((mom0_sliced, mom0_hdr)).run()
    test_T = PowerSpectrum((mom0_sliced.T, mom0_hdr)).run()

    npt.assert_almost_equal(test.slope, test_T.slope, decimal=7)
