# Licensed under an MIT open source license - see LICENSE
from __future__ import print_function, absolute_import, division

import pytest

import numpy as np
import numpy.testing as npt
import astropy.units as u
import os

from ..statistics import StatMoments, StatMoments_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_moments():
    tester = StatMoments(dataset1["moment0"])
    tester.run(verbose=True, save_name='test.png')
    os.system("rm test.png")

    assert np.allclose(tester.kurtosis_hist[1],
                       computed_data['kurtosis_nondist_val'])
    assert np.allclose(tester.skewness_hist[1],
                       computed_data['skewness_nondist_val'])

    # TODO: Add more test comparisons. Save the total moments over the whole
    # arrays, portions of the local arrays, and the histogram values.

    # Test loading and saving
    tester.save_results("statmom_output.pkl", keep_data=False)

    saved_tester = StatMoments.load_results("statmom_output.pkl")

    # Remove the file
    os.remove("statmom_output.pkl")

    assert np.allclose(saved_tester.kurtosis_hist[1],
                       computed_data['kurtosis_nondist_val'])
    assert np.allclose(saved_tester.skewness_hist[1],
                       computed_data['skewness_nondist_val'])


def test_moments_units():

    distance = 250 * u.pc

    radius = 5 * u.pix

    tester = StatMoments(dataset1["moment0"], radius=radius)
    tester.run()

    # Angular units
    radius = radius.value * dataset1['moment0'][1]['CDELT2'] * u.deg
    tester2 = StatMoments(dataset1["moment0"], radius=radius)
    tester2.run()

    # Physical units
    radius = radius.to(u.rad).value * distance
    tester3 = StatMoments(dataset1["moment0"], radius=radius,
                          distance=distance)
    tester3.run()

    npt.assert_allclose(tester.mean, tester2.mean)
    npt.assert_allclose(tester.mean, tester3.mean)

    npt.assert_allclose(tester.variance, tester2.variance)
    npt.assert_allclose(tester.variance, tester3.variance)

    npt.assert_allclose(tester.skewness, tester2.skewness)
    npt.assert_allclose(tester.skewness, tester3.skewness)

    npt.assert_allclose(tester.kurtosis, tester2.kurtosis)
    npt.assert_allclose(tester.kurtosis, tester3.kurtosis)


def test_moments_nonperiodic():

    tester = StatMoments(dataset1["moment0"])
    tester.run(periodic=False)

    assert np.allclose(tester.kurtosis_hist[1],
                       computed_data['kurtosis_nonper_val'])
    assert np.allclose(tester.skewness_hist[1],
                       computed_data['skewness_nonper_val'])


def test_moment_distance():
    tester_dist = \
        StatMoments_Distance(dataset1["moment0"],
                             dataset2["moment0"])
    tester_dist.distance_metric(verbose=True, save_name='test.png')
    os.system("rm test.png")

    assert np.allclose(tester_dist.moments1.kurtosis_hist[1],
                       computed_data['kurtosis_val'])
    assert np.allclose(tester_dist.moments1.skewness_hist[1],
                       computed_data['skewness_val'])

    npt.assert_almost_equal(tester_dist.kurtosis_distance,
                            computed_distances['kurtosis_distance'])
    npt.assert_almost_equal(tester_dist.skewness_distance,
                            computed_distances['skewness_distance'])

    # With StatMoments classes as inputs

    tester_dist2 = \
        StatMoments_Distance(tester_dist.moments1,
                             tester_dist.moments2)
    tester_dist2.distance_metric()

    assert np.allclose(tester_dist2.moments1.kurtosis_hist[1],
                       computed_data['kurtosis_val'])
    assert np.allclose(tester_dist2.moments1.skewness_hist[1],
                       computed_data['skewness_val'])

    npt.assert_almost_equal(tester_dist2.kurtosis_distance,
                            computed_distances['kurtosis_distance'])
    npt.assert_almost_equal(tester_dist2.skewness_distance,
                            computed_distances['skewness_distance'])

    # With fresh StatMoments classes
    tester = StatMoments(dataset1["moment0"])
    tester2 = StatMoments(dataset2["moment0"])

    tester_dist3 = \
        StatMoments_Distance(tester,
                             tester2)
    tester_dist3.distance_metric()

    assert np.allclose(tester_dist3.moments1.kurtosis_hist[1],
                       computed_data['kurtosis_val'])
    assert np.allclose(tester_dist3.moments1.skewness_hist[1],
                       computed_data['skewness_val'])

    npt.assert_almost_equal(tester_dist3.kurtosis_distance,
                            computed_distances['kurtosis_distance'])
    npt.assert_almost_equal(tester_dist3.skewness_distance,
                            computed_distances['skewness_distance'])
