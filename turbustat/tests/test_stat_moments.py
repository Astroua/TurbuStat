# Licensed under an MIT open source license - see LICENSE


'''
Test functions for Kurtosis
'''

import pytest

import numpy as np
import numpy.testing as npt

from ..statistics import StatMoments, StatMoments_Distance
from ._testing_data import \
    dataset1, dataset2, computed_data, computed_distances


def test_moments():
    tester = StatMoments(dataset1["moment0"])
    tester.run()

    # TODO: Add more test comparisons. Save the total moments over the whole
    # arrays, portions of the local arrays, and the histogram values.


def test_moments_units():
    pass


def test_moments_nonperiodic():
    pass


def test_moments_custombins():
    pass


def test_moment_distance():
    tester_dist = \
        StatMoments_Distance(dataset1["moment0"],
                             dataset2["moment0"])
    tester_dist.distance_metric()

    assert np.allclose(tester_dist.moments1.kurtosis_hist[1],
                       computed_data['kurtosis_val'])
    assert np.allclose(tester_dist.moments1.skewness_hist[1],
                       computed_data['skewness_val'])

    npt.assert_almost_equal(tester_dist.kurtosis_distance,
                            computed_distances['kurtosis_distance'])
    npt.assert_almost_equal(tester_dist.skewness_distance,
                            computed_distances['skewness_distance'])
